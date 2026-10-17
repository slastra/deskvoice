"""Lower everything else that's playing while deskvoice speaks or listens.

Streams are remembered by application name, not sink-input index: Chrome
and others tear a stream down on pause and make a new one on play, and
PipeWire's stream-restore hands the new stream the last volume written for
that application. So originals are keyed by application, and every write
re-lists the live streams.

Two things want audio ducked:
  playing  - any channel is speaking (set by the speaker)
  holds    - a deadline from a client ("listening for 16 s", "thinking for
             90 s", "follow-up window"); hold(0) releases
Audio is ducked while either wants it, and comes back exactly afterwards.
"""

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger("deskvoice")

MAX_HOLD = 120.0        # a lost "done" costs a quiet song for two minutes, never the evening
MAX_DUCKED = 180.0      # belt and braces: nothing keeps audio down longer than this
TAIL = 0.3              # after the last sentence, before audio comes back
# deskvoice's own streams (speech, cues) are never ducked; config adds more.
ALWAYS_EXCLUDE = ("deskvoice",)


def pactl_streams():
    out = subprocess.run(["pactl", "-f", "json", "list", "sink-inputs"],
                         capture_output=True, timeout=2, check=True).stdout
    return json.loads(out)


def pactl_set(index, pct):
    subprocess.run(["pactl", "set-sink-input-volume", str(index), f"{pct}%"],
                   capture_output=True, timeout=2)


def app_of(s):
    return (s.get("properties", {}).get("application.name") or "").lower()


def volume_of(s):
    for ch in (s.get("volume") or {}).values():
        return ch.get("value", 65536) / 65536
    return 1.0


def fading_file(path):
    """A check for another program that fades the same volumes (a presence
    sensor dimming the desk, say): while its JSON status file says "fade" > 0,
    two owners writing the same volumes would leave them wrong, so deskvoice
    stands aside. No path: never."""
    if not path:
        return lambda: False
    p = Path(path).expanduser()

    def fading():
        try:
            return json.loads(p.read_text()).get("fade", 0) > 0
        except (OSError, ValueError):
            return False
    return fading


class Ducker:
    def __init__(self, level=0.3, exclude=(), list_streams=pactl_streams, set_volume=pactl_set,
                 fading=lambda: False, clock=time.monotonic, sleep=time.sleep):
        self.level = level
        self.exclude = tuple(e.lower() for e in (*ALWAYS_EXCLUDE, *exclude))
        self._list, self._set, self._fading = list_streams, set_volume, fading
        self.clock, self.sleep = clock, sleep
        # Two locks: _in guards what clients ask for and is never held for long;
        # _lock guards the volume work, which sleeps through ramps (~0.7 s). A
        # request must never wait on a ramp: the socket loop calls hold().
        self._in = threading.Lock()
        self._lock = threading.Lock()
        self._playing = False
        self._quiet_since = None           # when playing last went false (for the tail)
        self._hold_until = 0.0
        self._orig = None                  # app -> original volume while ducked
        self._ducked_at = 0.0
        self._wake = threading.Event()
        self._stopping = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ducker")

    def start(self):
        self._thread.start()

    def close(self):
        self._stopping = True
        self._wake.set()
        self._thread.join(3)
        with self._lock:
            self._restore()

    # --- inputs -----------------------------------------------------------

    def playing(self, on):
        with self._in:
            if on == self._playing:
                return
            self._playing = on
            self._quiet_since = None if on else self.clock()
        self._wake.set()

    def hold(self, seconds):
        """Keep audio ducked for `seconds` from now; 0 releases the hold."""
        with self._in:
            self._hold_until = self.clock() + min(max(seconds, 0), MAX_HOLD) if seconds > 0 else 0.0
        self._wake.set()

    def holding(self):
        """A client hold (listening, thinking) is in force."""
        with self._in:
            return self.clock() < self._hold_until

    def status(self):
        orig = self._orig                  # a snapshot; the ramp may be mid-way
        with self._in:
            hold = max(0.0, round(self._hold_until - self.clock(), 1)) if self._hold_until else 0
        return {"ducked": bool(orig), "level": self.level, "apps": sorted(orig or {}), "hold": hold}

    # --- the loop ---------------------------------------------------------

    def _wanted(self, now):
        tail = self._quiet_since is not None and now - self._quiet_since < TAIL
        return self._playing or tail or now < self._hold_until

    def _loop(self):
        while not self._stopping:
            self._wake.wait(0.1)
            self._wake.clear()
            with self._lock:
                self.tick()

    def tick(self):
        """One evaluation (work lock held). Public for tests."""
        now = self.clock()
        with self._in:
            want = self._wanted(now)
        if want and self._orig is None:
            self._duck(now)
        elif self._orig is not None and (not want or now - self._ducked_at > MAX_DUCKED):
            if want:
                log.warning("duck: down for %.0f s, restoring anyway", now - self._ducked_at)
                with self._in:
                    self._hold_until, self._playing = 0.0, False
            self._restore()

    def _streams(self, corked=False):
        """Streams to duck: playing ones (corked=False), or every remembered
        app's, paused or not, when restoring. A stream paused while ducked
        must still get its volume back, or it resumes at 30%."""
        try:
            all_ = self._list()
        except Exception as e:
            log.warning("duck: pactl: %s", e)
            return []
        keep = []
        for s in all_:
            name = app_of(s)
            if (s.get("corked") and not corked) or not s.get("volume") or not name or any(e in name for e in self.exclude):
                continue
            keep.append(s)
        return keep

    def _set_all(self, frac, corked=False):
        for s in self._streams(corked):
            v = self._orig.get(app_of(s))
            if v is not None:
                self._set(s["index"], int(v * frac * 100 + 0.5))

    def _duck(self, now):
        self._orig = {}
        self._ducked_at = now
        if self._fading():
            return                         # {} marks "ducked" for this spell without touching anything
        for s in self._streams():
            self._orig.setdefault(app_of(s), volume_of(s))
        if not self._orig:
            return
        for i in (1, 2, 3):                # quick but not a click: ~120 ms
            self._set_all(1 - (1 - self.level) * i / 3)
            if i < 3:
                self.sleep(0.06)
        log.info("duck: lowered %s to %.0f%%", {k: round(v, 2) for k, v in self._orig.items()}, self.level * 100)

    def _restore(self):
        if not self._orig:
            self._orig = None
            return
        steps = 6
        for i in range(1, steps + 1):      # ~600 ms back up
            self._set_all(self.level + (1 - self.level) * i / steps, corked=True)
            self.sleep(0.1)
        orig = self._orig
        self._orig = None
        log.info("duck: restored %s", {k: round(v, 2) for k, v in orig.items()})

        # A player resumed while ducked may make its stream a moment later.
        def late():
            self.sleep(1.0)
            with self._lock:
                if self._orig is None:
                    self._orig = orig
                    self._set_all(1.0, corked=True)
                    self._orig = None
        threading.Thread(target=late, daemon=True).start()
