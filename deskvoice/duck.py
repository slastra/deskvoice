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

What keeps it from getting stuck:
  - the originals are written to a state file while ducked, so a crash or a
    kill is undone on the next start (restore_saved)
  - after a restore, each app stays "pending" for ten minutes: a stream it
    makes later that shows up at the ducked volume (PipeWire remembering
    the last volume written for that app) is set back to the original
  - a stream whose volume someone changed while it was ducked is left as
    they set it, now and when it comes back
  - when another program starts fading the same volumes (yield_to), the duck
    is restored first and stands aside, so that program never records a
    ducked volume as the original
  - holds expire (two minutes), any duck ends after three, and force_restore
    (`deskvoice unduck`) undoes everything at once
"""

import json
import logging
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger("deskvoice")

MAX_HOLD = 120.0        # a lost "done" costs a quiet song for two minutes, never the evening
MAX_DUCKED = 180.0      # belt and braces: nothing keeps audio down longer than this
TAIL = 0.3              # after the last sentence, before audio comes back
PENDING = 600.0         # how long after a restore a reappearing stream is still put back
PENDING_CHECK = 2.0     # how often pending apps are looked for
SAME = 0.02             # volumes this close count as the one deskvoice set
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


def present_file(path, max_age=15.0):
    """A check for a presence sensor's JSON status file: is anyone at the desk?
    False only when the file is fresh (written in the last `max_age` seconds)
    and says the sensor works ("sensor_ok": true), knows ("known": true), and
    nobody is there ("present": false), unless keyboard or mouse input is
    live ("input": "active", deskpresence's last 20 s): the radar can miss
    someone sitting still. Missing, stale, unreadable or incomplete: True, so
    a broken sensor never deafens the wake word.
    No path: always True."""
    if not path:
        return lambda: True
    p = Path(path).expanduser()

    def present():
        try:
            if time.time() - p.stat().st_mtime > max_age:
                return True
            data = json.loads(p.read_text())
            return not (data.get("known") is True and data.get("sensor_ok") is True
                        and data.get("present") is False and data.get("input") != "active")
        except (OSError, ValueError, AttributeError):
            return True
    return present


class Ducker:
    def __init__(self, level=0.3, exclude=(), list_streams=pactl_streams, set_volume=pactl_set,
                 fading=lambda: False, clock=time.monotonic, sleep=time.sleep, state_file=None):
        self.level = level
        self.exclude = tuple(e.lower() for e in (*ALWAYS_EXCLUDE, *exclude))
        self._list, self._set, self._fading = list_streams, set_volume, fading
        self.clock, self.sleep = clock, sleep
        self.state_file = Path(state_file) if state_file else None
        # Two locks: _in guards what clients ask for and is never held for long;
        # _lock guards the volume work, which sleeps through ramps (~0.7 s). A
        # request must never wait on a ramp: the socket loop calls hold().
        self._in = threading.Lock()
        self._lock = threading.Lock()
        self._playing = False
        self._quiet_since = None           # when playing last went false (for the tail)
        self._hold_until = 0.0
        self._orig = None                  # app -> original volume while ducked ({} = ducked, nothing touched)
        self._ducked_at = 0.0
        self._pending = {}                 # app -> (original, ducked volume, until): put back if it reappears
        self._pending_checked = 0.0
        self._yielding = False             # another program is fading: stay out of its way
        self._wake = threading.Event()
        self._stopping = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ducker")

    def start(self):
        self.restore_saved()
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

    def is_playing(self):
        with self._in:
            return self._playing

    def force_restore(self):
        """Everything back, now: holds and the speaking flag cleared, the duck
        restored. (`deskvoice unduck`, and the watchdog's last resort.)"""
        with self._in:
            self._hold_until, self._playing, self._quiet_since = 0.0, False, None
        with self._lock:
            self._restore()

    def status(self):
        orig = self._orig                  # a snapshot; the ramp may be mid-way
        with self._in:
            hold = max(0.0, round(self._hold_until - self.clock(), 1)) if self._hold_until else 0
        return {"ducked": bool(orig), "level": self.level, "apps": sorted(orig or {}), "hold": hold,
                "pending": sorted(self._pending), "yielding": self._yielding}

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
        fading = self._fading()
        if fading and self._orig:
            log.info("duck: another program is fading volumes; restoring and standing aside")
            self._restore()
        self._yielding = fading
        if want and self._orig is None and not fading:
            self._duck(now)
        elif self._orig is not None and (not want or now - self._ducked_at > MAX_DUCKED):
            if want:
                log.warning("duck: down for %.0f s, restoring anyway", now - self._ducked_at)
                with self._in:
                    self._hold_until, self._playing = 0.0, False
            self._restore()
        if self._pending and self._orig is None and now - self._pending_checked >= PENDING_CHECK:
            self._pending_checked = now
            self._sweep_pending(now)

    # --- volumes ----------------------------------------------------------

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

    def _ducked_volume(self, orig):
        return int(orig * self.level * 100 + 0.5) / 100

    def _duck(self, now):
        self._orig = {}
        self._ducked_at = now
        for s in self._streams():
            app = app_of(s)
            # An app still pending from the last duck is recorded at its real
            # original, not at the ducked volume its new stream inherited.
            pend = self._pending.pop(app, None)
            v = volume_of(s)
            if pend and abs(v - pend[1]) <= SAME:
                v = pend[0]
            self._orig.setdefault(app, v)
        if not self._orig:
            return
        self._save()
        for i in (1, 2, 3):                # quick but not a click: ~120 ms
            frac = 1 - (1 - self.level) * i / 3
            for s in self._streams():
                v = self._orig.get(app_of(s))
                if v is not None:
                    self._set(s["index"], int(v * frac * 100 + 0.5))
            if i < 3:
                self.sleep(0.06)
        log.info("duck: lowered %s to %.0f%%", {k: round(v, 2) for k, v in self._orig.items()}, self.level * 100)

    def _restore(self):
        if not self._orig:
            self._orig = None
            self._unsave()
            return
        orig = self._orig
        # Streams someone turned up or down while ducked keep their volume.
        touched = set()
        for s in self._streams(corked=True):
            app = app_of(s)
            if app in orig and abs(volume_of(s) - self._ducked_volume(orig[app])) > SAME + 0.01:
                touched.add(s["index"])
        steps = 6
        for i in range(1, steps + 1):      # ~600 ms back up
            frac = self.level + (1 - self.level) * i / steps
            for s in self._streams(corked=True):
                v = orig.get(app_of(s))
                if v is not None and s["index"] not in touched:
                    self._set(s["index"], int(v * frac * 100 + 0.5))
            self.sleep(0.1)
        until = self.clock() + PENDING
        for app, v in orig.items():
            self._pending[app] = (v, self._ducked_volume(v), until)
        self._orig = None
        self._unsave()
        log.info("duck: restored %s%s", {k: round(v, 2) for k, v in orig.items()},
                 f" (left {len(touched)} stream(s) as you set them)" if touched else "")

    def _sweep_pending(self, now):
        """Streams that reappeared at the ducked volume go back to the original."""
        for app in [a for a, (_, _, until) in self._pending.items() if until < now]:
            del self._pending[app]
        if not self._pending:
            return
        for s in self._streams(corked=True):
            pend = self._pending.get(app_of(s))
            if pend and abs(volume_of(s) - pend[1]) <= SAME:
                self._set(s["index"], int(pend[0] * 100 + 0.5))
                log.info("duck: %s came back at the ducked volume; set to %.0f%%", app_of(s), pend[0] * 100)

    # --- surviving a crash -------------------------------------------------

    def _save(self):
        if not self.state_file:
            return
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps({"level": self.level, "orig": self._orig, "ts": time.time()}))
        except OSError as e:
            log.warning("duck: saving the originals: %s", e)

    def _unsave(self):
        if self.state_file:
            self.state_file.unlink(missing_ok=True)

    def restore_saved(self):
        """A duck left behind by a crash or a kill: put those apps back. Only
        streams still at the ducked volume are touched; the rest go pending."""
        if not self.state_file or not self.state_file.exists():
            return
        try:
            saved = json.loads(self.state_file.read_text())
            orig, level = saved.get("orig") or {}, saved.get("level", self.level)
        except (OSError, ValueError):
            self._unsave()
            return
        fixed = []
        for s in self._streams(corked=True):
            app = app_of(s)
            if app in orig and abs(volume_of(s) - int(orig[app] * level * 100 + 0.5) / 100) <= SAME:
                self._set(s["index"], int(orig[app] * 100 + 0.5))
                fixed.append(app)
        until = self.clock() + PENDING
        for app, v in orig.items():
            self._pending[app] = (v, int(v * level * 100 + 0.5) / 100, until)
        self._unsave()
        log.warning("duck: found a duck left by the last run; restored %s", sorted(set(fixed)) or "nothing yet (pending)")
