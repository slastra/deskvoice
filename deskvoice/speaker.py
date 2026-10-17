"""The desk's one speaker: channels, priority, render-ahead, playback, stop.

An utterance is an id with one or more sentences. `say` adds a sentence and
`final` (or a later `end`) closes it, so a client can stream sentences as a
model writes them. Every sentence is rendered as soon as it arrives, assistant
sentences first, so the next one is ready while the current one plays.

Priority: an assistant utterance always goes next. A cue that's playing
finishes its sentence and then waits behind it. A cue that hasn't started
within its max_wait is dropped, and the caller is told, so a client never
claims to have said something it didn't.
"""

import itertools
import logging
import queue
import subprocess
import threading
import time
from collections import deque

import numpy as np

from .protocol import CHANNELS

log = logging.getLogger("deskvoice")

PRIORITY = {ch: i for i, ch in enumerate(CHANNELS)}
STREAM_IDLE = 30.0     # an open utterance with no new sentence for this long is treated as ended


class Sentence:
    __slots__ = ("text", "samples", "rate", "error", "ready")

    def __init__(self, text):
        self.text = text
        self.samples = None
        self.rate = 0
        self.error = None
        self.ready = threading.Event()

    @property
    def secs(self):
        return len(self.samples) / self.rate if self.samples is not None and self.rate else 0.0


class Utterance:
    def __init__(self, uid, channel, voice, max_wait, notify, now):
        self.id, self.channel, self.voice = uid, channel, voice
        self.max_wait = max_wait          # None: wait as long as it takes
        self.notify = notify
        self.created = self.last_activity = now
        self.sentences = []
        self.next = 0                     # index of the next sentence to play
        self.closed = self.started = self.done = self.cancelled = False

    def emit(self, event, **fields):
        try:
            self.notify({"event": event, "id": self.id, **fields})
        except Exception:                 # a gone client must never stall the speaker
            log.exception("notify %s", event)


class PwPlay:
    """Plays float32 mono samples through pw-play, tagged so ducking can tell
    deskvoice's own streams from everyone else's."""

    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None

    def play(self, samples, rate):
        proc = subprocess.Popen(
            ["pw-play", "--raw", "--format", "f32", "--rate", str(rate), "--channels", "1",
             "-P", "{ application.name = deskvoice }", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with self._lock:
            self._proc = proc
        try:
            proc.stdin.write(np.ascontiguousarray(samples, dtype=np.float32).tobytes())
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass                          # killed mid-sentence by stop()
        proc.wait()
        with self._lock:
            self._proc = None

    def kill(self):
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.kill()


class Speaker:
    def __init__(self, render, player=None, clock=time.monotonic):
        self.render = render              # (voice, text) -> (float32 samples, rate)
        self.player = player or PwPlay()
        self.on_busy = lambda busy: None  # the ducker listens: True when speech starts, False when it all stops
        self.on_said = lambda channel, voice, text: None   # an utterance that was heard, for "recent"
        self.voices = {"assistant": "", "cue": ""}   # each channel's voice: the configured one, then the last used
        self._quiet_at = 0.0              # when speech last stopped, for speaking()'s tail
        self.clock = clock
        self.cv = threading.Condition()
        self.queues = {ch: deque() for ch in CHANNELS}   # utterances waiting to (re)start
        self.by_id = {}
        self.current = None
        self._renders = queue.PriorityQueue()
        self._seq = itertools.count()
        self._stopping = False
        self._threads = [threading.Thread(target=f, daemon=True, name=f.__name__)
                         for f in (self._render_loop, self._play_loop, self._janitor_loop)]

    def start(self):
        for t in self._threads:
            t.start()

    def close(self):
        with self.cv:
            self._stopping = True
            self.cv.notify_all()
        self.player.kill()

    # --- requests ---------------------------------------------------------

    def say(self, uid, text, voice, channel, final=True, max_wait=None, notify=lambda e: None):
        """Add a sentence to utterance `uid`, creating it if new."""
        if channel not in PRIORITY:
            raise ValueError(f"unknown channel {channel!r}")
        with self.cv:
            u = self.by_id.get(uid)
            if u is None:
                self.voices[channel] = voice
                u = Utterance(uid, channel, voice, max_wait, notify, self.clock())
                self.by_id[uid] = u
                self.queues[channel].append(u)
                u.emit("queued")
            elif u.closed:
                raise ValueError(f"utterance {uid} has already ended")
            s = Sentence(text)
            u.sentences.append(s)
            u.last_activity = self.clock()
            if final:
                u.closed = True
            self._renders.put((PRIORITY[u.channel], next(self._seq), u, s))
            self.cv.notify_all()

    def end(self, uid):
        with self.cv:
            u = self.by_id.get(uid)
            if u:
                u.closed = True
                self.cv.notify_all()

    def stop(self, channel=None, uid=None):
        """Cut matching utterances short: the playing one at once, queued ones
        before they start. Returns how many were stopped."""
        with self.cv:
            hits = [u for u in self.by_id.values()
                    if (uid is None or u.id == uid) and (channel is None or u.channel == channel)]
            for u in hits:
                self._finish(u, "stopped")
            if self.current in hits:
                self.player.kill()
            self.cv.notify_all()
            return len(hits)

    def speaking(self, tail=0.3):
        """Is anything playing, or did it stop within `tail` seconds? The listener
        uses it to hold the wake word to its stricter barge-in threshold."""
        return self.current is not None or self.clock() - self._quiet_at < tail

    def status(self):
        with self.cv:
            cur = self.current
            return {
                "playing": {"id": cur.id, "channel": cur.channel, "voice": cur.voice,
                            "text": cur.sentences[min(cur.next, len(cur.sentences) - 1)].text if cur.sentences else ""}
                if cur and cur.started else None,
                "queued": {ch: len(q) for ch, q in self.queues.items()},
                "voices": dict(self.voices),
            }

    # --- internals (call with self.cv held) ---------------------------------

    def _finish(self, u, event, **fields):
        if u.done:
            return
        u.done = True
        u.cancelled = event != "finished"
        for q in self.queues.values():
            if u in q:
                q.remove(u)
        self.by_id.pop(u.id, None)
        u.emit(event, **fields)
        if u.started and event in ("finished", "stopped"):
            said = " ".join(x.text for x in u.sentences[:max(u.next, 1)])
            try:
                self.on_said(u.channel, u.voice, said)
            except Exception:
                log.exception("on_said")

    def _next_utterance(self):
        for ch in CHANNELS:
            for u in self.queues[ch]:
                if u.sentences:
                    return u
        return None

    def _assistant_waiting(self):
        return any(u.sentences for u in self.queues["assistant"])

    # --- workers ------------------------------------------------------------

    def _render_loop(self):
        while True:
            _, _, u, s = self._renders.get()
            if not u.cancelled and not any(c.isalnum() for c in s.text):
                s.samples, s.rate = np.zeros(0, dtype=np.float32), 24000   # "---", "...": nothing to say
            elif not u.cancelled:
                try:
                    s.samples, s.rate = self.render(u.voice, s.text)
                except Exception as e:  # unknown voice, engine failure: tell the caller
                    log.warning("render %s %r: %s", u.voice, s.text[:40], e)
                    s.error = str(e) or type(e).__name__
            s.ready.set()
            with self.cv:
                self.cv.notify_all()

    def _play_loop(self):
        while True:
            with self.cv:
                while not self._stopping and (u := self._next_utterance()) is None:
                    self.cv.wait(0.5)
                if self._stopping:
                    return
                self.queues[u.channel].remove(u)
                self.current = u
            self.on_busy(True)
            try:
                self._play(u)
            except Exception:
                log.exception("playing %s", u.id)
                with self.cv:
                    self._finish(u, "error", message="playback failed")
            with self.cv:
                self.current = None
                idle = self._next_utterance() is None
                self.cv.notify_all()
            if idle:
                self._quiet_at = self.clock()
                self.on_busy(False)

    def _play(self, u):
        errors = []
        while True:
            with self.cv:
                while u.next >= len(u.sentences) and not u.closed and not u.cancelled:
                    if self.clock() - u.last_activity > STREAM_IDLE:
                        u.closed = True
                        break
                    self.cv.wait(0.2)
                if u.cancelled:
                    return
                if u.next >= len(u.sentences):
                    if u.started or not errors:
                        self._finish(u, "finished")   # (or there was nothing pronounceable)
                    else:                 # every sentence failed to render
                        self._finish(u, "error", message="; ".join(errors))
                    return
                # A cue between sentences steps aside for an assistant reply.
                if u.channel != "assistant" and u.started and self._assistant_waiting():
                    self.queues[u.channel].appendleft(u)
                    return
                s = u.sentences[u.next]
            while not s.ready.wait(0.1):
                if u.cancelled:
                    return
            if u.cancelled:
                return
            if s.error:
                errors.append(s.error)
                u.next += 1
                continue
            if len(s.samples) == 0:
                u.next += 1
                continue
            with self.cv:
                if not u.started:
                    u.started = True
                    u.emit("started", secs=round(s.secs, 2))
            self.player.play(s.samples, s.rate)
            with self.cv:
                u.next += 1
                u.last_activity = self.clock()

    def _janitor_loop(self):
        """Drop cues that waited too long to start, while something else plays."""
        while True:
            time.sleep(0.1)
            with self.cv:
                if self._stopping:
                    return
                now = self.clock()
                for ch in CHANNELS:
                    for u in list(self.queues[ch]):
                        if not u.started and u.max_wait is not None and now - u.created > u.max_wait:
                            self._finish(u, "dropped", reason="busy")
