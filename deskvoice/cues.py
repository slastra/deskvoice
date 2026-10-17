"""Short sound cues for the listener: the wake word heard, dictation on, done,
an error, nothing heard, a follow-up window closing.

  builtin  deskvoice synthesizes two-note chirps itself (numpy, played with
           pw-play; tagged so ducking leaves them alone)
  osc      sends `/emotune <name>` over OSC/UDP to a cue daemon that plays
           recipes by name (emotune, or anything speaking that message)
  none     silent

Cues are fire-and-forget: a missing daemon or a failed pw-play costs a sound,
never a request.
"""

import socket
import subprocess
import threading

import numpy as np

RATE = 24000


def _note(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def _tone(freq, dur, soft=False, amp=0.16):
    t = np.arange(int(RATE * dur)) / RATE
    wave = np.sin(2 * np.pi * freq * t)
    if not soft:                      # a little brightness: odd harmonics, gently
        wave += 0.18 * np.sin(2 * np.pi * 3 * freq * t) + 0.06 * np.sin(2 * np.pi * 5 * freq * t)
    attack = np.minimum(1.0, t / 0.006)
    decay = np.exp(-t / (dur * (0.9 if soft else 0.55)))
    return (amp * wave * attack * decay).astype(np.float32)


def _two(a, b, step=0.08, ring=0.32, soft=False, amp=0.16):
    """Note a, then note b ringing out: the shape of every cue here."""
    first = _tone(_note(a), step * 2, soft, amp)
    second = _tone(_note(b), ring, soft, amp)
    out = np.zeros(int(RATE * step) + len(second), dtype=np.float32)
    out[:len(first)] += first[:len(out)]
    out[int(RATE * step):] += second
    return out


# name -> samples, built once
SOUNDS = {
    "listen": _two(76, 83),                           # rising fifth: I'm listening
    "dictate": _two(74, 86, step=0.09, soft=True, amp=0.2),   # warm rising octave: dictation on
    "ok": _two(72, 84),                               # octave leap: done
    "error": _two(60, 55, step=0.2, ring=0.5, soft=True, amp=0.24),   # falling: that failed
    "nomatch": _tone(_note(72), 0.24, soft=True, amp=0.14),          # one neutral note
    "bye": _two(76, 71, step=0.1, ring=0.3, soft=True, amp=0.12),   # soft falling fourth
}


def _osc(address, arg):
    def pad(b):
        return b + b"\0" * (4 - len(b) % 4)
    return pad(address.encode()) + pad(b",s") + pad(arg.encode())


class Cues:
    def __init__(self, cfg):
        self.backend = cfg.get("backend", "builtin")
        self.names = cfg.get("names", {})
        host, _, port = (cfg.get("osc") or "127.0.0.1:57120").rpartition(":")
        self.addr = (host or "127.0.0.1", int(port or 57120))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def play(self, event):
        """Play the cue for an event ("listen", "ok", ...)."""
        name = self.names.get(event, event)
        if self.backend == "osc":
            try:
                self.sock.sendto(_osc("/emotune", name), self.addr)
            except OSError:
                pass
        elif self.backend == "builtin":
            samples = SOUNDS.get(name)
            if samples is not None:
                threading.Thread(target=self._pw_play, args=(samples,), daemon=True).start()

    @staticmethod
    def _pw_play(samples):
        try:
            p = subprocess.Popen(
                ["pw-play", "--raw", "--format", "f32", "--rate", str(RATE), "--channels", "1",
                 "-P", "{ application.name = deskvoice-cue }", "-"],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            p.communicate(samples.tobytes(), timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
