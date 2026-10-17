"""Voices: Kokoro held in-process (CUDA first), Piper as resident subprocesses.

Every voice renders text to float32 mono samples and a sample rate, so the
speaker plays them all the same way.

Voice names: a bare Kokoro style ("af_heart") or "piper:<model stem>"
("piper:en_GB-cori-high").
"""

import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("deskvoice")
# kokoro's phonemizer warns "words count mismatch" on most sentences; it's
# harmless (it compares word counts before and after phonemizing) and floods
# the journal. Its get_logger() resets the level on every call but never
# propagate, and its own handler is a NullHandler, so this keeps it quiet.
logging.getLogger("phonemizer").propagate = False


SPEAKERS = {
    "af": "American female", "am": "American male", "bf": "British female", "bm": "British male",
    "ef": "Spanish female", "em": "Spanish male", "ff": "French female", "hf": "Hindi female",
    "hm": "Hindi male", "if": "Italian female", "im": "Italian male", "jf": "Japanese female",
    "jm": "Japanese male", "pf": "Portuguese female", "pm": "Portuguese male",
    "zf": "Chinese female", "zm": "Chinese male",
}


class UnknownVoice(Exception):
    pass


class Kokoro:
    """One Kokoro model for the life of the process. CUDA is named explicitly
    when wanted: left to itself onnxruntime tries TensorRT first, fails
    noisily, and falls back."""

    def __init__(self, model, voices, provider="auto", warm_voice="af_heart"):
        self.model_path, self.voices_path = Path(model), Path(voices)
        self.provider, self.warm_voice = provider, warm_voice
        self._lock = threading.Lock()
        self._kokoro = None
        self.providers = []
        self.styles = self._styles()

    def available(self):
        return self.model_path.exists() and self.voices_path.exists()

    def _styles(self):
        if not self.available():
            return []
        with np.load(str(self.voices_path)) as data:   # 27 MB bank, never the 311 MB model
            return sorted(data.files)

    def load(self):
        import onnxruntime as ort
        from kokoro_onnx import Kokoro as K
        ort.set_default_logger_severity(3)
        have = ort.get_available_providers()
        want_cuda = self.provider != "cpu"
        providers = [p for p in ("CUDAExecutionProvider",) if want_cuda and p in have] + ["CPUExecutionProvider"]
        t = time.monotonic()
        session = ort.InferenceSession(str(self.model_path), providers=providers)
        self._kokoro = K.from_session(session, str(self.voices_path))
        self.providers = session.get_providers()
        log.info("kokoro loaded on %s in %.2fs (%d voices)", self.providers[0], time.monotonic() - t, len(self.styles))
        warm = self.warm_voice if self.warm_voice in self.styles else self.styles[0]
        self.render(warm, "Ready.")   # the first inference builds CUDA kernels; pay it now

    def render(self, style, text):
        if style not in self.styles:
            raise UnknownVoice(style)
        lang = "en-gb" if style.startswith("b") else "en-us"
        with self._lock:
            samples, rate = self._kokoro.create(text, voice=style, speed=1.0, lang=lang)
        return np.asarray(samples, dtype=np.float32), rate


class PiperVoice:
    """A resident `piper-tts --json-input`: each sentence costs its synthesis
    (20-150 ms) instead of reloading the voice (150 ms). It writes each WAV and
    prints the path when done; a timeout or exit restarts it next time."""

    def __init__(self, model):
        self.model = model
        self._lock = threading.Lock()
        self._proc = None
        self._dir = Path(tempfile.mkdtemp(prefix="deskvoice-piper-", dir=os.environ.get("XDG_RUNTIME_DIR")))

    def _start(self):
        self._proc = subprocess.Popen(
            ["piper-tts", "-m", str(self.model), "--json-input", "-q"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def stop(self):
        if self._proc:
            self._proc.kill()
            self._proc = None

    def render(self, text, timeout=15.0):
        import soundfile as sf
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start()
            out = self._dir / f"{time.monotonic_ns()}.wav"
            try:
                self._proc.stdin.write(json.dumps({"text": text, "output_file": str(out)}) + "\n")
                self._proc.stdin.flush()
                done = threading.Event()
                line = {}

                def read():
                    line["v"] = self._proc.stdout.readline().strip()
                    done.set()
                threading.Thread(target=read, daemon=True).start()
                if not done.wait(timeout) or line.get("v") != str(out):
                    self.stop()
                    raise RuntimeError(f"piper-tts gave {line.get('v')!r} for {out.name}")
                samples, rate = sf.read(str(out), dtype="float32")
                return samples, rate
            except (BrokenPipeError, OSError):
                self.stop()
                raise
            finally:
                out.unlink(missing_ok=True)


class Voices:
    def __init__(self, kokoro, piper_dirs=()):
        self.kokoro = kokoro
        self.piper_models = self._discover_piper([Path(d).expanduser() for d in piper_dirs])
        self._piper = {}
        self._piper_lock = threading.Lock()

    @staticmethod
    def _discover_piper(roots):
        """Every <name>.onnx with its .onnx.json under the voice dirs, in
        order; the first of a name wins."""
        found = {}
        for root in roots:
            if root.is_dir():
                for onnx in sorted(root.rglob("*.onnx")):
                    if onnx.with_suffix(".onnx.json").exists():
                        found.setdefault(onnx.stem, onnx)
        return found

    def load(self):
        if self.kokoro.available():
            self.kokoro.load()
        else:
            log.warning("kokoro model not found at %s (deskvoice fetch-models downloads it); Piper voices only",
                        self.kokoro.model_path)
        log.info("piper: %d voices", len(self.piper_models))

    def list(self):
        out = [{"name": s, "engine": "kokoro", "label": f"{s.split('_', 1)[1].capitalize()} ({SPEAKERS.get(s[:2], s[:2])})"}
               for s in self.kokoro.styles]
        out += [{"name": "piper:" + stem, "engine": "piper", "label": stem} for stem in self.piper_models]
        return out

    def check(self, name):
        if name.startswith("piper:"):
            if name[6:] not in self.piper_models:
                raise UnknownVoice(name)
        elif name not in self.kokoro.styles:
            raise UnknownVoice(name)

    def render(self, name, text):
        self.check(name)
        if name.startswith("piper:"):
            stem = name[6:]
            with self._piper_lock:
                voice = self._piper.setdefault(stem, PiperVoice(self.piper_models[stem]))
            return voice.render(text)
        return self.kokoro.render(name, text)

    def close(self):
        for v in self._piper.values():
            v.stop()
