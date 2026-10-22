"""Settings: ~/.config/deskvoice/config.toml over built-in defaults.

Every key is optional; a missing file means the defaults. Command-line flags
to `deskvoice serve` override the file. See config.example.toml for every
key with its default.
"""

import copy
import os
import tomllib
from pathlib import Path

DATA = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "deskvoice"
STATE = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state") / "deskvoice"
CONFIG = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "deskvoice" / "config.toml"

DEFAULTS = {
    "voices": {
        # Kokoro style names ("af_heart"), or "piper:<model stem>"
        "assistant": "af_heart",       # the assistant channel's default voice
        "cue": "af_sky",               # notifications and cues
        "reader": "bm_george",         # synth requests that don't name one
    },
    "kokoro": {
        "model": str(DATA / "models" / "kokoro-v1.0.onnx"),
        "voices": str(DATA / "models" / "voices-v1.0.bin"),
        "provider": "auto",            # auto (CUDA if available) | cpu
    },
    "piper": {
        # searched in order for <name>.onnx with its .onnx.json; first name wins
        "voice_dirs": ["/usr/share/piper-voices", str(Path.home() / ".local/share/piper-voices")],
    },
    "speaker": {
        "cue_max_wait": 8.0,           # seconds a cue may wait behind other speech before it's dropped
    },
    "duck": {
        "level": 0.3,                  # other apps' volume while deskvoice speaks or listens (1 = never duck)
        # application.name substrings never ducked (deskvoice's own streams always excepted)
        "exclude": ["speech-dispatcher"],
        # a JSON file whose "fade" > 0 means another program is fading volumes
        # right now; ducking stands aside meanwhile ("" = no such program)
        "yield_to": "",
    },
    "listen": {
        "enabled": False,              # the wake-word listener (also: deskvoice serve --listen)
        # an openWakeWord model: a path to .onnx, or a built-in name ("hey_jarvis")
        "wake_model": "hey_jarvis",
        "threshold": 0.5,
        "barge_threshold": 0.7,        # while deskvoice is talking; 0 ignores the mic then
        # regex stripped from the front of a transcript (the wake phrase as the
        # recognizer tends to spell it), so clients get only the request
        "wake_prefix": r"^\s*hey[,.]?\s+jarvis\b[,.?!]*\s*",
        "asr_model": str(DATA / "models" / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"),
        "asr_threads": 4,
        "target": "",                  # PipeWire source node ("" = default)
        "start_timeout": 4.0,          # seconds to wait for speech after the wake word
        "end_silence": 0.8,            # seconds of quiet that end a request
        "max_utterance": 10.0,
        "clips_dir": str(DATA / "wake-clips"),   # trigger and near-miss clips for retraining ("" = none)
        # a presence sensor's JSON status file ({"known", "sensor_ok", "present"});
        # wakes are ignored while a fresh one says nobody is there ("" = never)
        "presence_file": "",
    },
    "mqtt": {
        "enabled": True,               # where heard requests go (the listener needs somewhere to send them)
        "host": "localhost",
        "port": 1883,
        "prefix": "deskvoice",         # topics: <prefix>/heard, <prefix>/listen/state, ...
    },
    "dictate": {
        "silence": 2.5,                # a voice dictation ends after this much quiet
        # A voice dictation's raw text goes to this command on stdin (with
        # {app} and {title} filled from the focused window when known) to be
        # cleaned up and delivered. Empty: deskvoice pastes the raw text into
        # the focused window itself (wl-copy + wtype).
        "finish_command": [],
    },
    "cues": {
        "backend": "builtin",          # builtin (deskvoice's own chirps) | osc | none
        "osc": "127.0.0.1:57120",      # backend "osc": sends /emotune <name> (an emotune-style daemon)
        # cue names, by event; the builtin backend knows these names
        "names": {"listen": "listen", "dictate": "dictate", "ok": "ok", "error": "error",
                  "nomatch": "nomatch", "bye": "bye"},
    },
    "status": {
        # name -> file: the status feed reports whether each file exists, so a
        # widget can show and toggle a mute flag another tool honours
        "flags": {},
    },
}


def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def expand(value):
    if isinstance(value, str) and value.startswith("~"):
        return os.path.expanduser(value)
    return value


def load(path=None):
    path = Path(path) if path else CONFIG
    data = {}
    if path.exists():
        with path.open("rb") as f:
            data = tomllib.load(f)
    cfg = merge(DEFAULTS, data)
    for section in cfg.values():
        if isinstance(section, dict):
            for k, v in section.items():
                section[k] = [expand(x) for x in v] if isinstance(v, list) else expand(v)
    return cfg
