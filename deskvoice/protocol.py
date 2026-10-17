"""The wire format, shared by the server and the stdlib-only client.

One JSON object per line over a unix socket. Requests carry an `op`; anything
about an utterance carries its `id`. The server answers requests and pushes
utterance events (queued, started, finished, dropped, error) to the connection
that asked.
"""

import json
import os

CHANNELS = ("assistant", "cue")          # highest priority first; "reader" arrives in M4
DEFAULT_MAX_WAIT = 8.0                   # seconds a cue may wait to start before it's dropped


def socket_path():
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return os.path.join(runtime, "deskvoice.sock")


def encode(msg):
    return (json.dumps(msg, separators=(",", ":")) + "\n").encode()


def decode(line):
    return json.loads(line)
