"""deskvoice: talk to the running service. Standard library only, so the
launcher starts with the system python in a few tens of milliseconds.

  deskvoice say [--voice V] [--channel cue|assistant] [--wait S] [--json] TEXT
        Blocks until the utterance starts playing (or is dropped), then prints one
        line. Exit 0 started, 3 dropped or failed, 2 service unavailable. Playback
        carries on in the service; the caller never waits for the audio.
  deskvoice stop [--channel C] [--id ID]
  deskvoice voices [--engine kokoro|piper]
  deskvoice duck SECONDS         keep other audio lowered that long (0 releases)
  deskvoice unduck               bring all ducked audio back now
  deskvoice status
  deskvoice fetch-models [--dir DIR] [--no-asr]
        download Kokoro (voices) and Parakeet (speech recognition) into
        ~/.local/share/deskvoice/models, where the default config looks
  deskvoice serve [--config FILE] [--listen]     run the service
"""

import argparse
import json
import socket
import sys
import time
import uuid

from . import protocol

EXIT_OK, EXIT_FAILED, EXIT_UNAVAILABLE = 0, 3, 2

MODELS = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}
ASR = ("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
       "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
       "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2")


def _download(url, dest):
    import shutil
    import urllib.request
    tmp = dest.with_name(dest.name + ".part")
    print(f"  {dest.name} ...", end="", flush=True)
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("content-length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dest.name} {done * 100 // total}%", end="", flush=True)
    shutil.move(tmp, dest)
    print(f"\r  {dest.name} done ({done >> 20} MB)")


def fetch_models(args):
    import os
    import tarfile
    from pathlib import Path
    data = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "deskvoice" / "models"
    root = Path(args.dir).expanduser() if args.dir else data
    root.mkdir(parents=True, exist_ok=True)
    print(f"models -> {root}")
    for name, url in MODELS.items():
        if (root / name).exists():
            print(f"  {name} already there")
        else:
            _download(url, root / name)
    if not args.no_asr:
        name, url = ASR
        if (root / name).is_dir():
            print(f"  {name} already there")
        else:
            archive = root / (name + ".tar.bz2")
            _download(url, archive)
            print(f"  unpacking {archive.name} ...")
            with tarfile.open(archive) as t:
                t.extractall(root, filter="data")
            archive.unlink()
    print("done. If you used --dir, point [kokoro] and [listen] asr_model at it in the config.")
    return EXIT_OK


class Unavailable(Exception):
    pass


class Connection:
    def __init__(self, path=None, timeout=5.0):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        try:
            self.sock.connect(path or protocol.socket_path())
        except OSError as e:
            raise Unavailable(str(e)) from e
        self.buf = b""

    def send(self, msg):
        self.sock.sendall(protocol.encode(msg))

    def recv(self, deadline):
        while b"\n" not in self.buf:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError
            self.sock.settimeout(left)
            chunk = self.sock.recv(65536)
            if not chunk:
                raise Unavailable("service closed the connection")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return protocol.decode(line)

    def request(self, msg, timeout=5.0):
        msg = {**msg, "req": uuid.uuid4().hex[:8]}
        self.send(msg)
        deadline = time.monotonic() + timeout
        while True:
            reply = self.recv(deadline)
            if reply.get("req") == msg["req"]:
                return reply


def say(args):
    uid = uuid.uuid4().hex
    msg = {"op": "say", "id": uid, "text": args.text, "channel": args.channel, "final": True}
    if args.voice:
        msg["voice"] = args.voice
    if args.wait is not None:
        msg["max_wait"] = args.wait
    conn = Connection()
    conn.send(msg)
    # Render (~0.1-0.3 s) plus however long it may wait behind other speech.
    deadline = time.monotonic() + (args.wait if args.wait is not None else protocol.DEFAULT_MAX_WAIT) + 20
    while True:
        try:
            ev = conn.recv(deadline)
        except TimeoutError:
            return report(args, {"event": "error", "message": "no answer from deskvoice"}, EXIT_FAILED)
        if ev.get("id") != uid:
            continue
        kind = ev.get("event")
        if kind == "started":
            return report(args, ev, EXIT_OK)
        if kind in ("dropped", "error", "stopped"):
            return report(args, ev, EXIT_FAILED)


def report(args, ev, code):
    if args.json:
        print(json.dumps(ev))
    elif ev.get("event") == "started":
        print(f"spoke {ev.get('secs', 0):.1f}s")
    elif ev.get("event") == "dropped":
        print(f"dropped: {ev.get('reason', '?')}")
    else:
        print(f"{ev.get('event')}: {ev.get('message', ev.get('reason', ''))}")
    return code


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["serve"]:
        from .server import main as serve   # the heavy imports, only for the service
        return serve(argv[1:])
    ap = argparse.ArgumentParser(prog="deskvoice", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("say")
    p.add_argument("--voice", default=None, help="a Kokoro style or piper:<model> (default: the channel's)")
    p.add_argument("--channel", default="cue", choices=protocol.CHANNELS)
    p.add_argument("--wait", type=float, default=None, help="seconds it may wait to start before it's dropped")
    p.add_argument("--json", action="store_true")
    p.add_argument("text", nargs="+")
    p = sub.add_parser("stop")
    p.add_argument("--channel", choices=protocol.CHANNELS)
    p.add_argument("--id")
    p = sub.add_parser("voices")
    p.add_argument("--engine", choices=("kokoro", "piper"))
    sub.add_parser("status")
    p = sub.add_parser("fetch-models")
    p.add_argument("--dir", help="where to put them (default ~/.local/share/deskvoice/models)")
    p.add_argument("--no-asr", action="store_true", help="voices only: skip Parakeet (~480 MB)")
    p = sub.add_parser("duck")
    p.add_argument("seconds", type=float)
    sub.add_parser("unduck")
    args = ap.parse_args(argv)

    if args.cmd == "fetch-models":
        return fetch_models(args)
    try:
        if args.cmd == "say":
            args.text = " ".join(args.text)
            return say(args)
        conn = Connection()
        if args.cmd == "stop":
            r = conn.request({"op": "stop", "channel": args.channel, "id": args.id})
            print(f"stopped {r.get('stopped', 0)}")
        elif args.cmd == "voices":
            for v in conn.request({"op": "voices"})["voices"]:
                if args.engine in (None, v["engine"]):
                    print(f"{v['name']:32s} {v['label']}")
        elif args.cmd == "duck":
            conn.request({"op": "duck", "seconds": args.seconds})
        elif args.cmd == "unduck":
            conn.request({"op": "unduck"})
            print("restored")
        elif args.cmd == "status":
            print(json.dumps(conn.request({"op": "status"}), indent=2))
        return EXIT_OK
    except Unavailable as e:
        print(f"unavailable: {e}")
        return EXIT_UNAVAILABLE


if __name__ == "__main__":
    sys.exit(main())
