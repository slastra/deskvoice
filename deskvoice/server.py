"""deskvoice serve: the unix-socket front of the speaker and the listener.

Requests (one JSON object per line; every reply echoes the request's "req"):
  {"op": "say", "id", "text", "voice"?, "channel"?, "final": true, "max_wait"?, "stop_on_close"?}
        queue a sentence; the same id streams more sentences into one utterance;
        stop_on_close: stop it if this connection closes first (a killed client)
  {"op": "end", "id"}                  that utterance is complete
  {"op": "stop", "channel"?, "id"?}    cut speech short (barge-in, cancel)
  {"op": "voices"}  {"op": "status"}
  {"op": "duck", "seconds"}            keep other apps' audio lowered that long; 0 releases
                                       (speech ducks by itself; this is for listening, thinking)
  {"op": "unduck"}                     everything back now: holds cleared, ducked audio restored
  {"op": "synth", "text", "voice"?, "out"}   render a WAV to `out` (absolute) for a client
                                       that plays it itself -> {"ok", "secs", "ms"}
  {"op": "lease", "channel": "reader"} hold the reader channel on this connection: deskvoice
                                       pushes {"event": "yield"} when anything else speaks or the
                                       mic listens, {"event": "resume"} after 0.5 s of quiet
  {"op": "dictate", "action": "start"|"stop"|"cancel", "save"?}  a dictation session on the
                                       shared mic: events state, level, time, partial, raw, error;
                                       closing the connection cancels it
  {"op": "transcribe", "path"}         Parakeet on a 16 kHz WAV -> {"text"}
  {"op": "subscribe"}                  push {"event": "status", ...} whenever anything changes,
                                       and {"event": "heard", ...} for each request heard
  {"op": "mic", "enabled"}             the wake word on or off (off = no capture; remembered)
  {"op": "followup", "seconds"}        listen again without the wake word
  {"op": "flag", "name", "on"}         create or remove one of the configured status flag files
Utterance events (queued, started, finished, stopped, dropped, error) go to
the connection that sent the say, and keep playing if that connection closes
(unless the say asked for stop_on_close).
"""

import argparse
import asyncio
import collections
import logging
import os
import signal
import threading
import time
from pathlib import Path

from . import protocol
from .duck import Ducker
from .engines import UnknownVoice, Voices
from .speaker import Speaker

log = logging.getLogger("deskvoice")


class Server:
    def __init__(self, voices, speaker, ducker=None, listener=None, cfg=None):
        from . import config
        self.voices, self.speaker, self.ducker, self.listener = voices, speaker, ducker, listener
        self.cfg = cfg or config.load("/nonexistent")
        self.started = time.time()
        self.leases = set()               # send functions of connections holding the reader lease
        self.yielding = False
        self.sessions = {}                # notify -> the Dictation that connection started
        self.subscribers = set()
        self.owned = {}                   # notify -> ids of its stop_on_close utterances
        self.finishing = 0                # voice dictations being cleaned up and pasted
        self.recent = collections.deque(maxlen=8)   # heard / said / dictated, newest last
        self._recent_lock = threading.Lock()

    async def handle(self, reader, writer):
        loop = asyncio.get_running_loop()

        def send(msg):
            if not writer.is_closing():
                writer.write(protocol.encode(msg))

        def notify(msg):                  # safe from speaker threads and from the loop
            loop.call_soon_threadsafe(send, msg)

        try:
            while line := await reader.readline():
                msg = {}
                try:
                    msg = protocol.decode(line)
                    if not isinstance(msg, dict):
                        raise ValueError("expected a JSON object")
                    if msg.get("op") == "synth":
                        # Rendering takes 0.1-0.3 s on the GPU: off the event loop.
                        reply = await loop.run_in_executor(None, self.synth, msg)
                    elif msg.get("op") == "transcribe":
                        reply = await loop.run_in_executor(None, self.transcribe, msg)
                    else:
                        reply = self.dispatch(msg, notify)
                except (ValueError, KeyError, UnknownVoice) as e:
                    msg = msg if isinstance(msg, dict) else {}
                    reply = {"event": "error", "message": _describe(e)}
                    if "id" in msg:
                        reply["id"] = msg["id"]
                if reply is not None:
                    if "req" in msg:
                        reply["req"] = msg["req"]
                    send(reply)
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            if notify in self.leases:
                self.leases.discard(notify)
                log.info("lease: reader released")
            self.subscribers.discard(notify)
            for uid in self.owned.pop(notify, ()):
                if self.speaker.stop(uid=uid):    # a no-op once it has finished
                    log.info("say: %s stopped, its client went away", uid)
            session = self.sessions.pop(notify, None)
            if session and not session.done.is_set():
                session.cancel.set()       # the client went away: never keep audio nobody asked for
            writer.close()

    def dispatch(self, msg, notify):
        op = msg.get("op")
        if op == "say":
            text = (msg.get("text") or "").strip()
            if not text:
                raise ValueError("nothing to say")
            channel = msg.get("channel") or "cue"
            voice = msg.get("voice") or self.cfg["voices"].get(channel) or self.cfg["voices"]["cue"]
            self.voices.check(voice)
            max_wait = msg.get("max_wait", self.cfg["speaker"]["cue_max_wait"] if channel == "cue" else None)
            self.speaker.say(msg["id"], text, voice, channel, final=msg.get("final", True),
                             max_wait=max_wait, notify=notify)
            log.info("say: %s %s %r", channel, voice, text[:60])
            if msg.get("stop_on_close"):
                self.owned.setdefault(notify, set()).add(msg["id"])
            return None                   # "queued" comes as an event
        if op == "dictate":
            return self.dictate(msg, notify)
        if op == "subscribe":
            self.subscribers.add(notify)
            return {"event": "status", **self.snapshot()}
        if op == "followup":
            if not self.listener:
                raise ValueError("the listener isn't running")
            self.listener.out.followup.put(float(msg.get("seconds", 6)))
            return {"ok": True}
        if op == "flag":
            path = self.cfg["status"]["flags"].get(msg.get("name", ""))
            if not path:
                raise ValueError(f"no status flag named {msg.get('name')!r} in the config")
            p = Path(path).expanduser()
            if msg.get("on"):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
            else:
                p.unlink(missing_ok=True)
            return {"ok": True, "on": p.exists()}
        if op == "mic":
            if not self.listener:
                raise ValueError("the listener isn't running")
            self.listener.set_mic(bool(msg.get("enabled")))
            return {"ok": True, "mic": self.listener.mic_on.is_set()}
        if op == "lease":
            if msg.get("channel") != "reader":
                raise ValueError("only the reader channel can be leased")
            if notify not in self.leases:
                log.info("lease: reader held")
            self.leases.add(notify)
            return {"ok": True, "yield": self.yielding}
        if op == "end":
            self.speaker.end(msg["id"])
            return {"ok": True}
        if op == "stop":
            return {"ok": True, "stopped": self.speaker.stop(channel=msg.get("channel"), uid=msg.get("id"))}
        if op == "duck":
            if self.ducker:
                self.ducker.hold(float(msg.get("seconds", 0)))
            return {"ok": True, "ducking": self.ducker is not None}
        if op == "unduck":
            if self.ducker:
                threading.Thread(target=self.ducker.force_restore, daemon=True).start()  # ramps take ~0.6 s
            return {"ok": True}
        if op == "voices":
            return {"voices": self.voices.list()}
        if op == "status":
            k = self.voices.kokoro
            return {
                "uptime": round(time.time() - self.started),
                "kokoro": {"loaded": bool(k.providers), "provider": k.providers[0] if k.providers else None,
                           "voices": len(k.styles)},
                "piper": {"voices": len(self.voices.piper_models), "resident": sorted(self.voices._piper)},
                "duck": self.ducker.status() if self.ducker else None,
                "listen": self.listener.status() if self.listener else None,
                **self.speaker.status(),
            }
        raise ValueError(f"unknown op {op!r}")


    def synth(self, msg):
        import soundfile as sf
        try:
            text, out = (msg.get("text") or "").strip(), msg.get("out") or ""
            voice = msg.get("voice") or self.cfg["voices"]["reader"]
            if not os.path.isabs(out):
                raise ValueError("out must be an absolute path")
            t = time.monotonic()
            if any(c.isalnum() for c in text):
                samples, rate = self.voices.render(voice, text)
            else:                          # nothing pronounceable: a breath of silence
                import numpy as np
                samples, rate = np.zeros(2400, dtype="float32"), 24000
            sf.write(out, samples, rate)
            secs, ms = round(len(samples) / rate, 3), round((time.monotonic() - t) * 1000)
            log.info("synth: %s %.1fs in %d ms %r", voice, secs, ms, text[:60])
            return {"ok": True, "secs": secs, "ms": ms}
        except (ValueError, UnknownVoice, OSError, RuntimeError) as e:
            return {"event": "error", "message": _describe(e)}

    def dictate(self, msg, notify):
        from .listen import Dictation
        action = msg.get("action")
        session = self.sessions.get(notify)
        if action == "start":
            if not self.listener:
                raise ValueError("the listener isn't running, so there's no mic to dictate from")
            if session and not session.done.is_set():
                raise ValueError("already dictating on this connection")
            session = Dictation(emit=lambda name, value: notify({"event": name, "value": value}),
                                save=msg.get("save"))
            self.sessions[notify] = session
            self.listener.request_dictation(session)
            log.info("dictate: start (socket)")
            return {"ok": True}
        if not session:
            raise ValueError("no dictation on this connection")
        if action == "stop":
            session.stop.set()
        elif action == "cancel":
            session.cancel.set()
        else:
            raise ValueError(f"unknown dictate action {action!r}")
        log.info("dictate: %s (socket)", action)
        return {"ok": True}

    def transcribe(self, msg):
        import soundfile as sf
        try:
            if not self.listener:
                raise ValueError("the listener isn't running")
            audio, rate = sf.read(msg.get("path") or "", dtype="int16")
            if rate != 16000:
                raise ValueError(f"expected 16 kHz audio, got {rate}")
            if audio.ndim > 1:
                audio = audio[:, 0]
            t = time.monotonic()
            text = self.listener.transcribe(audio)
            ms = round((time.monotonic() - t) * 1000)
            log.info("transcribe: %.1fs of audio in %d ms", len(audio) / rate, ms)
            return {"text": text, "ms": ms}
        except (ValueError, OSError, RuntimeError) as e:
            return {"event": "error", "message": str(e)}

    def remember(self, kind, text, **fields):
        """Something heard, said or dictated, for the widget's recent list."""
        text = (text or "").strip()
        if text:
            with self._recent_lock:
                self.recent.append({"kind": kind, "text": text[:400], "ts": round(time.time()), **fields})

    def thinking(self, playing, listen):
        """What deskvoice is waiting on, if anything, so the widget can show
        work in progress: "reply" while a client works out an answer (its duck
        hold, with nothing speaking and the mic not listening), "dictation"
        while a voice dictation is cleaned up and pasted, "transcribing" for the
        moment after you stop."""
        if self.finishing:
            return "dictation"
        if listen and listen["state"] == "transcribing":
            return "transcribing"
        if playing or not listen or listen["state"] not in ("waiting", "mic off"):
            return None
        return "reply" if self.ducker and self.ducker.holding() else None

    def snapshot(self):
        sp = self.speaker.status()
        playing = sp["playing"]
        listen = self.listener.status() if self.listener else None
        with self._recent_lock:
            recent = list(self.recent)
        return {
            "speaking": {"channel": playing["channel"], "voice": playing["voice"], "text": playing["text"]}
            if playing else None,
            "queued": sp["queued"],
            "voices": sp["voices"],
            "listen": {"state": listen["state"], "mic": listen["mic"], "last_wake": listen["last_wake"]}
            if listen else None,
            "dictation": listen["dictation"] if listen else None,
            "thinking": self.thinking(playing, listen),
            "duck": self.ducker.status() if self.ducker else None,
            "reader_held": self.yielding,
            "flags": {name: os.path.exists(os.path.expanduser(path))
                      for name, path in self.cfg["status"]["flags"].items()},
            "recent": recent,
        }

    async def watch_status(self):
        """Push a snapshot to subscribers whenever it changes (checked 6x a second;
        nothing is sent while nothing changes)."""
        import json as _json
        last = None
        while True:
            await asyncio.sleep(0.15)
            if not self.subscribers:
                last = None
                continue
            snap = self.snapshot()
            key = _json.dumps(snap, sort_keys=True)
            if key != last:
                last = key
                for notify in list(self.subscribers):
                    notify({"event": "status", **snap})

    def busy(self):
        """Is anything voice-related happening that the reader should wait out?"""
        if self.speaker.speaking(tail=0.0):
            return True
        if self.ducker and self.ducker.holding():
            return True
        return bool(self.listener and self.listener.state in ("listening", "transcribing", "dictating"))

    async def watch_leases(self):
        """Tell reader-lease holders to yield the moment something else speaks or
        the mic listens, and to resume after 0.5 s of quiet. Also the ducker's
        watchdog: a "speaking" flag left set while nothing has played for 2 s
        is cleared, so a playback hiccup can't hold audio down."""
        quiet_since = None
        stale_since = None
        while True:
            await asyncio.sleep(0.1)
            if self.ducker:
                if self.ducker.is_playing() and not self.speaker.speaking(tail=0.5):
                    stale_since = stale_since or time.monotonic()
                    if time.monotonic() - stale_since > 2:
                        log.warning("duck: speaking flag stuck with nothing playing; clearing it")
                        self.ducker.playing(False)
                        stale_since = None
                else:
                    stale_since = None
            if self.busy():
                quiet_since = None
                if not self.yielding:
                    self.yielding = True
                    self._broadcast({"event": "yield"})
            elif self.yielding:
                now = time.monotonic()
                quiet_since = quiet_since or now
                if now - quiet_since >= 0.5:
                    self.yielding = False
                    self._broadcast({"event": "resume"})

    def heard(self, payload):
        """A request the listener heard: recent list, and an event for subscribers."""
        self.remember("heard", payload.get("text"))
        for notify in list(self.subscribers):
            notify({"event": "heard", **payload})

    def _broadcast(self, msg):
        for send in list(self.leases):
            send(msg)


def _describe(e):
    if isinstance(e, UnknownVoice):
        return f"unknown voice {e.args[0]!r}"
    if isinstance(e, KeyError):
        return f"missing {e.args[0]!r}"
    return str(e)


def finish_voice_dictation(cfg, raw, cue, srv=None):
    """A voice dictation is done: deliver it (deliver.py) and chirp the result.
    DESKVOICE_DICTATE_DRYRUN=1 does everything but paste (for testing)."""
    from . import deliver
    if srv:
        srv.finishing += 1
    try:
        summary, ok = deliver.finish(raw, cfg["dictate"], dry_run=bool(os.environ.get("DESKVOICE_DICTATE_DRYRUN")))
    except Exception as e:                # a broken finish command must not kill the listener
        summary, ok = {"error": str(e)}, False
    finally:
        if srv:
            srv.finishing -= 1
    if not ok:
        log.warning("dictate: %s", summary.get("error"))
        cue("error")
        return
    how = "pasted and submitted" if summary.get("pasted") and summary.get("submitted") else \
        "submitted" if summary.get("submitted") else "pasted" if summary.get("pasted") else "cleaned (dry run)"
    log.info("dictate: %s into %s: %r", how, summary.get("app") or "?", (summary.get("text") or "")[:120])
    if srv:
        srv.remember("dictated", summary.get("text") or ("(enter)" if summary.get("submitted") else ""),
                     app=summary.get("app", ""), submitted=bool(summary.get("submitted")))
    cue("ok")


def listener_args(cfg):
    """The listener's settings, flattened from the config."""
    from types import SimpleNamespace
    lc, mq = cfg["listen"], cfg["mqtt"]
    return SimpleNamespace(
        wake_model=lc["wake_model"], threshold=lc["threshold"], barge_threshold=lc["barge_threshold"],
        wake_prefix=lc["wake_prefix"], asr=lc["asr_model"], asr_threads=lc["asr_threads"],
        target=lc["target"] or None, start_timeout=lc["start_timeout"], end_silence=lc["end_silence"],
        max_utterance=lc["max_utterance"], clips_dir=lc["clips_dir"], presence_file=lc["presence_file"],
        mqtt_enabled=mq["enabled"], mqtt_host=mq["host"], mqtt_port=mq["port"], mqtt_prefix=mq["prefix"],
        dictate_silence=cfg["dictate"]["silence"], cues=cfg["cues"])


def start_listener(cfg, speaker, ducker, srv=None):
    """The wake-word listener. Loading takes a couple of seconds, so it happens
    on the listener's own thread; speech is up before it."""
    from .listen import Listener
    args = listener_args(cfg)
    box = {}

    def barge():
        speaker.stop(channel="assistant")
        speaker.stop(channel="cue")

    def run():
        while True:
            try:
                listener = Listener(args, speaking=speaker.speaking, barge=barge,
                                    duck=ducker.hold if ducker else (lambda s: None))
                listener.on_voice_dictation = lambda raw: threading.Thread(
                    target=finish_voice_dictation, args=(cfg, raw, listener.cues.play, srv), daemon=True).start()
                if srv:
                    listener.on_heard = srv.heard
                box["l"] = listener
                listener.run()
                return
            except Exception:
                log.exception("listen: failed; retrying in 5 s")
                time.sleep(5)
    threading.Thread(target=run, daemon=True, name="listener").start()
    return box


async def serve(cfg, path):
    from .duck import fading_file
    from .engines import Kokoro
    kc = cfg["kokoro"]
    voices = Voices(Kokoro(kc["model"], kc["voices"], kc["provider"], warm_voice=cfg["voices"]["assistant"]),
                    cfg["piper"]["voice_dirs"])
    voices.load()                         # Kokoro warmed (~1 s on CUDA), then every cue is fast
    speaker = Speaker(voices.render)
    speaker.voices.update({ch: cfg["voices"][ch] for ch in ("assistant", "cue")})
    ducker = None
    dc = cfg["duck"]
    if dc["level"] < 1:
        from . import config as _config
        ducker = Ducker(level=dc["level"], exclude=dc["exclude"], fading=fading_file(dc["yield_to"]),
                        state_file=_config.STATE / "duck.json")
        speaker.on_busy = ducker.playing
        ducker.start()
        log.info("ducking other audio to %.0f%% while speaking or listening", dc["level"] * 100)
    srv = Server(voices, speaker, ducker, cfg=cfg)
    speaker.on_said = lambda channel, voice, text: srv.remember("said", text, channel=channel, voice=voice)
    speaker.start()
    listener_box = start_listener(cfg, speaker, ducker, srv) if cfg["listen"]["enabled"] else None

    if os.path.exists(path):
        os.unlink(path)
    server = await asyncio.start_unix_server(srv.handle, path=path)
    os.chmod(path, 0o600)
    log.info("listening on %s", path)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    async def attach_listener():
        while listener_box is not None and "l" not in listener_box:
            await asyncio.sleep(0.5)
        if listener_box:
            srv.listener = listener_box["l"]
    attach = asyncio.create_task(attach_listener())
    leases = asyncio.create_task(srv.watch_leases())
    status = asyncio.create_task(srv.watch_status())
    try:
        async with server:
            await stop.wait()
    finally:
        for task in (attach, leases, status):
            task.cancel()
        shutdown(srv.listener, speaker, ducker, voices, path)


def shutdown(listener, speaker, ducker, voices, path):
    """Close everything, each step on its own: one that fails (the MQTT
    goodbye with the broker gone, say) must not skip the rest, above all the
    ducker's restore, which is what keeps someone's music from staying at 30%."""
    def unlink():
        if os.path.exists(path):
            os.unlink(path)
    steps = (("listener", listener.close if listener else None), ("speaker", speaker.close),
             ("ducker", ducker.close if ducker else None), ("voices", voices.close), ("socket", unlink))
    for name, step in steps:
        if step is None:
            continue
        try:
            step()
        except Exception:
            log.exception("shutdown: closing the %s failed; carrying on", name)


def main(argv=None):
    from . import config
    ap = argparse.ArgumentParser(prog="deskvoice serve",
                                 description="Run deskvoice. Settings come from the config file; see config.example.toml.")
    ap.add_argument("--config", default=str(config.CONFIG), help="config file (default %(default)s)")
    ap.add_argument("--socket", default=protocol.socket_path())
    ap.add_argument("--listen", action="store_true", help="run the wake-word listener (same as listen.enabled)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = config.load(args.config)
    if args.listen:
        cfg["listen"]["enabled"] = True
    asyncio.run(serve(cfg, args.socket))


if __name__ == "__main__":
    main()
