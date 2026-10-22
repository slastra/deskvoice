import asyncio
import os
import tempfile
import threading

import pytest

from deskvoice import client, protocol
from deskvoice.engines import UnknownVoice
from deskvoice.server import Server
from deskvoice.speaker import Speaker
from tests.test_speaker import FakePlayer, fake_render


class FakeKokoro:
    providers = ["CUDAExecutionProvider"]
    styles = ["af_heart", "af_sky"]


class FakeVoices:
    kokoro = FakeKokoro()
    piper_models = {"en_GB-cori-high": "x"}
    _piper = {}

    def check(self, name):
        if name not in ("af_heart", "af_sky", "piper:en_GB-cori-high"):
            raise UnknownVoice(name)

    def list(self):
        return [{"name": n, "engine": "kokoro", "label": n} for n in self.kokoro.styles]

    def render(self, name, text):
        import numpy as np
        self.check(name)
        return np.zeros(2400, dtype=np.float32), 24000


@pytest.fixture
def running(monkeypatch):
    sock = os.path.join(tempfile.mkdtemp(), "deskvoice.sock")
    monkeypatch.setattr(protocol, "socket_path", lambda: sock)
    sp = Speaker(fake_render(), player=FakePlayer())
    sp.start()
    srv = Server(FakeVoices(), sp)
    sp.on_said = lambda channel, voice, text: srv.remember("said", text, channel=channel, voice=voice)
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    server = asyncio.run_coroutine_threadsafe(asyncio.start_unix_server(srv.handle, path=sock), loop).result(2)
    watcher = asyncio.run_coroutine_threadsafe(srv.watch_leases(), loop)
    statuser = asyncio.run_coroutine_threadsafe(srv.watch_status(), loop)
    yield sp
    sp.close()

    async def shut():
        watcher.cancel()
        statuser.cancel()
        server.close()
        await server.wait_closed()
    asyncio.run_coroutine_threadsafe(shut(), loop).result(2)
    loop.call_soon_threadsafe(loop.stop)
    t.join(2)


def test_cli_say_reports_start(running, capsys):
    assert client.main(["say", "--voice", "af_sky", "Build passed."]) == client.EXIT_OK
    assert capsys.readouterr().out.strip() == "spoke 0.1s"


def test_cli_say_unknown_voice(running, capsys):
    assert client.main(["say", "--voice", "nobody", "Hello."]) == client.EXIT_FAILED
    assert "unknown voice" in capsys.readouterr().out


def test_cli_dropped_when_busy(running, capsys):
    running.say("long", "Long reply.|1500", "af_heart", "assistant")
    assert client.main(["say", "--wait", "0.2", "--json", "Too late."]) == client.EXIT_FAILED
    assert '"reason": "busy"' in capsys.readouterr().out


def test_status_and_voices(running, capsys):
    assert client.main(["status"]) == client.EXIT_OK
    out = capsys.readouterr().out
    assert '"provider": "CUDAExecutionProvider"' in out
    assert client.main(["voices"]) == client.EXIT_OK
    assert "af_heart" in capsys.readouterr().out


def test_service_down(monkeypatch, capsys):
    monkeypatch.setattr(protocol, "socket_path", lambda: "/nonexistent/deskvoice.sock")
    assert client.main(["say", "Hello."]) == client.EXIT_UNAVAILABLE
    assert client.main(["status"]) == client.EXIT_UNAVAILABLE


def test_synth_writes_a_wav(running, tmp_path):
    out = tmp_path / "00001.wav"
    conn = client.Connection()
    r = conn.request({"op": "synth", "text": "Chapter one.", "voice": "af_heart", "out": str(out)})
    assert r["ok"] and r["secs"] == 0.1 and out.exists()
    r = conn.request({"op": "synth", "text": "Hi.", "voice": "nobody", "out": str(out)})
    assert r["event"] == "error" and "unknown voice" in r["message"]
    r = conn.request({"op": "synth", "text": "Hi.", "voice": "af_heart", "out": "relative.wav"})
    assert r["event"] == "error"


def test_reader_lease_yields_and_resumes(running):
    import time as _t
    conn = client.Connection()
    r = conn.request({"op": "lease", "channel": "reader"})
    assert r["ok"] and r["yield"] is False
    running.say("as", "A reply.|400", "af_heart", "assistant")
    deadline = _t.monotonic() + 3
    events = []
    while len(events) < 2:
        ev = conn.recv(deadline)
        if ev.get("event") in ("yield", "resume"):
            events.append((ev["event"], _t.monotonic()))
    assert [e for e, _ in events] == ["yield", "resume"]
    assert events[1][1] - events[0][1] >= 0.4 + 0.5 - 0.2   # the reply, half a second of quiet, less 0.1 s polls


def test_subscribe_pushes_changes_and_recent(running):
    import time as _t
    conn = client.Connection()
    first = conn.request({"op": "subscribe"})
    assert first["event"] == "status" and first["speaking"] is None and first["recent"] == []
    running.say("as", "Hello from the assistant.|300", "af_heart", "assistant")
    deadline = _t.monotonic() + 3
    saw_speaking = saw_recent = False
    while not (saw_speaking and saw_recent):
        ev = conn.recv(deadline)
        if ev.get("event") != "status":
            continue
        if ev["speaking"] and ev["speaking"]["channel"] == "assistant":
            saw_speaking = ev["speaking"]["text"] == "Hello from the assistant.|300"   # the fake keeps its length marker
        if ev["recent"] and ev["recent"][-1]["kind"] == "said":
            saw_recent = ev["recent"][-1]["text"] == "Hello from the assistant.|300" and ev["recent"][-1]["channel"] == "assistant"
    assert saw_speaking and saw_recent


def test_voice_dictation_finish_runs_and_counts_as_thinking(tmp_path, monkeypatch):
    """The finisher end to end with a stand-in dictate.py: it reports clean
    text, the cue is doc_ok, recent gets the dictation, and while it runs the
    snapshot says thinking: dictation."""
    import types
    from deskvoice import server as srvmod
    script = tmp_path / "dictate.py"
    script.write_text(
        "import sys, json, time\n"
        "raw = sys.stdin.read()\n"
        "time.sleep(0.3)\n"
        "print('clean', json.dumps(raw.strip().capitalize()))\n"
        "print('pasted', 'true')\n")
    sp = Speaker(fake_render(), player=FakePlayer())
    s = Server(FakeVoices(), sp)
    s.listener = types.SimpleNamespace(status=lambda: {"state": "waiting", "mic": True, "last_wake": None, "dictation": None})
    seen, cues = [], []
    import threading as _th
    cfg = {"dictate": {"finish_command": ["python3", str(script)]}}
    t = _th.Thread(target=srvmod.finish_voice_dictation, args=(cfg, "buy milk", cues.append, s))
    t.start()
    import time as _t
    _t.sleep(0.15)
    seen.append(s.snapshot()["thinking"])
    t.join(5)
    assert seen == ["dictation"] and cues == ["ok"]
    assert s.snapshot()["thinking"] is None
    assert s.recent[-1]["kind"] == "dictated" and s.recent[-1]["text"] == "Buy milk"


def test_thinking_while_a_client_works_out_a_reply():
    import types
    sp = Speaker(fake_render(), player=FakePlayer())
    holding = {"v": True}
    s = Server(FakeVoices(), sp, ducker=types.SimpleNamespace(holding=lambda: holding["v"], status=lambda: {}))
    listen = {"state": "waiting", "mic": True, "last_wake": None, "dictation": None}
    s.listener = types.SimpleNamespace(status=lambda: listen)
    assert s.snapshot()["thinking"] == "reply"          # heard, the brain holds the duck, nothing speaks
    listen["state"] = "listening"
    assert s.snapshot()["thinking"] is None             # the mic is open: that's listening, not thinking
    listen["state"] = "transcribing"
    assert s.snapshot()["thinking"] == "transcribing"
    listen["state"], holding["v"] = "waiting", False
    assert s.snapshot()["thinking"] is None             # nothing held: idle


def test_the_example_config_documents_the_defaults():
    from pathlib import Path
    from deskvoice import config
    example = Path(__file__).resolve().parent.parent / "config.example.toml"
    assert config.load(example) == config.load("/nonexistent")


def test_flags_are_reported_and_settable(tmp_path):
    from deskvoice import config
    flag = tmp_path / "quiet"
    cfg = config.merge(config.load("/nonexistent"), {"status": {"flags": {"quiet": str(flag)}}})
    sp = Speaker(fake_render(), player=FakePlayer())
    s = Server(FakeVoices(), sp, cfg=cfg)
    assert s.snapshot()["flags"] == {"quiet": False}
    assert s.dispatch({"op": "flag", "name": "quiet", "on": True}, lambda m: None)["on"] is True
    assert s.snapshot()["flags"] == {"quiet": True}
    s.dispatch({"op": "flag", "name": "quiet", "on": False}, lambda m: None)
    assert not flag.exists()


def test_watchdog_clears_a_stuck_speaking_flag():
    import asyncio, types
    flags = {"playing": True}
    ducker = types.SimpleNamespace(
        is_playing=lambda: flags["playing"],
        playing=lambda on: flags.__setitem__("playing", on),
        holding=lambda: False, status=lambda: {})
    sp = Speaker(fake_render(), player=FakePlayer())        # idle: nothing is playing
    s = Server(FakeVoices(), sp, ducker=ducker)

    async def run():
        task = asyncio.create_task(s.watch_leases())
        await asyncio.sleep(2.6)
        task.cancel()
    asyncio.run(run())
    assert flags["playing"] is False


class DisconnectedMQTT:
    """paho with the broker gone: wait_for_publish raises."""

    def __init__(self):
        self.calls = []

    def publish(self, topic, payload, qos=0, retain=False):
        import types

        def wait_for_publish(timeout=None):
            raise RuntimeError("The client is not currently connected.")
        return types.SimpleNamespace(wait_for_publish=wait_for_publish)

    def disconnect(self):
        self.calls.append("disconnect")

    def loop_stop(self):
        self.calls.append("loop_stop")


def test_listener_close_survives_a_disconnected_broker():
    from deskvoice.listen import Publisher
    out = Publisher("localhost", 1883, "deskvoice", enabled=False)
    out.client = mqtt = DisconnectedMQTT()
    out.close()
    assert mqtt.calls == ["loop_stop"]


def test_shutdown_restores_ducked_audio_even_when_a_close_fails(tmp_path):
    """The reboot that left music ducked: the listener's close raised, and the
    ducker's restore after it never ran."""
    import types
    from deskvoice.server import shutdown
    closed = []

    def failing():
        closed.append("listener")
        raise RuntimeError("The client is not currently connected.")
    sock = tmp_path / "deskvoice.sock"
    sock.touch()
    shutdown(types.SimpleNamespace(close=failing),
             types.SimpleNamespace(close=lambda: closed.append("speaker")),
             types.SimpleNamespace(close=lambda: closed.append("ducker")),
             types.SimpleNamespace(close=lambda: closed.append("voices")), str(sock))
    assert closed == ["listener", "speaker", "ducker", "voices"]
    assert not sock.exists()


def test_shutdown_with_a_real_listener_and_a_gone_broker(tmp_path):
    import types
    from deskvoice.listen import Listener, Publisher
    from deskvoice.server import shutdown
    listener = object.__new__(Listener)
    listener._stop, listener.mic = threading.Event(), None
    listener.out = Publisher("localhost", 1883, "deskvoice", enabled=False)
    listener.out.client = DisconnectedMQTT()
    closed = []
    shutdown(listener, types.SimpleNamespace(close=lambda: None),
             types.SimpleNamespace(close=lambda: closed.append("ducker")),
             types.SimpleNamespace(close=lambda: None), str(tmp_path / "gone.sock"))
    assert closed == ["ducker"] and listener._stop.is_set()


def test_each_op_is_logged_once(running, tmp_path, caplog):
    import logging
    import time as _t
    caplog.set_level(logging.INFO, logger="deskvoice")
    assert client.main(["say", "--voice", "af_sky", "Build passed."]) == client.EXIT_OK
    conn = client.Connection()
    conn.request({"op": "synth", "text": "Chapter one.", "voice": "af_heart", "out": str(tmp_path / "a.wav")})
    conn.request({"op": "lease", "channel": "reader"})
    conn.request({"op": "lease", "channel": "reader"})
    conn.sock.close()
    deadline = _t.monotonic() + 2
    while "lease: reader released" not in caplog.messages and _t.monotonic() < deadline:
        _t.sleep(0.02)
    msgs = caplog.messages
    assert msgs.count("say: cue af_sky 'Build passed.'") == 1
    synths = [m for m in msgs if m.startswith("synth: ")]
    assert len(synths) == 1 and synths[0].startswith("synth: af_heart 0.1s in ") and synths[0].endswith("'Chapter one.'")
    assert msgs.count("lease: reader held") == 1 and msgs.count("lease: reader released") == 1


def _when_playing(sp, then, timeout=3.0):
    """Call `then` once the speaker has started playing something."""
    import time as _t

    def watch():
        deadline = _t.monotonic() + timeout
        while _t.monotonic() < deadline:
            if sp.status()["playing"]:
                then()
                return
            _t.sleep(0.01)
    threading.Thread(target=watch, daemon=True).start()


def _quiet_within(sp, seconds):
    import time as _t
    deadline = _t.monotonic() + seconds
    while _t.monotonic() < deadline:
        if sp.current is None:
            return True
        _t.sleep(0.01)
    return False


def test_cli_say_block_waits_for_the_end(running, capsys):
    import time as _t
    t = _t.monotonic()
    assert client.main(["say", "--block", "Hello there.|400"]) == client.EXIT_OK
    assert _t.monotonic() - t >= 0.35
    assert capsys.readouterr().out.strip() == "finished"


def test_cli_say_block_reads_stdin(running, monkeypatch, capsys):
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO("Line one\nline two.|100\n"))
    assert client.main(["say", "--block", "--channel", "cue", "-"]) == client.EXIT_OK
    assert running.player.played[-1] == "Line one line two."


def test_cli_say_block_dropped(running, capsys):
    running.say("long", "Long reply.|1500", "af_heart", "assistant")
    assert client.main(["say", "--block", "--wait", "0.2", "Too late."]) == client.EXIT_FAILED
    assert capsys.readouterr().out.strip() == "dropped: busy"


def test_cli_say_block_stopped_elsewhere(running, capsys):
    _when_playing(running, lambda: running.stop(channel="cue"))
    assert client.main(["say", "--block", "Interrupted.|3000"]) == client.EXIT_FAILED
    assert capsys.readouterr().out.startswith("stopped")


def test_cli_say_block_sigterm_stops_its_speech(running):
    import signal
    before = signal.getsignal(signal.SIGTERM)
    _when_playing(running, lambda: os.kill(os.getpid(), signal.SIGTERM))
    assert client.main(["say", "--block", "A long notification.|5000"]) == 128 + signal.SIGTERM
    assert _quiet_within(running, 1.0)
    assert signal.getsignal(signal.SIGTERM) is before    # handlers put back


def _wait_playing(sp, timeout=3.0):
    import time as _t
    deadline = _t.monotonic() + timeout
    while not sp.status()["playing"]:
        assert _t.monotonic() < deadline, "never started"
        _t.sleep(0.01)


def test_stop_on_close_stops_when_the_client_dies(running):
    conn = client.Connection()
    conn.send({"op": "say", "id": "k1", "text": "Killed mid sentence.|5000", "stop_on_close": True})
    _wait_playing(running)
    conn.sock.close()
    assert _quiet_within(running, 1.0)


def test_without_stop_on_close_speech_outlives_its_client(running):
    conn = client.Connection()
    conn.send({"op": "say", "id": "k2", "text": "Fire and forget.|1500"})
    _wait_playing(running)
    conn.sock.close()
    assert not _quiet_within(running, 0.6)
