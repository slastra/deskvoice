from deskvoice import stream_restore as sr


def stream(props, vol=1.0, index=7, nchan=2):
    chans = ["front-left", "front-right"][:nchan] if nchan <= 2 else [f"c{i}" for i in range(nchan)]
    return {"index": index, "properties": props,
            "volume": {c: {"value": int(vol * 65536)} for c in chans}}


def test_identity_follows_wireplumbers_key_order():
    assert sr.identity(stream({"application.id": "org.x", "application.name": "X"})) == {"application.id": "org.x"}
    assert sr.identity(stream({"application.name": "Firefox", "media.name": "Song"})) == {"application.name": "Firefox"}
    assert sr.identity(stream({"media.role": "Notification", "application.name": "X"})) == {"media.role": "Notification"}
    assert sr.identity(stream({"media.name": "only"})) is None   # pw-play would key itself higher


class FakeProc:
    def __init__(self):
        self.killed = False

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


def run(held, original=0.6, lowest=0.18, appears=True):
    """A stand-in that shows up holding `held` (WirePlumber's saved value)."""
    state = {"vol": held, "token": None, "set": None}
    proc = FakeProc()

    def spawn(cmd, **kw):
        spa = cmd[cmd.index("-P") + 1]
        state["token"] = spa.split('"desk.volume-proxy" = "')[1].split('"')[0]
        state["cmd"] = cmd
        return proc

    def list_streams():
        if not appears or state["token"] is None:
            return []
        return [stream({"application.name": "Firefox", sr.PROXY_MARK: state["token"]}, state["vol"], index=42)]

    def set_volume(index, pct):
        state["set"] = (index, pct)
        state["vol"] = pct / 100

    t = [0.0]

    def clock():
        t[0] += 0.05
        return t[0]
    result = sr.repair({"application.name": "Firefox"}, 2, original, lowest, list_streams=list_streams,
                       set_volume=set_volume, spawn=spawn, sleep=lambda s: None, clock=clock)
    return result, state, proc


def test_repair_sets_a_ducked_saved_volume_back():
    result, state, proc = run(0.18)
    assert result == "fixed" and state["set"] == (42, 60) and proc.killed
    assert "--channels" in state["cmd"] and state["cmd"][state["cmd"].index("--channels") + 1] == "2"


def test_repair_also_fixes_a_value_caught_mid_ramp():
    assert run(0.40)[0] == "fixed"


def test_repair_leaves_a_value_somebody_set():
    result, state, proc = run(0.9)                # turned UP by someone: not ours
    assert result == "kept" and state["set"] is None and proc.killed
    assert run(0.05)[0] == "kept"                 # below anything we wrote


def test_repair_is_a_no_op_when_already_right():
    result, state, _ = run(0.6)
    assert result == "ok" and state["set"] is None


def test_repair_fails_cleanly_when_the_stand_in_never_shows():
    result, state, proc = run(0.18, appears=False)
    assert result == "failed" and proc.killed
