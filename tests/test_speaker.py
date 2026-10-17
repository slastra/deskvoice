import threading
import time

import numpy as np
import pytest

from deskvoice.speaker import Speaker


class FakePlayer:
    """Records what plays; each sentence 'plays' for its sample count in ms."""

    def __init__(self):
        self.played = []
        self.killed = threading.Event()
        self._lock = threading.Lock()

    def play(self, samples, rate):
        text = samples.text
        with self._lock:
            self.played.append(text)
        self.killed.clear()
        self.killed.wait(len(samples) / 1000)

    def kill(self):
        self.killed.set()


class Tagged(np.ndarray):
    pass


def fake_render(delay=0.0, fail=()):
    def render(voice, text):
        time.sleep(delay)
        if text in fail:
            raise RuntimeError(f"cannot say {text}")
        a = np.zeros(int(text.split("|")[1]) if "|" in text else 50, dtype=np.float32).view(Tagged)
        a.text = text.split("|")[0]
        return a, 1000
    return render


class Events:
    def __init__(self):
        self.items = []
        self.cv = threading.Condition()

    def __call__(self, ev):
        with self.cv:
            self.items.append(ev)
            self.cv.notify_all()

    def wait(self, uid, kind, timeout=3.0):
        deadline = time.monotonic() + timeout
        with self.cv:
            while True:
                for ev in self.items:
                    if ev["id"] == uid and ev["event"] == kind:
                        return ev
                left = deadline - time.monotonic()
                if left <= 0:
                    raise AssertionError(f"no {kind} for {uid}; got {self.items}")
                self.cv.wait(left)

    def kinds(self, uid):
        return [e["event"] for e in self.items if e["id"] == uid]


@pytest.fixture
def setup():
    player = FakePlayer()
    sp = Speaker(fake_render(), player=player)
    sp.start()
    yield sp, player, Events()
    sp.close()


def test_single_cue_lifecycle(setup):
    sp, player, ev = setup
    sp.say("a", "Build passed.", "af_sky", "cue", notify=ev)
    started = ev.wait("a", "started")
    ev.wait("a", "finished")
    assert started["secs"] == 0.05
    assert ev.kinds("a") == ["queued", "started", "finished"]
    assert player.played == ["Build passed."]


def test_streamed_sentences_play_in_order(setup):
    sp, player, ev = setup
    sp.say("r", "One.", "af_heart", "assistant", final=False, notify=ev)
    sp.say("r", "Two.", "af_heart", "assistant", final=False, notify=ev)
    time.sleep(0.05)
    sp.say("r", "Three.", "af_heart", "assistant", final=True, notify=ev)
    ev.wait("r", "finished")
    assert player.played == ["One.", "Two.", "Three."]


def test_assistant_goes_before_a_waiting_cue(setup):
    sp, player, ev = setup
    sp.say("c1", "Long cue.|300", "af_sky", "cue", notify=ev)          # playing
    ev.wait("c1", "started")
    sp.say("c2", "Waiting cue.", "af_sky", "cue", notify=ev)
    sp.say("as", "Reply.", "af_heart", "assistant", notify=ev)
    ev.wait("c2", "finished")
    assert player.played == ["Long cue.", "Reply.", "Waiting cue."]


def test_cue_dropped_after_max_wait(setup):
    sp, player, ev = setup
    sp.say("as", "A long reply.|800", "af_heart", "assistant", notify=ev)
    ev.wait("as", "started")
    sp.say("c", "Too late.", "af_sky", "cue", max_wait=0.2, notify=ev)
    dropped = ev.wait("c", "dropped")
    assert dropped["reason"] == "busy"
    assert "Too late." not in player.played


def test_stop_cuts_the_current_utterance(setup):
    sp, player, ev = setup
    sp.say("as", "First.|2000", "af_heart", "assistant", final=False, notify=ev)
    sp.say("as", "Never said.", "af_heart", "assistant", final=True, notify=ev)
    ev.wait("as", "started")
    t = time.monotonic()
    assert sp.stop(channel="assistant") == 1
    ev.wait("as", "stopped")
    time.sleep(0.1)
    assert time.monotonic() - t < 0.5
    assert player.played == ["First."]


def test_render_failure_is_reported():
    player = FakePlayer()
    sp = Speaker(fake_render(fail={"Broken."}), player=player)
    sp.start()
    ev = Events()
    try:
        sp.say("x", "Broken.", "af_heart", "cue", notify=ev)
        err = ev.wait("x", "error")
        assert "cannot say" in err["message"]
        sp.say("y", "Fine.", "af_heart", "cue", notify=ev)
        ev.wait("y", "finished")
    finally:
        sp.close()


def test_say_after_end_is_refused(setup):
    sp, player, ev = setup
    sp.say("z", "Hello.|400", "af_heart", "assistant", final=True, notify=ev)
    with pytest.raises(ValueError):
        sp.say("z", "More.", "af_heart", "assistant", notify=ev)


def test_unpronounceable_sentences_are_skipped(setup):
    sp, player, ev = setup
    sp.say("m", "Here's a story.", "af_heart", "assistant", final=False, notify=ev)
    sp.say("m", "---", "af_heart", "assistant", final=False, notify=ev)
    sp.say("m", "The end.", "af_heart", "assistant", final=True, notify=ev)
    ev.wait("m", "finished")
    assert player.played == ["Here's a story.", "The end."]
    assert "error" not in ev.kinds("m")
