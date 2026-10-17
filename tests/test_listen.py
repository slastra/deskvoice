import queue
import threading
from types import SimpleNamespace

import numpy as np

from deskvoice import listen


class FakeMic:
    def __init__(self, n=400):
        self.n, self.dropped = n, 0

    def read(self):
        if self.n == 0:
            raise listen.MicGone("out of frames")
        self.n -= 1
        return np.zeros(listen.FRAME, dtype=np.int16)

    def drain(self):
        pass


class FakeWake:
    """Scores in order, then zeros."""

    def __init__(self, scores):
        self.scores = list(scores)

    def reset(self):
        pass

    def predict(self, frame):
        return {"w": self.scores.pop(0) if self.scores else 0.0}


class FakeVAD:
    """Speech for `speech` frames after `delay` frames, then silence."""

    def __init__(self, delay=2, speech=10):
        self.delay, self.speech = delay, speech
        self.i = 0

    def reset_states(self):
        self.i = 0

    def predict(self, frame, frame_size=640):
        self.i += 1
        return 0.9 if self.delay < self.i <= self.delay + self.speech else 0.0


class FakeOut:
    def __init__(self):
        self.states, self.heards = [], []
        self.followup = queue.Queue()

    def brain_speaking(self):
        return False

    def state(self, s):
        self.states.append(s)

    def heard(self, p):
        self.heards.append(p)


def make(scores, transcript="What's up doc, turn on the office light.", speaking=False, vad=None):
    l = object.__new__(listen.Listener)
    l.args = SimpleNamespace(threshold=0.4, barge_threshold=0.6, start_timeout=4.0, end_silence=0.8,
                             max_utterance=10.0, dictate_silence=2.5)
    l.wake, l.wake_name, l.vad = FakeWake(scores), "w", vad or FakeVAD()
    l.asr = lambda audio: transcript
    l.out, l.clips, l.mic = FakeOut(), None, FakeMic()
    l.cues = SimpleNamespace(play=lambda name: None)
    import re as _re
    l.wake_prefix = _re.compile(r"^\s*(what'?s\s+up|sup)[,.]?\s+doc\b[,.?!]*\s*", _re.I)
    l.state, l.last_wake, l._stop = "waiting", None, threading.Event()
    l.dictations, l.asr_lock = queue.Queue(), threading.Lock()
    l.mic_on = threading.Event()
    l.mic_on.set()
    l.dictation_view, l.heard_log = None, []
    l.on_heard = lambda payload: l.heard_log.append(payload["text"])
    l.voice_dictations = []
    l.on_voice_dictation = l.voice_dictations.append
    events = []
    l.speaking_hook = lambda: speaking
    l.barge_hook = lambda: events.append("barge")
    l.duck = lambda s: events.append(f"duck {s}")
    return l, events


def test_wake_then_request_is_published_without_the_wake_phrase():
    l, events = make([0.1, 0.2, 0.9])
    l.turn()
    assert l.out.states == ["listening"]
    assert events == ["duck 16"]
    h = l.out.heards[0]
    assert h["text"] == "turn on the office light." and h["wake_score"] == 0.9 and h["barge"] is False


def test_barge_in_stops_speech_locally_and_tells_the_brain():
    l, events = make([0.5, 0.7], speaking=True)   # 0.5 is under the 0.6 barge threshold
    l.turn()
    assert events[:2] == ["duck 16", "barge"]
    assert l.out.states == ["barge", "listening"]
    assert l.out.heards[0]["barge"] is True


def test_wake_under_threshold_while_speaking_is_ignored():
    l, events = make([0.5, 0.55, 0.5], speaking=True)
    l.mic = FakeMic(n=5)
    try:
        l.turn()
    except listen.MicGone:
        pass
    assert events == [] and l.out.states == []


def test_no_speech_after_the_wake_releases_the_duck():
    l, events = make([0.9], vad=FakeVAD(delay=10_000))
    l.turn()
    assert events == ["duck 16", "duck 0"]
    assert l.out.states == ["listening", "idle"]
    assert l.out.heards == []


def test_followup_window_opens_without_the_wake_word():
    l, events = make([], transcript="yes")
    l.out.followup.put(6.0)
    l.turn()
    assert l.out.heards[0]["followup"] is True and l.out.heards[0]["wake_score"] is None


class ScriptedASR:
    """Hands out transcripts in order, one per call."""

    def __init__(self, *texts):
        self.texts = list(texts)

    def __call__(self, audio):
        return self.texts.pop(0) if self.texts else ""


class PhrasesVAD:
    """Speech in bursts: `pattern` is a list of (speech_frames, silent_frames)."""

    def __init__(self, pattern):
        self.frames = []
        for speech, quiet in pattern:
            self.frames += [0.9] * speech + [0.0] * quiet
        self.i = 0

    def reset_states(self):
        pass

    def predict(self, frame, frame_size=640):
        p = self.frames[self.i] if self.i < len(self.frames) else 0.0
        self.i += 1
        return p


def test_voice_dictation_is_not_sent_to_the_brain():
    # The request ("please dictate: buy milk"), then two phrases, then a long pause.
    l, events = make([0.9])
    l.vad = PhrasesVAD([(10, 12), (8, 12), (8, 40)])
    l.asr = ScriptedASR("What's up doc, please dictate: buy milk.", "And eggs.", "And bread.")
    l.turn()
    assert l.out.heards == []
    assert l.voice_dictations == ["buy milk. And eggs. And bread."]
    assert l.out.states[-1] == "idle"


def test_stop_dictating_ends_it_and_is_left_out():
    l, events = make([0.9])
    l.vad = PhrasesVAD([(10, 12), (8, 12), (6, 12), (8, 12)])
    l.asr = ScriptedASR("Please dictate.", "Remember the meeting at three.", "Stop dictating.", "Never heard.")
    l.turn()
    assert l.voice_dictations == ["Remember the meeting at three."]


def test_a_request_mentioning_dictation_mid_sentence_is_not_dictation():
    l, events = make([0.9], transcript="What's up doc, how do I dictate on my phone?")
    l.turn()
    assert l.voice_dictations == [] and l.out.heards[0]["text"] == "how do I dictate on my phone?"


def test_socket_dictation_session_runs_until_stopped():
    l, events = make([])
    l.vad = PhrasesVAD([(8, 10), (8, 10)] + [(0, 1)] * 100)
    l.asr = ScriptedASR("First phrase.", "Second phrase.")
    got = []
    session = listen.Dictation(emit=lambda n, v: got.append((n, v)))
    l.mic = FakeMic(n=60)
    l.request_dictation(session)
    # stop once both phrases have been transcribed
    orig = session.emit
    def emit(n, v):
        orig(n, v)
        if n == "partial" and v == "First phrase. Second phrase.":
            session.stop.set()
    session.emit = emit
    l.turn()
    names = [n for n, _ in got]
    assert names[0] == "state" and "level" in names and "time" in names
    assert ("raw", "First phrase. Second phrase.") in got
    assert l.state == "waiting"


def test_enter_on_its_own_ends_voice_dictation_and_stays_for_the_finish():
    l, events = make([0.9])
    l.vad = PhrasesVAD([(10, 12), (8, 12), (5, 12), (8, 12)])
    l.asr = ScriptedASR("Please dictate.", "Ship it tonight.", "Enter.", "Never heard.")
    l.turn()
    assert l.voice_dictations == ["Ship it tonight. Enter."]


def test_wake_word_off_closes_the_turn():
    l, events = make([0.9])
    l.mic_on.clear()
    try:
        l.turn()
        raise AssertionError("expected MicGone")
    except listen.MicGone:
        pass
    assert l.out.heards == [] and events == []


def test_heard_requests_reach_recent():
    l, events = make([0.9])
    l.turn()
    assert l.heard_log == ["turn on the office light."]


def test_deliver_split_submit():
    from deskvoice.deliver import split_submit
    assert split_submit("See you at three. Enter.") == ("See you at three.", True)
    assert split_submit("Looks good, hit enter") == ("Looks good", True)
    assert split_submit("Type the password you enter.") == ("Type the password you enter.", False)
    assert split_submit("The function should return.") == ("The function should return.", False)
    assert split_submit("Enter.") == ("", True)
