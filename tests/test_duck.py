from deskvoice.duck import MAX_DUCKED, TAIL, Ducker, fading_file


class FakeAudio:
    """A fake pactl: streams by index, volumes as percent."""

    def __init__(self, streams):
        self.streams = streams                  # index -> (app, volume 0..1, corked)

    def list(self):
        return [{"index": i, "corked": c, "volume": {"mono": {"value": int(v * 65536)}},
                 "properties": {"application.name": app}} for i, (app, v, c) in self.streams.items()]

    def set(self, index, pct):
        app, _, c = self.streams[index]
        self.streams[index] = (app, pct / 100, c)

    def vol(self, index):
        return round(self.streams[index][1], 2)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make(streams, fading=False):
    audio, clock = FakeAudio(streams), Clock()
    d = Ducker(level=0.3, list_streams=audio.list, set_volume=audio.set, fading=lambda: fading,
               clock=clock, sleep=lambda s: None)
    return d, audio, clock


def test_ducks_while_playing_and_restores_exactly():
    d, audio, clock = make({1: ("Firefox", 0.65, False), 2: ("deskvoice", 1.0, False), 3: ("mpv", 0.8, True)})
    d.playing(True)
    d.tick()
    assert audio.vol(1) == 0.19                 # 30% of 65% (64.999% as pactl stores it), whole percent
    assert audio.vol(2) == 1.0                  # deskvoice never ducks itself
    assert audio.vol(3) == 0.8                  # paused (corked) streams are left alone
    d.playing(False)
    d.tick()
    assert audio.vol(1) == 0.19                 # the tail keeps it down a moment
    clock.t += TAIL + 0.01
    d.tick()
    assert audio.vol(1) == 0.65


def test_hold_keeps_it_down_between_utterances():
    d, audio, clock = make({1: ("Firefox", 0.5, False)})
    d.hold(16)                                  # a client: listening
    d.tick()
    assert audio.vol(1) == 0.15
    clock.t += 10
    d.tick()
    assert audio.vol(1) == 0.15
    d.hold(0)                                   # nothing came of it
    d.tick()
    assert audio.vol(1) == 0.5


def test_hold_expires():
    d, audio, clock = make({1: ("Firefox", 0.5, False)})
    d.hold(5)
    d.tick()
    clock.t += 5.1
    d.tick()
    assert audio.vol(1) == 0.5


def test_never_down_longer_than_the_cap():
    d, audio, clock = make({1: ("Firefox", 0.5, False)})
    d.playing(True)                             # a "done" that never arrives
    d.tick()
    clock.t += MAX_DUCKED + 1
    d.tick()
    assert audio.vol(1) == 0.5


def test_stands_aside_while_another_program_fades():
    d, audio, clock = make({1: ("Firefox", 0.5, False)}, fading=True)
    d.playing(True)
    d.tick()
    assert audio.vol(1) == 0.5
    assert not d.status()["ducked"]


def test_restore_reaches_a_recreated_stream_by_app_name():
    d, audio, clock = make({1: ("Firefox", 0.6, False)})
    d.playing(True)
    d.tick()
    # Firefox tears its stream down and makes a new one at the ducked volume.
    audio.streams = {7: ("Firefox", 0.18, False)}
    d.playing(False)
    clock.t += TAIL + 0.01
    d.tick()
    assert audio.vol(7) == 0.6


def test_a_stream_paused_while_ducked_still_gets_its_volume_back():
    d, audio, clock = make({1: ("Firefox", 0.6, False)})
    d.playing(True)
    d.tick()
    audio.streams[1] = ("Firefox", audio.vol(1), True)      # paused while ducked
    d.playing(False)
    clock.t += TAIL + 0.01
    d.tick()
    assert audio.vol(1) == 0.6


def test_excluded_and_own_streams_are_never_ducked():
    audio, clock = FakeAudio({1: ("my-reader", 1.0, False), 2: ("deskvoice-cue", 1.0, False)}), Clock()
    d = Ducker(level=0.3, exclude=["my-reader"], list_streams=audio.list, set_volume=audio.set,
               clock=clock, sleep=lambda s: None)
    d.playing(True)
    d.tick()
    assert audio.vol(1) == 1.0 and audio.vol(2) == 1.0


def test_requests_never_wait_on_a_ramp():
    import threading, time
    d, audio, clock = make({1: ("Firefox", 0.5, False)})
    d.sleep = time.sleep                         # real ramps: ~0.7 s to restore
    d.playing(True)
    with d._lock:
        d.tick()
    d.playing(False)
    clock.t += 1
    worker = threading.Thread(target=lambda: (d._lock.acquire(), d.tick(), d._lock.release()))
    worker.start()
    time.sleep(0.05)                             # the restore ramp is under way
    t = time.monotonic()
    d.hold(16)
    assert time.monotonic() - t < 0.05
    worker.join()


def test_fading_file(tmp_path):
    f = tmp_path / "status.json"
    assert fading_file("")() is False
    check = fading_file(str(f))
    assert check() is False                     # missing file: not fading
    f.write_text('{"fade": 0.4}')
    assert check() is True
    f.write_text('{"fade": 0}')
    assert check() is False


def test_a_crash_while_ducked_is_undone_on_the_next_start(tmp_path):
    state = tmp_path / "duck.json"
    audio, clock = FakeAudio({1: ("Firefox", 0.6, False)}), Clock()
    d = Ducker(level=0.3, list_streams=audio.list, set_volume=audio.set, clock=clock,
               sleep=lambda s: None, state_file=state)
    d.playing(True)
    d.tick()
    assert audio.vol(1) == 0.18 and state.exists()
    # killed here: no restore. The next run finds the file.
    d2 = Ducker(level=0.3, list_streams=audio.list, set_volume=audio.set, clock=clock,
                sleep=lambda s: None, state_file=state)
    d2.restore_saved()
    assert audio.vol(1) == 0.6 and not state.exists()


def test_a_stream_that_comes_back_later_at_the_ducked_volume_is_put_back(tmp_path):
    d, audio, clock = make({1: ("Firefox", 0.6, False)})
    d.playing(True)
    d.tick()
    audio.streams = {}                           # the video ended: no stream at restore time
    d.playing(False)
    clock.t += TAIL + 0.01
    d.tick()
    assert d.status()["pending"] == ["firefox"]
    clock.t += 30                                # half a minute later a new video starts...
    audio.streams = {5: ("Firefox", 0.18, False)}   # ...at the volume PipeWire remembered
    d.tick()
    assert audio.vol(5) == 0.6


def test_a_volume_you_changed_while_ducked_is_left_alone():
    d, audio, clock = make({1: ("Firefox", 0.6, False)})
    d.playing(True)
    d.tick()
    audio.set(1, 45)                             # you turned it up mid-duck
    d.playing(False)
    clock.t += TAIL + 0.01
    d.tick()
    assert audio.vol(1) == 0.45
    clock.t += 5                                 # and the pending sweep doesn't override it either
    d.tick()
    assert audio.vol(1) == 0.45


def test_a_fade_starting_mid_duck_gets_the_originals_back_first():
    fading = {"v": False}
    audio, clock = FakeAudio({1: ("Firefox", 0.6, False)}), Clock()
    d = Ducker(level=0.3, list_streams=audio.list, set_volume=audio.set, fading=lambda: fading["v"],
               clock=clock, sleep=lambda s: None)
    d.playing(True)
    d.tick()
    assert audio.vol(1) == 0.18
    fading["v"] = True                           # the presence sensor starts dimming the desk
    d.tick()
    assert audio.vol(1) == 0.6 and d.status()["yielding"]
    d.tick()
    assert audio.vol(1) == 0.6                   # and doesn't duck again while it fades


def test_force_restore_clears_everything():
    d, audio, clock = make({1: ("Firefox", 0.6, False)})
    d.hold(60)
    d.playing(True)
    d.tick()
    d.force_restore()
    assert audio.vol(1) == 0.6 and not d.holding() and not d.is_playing()
