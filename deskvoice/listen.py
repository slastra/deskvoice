"""The listener: mic -> wake word -> record until silence -> Parakeet -> clients.

A wake word (openWakeWord) opens a request; Silero VAD records it until you
pause; Parakeet (sherpa-onnx) transcribes it locally. The text goes to MQTT
(when enabled) and to socket subscribers as a "heard" event. Living beside the
speaker, it knows when deskvoice is talking (the wake word then needs the
stricter barge-in threshold, and a hit stops the speech) and ducks other audio
from the wake word on.

MQTT topics (all JSON), under the configured prefix (default "deskvoice"):
  <prefix>/heard           out  {id, text, raw, barge, wake_score, followup, audio_s, asr_ms, ts}
  <prefix>/listen/status   out  "online" / "offline" (retained, offline is the last will)
  <prefix>/listen/state    out  {"state": "listening"|"idle"|"barge"}
  <prefix>/speaking        in   {"active": bool}  a client speaking with its own TTS, for
                                barge-in (treated like deskvoice speaking)
  <prefix>/listen          in   {"followup": seconds}  listen again without the wake word

"dictate ..." after the wake word starts a voice dictation instead (see dictate()).
Trigger and near-miss clips can be kept for retraining the wake word (clips_dir).
"""

import collections
import json
import logging
import queue
import re
import subprocess
import threading
import time
import uuid
import wave
from pathlib import Path

import numpy as np

from .cues import Cues
from .duck import present_file

log = logging.getLogger("deskvoice")

RATE = 16000
FRAME = 1280                                     # 80 ms, openWakeWord's chunk size
FRAME_S = FRAME / RATE


# "what's up doc, please dictate: buy milk and eggs" is dictation, not a request for the
# brain: whatever follows "dictate" starts the text, and listening carries on until a
# long pause or "stop dictating".
DICTATE_INTENT = re.compile(
    r"^\s*(?:(?:ok(?:ay)?|hey|now)[,.]?\s+)?(?:please[,.]?\s+)?(?:can you\s+)?"
    r"(?:dictate|take\s+(?:a\s+|some\s+)?dictation|start\s+dictat(?:ing|ion))\b[\s:,.!?-]*(.*)$",
    re.I | re.S)
# "Enter" said as its own phrase ends a voice dictation at once and stays in the text,
# so the finish step presses Return after pasting.
ENTER_ALONE = re.compile(r"^\s*(?:(?:and|now|then)[,.]?\s+)?(?:press\s+|hit\s+)?(?:enter|return)[.!]?\s*$", re.I)
STOP_DICTATING = re.compile(
    r"^\s*(?:(?:ok(?:ay)?|and)[,.]?\s+)?(?:stop|end|finish|done)\s+(?:the\s+)?dictat(?:ing|ion)\b|^\s*that'?s\s+all\b",
    re.I)




class MicGone(Exception):
    pass


STATE_FILE = Path.home() / ".local/state/deskvoice/state.json"


def load_mic_setting():
    try:
        return bool(json.loads(STATE_FILE.read_text()).get("mic", True))
    except (OSError, ValueError):
        return True


def save_mic_setting(on):
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
        data["mic"] = bool(on)
        STATE_FILE.write_text(json.dumps(data) + "\n")
    except (OSError, ValueError) as e:
        log.warning("listen: saving the mic setting: %s", e)


class Mic:
    """Continuous pw-record capture, delivered as 80 ms int16 frames."""

    def __init__(self, target=None):
        cmd = ["pw-record", "--rate", str(RATE), "--channels", "1", "--format", "s16",
               "-P", "{ application.name = deskvoice }"]
        if target:
            cmd += ["--target", target]
        self.proc = subprocess.Popen(cmd + ["-"], stdout=subprocess.PIPE)
        self.frames = queue.Queue(maxsize=200)
        self.dropped = 0
        threading.Thread(target=self._pump, daemon=True, name="mic").start()

    def _pump(self):
        while chunk := self.proc.stdout.read(FRAME * 2):
            if len(chunk) == FRAME * 2:
                try:
                    self.frames.put_nowait(np.frombuffer(chunk, dtype=np.int16))
                except queue.Full:                # consumer stalled (ASR); drop oldest
                    self.dropped += 1
                    self.frames.get_nowait()
                    self.frames.put_nowait(np.frombuffer(chunk, dtype=np.int16))
        self.frames.put(None)

    def read(self):
        frame = self.frames.get()
        if frame is None:
            raise MicGone("pw-record exited")
        return frame

    def drain(self):
        while not self.frames.empty():
            self.frames.get_nowait()

    def close(self):
        self.proc.terminate()


class Transcriber:
    def __init__(self, model_dir, threads=4):
        import sherpa_onnx
        m = Path(model_dir)
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(m / "encoder.int8.onnx"), decoder=str(m / "decoder.int8.onnx"),
            joiner=str(m / "joiner.int8.onnx"), tokens=str(m / "tokens.txt"),
            model_type="nemo_transducer", num_threads=threads)

    def __call__(self, audio):
        stream = self.rec.create_stream()
        stream.accept_waveform(RATE, audio.astype(np.float32) / 32768)
        self.rec.decode_stream(stream)
        return stream.result.text.strip()


class Publisher:
    """MQTT out, plus followup requests (and a client's own-TTS speaking flag) in.
    With MQTT off it only keeps the followup queue, fed over the socket."""

    def __init__(self, host, port, prefix, enabled=True):
        self.brain_speaking_until = 0.0
        self.followup = queue.Queue()
        self.prefix = prefix.rstrip("/")
        self.client = None
        if not enabled:
            return
        import paho.mqtt.client as mqtt
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"deskvoice-listen-{uuid.uuid4().hex[:6]}")
        c.will_set(f"{self.prefix}/listen/status", "offline", qos=1, retain=True)
        c.on_connect = self._on_connect
        c.on_message = self._on_message
        c.reconnect_delay_set(1, 30)
        c.connect_async(host, port, keepalive=30)
        c.loop_start()
        self.client = c

    def _on_connect(self, client, userdata, flags, reason, props):
        log.info("listen: mqtt connected (%s), topics under %s/", reason, self.prefix)
        client.publish(f"{self.prefix}/listen/status", "online", qos=1, retain=True)
        client.subscribe([(f"{self.prefix}/speaking", 1), (f"{self.prefix}/listen", 1)])

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload or b"{}")
        except ValueError:
            return
        if msg.topic == f"{self.prefix}/speaking":
            # A stuck "active" can't deafen us forever: cap it at 60 s.
            self.brain_speaking_until = time.monotonic() + 60 if data.get("active") else 0.0
        elif msg.topic == f"{self.prefix}/listen" and data.get("followup"):
            self.followup.put(float(data["followup"]))

    def brain_speaking(self):
        return time.monotonic() < self.brain_speaking_until

    def state(self, state):
        if self.client:
            self.client.publish(f"{self.prefix}/listen/state", json.dumps({"state": state}), qos=0)

    def heard(self, payload):
        if self.client:
            self.client.publish(f"{self.prefix}/heard", json.dumps(payload), qos=1)

    def close(self):
        if not self.client:
            return
        try:
            self.client.publish(f"{self.prefix}/listen/status", "offline", qos=1, retain=True).wait_for_publish(2)
            self.client.disconnect()              # clean disconnect: the broker skips the last will
        except (RuntimeError, ValueError) as e:   # not connected: the broker's last will says offline
            log.info("listen: mqtt offline status not sent: %s", e)
        finally:
            self.client.loop_stop()


class Clips:
    """Saves 2 s of audio around every wake trigger and near-miss, for review and retraining.

    A near-miss is a score that climbs past `near` without reaching the threshold: either
    you said the phrase and it wasn't caught, or something else came close. Triggers are
    kept too, so false wakes (a video, the TV) can be labeled as negatives.
    """

    CLIP_FRAMES = int(2.0 / FRAME_S)            # 25 frames: the length of the training takes

    def __init__(self, directory, near=0.15, keep=1500, gap=3.0):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index = self.dir / "clips.jsonl"
        self.near, self.keep, self.gap = near, keep, gap
        self.ring = collections.deque(maxlen=int(4.0 / FRAME_S))
        self.peak = None                        # (score, frames to the end of the clip) while near
        self.last_near = 0.0
        self.saved = 0

    def push(self, frame):
        self.ring.append(frame)

    def _write(self, kind, score, frames, speaking, extra=None):
        ts = time.time()
        name = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(ts))}-{int(ts * 1000) % 1000:03d}_{kind}_{score:.2f}.wav"
        with wave.open(str(self.dir / name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(np.concatenate(frames).tobytes())
        self.note(name, kind=kind, score=round(score, 3), ts=ts, speaking=speaking, **(extra or {}))
        self.saved += 1
        if self.saved % 50 == 0:
            self.prune()
        return name

    def note(self, name, **fields):
        with self.index.open("a") as f:
            f.write(json.dumps({"file": name, **fields}) + "\n")

    def trigger(self, score, speaking=False):
        """The 2 s ending now: the model fires at the end of the phrase."""
        self.peak = None
        return self._write("trigger", score, list(self.ring)[-self.CLIP_FRAMES:], speaking)

    def observe(self, score, threshold, speaking=False):
        """Track near-misses; save one when its score falls away."""
        if self.near <= score < threshold:
            if self.peak is None or score > self.peak[0]:
                self.peak = (score, 0)
            else:
                self.peak = (self.peak[0], self.peak[1] + 1)
            if self.peak[1] < 12:               # keep following for up to ~1 s past the peak
                return
        if self.peak is None:
            return
        score_peak, after = self.peak
        self.peak = None
        if time.monotonic() - self.last_near < self.gap:
            return
        self.last_near = time.monotonic()
        # End the clip 3 frames (240 ms) after the peak, like a trigger would.
        frames = list(self.ring)
        end = len(frames) - max(after - 3, 0)
        clip = frames[max(end - self.CLIP_FRAMES, 0):end]
        if len(clip) == self.CLIP_FRAMES:
            self._write("near", score_peak, clip, speaking)

    def prune(self):
        wavs = sorted(self.dir.glob("*.wav"))
        for old in wavs[:-self.keep] if len(wavs) > self.keep else []:
            old.unlink(missing_ok=True)


class Dictation:
    """One dictation session on the shared mic. Socket clients (a dictation
    hotkey) stop or cancel it; a voice session ends itself on a long pause or
    "stop dictating". Events go to `emit(name, value)`: state, level, time,
    partial, raw, error - the names dictate.py already speaks."""

    def __init__(self, emit=lambda name, value: None, voice=False, first_text="", save=None,
                 end_silence=2.5, start_timeout=6.0, max_seconds=120.0):
        self.emit, self.voice, self.save = emit, voice, save
        self.first_text = first_text.strip()
        self.end_silence, self.start_timeout, self.max_seconds = end_silence, start_timeout, max_seconds
        self.stop = threading.Event()
        self.cancel = threading.Event()
        self.raw = None
        self.done = threading.Event()


class Listener:
    """The wake-word loop. Hooks into the rest of deskvoice:
      speaking()   is deskvoice (or the brain's own Piper) talking right now
      barge()      the wake word interrupted it: stop speaking
      duck(secs)   keep other audio down while listening (0 releases)
    """

    def __init__(self, args, speaking=lambda: False, barge=lambda: None, duck=lambda s: None):
        import openwakeword.utils
        from openwakeword.model import Model
        from openwakeword.vad import VAD
        self.args = args
        self.speaking_hook, self.barge_hook, self.duck = speaking, barge, duck
        self.cues = Cues(args.cues)
        self.wake_prefix = re.compile(args.wake_prefix, re.I) if args.wake_prefix else None
        # The wake model is a path to your own .onnx, or a built-in openWakeWord
        # name ("hey_jarvis"). Either way the shared feature and VAD models are
        # fetched if missing (a no-op once present).
        wake = str(args.wake_model)
        if Path(wake).expanduser().exists():
            openwakeword.utils.download_models(model_names=["deskvoice-none"])  # matches no stock word
            wake = str(Path(wake).expanduser())
        else:
            openwakeword.utils.download_models(model_names=[wake])
        self.wake = Model(wakeword_models=[wake], inference_framework="onnx")
        self.wake_name = next(iter(self.wake.models))
        self.vad = VAD()
        t = time.monotonic()
        self.asr = Transcriber(args.asr, args.asr_threads)
        log.info("listen: parakeet loaded in %.1fs, wake model %s", time.monotonic() - t, Path(str(args.wake_model)).name)
        self.out = Publisher(args.mqtt_host, args.mqtt_port, args.mqtt_prefix, args.mqtt_enabled)
        self.clips = Clips(args.clips_dir) if args.clips_dir else None
        self.present = present_file(args.presence_file)   # is anyone at the desk?
        self.mic = None
        self.state = "starting"
        self.last_wake = None                    # (score, unix time)
        self._stop = threading.Event()
        self.dictations = queue.Queue()          # sessions asked for over the socket (Super+X)
        # The wake word's mic switch. Off means no capture at all (not just
        # ignoring it); Super+X still opens the mic for its own session.
        self.mic_on = threading.Event()
        if load_mic_setting():
            self.mic_on.set()
        self.dictation_view = None               # {"partial", "secs", "voice"} while dictating
        self.on_heard = lambda payload: None     # deskvoice: recent list and socket "heard" events
        self.on_voice_dictation = lambda raw: None   # deskvoice cleans and pastes it
        self.asr_lock = threading.Lock()         # the socket's transcribe op shares Parakeet

    def set_mic(self, on):
        if on:
            self.mic_on.set()
        else:
            self.mic_on.clear()
        save_mic_setting(on)
        log.info("listen: wake word %s", "on" if on else "off (mic closed)")

    def request_dictation(self, session):
        """Queue a socket session; it starts as soon as the wake loop is free."""
        self.dictations.put(session)

    def transcribe(self, audio):
        with self.asr_lock:
            return self.asr(audio)

    def speaking(self):
        return self.speaking_hook() or self.out.brain_speaking()

    def status(self):
        return {"state": self.state, "mic": self.mic_on.is_set(), "last_wake": self.last_wake,
                "dropped_frames": self.mic.dropped if self.mic else 0,
                "threshold": self.args.threshold, "barge_threshold": self.args.barge_threshold,
                "dictation": self.dictation_view}

    def wait_for_wake(self):
        """Block until the wake word fires (or a follow-up is requested).

        Returns (score, followup_s, barge). While deskvoice talks, the wake word still
        counts at the stricter --barge-threshold, which interrupts it (barge-in);
        --barge-threshold 0 ignores the mic while it talks.
        """
        self.wake.reset()
        was_speaking = False
        while not self._stop.is_set():
            try:
                return None, self.out.followup.get_nowait(), False
            except queue.Empty:
                pass
            if not self.dictations.empty():
                return None, 0.0, False          # turn() picks the session up
            if not self.mic_on.is_set():
                raise MicGone("wake word turned off")
            frame = self.mic.read()
            if self.clips:
                self.clips.push(frame)
            speaking = self.speaking()
            if speaking != was_speaking:
                self.wake.reset()                 # a fresh window either side of a reply
                was_speaking = speaking
            if speaking and not self.args.barge_threshold:
                continue                          # don't hear our own replies
            score = float(self.wake.predict(frame)[self.wake_name])
            threshold = self.args.barge_threshold if speaking else self.args.threshold
            if self.clips:
                self.clips.observe(score, threshold, speaking)
            if score >= threshold:
                if not speaking and not self.present():
                    # Nobody at the desk: a video, the TV, the room. Barge-in
                    # (deskvoice is talking) is never blocked.
                    log.info("listen: wake %.2f ignored, nobody at the desk", score)
                    if self.clips:
                        self.clips.note(self.clips.trigger(score), outcome="ignored: nobody at the desk")
                    self.wake.reset()
                    continue
                return score, 0.0, speaking
        raise MicGone("stopping")

    def record(self, start_timeout):
        """Record until speech then silence. Returns int16 audio, or None if nobody spoke."""
        self.vad.reset_states()
        frames, speech_frames, silent = [], 0, 0
        started = time.monotonic()
        max_frames = int(self.args.max_utterance / FRAME_S)
        end_silence = int(self.args.end_silence / FRAME_S)
        while len(frames) < max_frames:
            frame = self.mic.read()
            frames.append(frame)
            if self.vad.predict(frame, frame_size=640) >= 0.5:
                speech_frames += 1
                silent = 0
            elif speech_frames:
                silent += 1
                if silent >= end_silence:
                    break
            elif time.monotonic() - started > start_timeout:
                return None
        if speech_frames < 3:                     # under 240 ms of speech: a cough, not a request
            return None
        return np.concatenate(frames)

    def dictate(self, session):
        """Record and transcribe until stopped: each phrase is transcribed at its
        pause, so stopping costs only the last phrase (~0.2 s) however long it ran."""
        self.state = "dictating"
        self.duck(session.max_seconds + 10)
        def emit(name, value):
            session.emit(name, value)
            if name == "partial":
                self.dictation_view = {**(self.dictation_view or {}), "partial": value}
        self.dictation_view = {"partial": session.first_text, "secs": 0, "voice": session.voice}
        emit("state", "recording")
        segments = [session.first_text] if session.first_text else []
        heard_speech = bool(segments)
        seg, every, speech_frames, silent = [], [], 0, 0
        pause = int(0.6 / FRAME_S)               # a phrase ends at 0.6 s of quiet
        end = int(session.end_silence / FRAME_S)
        start_wait = int(session.start_timeout / FRAME_S)
        last_time = -1.0
        self.vad.reset_states()

        def flush():
            nonlocal seg, speech_frames
            if speech_frames >= 3:
                text = self.transcribe(np.concatenate(seg)).strip()
                if text and STOP_DICTATING.match(text):
                    return True
                if text and session.voice and ENTER_ALONE.match(text):
                    segments.append(text)          # finish turns it into a Return
                    emit("partial", " ".join(segments))
                    return True
                if text:
                    segments.append(text)
                    emit("partial", " ".join(segments))
            seg, speech_frames = [], 0
            return False

        stopped_by_voice = False
        try:
            while not session.stop.is_set() and not session.cancel.is_set() and not self._stop.is_set():
                frame = self.mic.read()
                every.append(frame)
                seg.append(frame)
                secs = len(every) * FRAME_S
                rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))) or 1.0
                emit("level", round(max(0.0, min(1.0, (20 * np.log10(rms / 32768) + 60) / 60)), 2))
                if secs - last_time >= 0.1:
                    emit("time", round(secs, 1))
                    last_time = secs
                    self.dictation_view = {**self.dictation_view, "secs": round(secs)}
                if self.vad.predict(frame, frame_size=640) >= 0.5:
                    speech_frames += 1
                    silent = 0
                    heard_speech = True
                else:
                    silent += 1
                    if speech_frames and silent >= pause:
                        if flush():
                            stopped_by_voice = True
                            break
                    if session.voice:
                        if heard_speech and silent >= end:
                            break
                        if not heard_speech and len(every) >= start_wait:
                            break
                if secs >= session.max_seconds:
                    emit("note", "Stopped at the two-minute limit")
                    break
            if session.cancel.is_set():
                return
            if not stopped_by_voice:
                emit("state", "transcribing")
                flush()
            session.raw = " ".join(segments).strip()
            if session.save and every:
                try:
                    Path(session.save).parent.mkdir(parents=True, exist_ok=True)
                    with wave.open(str(session.save), "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(RATE)
                        w.writeframes(np.concatenate(every).tobytes())
                except OSError as e:
                    log.warning("listen: saving the dictation clip: %s", e)
            if session.raw:
                emit("raw", session.raw)
            else:
                emit("error", "Nothing recognised")
            log.info("listen: dictation %.1fs, %d phrases%s", len(every) * FRAME_S, len(segments),
                     " (voice)" if session.voice else "")
        finally:
            session.done.set()
            self.dictation_view = None
            self.duck(0)
            self.state = "waiting"
            self.mic.drain()


    def _idle(self):
        self.out.state("idle")
        self.duck(0)
        self.state = "waiting"

    def turn(self):
        """One wake (or follow-up) and what came of it."""
        self.state = "waiting"
        score, followup, barge = self.wait_for_wake()
        if score is None and not followup:
            try:
                session = self.dictations.get_nowait()
            except queue.Empty:
                return
            self.dictate(session)
            return
        clip = None
        if score is not None and self.clips:
            clip = self.clips.trigger(score, speaking=barge)
        self.duck(16)                             # the recording lasts at most ~15 s
        if barge:
            self.barge_hook()                     # stop talking now, locally
            self.out.state("barge")               # and tell the brain to drop that request
        self.out.state("listening")
        self.state = "listening"
        if score is not None:
            self.last_wake = (round(score, 2), time.time())
            log.info("listen: wake %.2f%s", score, " (barge-in)" if barge else "")
            self.cues.play("listen")
        else:
            log.info("listen: follow-up window %.0fs", followup)
        audio = self.record(start_timeout=followup or self.args.start_timeout)
        if audio is None:
            self._idle()
            self.cues.play("bye" if followup else "nomatch")
            log.info("listen: no speech")
            if clip:
                self.clips.note(clip, outcome="no speech")
            return
        self.state = "transcribing"
        t = time.perf_counter()
        raw = self.transcribe(audio)
        asr_ms = (time.perf_counter() - t) * 1000
        text = (self.wake_prefix.sub("", raw) if self.wake_prefix else raw).strip()
        log.info("listen: heard %r -> %r (%.1fs audio, %.0f ms)", raw, text, len(audio) / RATE, asr_ms)
        m = DICTATE_INTENT.match(text)
        if m and not followup:
            if clip:
                self.clips.note(clip, heard_id="dictation", text=text)
            self.cues.play("dictate")            # go ahead: dictation is on
            session = Dictation(voice=True, first_text=m.group(1), end_silence=self.args.dictate_silence)
            self.dictate(session)
            if session.raw:
                self.on_voice_dictation(session.raw)
            else:
                self.cues.play("nomatch")
            self._idle()
            return
        if not text:
            self._idle()
            self.cues.play("nomatch")
            if clip:
                self.clips.note(clip, outcome="empty transcript", raw=raw)
            return
        heard_id = uuid.uuid4().hex
        if clip:
            self.clips.note(clip, heard_id=heard_id, text=text)
        payload = {
            "id": heard_id, "text": text, "raw": raw, "barge": barge,
            "wake_score": score, "followup": bool(followup),
            "audio_s": round(len(audio) / RATE, 2), "asr_ms": round(asr_ms),
            "ts": time.time(),
        }
        self.out.heard(payload)
        self.on_heard(payload)
        self.state = "waiting"
        self.mic.drain()                          # skip audio that queued up during ASR

    def run(self):
        log.info("listen: wake threshold %.2f, %.2f while speaking", self.args.threshold, self.args.barge_threshold)
        while not self._stop.is_set():
            if not self.mic_on.is_set():
                self._mic_off()
                continue
            self.mic = Mic(self.args.target)
            try:
                while not self._stop.is_set():
                    self.turn()
            except MicGone as e:
                if self._stop.is_set():
                    break
                if self.mic_on.is_set():
                    log.warning("listen: %s; reopening the mic in 2 s", e)
                    self.state = "no mic"
                    time.sleep(2)
            except Exception:
                log.exception("listen: turn failed; carrying on")
                time.sleep(1)
            finally:
                self.mic.close()
                self.mic = None

    def _mic_off(self):
        """Wake word off: no capture. A Super+X session opens the mic just for itself."""
        self.state = "mic off"
        while not self._stop.is_set() and not self.mic_on.is_set():
            try:
                session = self.dictations.get(timeout=0.25)
            except queue.Empty:
                continue
            self.mic = Mic(self.args.target)
            try:
                self.dictate(session)
            except MicGone as e:
                log.warning("listen: %s during dictation", e)
                session.emit("error", "the mic went away")
                session.done.set()
            finally:
                self.mic.close()
                self.mic = None
                self.state = "mic off"

    def close(self):
        self._stop.set()
        if self.mic:
            self.mic.close()                      # unblocks a read
        self.out.close()
