# deskvoice

One local speech service for the Linux desktop. Everything on your machine
that wants to speak or listen goes through it: an assistant, notification
cues, a document reader, dictation. deskvoice keeps one set of voice models
warm, decides who talks when, lowers your music while it does, and gives the
microphone to whoever needs it.

Nothing leaves your machine: voices are [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)
and [Piper](https://github.com/rhasspy/piper), the wake word is
[openWakeWord](https://github.com/dscripka/openWakeWord), and speech
recognition is NVIDIA's Parakeet running locally through
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).

## Why

Desktop speech tends to grow one program at a time, and each one loads its
own model and plays its own audio. Then a notification talks over the article
you're listening to, an assistant answers on top of both, every cue pays a
second of model loading, and two programs record the microphone without
knowing about each other. deskvoice is the one place that knows what's
playing and what's listening.

## What it does

- **Fast speech.** Kokoro stays loaded (on CUDA when available), so a short
  cue starts playing in about 150-200 ms. Piper voices stay resident too.
- **Channels with priority.** `assistant` speech goes first, `cue` speech
  waits its turn (and is dropped, with the caller told, if it waits too long).
  A document reader holding the `reader` lease is paused while anything else
  speaks or the mic listens, then resumes at the interrupted sentence.
- **Streaming.** Send sentences as a language model writes them; each renders
  while the previous one plays.
- **Ducking.** While deskvoice speaks or listens, other apps drop to a
  fraction of their own volume and come back exactly afterwards, even apps
  that recreate their audio streams (browsers do). It can't get stuck: the
  originals survive a crash, a stream that comes back later at the ducked
  volume is put back, a volume you change mid-duck is respected, and
  `deskvoice unduck` restores everything at once.
- **A wake word** with barge-in: say it while deskvoice is talking and the
  speech stops. Requests are transcribed locally (about 0.1-0.4 s for a short
  one) and published to MQTT and to socket subscribers.
- **Dictation.** By hotkey: a client opens a session, sees levels and partial
  text, and gets the transcript about 0.02 s after it says stop (each phrase is
  transcribed at its pause). By voice: "hey jarvis, dictate: see you at three,
  enter" pastes "See you at three" into the focused window and presses Return.
  A finish command of your own can clean the text up first.
- **A status feed** for bar widgets: who is speaking and what, whether the mic
  is listening, live dictation, recent exchanges, all pushed only on change.
- **A real mic switch.** Turn the wake word off and the microphone is closed,
  not ignored.

## Requirements

- Linux with PipeWire (`pw-play`, `pw-record`, `pactl`)
- Python 3.11 and [uv](https://docs.astral.sh/uv/)
- An NVIDIA GPU is optional: on a desktop CPU a short cue renders in about
  0.3 s (0.1 s on CUDA)
- For pasting dictation: Wayland with `wl-copy` and `wtype`
- For the wake word's requests: an MQTT broker (mosquitto), or a client
  subscribed over the socket

## Install

```sh
git clone https://github.com/slastra/deskvoice && cd deskvoice
uv tool install .                 # puts `deskvoice` in ~/.local/bin
# with an NVIDIA GPU instead:
#   uv tool install . --with onnxruntime-gpu --overrides overrides-gpu.txt
deskvoice fetch-models            # Kokoro and Parakeet into ~/.local/share/deskvoice/models (~1 GB)

mkdir -p ~/.config/deskvoice
cp config.example.toml ~/.config/deskvoice/config.toml    # optional: every key has a default
cp systemd/deskvoice.service ~/.config/systemd/user/
systemctl --user enable --now deskvoice

deskvoice say "Hello from deskvoice."
```

The wake word is off by default. Set `listen.enabled = true` (or run
`deskvoice serve --listen`); it starts with openWakeWord's built-in
"hey jarvis", and `listen.wake_model` takes the path to a model you trained
yourself.

## Using it

```sh
deskvoice say "Build passed."                         # a cue (waits up to 8 s behind other speech)
deskvoice say --channel assistant "Here's the weather."
deskvoice say --voice bm_george "A different voice."  # any Kokoro style, or piper:<model>
deskvoice voices                                      # what's installed
deskvoice stop --channel assistant                    # cut it short
deskvoice status
```

`deskvoice say` returns as soon as the speech starts (exit 0), or says why it
won't (exit 3: dropped, unknown voice; exit 2: the service isn't running), so
a notification hook can report honestly what it did. The client is standard
library only and starts in a few tens of milliseconds.

## Protocol

A unix socket at `$XDG_RUNTIME_DIR/deskvoice.sock`, one JSON object per line.
Requests may carry a `req` that the reply echoes.

| Request | What |
|---|---|
| `{"op": "say", "id", "text", "voice"?, "channel"?, "final"?, "max_wait"?}` | Queue a sentence. Sentences with the same `id` are one utterance, played in order; `final: false` keeps it open for more |
| `{"op": "end", "id"}` | That utterance is complete |
| `{"op": "stop", "channel"?, "id"?}` | Cut speech short |
| `{"op": "synth", "text", "voice"?, "out"}` | Render a WAV to `out` for a client that plays it itself |
| `{"op": "lease", "channel": "reader"}` | Hold the reader channel: deskvoice pushes `yield` and `resume` |
| `{"op": "duck", "seconds"}` | Keep other audio lowered (listening, thinking); `0` releases |
| `{"op": "unduck"}` | Bring all ducked audio back now |
| `{"op": "dictate", "action": "start" \| "stop" \| "cancel", "save"?}` | A dictation session on the shared mic; closing the connection cancels |
| `{"op": "transcribe", "path"}` | Transcribe a 16 kHz WAV |
| `{"op": "subscribe"}` | Push `status` on every change, and `heard` for each request |
| `{"op": "mic", "enabled"}` | Wake word on or off (off closes the mic; remembered) |
| `{"op": "followup", "seconds"}` | Listen again without the wake word |
| `{"op": "flag", "name", "on"}` | Create or remove a configured status flag file |
| `{"op": "voices"}`, `{"op": "status"}` | What's installed; what's happening |

Utterance events go to the connection that sent the `say`: `queued`,
`started {secs}`, `finished`, `stopped`, `dropped {reason}`, `error {message}`.
Dictation sessions send `state`, `level`, `time`, `partial`, `raw` and
`error`.

A streamed reply, from any language:

```json
{"op": "say", "id": "r1", "text": "Here's what I found.", "channel": "assistant", "final": false}
{"op": "say", "id": "r1", "text": "It's sunny with a high of 62.", "channel": "assistant", "final": false}
{"op": "end", "id": "r1"}
```

## Configuration

`~/.config/deskvoice/config.toml`. [`config.example.toml`](config.example.toml)
lists every key with its default and what it does: voices per channel, model
paths, the duck level and exclusions, the wake model and thresholds, MQTT,
the dictation finish command, cue sounds (built in, or an OSC cue daemon),
and status flags.

## Notes

- deskvoice's code is MIT licensed. The models it downloads carry their own
  licenses: Kokoro is Apache-2.0, Parakeet is CC-BY-4.0, and openWakeWord's
  built-in wake words are CC BY-NC-SA 4.0 (non-commercial). A wake word you
  train yourself is yours.
- The CPU and GPU builds of onnxruntime install the same module and overwrite
  each other, so a GPU install overrides away the CPU one that kokoro-onnx and
  openWakeWord ask for (`overrides-gpu.txt`; a checkout's `uv sync` does the
  same through `[tool.uv]`).
- Testing: `uv run pytest`. `DESKVOICE_DICTATE_DRYRUN=1` makes voice
  dictation clean up without pasting anything.
