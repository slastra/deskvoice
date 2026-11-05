"""Put an application's remembered volume right when it has no stream.

WirePlumber's stream-restore saves every volume change on a stream under
the application's identity, and hands that volume to the application's next
stream. Ducking writes volumes, so a stream that ENDS while ducked leaves
the ducked volume saved: deskvoice's own pw-play (one stream per sentence)
faded by deskpresence on 2026-11-04 stayed at 24% all day, and a video
paused while ducked came back at 30%.

There is no API to write a stream-restore entry directly, but opening a
stream with the same identity reaches it: this opens a silent stand-in
stream that WirePlumber files under the same key, sets its volume, and
closes it. Its initial volume is the one WirePlumber holds for the
application, so the stand-in also tells whether that value is ours to fix
or one somebody set on purpose.

WirePlumber 0.5 keys a stream by media.role when that is "Notification",
else by the first of application.id, application.name, media.name,
node.name (scripts/node/state-stream.lua, formKey). pw-play always sets an
application.name of its own, so only identities at or above that level can
be reproduced; media.name/node.name-keyed streams are not repaired.

Stand-ins carry PROXY_MARK so the duckers never take one for the app.
deskpresence (Go) implements the same contract in its audio.go.
"""

import json
import logging
import subprocess
import time
import uuid

log = logging.getLogger("deskvoice")

PROXY_MARK = "desk.volume-proxy"
SAME = 0.02


def identity(stream):
    """The properties WirePlumber keys this stream's saved volume by, or None
    when a stand-in can't reproduce them."""
    props = stream.get("properties") or {}
    if props.get("media.role") == "Notification":
        return {"media.role": "Notification"}
    for key in ("application.id", "application.name"):
        if props.get(key):
            return {key: props[key]}
    return None


def channels(stream):
    return max(1, len(stream.get("volume") or {}))


def is_proxy(stream):
    return PROXY_MARK in (stream.get("properties") or {})


def _volume(stream):
    for ch in (stream.get("volume") or {}).values():
        return ch.get("value", 65536) / 65536
    return 1.0


def _spa(props):
    return "{ " + " ".join(f"{json.dumps(k)} = {json.dumps(v)}" for k, v in props.items()) + " }"


def repair(ident, nchan, original, lowest, *, list_streams, set_volume,
           spawn=subprocess.Popen, sleep=time.sleep, clock=time.monotonic, timeout=1.5):
    """Set the saved volume of the application `ident` back to `original`,
    if what WirePlumber holds for it is one we wrote: between `lowest` (the
    deepest we went) and `original`. Returns "fixed", "ok" (already right),
    "kept" (somebody else's value: left alone) or "failed"."""
    token = uuid.uuid4().hex
    props = {**ident, PROXY_MARK: token}
    if "application.name" not in props:
        props["application.name"] = "volume repair"   # below application.id/role: not the key
    try:
        proc = spawn(["pw-play", "-P", _spa(props), "--raw", "--format", "f32", "--rate", "48000",
                      "--channels", str(nchan), "/dev/zero"],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        log.warning("volume repair: pw-play: %s", e)
        return "failed"
    try:
        deadline = clock() + timeout
        stand_in = None
        while clock() < deadline:
            try:
                stand_in = next((s for s in list_streams()
                                 if (s.get("properties") or {}).get(PROXY_MARK) == token), None)
            except Exception:
                stand_in = None
            if stand_in is not None:
                break
            sleep(0.03)
        if stand_in is None:
            return "failed"
        # stream-restore applies the saved volume just AFTER the node
        # appears (a first read can show 0): settle, read again, and read
        # once more before calling a value somebody else's
        def held_now():
            nonlocal stand_in
            sleep(0.15)
            stand_in = next((s for s in list_streams()
                             if (s.get("properties") or {}).get(PROXY_MARK) == token), stand_in)
            return _volume(stand_in)
        held = held_now()
        if not (lowest - SAME <= held < original + SAME):
            held = held_now()
        if abs(held - original) <= SAME:
            return "ok"
        if not (lowest - SAME <= held < original - SAME):
            return "kept"
        set_volume(stand_in["index"], int(original * 100 + 0.5))
        sleep(0.15)  # WirePlumber stores it on the params-changed event
        return "fixed"
    finally:
        proc.kill()
        try:
            proc.wait(1)
        except Exception:
            pass
