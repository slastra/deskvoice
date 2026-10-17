"""Delivering a voice dictation: into the focused window, or through your own
finish command (to clean the text up first, say with a language model).

A dictation that ends with "enter" (", enter", "press enter", or "Enter."
said on its own) presses Return after the paste; the word itself is never
typed. A bare "enter" only counts standing apart (after a pause or a comma,
which the recognizer punctuates), so "the password you enter" stays text.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time

log = logging.getLogger("deskvoice")

SUBMIT = re.compile(r"(?:[\s,;:-]*\b(?:and\s+)?(?:press|hit)\s+(?:enter|return)"
                    r"|(?:^|(?<=[.!?,;:]))\s*(?:enter|return))[.!]?\s*$", re.I)

# Terminals take Ctrl+Shift+V as a bracketed paste (a newline in the text
# can't press Return); everything else takes Ctrl+V.
TERMINALS = ("kitty", "foot", "alacritty", "org.wezfurlong.wezterm", "com.mitchellh.ghostty", "konsole",
             "gnome-terminal", "xterm")


def split_submit(raw):
    """(text without a trailing "enter", whether to press Return)."""
    m = SUBMIT.search(raw)
    if not m:
        return raw, False
    return raw[:m.start()].rstrip(" ,;:-"), True


def focused_window():
    """{"class", "title"} of the focused window, where the compositor says
    (Hyprland, sway); {} otherwise."""
    try:
        if shutil.which("hyprctl") and os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
            w = json.loads(subprocess.run(["hyprctl", "activewindow", "-j"], capture_output=True,
                                          text=True, timeout=2).stdout or "{}")
            return {"class": w.get("class", ""), "title": w.get("title", "")}
        if shutil.which("swaymsg") and os.environ.get("SWAYSOCK"):
            tree = json.loads(subprocess.run(["swaymsg", "-t", "get_tree"], capture_output=True,
                                             text=True, timeout=2).stdout or "{}")
            stack = [tree]
            while stack:
                n = stack.pop()
                if n.get("focused"):
                    return {"class": n.get("app_id") or n.get("window_properties", {}).get("class", ""),
                            "title": n.get("name") or ""}
                stack += n.get("nodes", []) + n.get("floating_nodes", [])
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return {}


def _quiet(cmd, **kw):
    # Never let a helper inherit a pipe someone waits on: wl-copy leaves a
    # child serving the clipboard, and an inherited stdout would hold that
    # reader open until the clipboard changed hands.
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, **kw)


def paste(text, app, submit):
    """wl-copy + wtype, the way a person would: paste, then Return if asked."""
    if text:
        _quiet(["wl-copy"], input=text.encode(), check=True)
    if not shutil.which("wtype"):
        log.warning("dictate: wtype isn't installed, so the text is on the clipboard but not pasted")
        return False
    if text:
        time.sleep(0.08)
        keys = ["-M", "ctrl", "-M", "shift", "-k", "v", "-m", "shift", "-m", "ctrl"] \
            if app.lower() in TERMINALS else ["-M", "ctrl", "-k", "v", "-m", "ctrl"]
        _quiet(["wtype", *keys], check=True)
    if submit:
        time.sleep(0.12)                 # let the paste land before Return
        _quiet(["wtype", "-k", "Return"], check=True)
    return True


def run_finish_command(command, raw, win, dry_run):
    """Your finish command: the raw text on stdin; {app}, {title} and
    {paste_flag} ("--paste", or "--dry-run" in a dry run) filled in its
    arguments. It reports on stdout one `<name> <json>` per line: clean,
    pasted, submitted, error. Returns (events, ok)."""
    fill = {"app": win.get("class", ""), "title": win.get("title", ""),
            "paste_flag": "--dry-run" if dry_run else "--paste"}
    cmd = [str(a).format(**fill) for a in command]
    with tempfile.TemporaryFile(mode="w+") as out:   # a file, not a pipe: see _quiet
        proc = subprocess.run(cmd, input=raw, stdout=out, stderr=subprocess.STDOUT, text=True, timeout=45)
        out.seek(0)
        printed = out.read()
    events = {}
    for line in printed.splitlines():
        name, _, value = line.partition(" ")
        try:
            events[name] = json.loads(value)
        except ValueError:
            pass
    if "error" in events or proc.returncode != 0:
        events.setdefault("error", printed.strip()[-200:] or f"exit {proc.returncode}")
        return events, False
    return events, True


def finish(raw, cfg, dry_run=False):
    """Deliver a voice dictation. Returns (summary dict for logs and recent, ok)."""
    win = focused_window()
    command = cfg.get("finish_command") or []
    if command:
        events, ok = run_finish_command(command, raw, win, dry_run)
        return {"text": events.get("clean") or "", "app": win.get("class", ""),
                "pasted": "pasted" in events, "submitted": "submitted" in events,
                "error": events.get("error")}, ok
    text, submit = split_submit(raw)
    pasted = False
    if not dry_run:
        try:
            pasted = paste(text, win.get("class", ""), submit)
        except (OSError, subprocess.SubprocessError) as e:
            return {"text": text, "app": win.get("class", ""), "error": str(e)}, False
    return {"text": text, "app": win.get("class", ""), "pasted": pasted and bool(text),
            "submitted": pasted and submit}, True
