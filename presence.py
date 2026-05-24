#!/usr/bin/env python3
"""Alyx local presence state controller.

This is the boring core beneath any future desktop widget:
mode/capability state lives in voice/presence-state.json, while helpers like
speak.py and listen.py can read it to avoid echo loops and privacy mistakes.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "presence-state.json"
LOCK_PATH = ROOT / ".presence-state.lock"

MODES = {
    "privacy": {
        "speech_enabled": False,
        "listening_enabled": False,
        "camera_enabled": False,
        "default_activity": "muted",
        "description": "Speech off, listening off, camera off/privacy.",
    },
    "talk_only": {
        "speech_enabled": True,
        "listening_enabled": False,
        "camera_enabled": False,
        "default_activity": "idle",
        "description": "Alyx may speak; mic/STT stays off.",
    },
    "listen_only": {
        "speech_enabled": False,
        "listening_enabled": True,
        "camera_enabled": False,
        "default_activity": "idle",
        "description": "Mic/STT on; Alyx replies text-only.",
    },
    "conversational": {
        "speech_enabled": True,
        "listening_enabled": True,
        "camera_enabled": False,
        "default_activity": "idle",
        "description": "Mic/STT and speech enabled; camera still separate.",
    },
}

ACTIVITIES = {"muted", "idle", "listening", "thinking", "speaking"}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


@contextmanager
def state_lock():
    """Best-effort local process lock for state updates."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "w", encoding="utf-8") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file, fcntl.LOCK_EX)
        except Exception:
            pass
        yield
        try:
            import fcntl

            fcntl.flock(lock_file, fcntl.LOCK_UN)
        except Exception:
            pass


def default_state() -> dict[str, Any]:
    mode = "privacy"
    spec = MODES[mode]
    return {
        "schema": "alyx.presence.v1",
        "mode": mode,
        "activity": spec["default_activity"],
        "speech_enabled": spec["speech_enabled"],
        "listening_enabled": spec["listening_enabled"],
        "camera_enabled": spec["camera_enabled"],
        "updated_at": now_iso(),
        "updated_by": "presence.py:init",
        "note": spec["description"],
    }


def normalize_state(state: dict[str, Any]) -> dict[str, Any]:
    mode = state.get("mode") if state.get("mode") in MODES else "privacy"
    spec = MODES[mode]
    activity = state.get("activity") if state.get("activity") in ACTIVITIES else spec["default_activity"]
    if mode == "privacy":
        activity = "muted"
    return {
        "schema": "alyx.presence.v1",
        "mode": mode,
        "activity": activity,
        "speech_enabled": bool(spec["speech_enabled"]),
        "listening_enabled": bool(spec["listening_enabled"]),
        "camera_enabled": bool(state.get("camera_enabled", spec["camera_enabled"])),
        "updated_at": state.get("updated_at") or now_iso(),
        "updated_by": state.get("updated_by") or "presence.py:normalize",
        "note": state.get("note") or spec["description"],
    }


def read_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return default_state()
    try:
        return normalize_state(json.loads(STATE_PATH.read_text(encoding="utf-8")))
    except Exception:
        return default_state()


def write_state(state: dict[str, Any]) -> dict[str, Any]:
    state = normalize_state(state)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = STATE_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, STATE_PATH)
    return state


def set_mode(mode: str, updated_by: str = "presence.py:set-mode") -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}. Expected one of: {', '.join(MODES)}")
    spec = MODES[mode]
    with state_lock():
        state = read_state()
        state.update(
            {
                "mode": mode,
                "activity": spec["default_activity"],
                "speech_enabled": spec["speech_enabled"],
                "listening_enabled": spec["listening_enabled"],
                "camera_enabled": False,
                "updated_at": now_iso(),
                "updated_by": updated_by,
                "note": spec["description"],
            }
        )
        return write_state(state)


def set_activity(activity: str, updated_by: str = "presence.py:set-activity") -> dict[str, Any]:
    if activity not in ACTIVITIES:
        raise ValueError(f"Unknown activity: {activity}. Expected one of: {', '.join(sorted(ACTIVITIES))}")
    with state_lock():
        state = read_state()
        if state["mode"] == "privacy":
            activity = "muted"
        state.update({"activity": activity, "updated_at": now_iso(), "updated_by": updated_by})
        return write_state(state)


def can_listen(state: dict[str, Any] | None = None) -> tuple[bool, str]:
    state = state or read_state()
    if not state.get("listening_enabled"):
        return False, f"listening disabled in mode '{state.get('mode')}'"
    if state.get("activity") == "speaking":
        return False, "Alyx is speaking; listener paused to avoid echo"
    if state.get("activity") == "muted":
        return False, "presence is muted"
    return True, "listening allowed"


def can_speak(state: dict[str, Any] | None = None) -> tuple[bool, str]:
    state = state or read_state()
    if not state.get("speech_enabled"):
        return False, f"speech disabled in mode '{state.get('mode')}'"
    return True, "speech allowed"


@contextmanager
def speaking_guard(enabled: bool = True):
    """Mark Alyx as speaking while audio playback runs, then restore activity.

    This is intentionally logical muting: listeners should check the state and
    pause/ignore STT instead of us fighting PipeWire device mute state.
    """
    if not enabled:
        yield
        return
    previous = read_state()
    allowed, _reason = can_speak(previous)
    if allowed:
        set_activity("speaking", updated_by="speak.py")
    try:
        yield
    finally:
        if allowed:
            current = read_state()
            if current.get("activity") == "speaking":
                restore_activity = previous.get("activity") if previous.get("activity") in ACTIVITIES else "idle"
                if previous.get("mode") == "privacy":
                    restore_activity = "muted"
                set_activity(restore_activity, updated_by="speak.py:restore")


def print_state(state: dict[str, Any]) -> None:
    print(json.dumps(state, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Control Alyx local presence state")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Print current presence state as JSON")
    init_p = sub.add_parser("init", help="Create state file if missing")
    init_p.add_argument("--force", action="store_true", help="Reset to privacy even if state exists")

    mode_p = sub.add_parser("mode", help="Set presence mode")
    mode_p.add_argument("mode", choices=sorted(MODES))

    activity_p = sub.add_parser("activity", help="Set transient activity")
    activity_p.add_argument("activity", choices=sorted(ACTIVITIES))

    sub.add_parser("can-listen", help="Exit 0 if listener may record now")
    sub.add_parser("can-speak", help="Exit 0 if speech may play now")

    args = parser.parse_args()
    command = args.command or "status"

    try:
        if command == "status":
            print_state(read_state())
            return 0
        if command == "init":
            with state_lock():
                if args.force or not STATE_PATH.exists():
                    print_state(write_state(default_state()))
                else:
                    print_state(read_state())
            return 0
        if command == "mode":
            print_state(set_mode(args.mode))
            return 0
        if command == "activity":
            print_state(set_activity(args.activity))
            return 0
        if command == "can-listen":
            allowed, reason = can_listen()
            print(reason)
            return 0 if allowed else 1
        if command == "can-speak":
            allowed, reason = can_speak()
            print(reason)
            return 0 if allowed else 1
    except Exception as exc:
        print(f"Error: {exc}", file=os.sys.stderr)
        return 1

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
