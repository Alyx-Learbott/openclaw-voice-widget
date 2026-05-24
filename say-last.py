#!/usr/bin/env python3
"""Speak the latest Agent assistant reply from an OpenClaw session.

Ubuntu-first helper: OpenClaw chat.history -> voice/speak.py -> local TTS -> speakers.
Piper remains the default fallback; Kokoro can be selected with --backend kokoro.

Usage:
  .venv-voice/bin/python voice/say-last.py
  .venv-voice/bin/python voice/say-last.py --backend kokoro
  .venv-voice/bin/python voice/say-last.py --no-play
  .venv-voice/bin/python voice/say-last.py --session-key agent:main:main
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DEFAULT_SESSION_KEY = "agent:main:main"


def resolve_openclaw_command() -> list[str]:
    override = os.environ.get("OPENCLAW_BIN", "").strip()
    if override:
        return [override]
    found = shutil.which("openclaw")
    if found:
        return [found]
    raise RuntimeError("Could not find the OpenClaw CLI. Set OPENCLAW_BIN if needed.")


def run_history(session_key: str, limit: int) -> dict:
    cmd = [
        *resolve_openclaw_command(),
        "gateway",
        "call",
        "chat.history",
        "--json",
        "--timeout",
        "30000",
        "--params",
        json.dumps({"sessionKey": session_key, "limit": limit}),
    ]
    result = subprocess.run(
        cmd,
        cwd=str(WORKSPACE),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        output = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
        raise RuntimeError(output or f"chat.history failed with exit code {result.returncode}")
    return json.loads(result.stdout)


def extract_text_from_message(message: dict) -> str:
    parts: list[str] = []
    text = message.get("text")
    if isinstance(text, str) and text.strip():
        parts.append(text.strip())

    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"].strip())

    joined = "\n".join(part for part in parts if part)
    cleaned_lines = []
    for line in joined.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "NO_REPLY":
            continue
        if stripped.startswith("MEDIA:"):
            continue
        if stripped.startswith("[[") and "]]" in stripped[:80]:
            # Strip OpenClaw delivery directives such as [[reply_to_current]].
            stripped = stripped.split("]]", 1)[1].strip()
            if not stripped:
                continue
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines).strip()


def latest_assistant_text(history: dict) -> str:
    messages = history.get("messages", [])
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).lower() != "assistant":
            continue
        text = extract_text_from_message(message)
        if text:
            return text
    return ""


def truncate_text(text: str, max_chars: int | None) -> str:
    if not max_chars or max_chars <= 0 or len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rstrip()
    # Prefer ending on a sentence boundary if one is nearby.
    boundary = max(clipped.rfind(". "), clipped.rfind("! "), clipped.rfind("? "), clipped.rfind("\n"))
    if boundary >= max_chars * 0.6:
        clipped = clipped[: boundary + 1].rstrip()
    return clipped + " …"


def speak_text(
    text: str,
    no_play: bool,
    print_text: bool,
    backend: str,
    voice: str,
    speed: float,
) -> None:
    if print_text:
        print(text)
    cmd = [sys.executable, str(ROOT / "speak.py"), "--backend", backend]
    if backend == "kokoro":
        cmd.extend(["--voice", voice, "--speed", str(speed)])
    if no_play:
        cmd.append("--no-play")
    # Use stdin so long replies do not become giant argv payloads.
    subprocess.run(cmd, cwd=str(WORKSPACE), input=text, text=True, encoding="utf-8", check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Speak the latest Agent assistant reply")
    parser.add_argument("--session-key", default=DEFAULT_SESSION_KEY)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--backend", choices=["piper", "kokoro"], default="piper", help="TTS backend to use")
    parser.add_argument("--voice", default="af_heart", help="Kokoro voice name")
    parser.add_argument("--speed", type=float, default=1.0, help="Kokoro speaking speed")
    parser.add_argument("--no-play", action="store_true", help="Synthesize only; do not play audio")
    parser.add_argument("--print", dest="print_text", action="store_true", help="Print the text before speaking")
    parser.add_argument("--max-chars", type=int, default=1200, help="Maximum characters to speak; use 0 for no limit")
    args = parser.parse_args()

    try:
        history = run_history(args.session_key, args.limit)
        text = latest_assistant_text(history)
        if not text:
            print("No assistant text reply found.", file=sys.stderr)
            return 1
        text = truncate_text(text, args.max_chars)
        speak_text(
            text,
            no_play=args.no_play,
            print_text=args.print_text,
            backend=args.backend,
            voice=args.voice,
            speed=args.speed,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
