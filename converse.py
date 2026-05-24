#!/usr/bin/env python3
"""One-turn Alyx voice conversation runner.

Flow:
  presence conversational -> listen.py -> transcript -> OpenClaw chat.send
  -> wait for assistant reply -> speak.py

This is deliberately one-turn first. A desktop widget or continuous loop can sit
on top later once this path is boring and reliable.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from presence import can_listen, set_activity, set_mode

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DEFAULT_SESSION_KEY = "agent:main:main"
PAIRING_HINT = (
    "OpenClaw CLI pairing/scope approval is required. Approve the pending local CLI device, "
    "then run the voice loop again."
)


def resolve_openclaw_command() -> list[str]:
    override = os.environ.get("OPENCLAW_BIN", "").strip()
    if override:
        return [override]
    found = shutil.which("openclaw")
    if found:
        return [found]
    raise RuntimeError("Could not find the OpenClaw CLI. Set OPENCLAW_BIN if needed.")


def run_openclaw_rpc(method: str, params: dict, expect_final: bool = False, timeout_ms: int = 600000) -> dict:
    openclaw_cmd = resolve_openclaw_command()
    cmd = [
        *openclaw_cmd,
        "gateway",
        "call",
        method,
        "--json",
        "--timeout",
        str(timeout_ms),
        "--params",
        json.dumps(params),
    ]
    if expect_final:
        cmd.insert(len(openclaw_cmd) + 2, "--expect-final")

    result = subprocess.run(
        cmd,
        cwd=str(WORKSPACE),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        output = "\n".join(part for part in [stdout, stderr] if part)
        lower = output.lower()
        if "pairing required" in lower or "scope upgrade pending approval" in lower:
            raise RuntimeError(PAIRING_HINT)
        raise RuntimeError(output or f"OpenClaw RPC {method} failed with exit code {result.returncode}")
    if not stdout:
        return {}
    return json.loads(stdout)


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
    cleaned_lines: list[str] = []
    for line in joined.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "NO_REPLY" or stripped.startswith("MEDIA:"):
            continue
        if stripped.startswith("[[") and "]]" in stripped[:80]:
            stripped = stripped.split("]]", 1)[1].strip()
            if not stripped:
                continue
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines).strip()


def truncate_text(text: str, max_chars: int | None) -> str:
    if not max_chars or max_chars <= 0 or len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rstrip()
    boundary = max(clipped.rfind(". "), clipped.rfind("! "), clipped.rfind("? "), clipped.rfind("\n"))
    if boundary >= max_chars * 0.6:
        clipped = clipped[: boundary + 1].rstrip()
    return clipped + " …"


def history_messages(session_key: str, limit: int = 80) -> list[dict]:
    payload = run_openclaw_rpc("chat.history", {"sessionKey": session_key, "limit": limit}, timeout_ms=30000)
    messages = payload.get("messages", [])
    return messages if isinstance(messages, list) else []


def assistant_text_messages(session_key: str, limit: int = 80) -> list[dict]:
    out: list[dict] = []
    for message in history_messages(session_key, limit=limit):
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).lower() != "assistant":
            continue
        text = extract_text_from_message(message)
        timestamp = message.get("timestamp")
        ts = int(timestamp) if isinstance(timestamp, (int, float)) else None
        out.append({"timestamp": ts, "text": text})
    return out


def transcribe_wav(
    wav_path: Path,
    model_name: str = "small.en",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = "en",
) -> tuple[str, dict]:
    """Transcribe an existing WAV file using faster-whisper.

    Returns (text, info_dict).
    """
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    kwargs: dict = {"beam_size": 5}
    if language:
        kwargs["language"] = language
    segments, info = model.transcribe(str(wav_path), **kwargs)
    text = " ".join(s.text.strip() for s in segments)
    info_dict = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
    }
    return text, info_dict


def message_signature(message: dict) -> tuple:
    return (message.get("timestamp"), message.get("text"))


def wait_for_assistant_reply(
    session_key: str,
    started_at_ms: int,
    baseline_signatures: set[tuple],
    timeout_seconds: float = 120.0,
    poll_interval: float = 1.0,
) -> str:
    deadline = time.time() + timeout_seconds
    saw_no_reply = False
    while time.time() < deadline:
        candidates: list[str] = []
        for message in assistant_text_messages(session_key):
            signature = message_signature(message)
            timestamp = message.get("timestamp")
            if signature in baseline_signatures:
                continue
            if timestamp is not None and timestamp < started_at_ms - 1000:
                continue
            text = str(message.get("text") or "").strip()
            if text:
                candidates.append(text)
        for text in reversed(candidates):
            if text == "NO_REPLY":
                saw_no_reply = True
                continue
            return text
        time.sleep(poll_interval)
    if saw_no_reply:
        return ""
    raise RuntimeError("Timed out waiting for an assistant reply.")


def run_listen(
    duration: int,
    model: str,
    source: str,
    ignore_presence: bool,
    stop_on_silence: bool,
    silence_seconds: float,
    silence_threshold: float,
    speech_threshold: float,
) -> str:
    cmd = [
        sys.executable,
        str(ROOT / "listen.py"),
        "--duration",
        str(duration),
        "--model",
        model,
        "--source",
        source,
        "--raw",
    ]
    if ignore_presence:
        cmd.append("--ignore-presence")
    if stop_on_silence:
        cmd.extend([
            "--stop-on-silence",
            "--silence-seconds", str(silence_seconds),
            "--silence-threshold", str(silence_threshold),
            "--speech-threshold", str(speech_threshold),
        ])
    result = subprocess.run(
        cmd,
        cwd=str(WORKSPACE),
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        output = (result.stdout or "").strip()
        raise RuntimeError(output or f"listen.py failed with exit code {result.returncode}")
    return result.stdout.strip()


def send_transcript(session_key: str, transcript: str, timeout_seconds: float) -> str:
    baseline_messages = assistant_text_messages(session_key)
    baseline_signatures = {message_signature(message) for message in baseline_messages}
    started_at_ms = int(time.time() * 1000)
    set_activity("thinking", updated_by="converse.py")
    try:
        run_openclaw_rpc(
            "chat.send",
            {
                "sessionKey": session_key,
                "message": transcript,
                "deliver": False,
                "idempotencyKey": str(uuid.uuid4()),
            },
            expect_final=True,
            timeout_ms=int(max(timeout_seconds + 30, 60) * 1000),
        )
        return wait_for_assistant_reply(
            session_key,
            started_at_ms,
            baseline_signatures,
            timeout_seconds=timeout_seconds,
        )
    finally:
        set_activity("idle", updated_by="converse.py:restore")


def speak_reply(text: str, backend: str, voice: str, speed: float, no_play: bool, max_chars: int) -> None:
    text = truncate_text(text, max_chars)
    print(f"\nAlyx: {text}\n")
    if no_play or not text:
        return
    cmd = [sys.executable, str(ROOT / "speak.py"), "--backend", backend]
    if backend == "kokoro":
        cmd.extend(["--voice", voice, "--speed", str(speed)])
    subprocess.run(cmd, cwd=str(WORKSPACE), input=text, text=True, encoding="utf-8", check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one voice conversation turn")
    parser.add_argument("--session-key", default=DEFAULT_SESSION_KEY, help="OpenClaw session key to send transcript into")
    parser.add_argument("--duration", type=int, default=30, help="Maximum recording duration in seconds")
    parser.add_argument("--model", default="small.en", help="faster-whisper model for listen.py")
    parser.add_argument("--source", default="POROSVOC", help="PipeWire source for listen.py")
    parser.add_argument("--backend", choices=["piper", "kokoro"], default="kokoro", help="TTS backend for replies")
    parser.add_argument("--voice", default="af_heart", help="Kokoro voice name")
    parser.add_argument("--speed", type=float, default=1.0, help="Kokoro speaking speed")
    parser.add_argument("--timeout", type=float, default=120.0, help="Seconds to wait for assistant reply")
    parser.add_argument("--max-chars", type=int, default=0, help="Max reply chars to speak; 0 means no limit")
    parser.add_argument("--no-send", action="store_true", help="Only listen/transcribe; do not send to OpenClaw")
    parser.add_argument("--no-play", action="store_true", help="Do not speak the assistant reply")
    parser.add_argument("--wav-in", help="Transcribe an existing WAV file instead of recording (for PTT widget use)")
    parser.add_argument("--ignore-presence", action="store_true", help="Bypass presence checks for smoke tests")
    parser.add_argument("--no-cue", action="store_true", help="Skip the terminal bell/listening cue")
    parser.add_argument(
        "--stop-on-silence", action="store_true",
        help="Experimental: stop after speech is followed by silence instead of waiting the full duration",
    )
    parser.add_argument(
        "--silence-seconds", type=float, default=4.0,
        help="Seconds of quiet after speech before stopping when --stop-on-silence is enabled",
    )
    parser.add_argument(
        "--silence-threshold", type=float, default=150.0,
        help="RMS level considered quiet for --stop-on-silence",
    )
    parser.add_argument(
        "--speech-threshold", type=float, default=900.0,
        help="RMS level considered speech for --stop-on-silence",
    )
    args = parser.parse_args()

    try:
        set_mode("conversational", updated_by="converse.py")
        if not args.ignore_presence:
            allowed, reason = can_listen()
            if not allowed:
                print(f"Listening blocked by presence state: {reason}", file=sys.stderr)
                return 3

        if args.wav_in:
            # PTT mode: transcribe existing WAV, skip recording
            wav_path = Path(args.wav_in).expanduser()
            if not wav_path.is_absolute():
                wav_path = WORKSPACE / wav_path
            if not wav_path.exists():
                print(f"WAV file not found: {wav_path}", file=sys.stderr)
                return 1
            print(f"Transcribing {wav_path.name}...", file=sys.stderr)
            try:
                text, info = transcribe_wav(
                    wav_path,
                    model_name=args.model,
                )
            except Exception as exc:
                print(f"Transcription error: {exc}", file=sys.stderr)
                return 1
            transcript = text.strip()
        else:
            if not args.no_cue:
                print("🔊 Cue: start speaking after the beep.", file=sys.stderr)
                print("", end="", file=sys.stderr, flush=True)
            if args.stop_on_silence:
                print(f"Listening for up to {args.duration}s; will stop after speech + {args.silence_seconds:.1f}s silence...", file=sys.stderr)
            else:
                print(f"Listening for {args.duration}s...", file=sys.stderr)
            transcript = run_listen(
                args.duration,
                args.model,
                args.source,
                args.ignore_presence,
                args.stop_on_silence,
                args.silence_seconds,
                args.silence_threshold,
                args.speech_threshold,
            )
        print(f"\nKevin: {transcript}\n")
        if not transcript:
            print("No transcript produced.", file=sys.stderr)
            return 2
        if args.no_send:
            print("Dry run: not sending transcript to OpenClaw.", file=sys.stderr)
            return 0

        reply = send_transcript(args.session_key, transcript, args.timeout)
        if not reply:
            print("Assistant returned no spoken reply.", file=sys.stderr)
            return 0
        speak_reply(reply, args.backend, args.voice, args.speed, args.no_play, args.max_chars)
        return 0
    except KeyboardInterrupt:
        set_activity("idle", updated_by="converse.py:interrupt")
        return 130
    except Exception as exc:
        set_activity("idle", updated_by="converse.py:error")
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
