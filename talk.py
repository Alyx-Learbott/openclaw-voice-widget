from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import uuid
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import winsound
from faster_whisper import WhisperModel
from piper import PiperVoice

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
CONFIG_PATH = ROOT / "config.json"
EXAMPLE_CONFIG_PATH = ROOT / "config.example.json"
DEFAULTS = json.loads(EXAMPLE_CONFIG_PATH.read_text(encoding="utf-8"))
PAIRING_HINT = (
    "OpenClaw CLI pairing is still required. Approve the pending local CLI device, "
    "then run the voice loop again."
)


def load_config() -> dict:
    config = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        config.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    return config


def abs_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE / path


def _write_audio_frames(path: Path, audio: np.ndarray, sample_rate: int, channels: int) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())


def record_to_wav(path: Path, sample_rate: int, channels: int, input_device=None) -> None:
    print("\nPress Enter to start recording.")
    input()
    frames: list[np.ndarray] = []

    def callback(indata, _frames, _time, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        frames.append(indata.copy())

    print("Recording... press Enter to stop.")
    with sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
        callback=callback,
        device=input_device,
    ):
        input()

    if not frames:
        raise RuntimeError("No audio captured.")

    audio = np.concatenate(frames, axis=0)
    _write_audio_frames(path, audio, sample_rate, channels)


def record_for_seconds(path: Path, sample_rate: int, channels: int, seconds: float, input_device=None) -> None:
    print(f"\nPress Enter to record for up to {seconds:.1f} seconds.")
    input()
    print(f"Recording for up to {seconds:.1f} seconds...")

    block_duration = 0.1
    blocksize = max(1, int(sample_rate * block_duration))
    speech_threshold = 0.015
    tail_silence_seconds = 0.8
    min_speech_seconds = 0.35

    frames: list[np.ndarray] = []
    speech_started = False
    speech_start_at = 0.0
    last_voiced_at = 0.0
    started_at = time.monotonic()

    with sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
        blocksize=blocksize,
        device=input_device,
    ) as stream:
        while time.monotonic() - started_at < seconds:
            chunk, overflowed = stream.read(blocksize)
            if overflowed:
                print("[audio] input overflow", file=sys.stderr)
            frames.append(chunk.copy())
            normalized = chunk.astype(np.float32) / 32768.0
            level = float(np.sqrt(np.mean(np.square(normalized))))
            now = time.monotonic()
            if level >= speech_threshold:
                if not speech_started:
                    speech_started = True
                    speech_start_at = now
                last_voiced_at = now
            elif speech_started and last_voiced_at:
                if now - last_voiced_at >= tail_silence_seconds and now - speech_start_at >= min_speech_seconds:
                    break

    if not frames:
        raise RuntimeError("No audio captured.")

    audio = np.concatenate(frames, axis=0)
    _write_audio_frames(path, audio, sample_rate, channels)


class LocalWhisper:
    def __init__(self, model_name: str, download_root: Path, compute_type: str):
        self.model = WhisperModel(
            model_name,
            device="cpu",
            compute_type=compute_type,
            download_root=str(download_root),
        )

    def transcribe(self, wav_path: Path, language: str) -> str:
        segments, _info = self.model.transcribe(
            str(wav_path),
            language=language,
            vad_filter=True,
            beam_size=1,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return re.sub(r"\s+", " ", text)


def prepare_text_for_speech(text: str) -> str:
    speech = text.strip()
    if not speech:
        return ""
    speech = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", speech)
    speech = re.sub(r"https?://\S+", "", speech)
    speech = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", speech)
    speech = re.sub(r"(?m)^\s*[-*+]\s+", "", speech)
    speech = re.sub(r"(?m)^\s*>\s*", "", speech)
    speech = speech.replace("```", " ")
    speech = speech.replace("**", "")
    speech = speech.replace("__", "")
    speech = speech.replace("`", "")
    speech = speech.replace("*", "")
    speech = speech.replace("_", " ")
    cleaned_chars: list[str] = []
    for char in speech:
        category = unicodedata.category(char)
        if category in {"So", "Sk", "Cs"}:
            continue
        cleaned_chars.append(char)
    speech = "".join(cleaned_chars)
    speech = speech.replace("&", " and ")
    speech = speech.replace("—", ", ")
    speech = speech.replace("–", ", ")
    speech = re.sub(r"\s+", " ", speech)
    return speech.strip()


class PiperSpeaker:
    def __init__(self, model_path: Path):
        self.voice = PiperVoice.load(str(model_path))

    def speak(self, text: str) -> None:
        speech_text = prepare_text_for_speech(text)
        if not speech_text:
            return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)
        try:
            with wave.open(str(wav_path), "wb") as wav_file:
                self.voice.synthesize_wav(speech_text, wav_file)
            winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except OSError:
                pass


def resolve_openclaw_command() -> list[str]:
    override = os.environ.get("OPENCLAW_BIN", "").strip()
    if override:
        override_path = Path(override)
        if override_path.suffix.lower() == ".ps1":
            return ["pwsh", "-File", str(override_path)]
        return [str(override_path)]

    candidates = [
        shutil.which("openclaw.cmd"),
        shutil.which("openclaw"),
        str(Path(os.environ.get("APPDATA", "")) / "npm" / "openclaw.cmd"),
        str(Path(os.environ.get("APPDATA", "")) / "npm" / "openclaw"),
        str(Path(os.environ.get("APPDATA", "")) / "npm" / "openclaw.ps1"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            if path.suffix.lower() == ".ps1":
                return ["pwsh", "-File", str(path)]
            return [str(path)]
    raise RuntimeError(
        "Could not find the OpenClaw CLI. Set OPENCLAW_BIN to openclaw.cmd if needed."
    )


def run_openclaw_rpc(method: str, params: dict, expect_final: bool = False) -> dict:
    openclaw_cmd = resolve_openclaw_command()
    cmd = [
        *openclaw_cmd,
        "gateway",
        "call",
        method,
        "--json",
        "--timeout",
        "600000",
        "--params",
        json.dumps(params),
    ]
    if expect_final:
        cmd.insert(len(openclaw_cmd) + 2, "--expect-final")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(WORKSPACE),
    )
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        output = "\n".join(part for part in [stdout, stderr] if part)
        if "pairing required" in output.lower() or "scope upgrade pending approval" in output.lower():
            raise RuntimeError(PAIRING_HINT)
        raise RuntimeError(output or f"OpenClaw RPC failed with exit code {result.returncode}")
    if not stdout:
        raise RuntimeError(stderr or "OpenClaw RPC returned no JSON output.")
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
    cleaned_lines = []
    for line in joined.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("MEDIA:"):
            continue
        if stripped.startswith("[[") and stripped.endswith("]]"):
            continue
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines).strip()


def assistant_text_messages(session_key: str, limit: int = 50) -> list[dict]:
    payload = run_openclaw_rpc("chat.history", {"sessionKey": session_key, "limit": limit})
    messages: list[dict] = []
    for message in payload.get("messages", []):
        if not isinstance(message, dict) or str(message.get("role", "")).lower() != "assistant":
            continue
        text = extract_text_from_message(message)
        timestamp = message.get("timestamp")
        ts = int(timestamp) if isinstance(timestamp, (int, float)) else None
        messages.append({"timestamp": ts, "text": text})
    return messages


def message_signature(message: dict) -> tuple:
    return (message.get("timestamp"), message.get("text"))


def wait_for_assistant_reply(
    session_key: str,
    started_at_ms: int,
    baseline_signatures: set[tuple],
    timeout_seconds: float = 90.0,
    poll_interval: float = 1.0,
) -> str:
    deadline = time.time() + timeout_seconds
    saw_no_reply = False
    while time.time() < deadline:
        messages = assistant_text_messages(session_key)
        candidates = []
        for message in messages:
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


def send_text(session_key: str, text: str) -> str:
    baseline_messages = assistant_text_messages(session_key)
    baseline_signatures = {message_signature(message) for message in baseline_messages}
    started_at_ms = int(time.time() * 1000)
    run_openclaw_rpc(
        "chat.send",
        {
            "sessionKey": session_key,
            "message": text,
            "deliver": False,
            "idempotencyKey": str(uuid.uuid4()),
        },
        expect_final=True,
    )
    return wait_for_assistant_reply(session_key, started_at_ms, baseline_signatures)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local voice loop for Agent")
    parser.add_argument("--session-key", help="Override the OpenClaw session key")
    parser.add_argument("--no-speak", action="store_true", help="Print replies without TTS playback")
    parser.add_argument("--list-devices", action="store_true", help="Print detected audio devices and exit")
    parser.add_argument("--seconds", type=float, help="Record each turn for a fixed number of seconds instead of start/stop with Enter")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return 0

    config = load_config()
    session_key = args.session_key or config["session_key"]
    stt_root = abs_path(config["stt_download_root"])
    tts_model_path = abs_path(config["tts_model_path"])

    print("Loading local STT and TTS...")
    transcriber = LocalWhisper(config["stt_model"], stt_root, config["stt_compute_type"])
    speaker = None if args.no_speak else PiperSpeaker(tts_model_path)
    tmp_dir = ROOT / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    input_device = config.get("input_device")

    print("Ready. Press Ctrl+C to quit.")
    try:
        while True:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(tmp_dir.resolve())) as tmp:
                wav_path = Path(tmp.name)
            try:
                if args.seconds and args.seconds > 0:
                    record_for_seconds(
                        wav_path,
                        int(config["sample_rate"]),
                        int(config["channels"]),
                        float(args.seconds),
                        input_device=input_device,
                    )
                else:
                    record_to_wav(
                        wav_path,
                        int(config["sample_rate"]),
                        int(config["channels"]),
                        input_device=input_device,
                    )
                transcript = transcriber.transcribe(wav_path, config["language"])
                if not transcript:
                    print("I didn't catch anything. Try again.")
                    continue
                print(f"\nYou: {transcript}")
                reply = send_text(session_key, transcript)
                if reply:
                    print(f"\nAgent: {reply}\n")
                    if speaker:
                        speaker.speak(reply)
                else:
                    print("\nAgent stayed quiet.\n")
            finally:
                try:
                    wav_path.unlink(missing_ok=True)
                except OSError:
                    pass
    except KeyboardInterrupt:
        print("\nBye.")
        return 0
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
