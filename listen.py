#!/usr/bin/env python3
"""Ubuntu-native speech-to-text helper for Agent.

Records audio from POROSVOC mic, transcribes with faster-whisper, outputs text.

Usage:
  python3 voice/listen.py                    # Record 10s, transcribe, print text
  python3 voice/listen.py --duration 5       # Record 5 seconds
  python3 voice/listen.py --no-transcribe    # Record only, save WAV
  python3 voice/listen.py --file audio.wav   # Transcribe existing file
  python3 voice/listen.py --model small.en   # Use small.en model
  python3 voice/listen.py --list-mics        # Show available PipeWire sources

The capture→transcribe pipeline:
  POROSVOC mic → pw-record (raw PCM) → faster-whisper (CTranslate2/CPU) → text
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import signal
import sys
import tempfile
import time
from pathlib import Path

from presence import can_listen, set_activity

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DEFAULT_TMP = ROOT / "tmp"
DEFAULT_MODEL = "small.en"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE = "int8"
DEFAULT_DURATION = 10
DEFAULT_RATE = 48000
DEFAULT_CHANNELS = 1

# POROSVOC is the verified working mic on this system.
# PipeWire source name for pw-record --target.
POROSVOC_SOURCE = "POROSVOC"
POROSVOC_FULL_NAME = "alsa_input.usb-POROSVOC_POROSVOC_PNC201_4MIC-00.mono-fallback"


def list_mics() -> list[dict[str, str]]:
    """List available PipeWire audio sources."""
    result = subprocess.run(
        ["pw-cli", "list-objects"],
        capture_output=True, text=True, timeout=5,
    )
    sources: list[dict[str, str]] = []
    name = desc = ""
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if "node.name" in stripped and "alsa_input" in stripped:
            name = stripped.split("=", 1)[1].strip().strip('"')
        elif "node.description" in stripped and "Audio/Source" not in stripped:
            desc = stripped.split("=", 1)[1].strip().strip('"')
        elif "media.class" in stripped and "Audio/Source" in stripped:
            if name:
                sources.append({"name": name, "description": desc or name})
                name = desc = ""
    return sources


def record_raw(
    output_path: Path,
    duration: int = DEFAULT_DURATION,
    source: str = POROSVOC_SOURCE,
    rate: int = DEFAULT_RATE,
    channels: int = DEFAULT_CHANNELS,
) -> Path:
    """Record raw PCM audio from a PipeWire source using pw-record.

    Returns the path to the raw PCM file (16-bit signed LE, mono).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "pw-record",
        "--rate", str(rate),
        "--channels", str(channels),
        "--format", "s16",
        "--target", source,
        str(output_path),
    ]

    try:
        subprocess.run(cmd, timeout=duration + 5)
    except subprocess.TimeoutExpired:
        # pw-record doesn't have a built-in duration limit, so we use
        # the system timeout command externally when needed.
        pass

    return output_path


def raw_to_wav(raw_path: Path, wav_path: Path, rate: int = DEFAULT_RATE, channels: int = DEFAULT_CHANNELS) -> Path:
    """Convert raw PCM (s16le) to WAV format."""
    import struct
    import wave

    with open(raw_path, "rb") as f:
        raw_data = f.read()

    if not raw_data:
        raise RuntimeError(f"Raw audio file is empty: {raw_path}")

    n_samples = len(raw_data) // (2 * channels)
    duration = n_samples / rate

    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(raw_data)

    return wav_path


def pcm_rms_s16le(data: bytes) -> float:
    """Return RMS level for mono/stereo signed 16-bit little-endian PCM."""
    if len(data) < 2:
        return 0.0
    if len(data) % 2:
        data = data[:-1]
    samples = len(data) // 2
    if not samples:
        return 0.0
    total = 0
    for i in range(0, len(data), 2):
        sample = int.from_bytes(data[i : i + 2], "little", signed=True)
        total += sample * sample
    return (total / samples) ** 0.5


def _run_pw_record_fixed(raw_path: Path, duration: int, source: str, rate: int, channels: int) -> subprocess.CompletedProcess:
    cmd = [
        "timeout", str(duration + 2),
        "pw-record",
        "--rate", str(rate),
        "--channels", str(channels),
        "--format", "s16",
        "--target", source,
        str(raw_path),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 10)


def _run_pw_record_until_silence(
    raw_path: Path,
    duration: int,
    source: str,
    rate: int,
    channels: int,
    silence_seconds: float,
    silence_threshold: float,
    speech_threshold: float,
    min_record_seconds: float,
) -> subprocess.CompletedProcess:
    """Record with pw-record, terminating early after speech followed by silence."""
    cmd = [
        "pw-record",
        "--rate", str(rate),
        "--channels", str(channels),
        "--format", "s16",
        "--target", source,
        str(raw_path),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
    start = time.time()
    last_offset = 0
    speech_seen = False
    silence_started: float | None = None
    last_rms = 0.0
    stop_reason = "max duration"
    silence_notice_printed = False
    quiet_notice_delay = min(1.0, max(0.3, silence_seconds / 3))
    # Hysteresis counters: require consecutive chunks above/below thresholds
    # before deciding speech started or silence started.
    consecutive_loud = 0
    consecutive_quiet = 0
    HYSTERESIS_CHUNKS = 3

    print("Listening…", file=sys.stderr, flush=True)
    try:
        while proc.poll() is None:
            elapsed = time.time() - start
            if elapsed >= duration:
                stop_reason = "max duration"
                break

            if raw_path.exists():
                size = raw_path.stat().st_size
                if size > last_offset:
                    with open(raw_path, "rb") as f:
                        f.seek(last_offset)
                        chunk = f.read(size - last_offset)
                    last_offset = size
                    last_rms = pcm_rms_s16le(chunk)

                    if last_rms >= speech_threshold:
                        consecutive_loud += 1
                        consecutive_quiet = 0
                    else:
                        consecutive_loud = 0
                        if last_rms <= silence_threshold:
                            consecutive_quiet += 1
                        else:
                            consecutive_quiet = 0

                    # Transition to speech only after a few consecutive loud chunks
                    if consecutive_loud >= HYSTERESIS_CHUNKS and not speech_seen:
                        print("Speech detected…", file=sys.stderr, flush=True)
                        speech_seen = True

                    if speech_seen:
                        if consecutive_loud >= HYSTERESIS_CHUNKS:
                            # Sustained speech (3+ consecutive loud chunks) — reset silence clock
                            silence_started = None
                            silence_notice_printed = False
                        elif last_rms > silence_threshold:
                            # A single loud-ish blip — don't reset the silence clock yet.
                            # Let it ride; if speech is real, the next few chunks will confirm.
                            pass
                        elif consecutive_quiet >= HYSTERESIS_CHUNKS and elapsed >= min_record_seconds:
                            # Sustained quiet — start/continue silence timer
                            if silence_started is None:
                                silence_started = time.time()
                            quiet_for = time.time() - silence_started
                            if not silence_notice_printed and quiet_for >= quiet_notice_delay:
                                print("Quiet detected; waiting before stop…", file=sys.stderr, flush=True)
                                silence_notice_printed = True
                            if quiet_for >= silence_seconds:
                                stop_reason = f"silence timeout ({silence_seconds:.1f}s)"
                                break
                        else:
                            # In hysteresis window — don't start or reset silence timer
                            pass

            time.sleep(0.1)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
        stdout_b, stderr_b = proc.communicate(timeout=2)

    print(f"Listening stopped: {stop_reason}.", file=sys.stderr, flush=True)
    stderr = stderr_b.decode("utf-8", errors="replace") if isinstance(stderr_b, bytes) else str(stderr_b or "")
    stdout = stdout_b.decode("utf-8", errors="replace") if isinstance(stdout_b, bytes) else str(stdout_b or "")
    return subprocess.CompletedProcess(cmd, proc.returncode or 0, stdout=stdout, stderr=stderr + f"\nlast_rms={last_rms:.1f}; stop_reason={stop_reason}")


def record_wav(
    output_path: Path,
    duration: int = DEFAULT_DURATION,
    source: str = POROSVOC_SOURCE,
    rate: int = DEFAULT_RATE,
    channels: int = DEFAULT_CHANNELS,
    stop_on_silence: bool = False,
    silence_seconds: float = 4.0,
    silence_threshold: float = 150.0,
    speech_threshold: float = 900.0,
    min_record_seconds: float = 2.0,
) -> Path:
    """Record audio and save as WAV.

    Uses pw-record for capture (verified working on this system),
    then converts raw PCM to WAV. Optional silence detection is intentionally
    conservative and experimental; fixed-duration recording remains available.
    """
    raw_path = output_path.with_suffix(".raw")
    raw_path.unlink(missing_ok=True)

    runner = _run_pw_record_until_silence if stop_on_silence else _run_pw_record_fixed
    if stop_on_silence:
        result = runner(
            raw_path,
            duration,
            source,
            rate,
            channels,
            silence_seconds,
            silence_threshold,
            speech_threshold,
            min_record_seconds,
        )
    else:
        result = runner(raw_path, duration, source, rate, channels)

    if not raw_path.exists() or raw_path.stat().st_size == 0:
        raw_path.unlink(missing_ok=True)
        if stop_on_silence:
            result = _run_pw_record_until_silence(
                raw_path,
                duration,
                POROSVOC_FULL_NAME,
                rate,
                channels,
                silence_seconds,
                silence_threshold,
                speech_threshold,
                min_record_seconds,
            )
        else:
            result = _run_pw_record_fixed(raw_path, duration, POROSVOC_FULL_NAME, rate, channels)

    if not raw_path.exists() or raw_path.stat().st_size == 0:
        raise RuntimeError(
            f"Recording failed (0 bytes). Source '{source}' may not be delivering audio.\n"
            f"pw-record stderr: {result.stderr.strip()}"
        )

    raw_to_wav(raw_path, output_path, rate, channels)
    raw_path.unlink(missing_ok=True)
    return output_path


def transcribe(
    audio_path: Path,
    model_name: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE,
    language: str | None = "en",
    beam_size: int = 5,
) -> tuple[str, dict]:
    """Transcribe audio file using faster-whisper.

    Returns (text, info_dict).
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    kwargs: dict = {"beam_size": beam_size}
    if language:
        kwargs["language"] = language

    segments, info = model.transcribe(str(audio_path), **kwargs)
    text = " ".join(s.text.strip() for s in segments)

    info_dict = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
    }
    return text, info_dict


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record from mic and transcribe with faster-whisper"
    )
    parser.add_argument(
        "--duration", "-d", type=int, default=DEFAULT_DURATION,
        help=f"Recording duration in seconds (default: {DEFAULT_DURATION})",
    )
    parser.add_argument(
        "--model", "-m", default=DEFAULT_MODEL,
        help=f"Whisper model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--device", default=DEFAULT_DEVICE,
        help=f"Compute device (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--compute-type", default=DEFAULT_COMPUTE,
        help=f"Compute type (default: {DEFAULT_COMPUTE})",
    )
    parser.add_argument(
        "--language", "-l", default="en",
        help="Language code for transcription (default: en)",
    )
    parser.add_argument(
        "--source", "-s", default=POROSVOC_SOURCE,
        help=f"PipeWire source name (default: {POROSVOC_SOURCE})",
    )
    parser.add_argument(
        "--wav-out", "-w",
        help="Save recorded WAV to this path (also kept even without this flag)",
    )
    parser.add_argument(
        "--no-transcribe", action="store_true",
        help="Record only, do not transcribe",
    )
    parser.add_argument(
        "--file", "-f",
        help="Transcribe an existing WAV file instead of recording",
    )
    parser.add_argument(
        "--list-mics", action="store_true",
        help="List available PipeWire audio sources and exit",
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="Print raw transcript without labels",
    )
    parser.add_argument(
        "--ignore-presence", action="store_true",
        help="Record even if presence state says listening is disabled",
    )
    parser.add_argument(
        "--stop-on-silence", action="store_true",
        help="Experimental: stop after speech is followed by silence instead of always using the full duration",
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
    parser.add_argument(
        "--min-record-seconds", type=float, default=1.5,
        help="Minimum recording time before silence detection can trigger (default: 1.5)",
    )
    args = parser.parse_args()

    if args.list_mics:
        sources = list_mics()
        if not sources:
            print("No PipeWire audio sources found.", file=sys.stderr)
            return 1
        for s in sources:
            print(f"  {s['name']}")
            print(f"    {s['description']}")
        return 0

    # Transcribe existing file
    if args.file:
        audio_path = Path(args.file).expanduser()
        if not audio_path.is_absolute():
            audio_path = WORKSPACE / audio_path
        if not audio_path.exists():
            print(f"File not found: {audio_path}", file=sys.stderr)
            return 1
        text, info = transcribe(
            audio_path,
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
        )
        if args.raw:
            print(text.strip())
        else:
            print(f"[{info['language']}] ({info['duration']:.1f}s) {text.strip()}")
        return 0

    # Record audio
    DEFAULT_TMP.mkdir(parents=True, exist_ok=True)

    if args.wav_out:
        wav_path = Path(args.wav_out).expanduser()
        if not wav_path.is_absolute():
            wav_path = WORKSPACE / wav_path
    else:
        wav_path = DEFAULT_TMP / f"listen-{time.strftime('%Y%m%d-%H%M%S')}.wav"

    if not args.ignore_presence:
        allowed, reason = can_listen()
        if not allowed:
            print(f"Listening blocked by presence state: {reason}", file=sys.stderr)
            return 3

    if args.stop_on_silence:
        print(
            f"Recording up to {args.duration}s from {args.source}; stopping after speech + {args.silence_seconds:.1f}s silence...",
            file=sys.stderr,
        )
    else:
        print(f"Recording {args.duration}s from {args.source}...", file=sys.stderr)
    interrupted = False
    try:
        if not args.ignore_presence:
            set_activity("listening", updated_by="listen.py")
        record_wav(
            wav_path,
            duration=args.duration,
            source=args.source,
            stop_on_silence=args.stop_on_silence,
            silence_seconds=args.silence_seconds,
            silence_threshold=args.silence_threshold,
            speech_threshold=args.speech_threshold,
            min_record_seconds=args.min_record_seconds,
        )
    except KeyboardInterrupt:
        # SIGINT from PTT widget — convert whatever we recorded before exiting
        interrupted = True
        raw_path = wav_path.with_suffix(".raw")
        if raw_path.exists() and raw_path.stat().st_size > 0 and not wav_path.exists():
            print("Interrupted; converting partial recording to WAV...", file=sys.stderr, flush=True)
            raw_to_wav(raw_path, wav_path)
            raw_path.unlink(missing_ok=True)
    except RuntimeError as exc:
        print(f"Recording error: {exc}", file=sys.stderr)
        return 1
    finally:
        if not args.ignore_presence:
            set_activity("idle", updated_by="listen.py:restore")

    if not wav_path.exists():
        print("No recording captured.", file=sys.stderr)
        return 2 if interrupted else 1

    file_size = wav_path.stat().st_size
    print(f"Recorded: {wav_path} ({file_size} bytes)", file=sys.stderr)

    if args.no_transcribe:
        print(str(wav_path))
        return 0

    # Transcribe
    print(f"Transcribing with {args.model}...", file=sys.stderr)
    start = time.time()
    try:
        text, info = transcribe(
            wav_path,
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
        )
    except Exception as exc:
        print(f"Transcription error: {exc}", file=sys.stderr)
        return 1
    elapsed = time.time() - start

    if args.raw:
        print(text.strip())
    else:
        print(f"[{info['language']}] ({info['duration']:.1f}s audio, {elapsed:.1f}s transcribe) {text.strip()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
