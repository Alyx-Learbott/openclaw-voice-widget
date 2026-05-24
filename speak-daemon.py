#!/usr/bin/env python3
"""Alyx TTS daemon — keeps Kokoro model in memory for fast synthesis.

Runs as a persistent process, listening on a Unix socket for synthesis requests.
This eliminates the ~5s cold-start per invocation.

Protocol (line-delimited JSON):
  Request:  {"action": "synthesize", "text": "...", "voice": "af_heart", "speed": 1.0, "wav_out": "/path/to/output.wav"}
  Response: {"status": "ok", "wav_path": "/path/to/output.wav", "duration": 2.5}
  Response: {"status": "error", "error": "..."}

  Request:  {"action": "ping"}
  Response: {"status": "ok", "backend": "kokoro"}

  Request:  {"action": "quit"}
  Response: {"status": "ok"}  (then exits)

Usage:
  python3 voice/speak-daemon.py                    # Start daemon, blocks until quit
  python3 voice/speak-daemon.py --socket-path /tmp/alyx-tts.sock  # Custom socket path
  python3 voice/speak-daemon.py --check             # Check if daemon is running
  python3 voice/speak-daemon.py --shutdown           # Request running daemon to quit
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DEFAULT_SOCKET_PATH = ROOT / "tmp" / "tts-daemon.sock"
DEFAULT_KOKORO_VOICE = "af_heart"
KOKORO_PYTHON = Path.home() / ".openclaw" / "tts" / "kokoro" / ".venv312" / "bin" / "python"

# Global pipeline — loaded once, reused for every request
_pipeline = None
_pipeline_voice = None


def load_pipeline():
    """Load the Kokoro pipeline into memory. Called once at startup."""
    global _pipeline
    if _pipeline is not None:
        return True

    # We import Kokoro via subprocess to use its isolated venv
    # But for the daemon, we need it in-process. Check if we can import directly.
    try:
        # The daemon should be run with the Kokoro venv Python
        from kokoro import KPipeline
        _pipeline = KPipeline(lang_code='a')
        return True
    except ImportError:
        # Fall back: try running within Kokoro's venv
        pass

    return False


def synthesize_kokoro(text: str, wav_path: Path, voice: str = DEFAULT_KOKORO_VOICE, speed: float = 1.0) -> dict:
    """Synthesize text using the in-process Kokoro pipeline."""
    import numpy as np
    import soundfile as sf

    wav_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = list(_pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+'))
    audio_parts = [chunk[2] for chunk in chunks]
    if not audio_parts:
        return {"status": "error", "error": "No audio produced"}
    audio = audio_parts[0] if len(audio_parts) == 1 else np.concatenate(audio_parts)
    sf.write(str(wav_path), audio, 24000)

    duration = len(audio) / 24000.0
    return {"status": "ok", "wav_path": str(wav_path), "duration": round(duration, 2)}


def synthesize_piper(text: str, wav_path: Path, model_path: Path) -> dict:
    """Synthesize text using Piper (fallback)."""
    import wave
    from piper import PiperVoice

    if not model_path.exists():
        return {"status": "error", "error": f"Piper model not found: {model_path}"}

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    voice = PiperVoice.load(str(model_path))
    with wave.open(str(wav_path), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

    return {"status": "ok", "wav_path": str(wav_path)}


def handle_request(data: dict) -> dict:
    """Process a single request and return a response dict."""
    action = data.get("action", "")

    if action == "ping":
        backend = "kokoro" if _pipeline is not None else "unknown"
        return {"status": "ok", "backend": backend}

    if action == "quit":
        return {"status": "ok", "shutdown": True}

    if action == "synthesize":
        text = data.get("text", "").strip()
        if not text:
            return {"status": "error", "error": "No text provided"}

        backend = data.get("backend", "kokoro")
        wav_out = data.get("wav_out")
        voice = data.get("voice", DEFAULT_KOKORO_VOICE)
        speed = data.get("speed", 1.0)

        if wav_out:
            wav_path = Path(wav_out)
            if not wav_path.is_absolute():
                wav_path = WORKSPACE / wav_path
        else:
            ROOT / "tmp" / "tts-daemon" / "tmp" .mkdir(parents=True, exist_ok=True)
            wav_path = ROOT / "tmp" / f"tts-daemon-{time.strftime('%Y%m%d-%H%M%S')}.wav"

        try:
            if backend == "kokoro" and _pipeline is not None:
                return synthesize_kokoro(text, wav_path, voice=voice, speed=speed)
            elif backend == "piper":
                model_path = Path(data.get("model", str(ROOT / "models" / "piper" / "en_US-lessac-medium.onnx")))
                if not model_path.is_absolute():
                    model_path = WORKSPACE / model_path
                return synthesize_piper(text, wav_path, model_path)
            else:
                return {"status": "error", "error": f"Backend '{backend}' not available"}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    return {"status": "error", "error": f"Unknown action: {action}"}


def send_to_daemon(socket_path: Path, data: dict, timeout: float = 30.0) -> dict:
    """Send a request to a running daemon and return the response."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(socket_path))
        sock.sendall((json.dumps(data) + "\n").encode("utf-8"))
        response_data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            response_data += chunk
            if b"\n" in response_data:
                break
        return json.loads(response_data.decode("utf-8").strip())
    except (ConnectionRefusedError, FileNotFoundError):
        return {"status": "error", "error": "Daemon not running"}
    except socket.timeout:
        return {"status": "error", "error": "Daemon timeout"}
    finally:
        sock.close()


def run_daemon(socket_path: Path, idle_timeout: float = 300.0):
    """Main daemon loop — load model, listen on socket, handle requests.
    
    Auto-shuts down after idle_timeout seconds with no requests.
    """
    print(f"Loading Kokoro model...", file=sys.stderr, flush=True)
    start = time.time()

    if not load_pipeline():
        # Try using the Kokoro venv Python to restart ourselves
        if KOKORO_PYTHON.exists():
            print(f"Restarting with Kokoro venv Python: {KOKORO_PYTHON}", file=sys.stderr, flush=True)
            os.execv(str(KOKORO_PYTHON), [str(KOKORO_PYTHON)] + sys.argv)
        else:
            print(f"ERROR: Cannot load Kokoro. Install Kokoro in {KOKORO_PYTHON}", file=sys.stderr, flush=True)
            sys.exit(1)

    load_time = time.time() - start
    print(f"Kokoro model loaded in {load_time:.1f}s", file=sys.stderr, flush=True)

    # Clean up stale socket
    if socket_path.exists():
        socket_path.unlink()

    socket_path.parent.mkdir(parents=True, exist_ok=True)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(1)
    server.settimeout(1.0)  # Check for shutdown every second

    print(f"TTS daemon listening on {socket_path} (idle timeout: {idle_timeout:.0f}s)", file=sys.stderr, flush=True)

    shutting_down = False
    last_request_time = time.time()

    def signal_handler(signum, frame):
        nonlocal shutting_down
        shutting_down = True
        print("Shutdown requested...", file=sys.stderr, flush=True)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while not shutting_down:
            # Check idle timeout
            if time.time() - last_request_time > idle_timeout:
                print(f"Idle timeout ({idle_timeout:.0f}s), shutting down.", file=sys.stderr, flush=True)
                break

            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue

            conn.settimeout(30.0)
            try:
                data = b""
                while True:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                    if b"\n" in data:
                        break

                if not data:
                    conn.close()
                    continue

                request = json.loads(data.decode("utf-8").strip())
                response = handle_request(request)

                # Serialize and add newline delimiter
                response_bytes = (json.dumps(response) + "\n").encode("utf-8")
                conn.sendall(response_bytes)
                conn.close()

                if request.get("action") == "quit" or response.get("shutdown"):
                    shutting_down = True

                # Reset idle timer on any successful request
                last_request_time = time.time()

            except Exception as exc:
                print(f"Error handling request: {exc}", file=sys.stderr, flush=True)
                try:
                    error_response = json.dumps({"status": "error", "error": str(exc)}) + "\n"
                    conn.sendall(error_response.encode("utf-8"))
                    conn.close()
                except Exception:
                    pass
    finally:
        server.close()
        if socket_path.exists():
            socket_path.unlink()
        print("TTS daemon stopped.", file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Alyx TTS daemon — keeps Kokoro model in memory")
    parser.add_argument("--socket-path", default=str(DEFAULT_SOCKET_PATH), help="Unix socket path")
    parser.add_argument("--check", action="store_true", help="Check if daemon is running")
    parser.add_argument("--shutdown", action="store_true", help="Request running daemon to quit")
    args = parser.parse_args()

    socket_path = Path(args.socket_path)

    if args.check:
        if not socket_path.exists():
            print("NOT RUNNING", file=sys.stderr)
            return 1
        result = send_to_daemon(socket_path, {"action": "ping"}, timeout=3.0)
        if result.get("status") == "ok":
            print(f"RUNNING (backend: {result.get('backend', 'unknown')})", file=sys.stderr)
            return 0
        else:
            print(f"NOT RESPONDING: {result.get('error', 'unknown')}", file=sys.stderr)
            return 1

    if args.shutdown:
        if not socket_path.exists():
            print("Daemon not running (no socket).", file=sys.stderr)
            return 0
        result = send_to_daemon(socket_path, {"action": "quit"}, timeout=5.0)
        print(f"Shutdown response: {result}", file=sys.stderr)
        # Wait for socket to disappear
        for _ in range(20):
            if not socket_path.exists():
                break
            time.sleep(0.25)
        return 0

    run_daemon(socket_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())