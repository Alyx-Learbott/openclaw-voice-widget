#!/usr/bin/env python3
"""Agent local voice control widget — GTK4/Adwaita desktop controller.

Five actions:
  1. Press to Talk (speech reply) — hold to record, release to send, Agent speaks reply
  2. Press to Talk (GUI reply) — hold to record, release to send, text-only reply
  3. Conversation mode — continuous listen/speak loop with auto-stop on 1 silent turn
  4. TTS warm toggle — pre-load Kokoro model for faster first reply
  5. Privacy kill switch — stops all recording, mutes mic

Architecture:
  - Widget is a thin controller over voice/presence.py and voice/converse.py.
  - PTT records via listen.py, then hands WAV to converse.py --wav-in.
  - Conversation mode runs converse.py --stop-on-silence in a loop, stopping after
    1 consecutive empty/silent turn.
  - TTS warm runs speak.py --warm-only to load the Kokoro model.
  - Status sync via voice/presence-state.json polling.
  - Visual: green = recording/conversation, red = privacy, neutral = idle.

Requirements:
  - Python 3.10+
  - PyGObject (gi) with GTK 4 and libadwaita
  - All existing voice/ dependencies

Run:
  cd /shared/openclaw_shared/workspace
  . .venv-voice/bin/activate
  python voice/widget.py
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
import fcntl

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk

VOICE_DIR = Path(__file__).resolve().parent
WORKSPACE = VOICE_DIR.parent
STATE_PATH = VOICE_DIR / "presence-state.json"
CONFIG_PATH = VOICE_DIR / "widget-config.json"
VENV_PYTHON = WORKSPACE / ".venv-voice" / "bin" / "python3"
SYSTEM_PYTHON = Path(sys.executable)

# Use venv Python for voice tools (faster_whisper, presence, etc)
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else str(SYSTEM_PYTHON)

APP_ID = "ai.openclaw.agent-voice"
LOCK_FILE = Path(f"/tmp/{APP_ID}.lock")

# ── Tunable settings ──────────────────────────────────────
# Defaults are loaded from voice/widget-config.json when present.
# Widget must be restarted for config changes to take effect.
DEFAULT_CONFIG = {
    "agent_name": "Agent",
    "conversation_silence_seconds": 2.5,
    "conversation_max_duration": 180,
    "ptt_stop_on_silence": False,
    "ptt_silence_seconds": 4.0,
    "ptt_max_duration": 30,
    "whisper_model": "small.en",
    "tts_backend": "kokoro",
    "tts_voice": "af_heart",
    "tts_speed": 1.0,
    "tts_daemon_idle_timeout": 300,
    "conversation_idle_timeout": 180,
}


def agent_label() -> str:
    """Return the configured display name for this agent/widget."""
    return AGENT_NAME


def load_widget_config() -> dict:
    """Load widget settings from JSON, falling back safely to defaults."""
    config = DEFAULT_CONFIG.copy()
    if CONFIG_PATH.exists():
        try:
            loaded = json.loads(CONFIG_PATH.read_text())
            if isinstance(loaded, dict):
                config.update({k: v for k, v in loaded.items() if k in DEFAULT_CONFIG})
        except Exception as e:
            print(f"[widget] Could not load {CONFIG_PATH}: {e}; using defaults", file=sys.stderr, flush=True)
    return config


def save_widget_config(config: dict) -> None:
    """Persist widget settings in a human-editable JSON file."""
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")


def apply_widget_config(config: dict) -> None:
    """Apply config values to the module-level settings used by actions."""
    global CONFIG, AGENT_NAME, CONV_SILENCE_SECONDS, CONV_MAX_DURATION
    global PTT_SILENCE_SECONDS, PTT_STOP_ON_SILENCE, PTT_MAX_DURATION
    global WHISPER_MODEL, TTS_BACKEND, TTS_VOICE, TTS_SPEED
    global IDLE_TIMEOUT_S, CONV_IDLE_TIMEOUT_S

    CONFIG = config
    AGENT_NAME = str(CONFIG.get("agent_name") or "Agent").strip() or "Agent"
    CONV_SILENCE_SECONDS = float(CONFIG["conversation_silence_seconds"])
    CONV_MAX_DURATION = int(CONFIG["conversation_max_duration"])
    PTT_SILENCE_SECONDS = float(CONFIG["ptt_silence_seconds"])
    PTT_STOP_ON_SILENCE = bool(CONFIG["ptt_stop_on_silence"])
    PTT_MAX_DURATION = int(CONFIG["ptt_max_duration"])
    WHISPER_MODEL = str(CONFIG["whisper_model"])
    TTS_BACKEND = str(CONFIG["tts_backend"])
    TTS_VOICE = str(CONFIG["tts_voice"])
    TTS_SPEED = float(CONFIG["tts_speed"])
    IDLE_TIMEOUT_S = int(CONFIG["tts_daemon_idle_timeout"])
    CONV_IDLE_TIMEOUT_S = int(CONFIG["conversation_idle_timeout"])


apply_widget_config(load_widget_config())

# Colors for status indication
COLOR_IDLE = "#888888"
COLOR_RECORDING = "#4CAF50"    # Green
COLOR_THINKING = "#FF9800"     # Orange
COLOR_SPEAKING = "#2196F3"     # Blue
COLOR_PRIVACY = "#F44336"      # Red
COLOR_WARMING = "#9C27B0"      # Purple


class VoiceWidget(Gtk.ApplicationWindow):
    """Main voice control window."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs, title=f"{agent_label()} Voice")

        self._recording_proc = None
        self._converse_proc = None
        self._converse_thread = None
        self._ptt_wav_path = None
        self._ptt_speech_reply = True
        self._tts_warmed = False
        self._conv_active = False
        self._poll_source_id = None
        self._conv_watchdog_source_id = None
        self._conv_last_meaningful_activity_time = 0.0

        self.set_default_size(320, 400)
        self.set_resizable(False)

        self._build_ui()
        self._start_polling()

    # ── UI construction ──────────────────────────────────────

    def _build_ui(self):
        main_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            margin_top=16,
            margin_bottom=16,
            margin_start=16,
            margin_end=16,
        )
        self.set_child(main_box)

        # Title + status
        self._title_label = Gtk.Label(label=f"{agent_label()} Voice Control")
        self._title_label.add_css_class("title-4")
        main_box.append(self._title_label)

        self._status_label = Gtk.Label(label="Idle")
        self._status_label.add_css_class("heading")
        main_box.append(self._status_label)

        # Status indicator (colored box)
        self._status_box = Gtk.Box()
        self._status_box.set_size_request(-1, 6)
        self._status_box.add_css_class("status-indicator")
        self._set_indicator_color(COLOR_IDLE)
        main_box.append(self._status_box)

        # ── Push to Talk section ──
        ptt_frame = Gtk.Frame(label="Push to Talk")
        ptt_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8,
                           margin_top=8, margin_bottom=8, margin_start=8, margin_end=8)
        ptt_frame.set_child(ptt_box)
        main_box.append(ptt_frame)

        self._ptt_gui_btn = Gtk.Button(label="⌨️ Press & Hold (GUI reply)")
        ptt_box.append(self._ptt_gui_btn)

        self._ptt_speech_btn = Gtk.Button(label="🎙️ Press & Hold (speech reply)")
        self._ptt_speech_btn.add_css_class("suggested-action")
        ptt_box.append(self._ptt_speech_btn)

        self._stop_spoken_btn = Gtk.Button(label="🛑 Stop Spoken Reply")
        self._stop_spoken_btn.connect("clicked", self._on_stop_spoken_reply_clicked)
        main_box.append(self._stop_spoken_btn)

        # ── Conversation mode ──
        conv_frame = Gtk.Frame(label="Conversation")
        conv_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8,
                           margin_top=8, margin_bottom=8, margin_start=8, margin_end=8)
        conv_frame.set_child(conv_box)
        main_box.append(conv_frame)

        self._conv_btn = Gtk.ToggleButton(label="💬 Start Conversation")
        self._conv_btn.add_css_class("suggested-action")
        self._conv_btn.connect("toggled", self._on_conversation_toggled)
        conv_box.append(self._conv_btn)

        # ── Toggles row ──
        toggle_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8, homogeneous=True)
        main_box.append(toggle_box)

        self._tts_warm_btn = Gtk.ToggleButton(label="🔊 TTS Warm")
        self._tts_warm_btn.connect("toggled", self._on_tts_warm_toggled)
        toggle_box.append(self._tts_warm_btn)

        self._privacy_btn = Gtk.ToggleButton(label="🔒 Privacy")
        self._privacy_btn.add_css_class("destructive-action")
        self._privacy_btn.connect("toggled", self._on_privacy_toggled)
        toggle_box.append(self._privacy_btn)

        self._settings_btn = Gtk.Button(label="⚙️ Settings")
        self._settings_btn.connect("clicked", self._on_settings_clicked)
        main_box.append(self._settings_btn)

        self._instructions_btn = Gtk.Button(label="❔ Instructions")
        self._instructions_btn.connect("clicked", self._on_instructions_clicked)
        main_box.append(self._instructions_btn)

        # ── Info bar ──
        self._info_label = Gtk.Label(label="Ready", wrap=True)
        self._info_label.add_css_class("dim-label")
        main_box.append(self._info_label)

        # Wire push-to-talk gesture signals
        self._setup_ptt_gesture(self._ptt_speech_btn, speech_reply=True)
        self._setup_ptt_gesture(self._ptt_gui_btn, speech_reply=False)

    def _set_indicator_color(self, color: str):
        """Apply a background color to the status indicator."""
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(f".status-indicator {{ background: {color}; border-radius: 3px; min-height: 6px; }}".encode())
        # Apply to the default display (GTK4 way)
        display = self.get_display()
        if display:
            Gtk.StyleContext.add_provider_for_display(display, css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

    # ── Spoken reply interruption ───────────────────────────

    def _on_stop_spoken_reply_clicked(self, _button):
        """Stop currently playing speech without shutting down the TTS daemon."""
        patterns = [
            r"python3? .*/voice/speak\.py",
            r"python3? voice/speak\.py",
            r"aplay .*voice/tmp/.*\.wav",
            r"pw-play .*voice/tmp/.*\.wav",
            r"paplay .*voice/tmp/.*\.wav",
        ]
        stopped = False
        for pattern in patterns:
            result = subprocess.run(
                ["pkill", "-f", pattern],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            stopped = stopped or result.returncode == 0

        self._set_presence_mode("conversational")
        subprocess.Popen(
            [str(PYTHON), str(VOICE_DIR / "presence.py"), "activity", "idle"],
            cwd=str(WORKSPACE),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._set_indicator_color(COLOR_IDLE)
        self._set_info("Spoken reply stopped" if stopped else "No spoken reply was playing")

    # ── Settings dialog ─────────────────────────────────────

    def _on_settings_clicked(self, _button):
        """Open a small settings dialog backed by widget-config.json."""
        dialog = Gtk.Dialog(title=f"{agent_label()} Voice Settings", transient_for=self, modal=True)
        dialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
        dialog.add_button("Save", Gtk.ResponseType.OK)
        dialog.set_default_size(380, 300)

        content = dialog.get_content_area()
        content.set_margin_top(12)
        content.set_margin_bottom(12)
        content.set_margin_start(12)
        content.set_margin_end(12)
        content.set_spacing(10)

        grid = Gtk.Grid(column_spacing=10, row_spacing=10)
        content.append(grid)

        def add_label(text: str, row: int):
            label = Gtk.Label(label=text, xalign=0)
            grid.attach(label, 0, row, 1, 1)
            return label

        add_label("Agent name", 0)
        agent_name = Gtk.Entry()
        agent_name.set_text(agent_label())
        grid.attach(agent_name, 1, 0, 1, 1)

        add_label("Conversation silence (sec)", 1)
        conv_silence = Gtk.SpinButton.new_with_range(1.0, 10.0, 0.5)
        conv_silence.set_value(CONV_SILENCE_SECONDS)
        grid.attach(conv_silence, 1, 1, 1, 1)

        add_label("Conversation idle timeout (sec)", 2)
        conv_idle = Gtk.SpinButton.new_with_range(30, 900, 30)
        conv_idle.set_value(CONV_IDLE_TIMEOUT_S)
        grid.attach(conv_idle, 1, 2, 1, 1)

        add_label("Whisper model", 3)
        whisper_model = Gtk.ComboBoxText()
        for model in ("base.en", "small.en"):
            whisper_model.append_text(model)
        whisper_model.set_active(1 if WHISPER_MODEL == "small.en" else 0)
        grid.attach(whisper_model, 1, 3, 1, 1)

        ptt_stop = Gtk.CheckButton(label="PTT stop on silence")
        ptt_stop.set_active(PTT_STOP_ON_SILENCE)
        grid.attach(ptt_stop, 0, 4, 2, 1)

        add_label("PTT silence (sec)", 5)
        ptt_silence = Gtk.SpinButton.new_with_range(1.0, 10.0, 0.5)
        ptt_silence.set_value(PTT_SILENCE_SECONDS)
        ptt_silence.set_sensitive(PTT_STOP_ON_SILENCE)
        grid.attach(ptt_silence, 1, 5, 1, 1)

        ptt_stop.connect("toggled", lambda btn: ptt_silence.set_sensitive(btn.get_active()))

        note = Gtk.Label(
            label="Saved settings apply to new turns immediately. Active recordings/conversations keep their current settings.",
            wrap=True,
            xalign=0,
        )
        note.add_css_class("dim-label")
        content.append(note)

        dialog.connect(
            "response",
            self._on_settings_response,
            agent_name,
            conv_silence,
            conv_idle,
            whisper_model,
            ptt_stop,
            ptt_silence,
        )
        dialog.present()

    def _on_settings_response(self, dialog, response, agent_name, conv_silence, conv_idle, whisper_model, ptt_stop, ptt_silence):
        if response == Gtk.ResponseType.OK:
            new_config = CONFIG.copy()
            new_config["agent_name"] = agent_name.get_text().strip() or "Agent"
            new_config["conversation_silence_seconds"] = round(conv_silence.get_value(), 1)
            new_config["conversation_idle_timeout"] = int(conv_idle.get_value())
            new_config["whisper_model"] = whisper_model.get_active_text() or WHISPER_MODEL
            new_config["ptt_stop_on_silence"] = bool(ptt_stop.get_active())
            new_config["ptt_silence_seconds"] = round(ptt_silence.get_value(), 1)

            try:
                save_widget_config(new_config)
                apply_widget_config(new_config)
                self._refresh_agent_labels()
                self._set_info("Settings saved")
            except Exception as e:
                self._set_info(f"Settings save failed: {e}")

        dialog.destroy()

    def _refresh_agent_labels(self):
        """Refresh visible labels after the configured agent name changes."""
        self.set_title(f"{agent_label()} Voice")
        if hasattr(self, "_title_label"):
            self._title_label.set_label(f"{agent_label()} Voice Control")

    # ── Instructions dialog ─────────────────────────────────

    def _on_instructions_clicked(self, _button):
        """Show concise human-facing instructions for the widget."""
        name = agent_label()
        text = f"""PTT means Push to Talk.

Press & Hold (speech reply): hold the button while speaking, release to send, and {name}'s reply is spoken out loud.

Press & Hold (GUI reply): hold the button while speaking, release to send, and {name}'s reply appears in the OpenClaw control chat window instead of being spoken out loud.

Conversation mode: keeps taking voice turns until you stop it, a silent turn is detected, or the conversation idle timeout is reached.

Stop Spoken Reply: stops the currently playing spoken audio reply without shutting down the Kokoro speech engine. Use this when you have read ahead and want to speak or press-and-hold again immediately.

Both your transcribed speech and {name}'s replies appear in the OpenClaw control chat window in every mode. Speech reply adds spoken audio; GUI reply is text-only.

TTS Warm: preloads the Kokoro speech engine so {name}'s first spoken reply starts faster.

Privacy: blocks recording/listening and stops active voice capture.

Conversation silence timeout: after you start speaking, this is how long the widget waits through silence before deciding your turn is finished. The timer does not begin until after your first spoken words are detected.

Conversation idle timeout: how long conversation mode can sit with no successful speech turns before it automatically stops.

Whisper model: chooses the speech transcription model. base.en is faster; small.en is usually cleaner.

PTT stop on silence: optional auto-stop for Push to Talk. Normally off, because releasing the button is the manual stop.

PTT silence timeout: only applies if PTT stop on silence is enabled."""

        dialog = Gtk.Dialog(title=f"{name} Voice Instructions", transient_for=self, modal=True)
        dialog.add_button("OK", Gtk.ResponseType.OK)
        dialog.set_default_size(460, 520)

        content = dialog.get_content_area()
        content.set_margin_top(12)
        content.set_margin_bottom(12)
        content.set_margin_start(12)
        content.set_margin_end(12)
        content.set_spacing(10)

        title = Gtk.Label(label=f"{name} Voice Instructions", xalign=0)
        title.add_css_class("title-4")
        content.append(title)

        body = Gtk.Label(label=text, wrap=True, xalign=0)
        body.set_selectable(True)
        content.append(body)

        dialog.connect("response", lambda d, _response: d.destroy())
        dialog.present()

    # ── Push to Talk handlers ────────────────────────────────

    def _setup_ptt_gesture(self, button: Gtk.Button, speech_reply: bool):
        gesture = Gtk.GestureClick()
        gesture.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        gesture.connect("pressed", self._on_ptt_press, speech_reply)
        gesture.connect("released", self._on_ptt_release, speech_reply)
        button.add_controller(gesture)

    def _on_ptt_press(self, gesture, n_press, x, y, speech_reply: bool):
        """Start recording on button press."""
        # Check privacy mode
        if self._privacy_btn.props.active:
            self._set_info("🔒 Privacy mode — recording blocked")
            return

        mode = "conversational" if speech_reply else "listen_only"
        self._set_presence_mode(mode)
        self._set_status("🔴 Recording…")
        self._set_indicator_color(COLOR_RECORDING)
        self._set_info(f"Recording ({'speech' if speech_reply else 'GUI'} reply)")

        print(f"[widget] PTT press: speech_reply={speech_reply}", file=sys.stderr, flush=True)
        self._start_ptt_recording(speech_reply)

    def _on_ptt_release(self, gesture, n_press, x, y, speech_reply: bool):
        """Stop recording and process on button release."""
        print(f"[widget] PTT release: speech_reply={speech_reply}", file=sys.stderr, flush=True)
        self._stop_ptt_recording(speech_reply)

    def _start_ptt_recording(self, speech_reply: bool):
        """Launch listen.py to record until we stop it."""
        wav_path = VOICE_DIR / "tmp" / f"ptt-{time.strftime('%Y%m%d-%H%M%S')}.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)

        # Removed --duration for PTT. It now records until terminated on release.
        cmd = [
            str(PYTHON), str(VOICE_DIR / "listen.py"),
            "--source", "POROSVOC",
            "--no-transcribe",
            "--ignore-presence",
            "--wav-out", str(wav_path),
        ]
        if PTT_STOP_ON_SILENCE:
            cmd.extend(["--stop-on-silence", "--silence-seconds", str(PTT_SILENCE_SECONDS)])

        print(f"[widget] Starting listen.py: {cmd}", file=sys.stderr, flush=True)
        self._ptt_wav_path = wav_path
        self._ptt_speech_reply = speech_reply
        self._recording_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(WORKSPACE),
        )
        print(f"[widget] listen.py PID: {self._recording_proc.pid}", file=sys.stderr, flush=True)

    def _stop_ptt_recording(self, speech_reply: bool):
        """Stop recording, transcribe, and send via converse.py."""
        proc = self._recording_proc
        self._recording_proc = None

        if proc is None or proc.poll() is not None:
            self._set_status("Idle")
            self._set_indicator_color(COLOR_IDLE)
            return

        # Send SIGINT to stop recording gracefully
        # listen.py will convert .raw to .wav even on interrupt
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

        wav_path = self._ptt_wav_path
        if wav_path is None or not wav_path.exists():
            # Check if raw file exists but conversion failed
            raw_path = wav_path.with_suffix(".raw") if wav_path else None
            if raw_path and raw_path.exists():
                # Attempt manual conversion
                try:
                    from listen import raw_to_wav
                    raw_to_wav(raw_path, wav_path)
                    raw_path.unlink(missing_ok=True)
                except Exception:
                    pass
            if wav_path is None or not wav_path.exists():
                self._set_status("Idle")
                self._set_indicator_color(COLOR_IDLE)
                self._set_info("No recording captured")
                self._set_presence_mode("conversational")
                return

        self._set_status("🧠 Transcribing…")
        self._set_indicator_color(COLOR_THINKING)

        def _transcribe_and_send():
            try:
                cmd = [
                    str(PYTHON), str(VOICE_DIR / "converse.py"),
                    "--session-key", "agent:main:main",
                    "--wav-in", str(wav_path),
                    "--model", WHISPER_MODEL,
                    "--backend", TTS_BACKEND if speech_reply else "piper",
                ]
                if not speech_reply:
                    cmd.append("--no-play")

                # Use a generous timeout for LLM roundtrip
                result = subprocess.run(
                    cmd,
                    capture_output=True, text=True,
                    cwd=str(WORKSPACE),
                    timeout=180,
                )
                exit_code = result.returncode
                if exit_code == 2:
                    GLib.idle_add(self._set_info, "No speech detected")
                elif exit_code != 0:
                    stderr = (result.stderr or "").strip()
                    GLib.idle_add(self._set_info, f"Error: {stderr[:120]}")
                else:
                    GLib.idle_add(self._set_info, "Turn complete")

            except subprocess.TimeoutExpired:
                # If the turn times out, we DON'T just give up.
                # converse.py usually logs what it got before timing out.
                # We inform the user and let the system handle the cleanup.
                GLib.idle_add(self._set_info, "Response timed out (too long)")
            except Exception as e:
                GLib.idle_add(self._set_info, f"Error: {e}")
            finally:
                # Clean up temp WAV
                if wav_path and wav_path.exists():
                    try:
                        wav_path.unlink()
                    except OSError:
                        pass
                GLib.idle_add(self._set_status, "Idle")
                GLib.idle_add(self._set_indicator_color, COLOR_IDLE)
                self._set_presence_mode("conversational")

        threading.Thread(target=_transcribe_and_send, daemon=True).start()

    # ── Conversation mode ────────────────────────────────────

    def _on_conversation_toggled(self, btn):
        if btn.props.active:
            # Privacy check
            if self._privacy_btn.props.active:
                self._set_info("🔒 Privacy mode — cannot start conversation")
                btn.props.active = False
                return
            self._conv_active = True
            self._conv_last_meaningful_activity_time = time.time()
            self._conv_btn.set_label("💬 Stop Conversation")
            self._set_status("💬 Conversation")
            self._set_indicator_color(COLOR_RECORDING)
            self._set_info("Starting conversation loop…")
            self._start_conversation_watchdog()
            self._start_conversation_loop()
        else:
            self._stop_conversation()

    def _start_conversation_loop(self):
        """Run converse.py in a loop, stopping after 1 consecutive silent turn."""
        if not self._conv_active:
            return

        # Dynamically use the CONFIG values from the settings dialog
        cmd = [
            str(PYTHON), str(VOICE_DIR / "converse.py"),
            "--session-key", "agent:main:main",
            "--stop-on-silence",
            "--silence-seconds", str(CONFIG["conversation_silence_seconds"]),
            "--duration", str(CONFIG["conversation_max_duration"]),
            "--model", WHISPER_MODEL,
        ]

        def _run_turn():
            try:
                self._converse_proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=str(WORKSPACE),
                )
                stdout, stderr = self._converse_proc.communicate(timeout=180)
                exit_code = self._converse_proc.returncode
                self._converse_proc = None

                if exit_code == 2:
                    # Empty transcript — 1 silent turn, stop loop
                    GLib.idle_add(self._set_info, "Silent turn — conversation paused")
                    GLib.idle_add(self._end_conversation)
                    return
                elif exit_code == 130:
                    # Keyboard interrupt — stop loop
                    GLib.idle_add(self._end_conversation)
                    return
                elif exit_code != 0:
                    stderr_text = (stderr or "").strip()
                    GLib.idle_add(self._set_info, f"Turn error: {stderr_text[:100]}")
                else:
                    transcript = self._extract_transcript_from_converse_stdout(stdout or "")
                    if self._is_meaningful_transcript(transcript):
                        # Successful meaningful user turn — reset idle timer.
                        self._conv_last_meaningful_activity_time = time.time()

                # Continue loop if still active
                if self._conv_active:
                    GLib.idle_add(self._start_conversation_loop)

            except subprocess.TimeoutExpired:
                if self._converse_proc:
                    self._converse_proc.terminate()
                GLib.idle_add(self._set_info, "Turn timed out")
                if self._conv_active:
                    GLib.idle_add(self._start_conversation_loop)
            except Exception as e:
                GLib.idle_add(self._set_info, f"Error: {e}")
                GLib.idle_add(self._end_conversation)

        self._converse_thread = threading.Thread(target=_run_turn, daemon=True)
        self._converse_thread.start()

    def _extract_transcript_from_converse_stdout(self, stdout: str) -> str:
        """Extract the user transcript printed by converse.py, if present."""
        marker = "Kevin:"
        if marker not in stdout:
            return ""
        after_marker = stdout.split(marker, 1)[1]
        before_reply = after_marker.split("\nAgent:", 1)[0]
        return before_reply.strip()

    def _is_meaningful_transcript(self, transcript: str) -> bool:
        """Ignore common tiny Whisper hallucinations when tracking idle time."""
        text = " ".join(transcript.lower().strip().strip(".!?,;:").split())
        if not text:
            return False
        hallucinations = {
            "you",
            "thank you",
            "thanks",
            "thank you thank you",
            "you you",
            "you you you",
        }
        if text in hallucinations:
            return False
        words = text.split()
        if len(words) <= 2 and len(text) <= 12:
            return False
        return True

    def _start_conversation_watchdog(self):
        """Start low-overhead idle enforcement for conversation mode."""
        self._stop_conversation_watchdog()
        self._conv_watchdog_source_id = GLib.timeout_add_seconds(10, self._conversation_idle_watchdog)

    def _stop_conversation_watchdog(self):
        if self._conv_watchdog_source_id is not None:
            GLib.source_remove(self._conv_watchdog_source_id)
            self._conv_watchdog_source_id = None

    def _conversation_idle_watchdog(self) -> bool:
        if not self._conv_active:
            self._conv_watchdog_source_id = None
            return False
        idle_for = time.time() - self._conv_last_meaningful_activity_time
        if idle_for >= CONV_IDLE_TIMEOUT_S:
            self._set_info(f"Conversation idle — no meaningful speech for {CONV_IDLE_TIMEOUT_S}s")
            self._end_conversation()
            self._conv_watchdog_source_id = None
            return False
        return True

    def _stop_active_converse_proc(self):
        if self._converse_proc and self._converse_proc.poll() is None:
            self._converse_proc.send_signal(signal.SIGINT)

    def _end_conversation(self):
        """Called from conversation loop when it stops itself (e.g. silent turn)."""
        if not self._conv_active:
            return
        self._conv_active = False
        self._stop_conversation_watchdog()
        self._stop_active_converse_proc()
        self._conv_btn.props.active = False
        self._conv_btn.set_label("💬 Start Conversation")
        self._set_status("Idle")
        self._set_indicator_color(COLOR_IDLE)
        self._set_presence_mode("conversational")

    def _stop_conversation(self):
        """Called when user toggles conversation off."""
        self._conv_active = False
        self._stop_conversation_watchdog()
        self._stop_active_converse_proc()
        self._conv_btn.set_label("💬 Start Conversation")
        self._set_status("Idle")
        self._set_indicator_color(COLOR_IDLE)
        self._set_info("Conversation ended")
        self._set_presence_mode("conversational")

    # ── TTS Warm toggle ──────────────────────────────────────

    def _on_tts_warm_toggled(self, btn):
        if btn.props.active:
            self._set_info("Starting Kokoro TTS daemon…")
            self._set_indicator_color(COLOR_WARMING)

            def _start_daemon():
                try:
                    # Check if daemon is already running
                    check = subprocess.run(
                        [str(PYTHON), str(VOICE_DIR / "speak-daemon.py"), "--check"],
                        capture_output=True, text=True, cwd=str(WORKSPACE), timeout=3,
                    )
                    if check.returncode == 0:
                        self._tts_warmed = True
                        GLib.idle_add(self._set_info, "Kokoro TTS daemon already running ✅")
                        GLib.idle_add(self._set_indicator_color, COLOR_IDLE)
                        return

                    # Start daemon in background with Kokoro's venv Python
                    kokoro_python = Path.home() / ".openclaw" / "tts" / "kokoro" / ".venv312" / "bin" / "python"
                    daemon_cmd = [
                        str(kokoro_python) if kokoro_python.exists() else str(PYTHON),
                        str(VOICE_DIR / "speak-daemon.py"),
                    ]
                    subprocess.Popen(
                        daemon_cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        cwd=str(WORKSPACE),
                    )

                    # Wait for daemon to become ready
                    for _ in range(30):  # 30 x 0.5s = 15s max
                        time.sleep(0.5)
                        check = subprocess.run(
                            [str(PYTHON), str(VOICE_DIR / "speak-daemon.py"), "--check"],
                            capture_output=True, text=True, cwd=str(WORKSPACE), timeout=3,
                        )
                        if check.returncode == 0:
                            self._tts_warmed = True
                            GLib.idle_add(self._set_info, "Kokoro TTS daemon ready ✅")
                            GLib.idle_add(self._set_indicator_color, COLOR_IDLE)
                            return

                    GLib.idle_add(self._set_info, "TTS daemon failed to start")
                    GLib.idle_add(self._set_indicator_color, COLOR_IDLE)

                except Exception as e:
                    GLib.idle_add(self._set_info, f"TTS daemon error: {e}")
                    GLib.idle_add(self._set_indicator_color, COLOR_IDLE)

            threading.Thread(target=_start_daemon, daemon=True).start()
        else:
            # Request daemon shutdown
            self._set_info("Stopping TTS daemon…")
            try:
                subprocess.run(
                    [str(PYTHON), str(VOICE_DIR / "speak-daemon.py"), "--shutdown"],
                    capture_output=True, text=True, cwd=str(WORKSPACE), timeout=5,
                )
            except Exception:
                pass
            self._tts_warmed = False
            self._set_info("TTS daemon stopped")

    # ── Privacy toggle ───────────────────────────────────────

    def _on_privacy_toggled(self, btn):
        if btn.props.active:
            self._set_presence_mode("privacy")
            self._set_status("🔒 Privacy")
            self._set_indicator_color(COLOR_PRIVACY)
            self._set_info("Mic off, speech off")
            # Stop any active recording or conversation
            if self._recording_proc and self._recording_proc.poll() is None:
                self._recording_proc.terminate()
                self._recording_proc = None
            if self._conv_active:
                self._conv_active = False
                self._conv_btn.props.active = False
                self._conv_btn.set_label("💬 Start Conversation")
                if self._converse_proc and self._converse_proc.poll() is None:
                    self._converse_proc.send_signal(signal.SIGINT)
        else:
            self._set_presence_mode("conversational")
            self._set_status("Idle")
            self._set_indicator_color(COLOR_IDLE)
            self._set_info("Privacy off")

    # ── Presence state helpers ────────────────────────────────

    def _set_presence_mode(self, mode: str):
        # Guard: ensure we don't spawn multiple presence updates in a tight loop
        subprocess.Popen(
            [str(PYTHON), str(VOICE_DIR / "presence.py"), "mode", mode],
            cwd=str(WORKSPACE),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _kill_voice_process(self, pattern: str):
        """Kill any existing process matching the pattern to prevent overlapping."""
        try:
            subprocess.run(["pkill", "-f", pattern], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def _set_status(self, text: str):
        self._status_label.set_label(text)

    def _set_info(self, text: str):
        self._info_label.set_label(text)

    def _start_polling(self):
        """Poll presence-state.json for external state changes."""
        self._poll_source_id = GLib.timeout_add(500, self._poll_state)

    def _poll_state(self) -> bool:
        """Called every 500ms to sync UI with presence state."""
        try:
            if STATE_PATH.exists():
                state = json.loads(STATE_PATH.read_text())
                activity = state.get("activity", "idle")
                # Update status based on activity from converse.py
                if activity == "listening" and not self._conv_active:
                    # Only update if we're not already showing something
                    pass
                elif activity == "thinking":
                    self._set_indicator_color(COLOR_THINKING)
                elif activity == "speaking":
                    self._set_indicator_color(COLOR_SPEAKING)
                elif activity == "idle":
                    if not self._conv_active and self._recording_proc is None:
                        if not self._privacy_btn.props.active:
                            self._set_indicator_color(COLOR_IDLE)
        except Exception:
            pass
        return True  # Continue polling

    def _stop_polling(self):
        if self._poll_source_id is not None:
            GLib.source_remove(self._poll_source_id)
            self._poll_source_id = None


def on_activate(app):
    win = VoiceWidget(application=app)
    win.present()


def main():
    # Recovery-First Lock: Kill existing instance if found, then start fresh
    lock_f = open(LOCK_FILE, 'w')
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        print("Another instance of Agent Voice is running. Attempting to replace...", file=sys.stderr)
        try:
            # Kill by process name to ensure we clear the zombie/frozen GUI
            subprocess.run(["pkill", "-f", "python3 .*voice/widget.py"], check=False)
            time.sleep(0.5) # Give the OS a moment to release the lock
            # Try to grab the lock again after killing
            fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception as e:
            print(f"Could not replace existing instance: {e}", file=sys.stderr)
            sys.exit(1)

    app = Adw.Application(application_id=APP_ID)
    app.connect("activate", on_activate)
    app.run(None)


if __name__ == "__main__":
    main()
