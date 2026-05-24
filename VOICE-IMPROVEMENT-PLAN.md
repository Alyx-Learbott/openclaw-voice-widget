# Alyx Voice Improvement Plan

Date: 2026-05-10
Status: research + planning complete; implementation not started

## Goal

Make Alyx's local voice smoother, warmer, and more emotionally expressive while keeping the current Piper path as the reliable fallback.

## Current baseline

- Runtime path: `voice/speak.py` → Piper Python API → WAV → ALSA/PipeWire player.
- Current model: `voice/models/piper/en_US-lessac-medium.onnx`.
- Model config defaults: sample rate 22050 Hz, `length_scale=1`, `noise_scale=0.667`, `noise_w=0.8`.
- Current helper already cleans Markdown/directives and supports `--wav-out`, `--no-play`, and custom `--model`.
- `say-last.py` fetches latest assistant text from OpenClaw history and pipes it to `speak.py`.

## Research findings

### Piper: best for reliability, limited for emotion

Piper is still the right always-available fallback: fast, local, CPU-friendly, simple. The Python API exposes synthesis controls through `SynthesisConfig`:

- `length_scale`: speaking speed / duration. Higher is slower; lower is faster.
- `noise_scale`: variation/noise in acoustic generation.
- `noise_w_scale`: variation in duration/prosody.
- `volume`: output gain.

Useful first-pass improvement: expose these as presets in `speak.py`, e.g. `calm`, `warm`, `bright`, `urgent`, `soft`, then A/B test generated WAVs. Piper does not appear to offer robust SSML-style emotion/emphasis controls; punctuation, line splitting, and synthesis parameters are the practical levers.

### Kokoro: best lightweight quality upgrade candidate

Kokoro-82M is open-weight, Apache-licensed, small, and designed for fast local inference. It supports multiple voices and a speed parameter, uses 24 kHz output, and should be much easier to run locally than the large expressive models. It is probably the best Phase 2 experiment after Piper tuning.

Caveat: likely more natural than Piper, but not deeply emotion-controllable. Good for smoother baseline voice, less good for explicit emotional acting.

### Zonos: best explicit emotion-control candidate

Zonos is open-weight and supports voice cloning/reference audio plus fine-grained conditioning: speaking rate, pitch variation, audio quality, and emotion vectors such as happiness, fear, sadness, anger, neutral, etc. It outputs at 44 kHz and has a Gradio/demo path.

Caveats: intended for GPU; CPU can run but is likely too slow for interactive use. Hybrid model has stricter GPU requirements; transformer may be more feasible. This is the most interesting candidate for an expressive Alyx identity, but probably not first implementation.

### F5-TTS: strong voice-cloning/quality candidate

F5-TTS v1 supports zero-shot generation from reference audio, CLI/Gradio use, chunk inference, and multi-style/multi-speaker generation. It has PyTorch, Docker, ROCm/AMD mentions, and an ONNX community path.

Caveats: pretrained model license is CC-BY-NC; less explicit emotion control than Zonos/Orpheus/Dia; setup is heavier than Kokoro.

### Orpheus: strong human-like/emotive candidate, heavy

Orpheus is a 3B LLM-backed TTS system with natural rhythm, zero-shot cloning, streaming, and simple emotion/intonation tags. It has local and no-GPU implementation references, but the main fast path uses vLLM/GPU-style serving.

Caveats: heavy for this SER9 setup, more moving parts, likely not the first local experiment unless we specifically want to evaluate tag-driven emotion.

### Dia: great for dialogue/nonverbals, not ideal as Alyx's default voice

Dia generates realistic dialogue, supports audio conditioning and tags like `(laughs)`, `(sighs)`, `(clears throat)`, etc. Good for storytime/dialogue experiments.

Caveats: GPU-tested; English-only; single-speaker default consistency requires audio prompt or fixed seed; input format is dialogue-oriented (`[S1]`, `[S2]`). Not ideal for everyday assistant speech.

### StyleTTS2: high quality but research-heavy

StyleTTS2 is high quality and style-focused, but packaging/finetuning looks heavier and less clean for the immediate goal. Good to keep in the background, not first.

## Recommended staged plan

### Phase 1 — Tune Piper without changing architecture

Purpose: immediate improvement, low risk.

1. Add optional synthesis parameters to `voice/speak.py`:
   - `--length-scale`
   - `--noise-scale`
   - `--noise-w-scale`
   - `--volume`
   - `--style {default,calm,warm,bright,urgent,soft}`
2. Add a tiny text preprocessor for speech rhythm:
   - split long replies into shorter spoken chunks;
   - convert em dash/semicolon/colon into natural pauses;
   - avoid speaking lists as one flat run-on paragraph;
   - optionally add period/comma pauses where Markdown cleanup removes structure.
3. Generate an A/B test set under `voice/tests/` with the same 6-10 sentences across presets.
4. Kevin listens and picks the least-annoying baseline.

Expected result: smoother pacing and less robotic delivery, but not true emotion.

### Phase 2 — Add a pluggable TTS backend interface

Purpose: make experiments safe and reversible.

1. Keep `speak.py` as the public CLI.
2. Internally route to `piper` by default.
3. Add backend option: `--backend piper|kokoro|zonos|f5|orpheus|dia` over time.
4. Preserve `--wav-out`, `--no-play`, direct ALSA playback, and text cleanup behavior.
5. Never remove Piper fallback.

Expected result: we can test better engines without breaking the current voice.

### Phase 3 — Try Kokoro as the first new local voice

Purpose: likely best effort/quality ratio.

1. Create an isolated environment, probably `.venv-tts-kokoro` or a subdir under `tools/tts/`.
2. Install Kokoro dependencies only after approval.
3. Generate samples using a few American English voices, especially warmer/friendlier ones.
4. Compare against tuned Piper on latency, naturalness, stability, and CPU load.
5. If clearly better, add `--backend kokoro` as optional.

Expected result: smoother and more natural than Piper if setup is clean.

### Phase 4 — Evaluate expressive engines offline

Purpose: find the eventual emotive Alyx voice.

Priority order:

1. Zonos transformer: explicit emotion + pitch/rate controls; best fit for controllable emotive voice if local performance is acceptable.
2. Orpheus: tag-driven human-like prosody; evaluate if GPU/CPU path is tolerable.
3. F5-TTS: voice identity/quality and reference-driven style; evaluate licensing and performance.
4. Dia: keep for story/dialogue/nonverbal experiments, not default assistant voice.
5. StyleTTS2: defer unless other paths fail.

Expected result: choose one expressive experimental backend, not five half-integrations.

## Second-pass revision after targeted research

Original instinct was “jump to Zonos/F5 for emotion.” After the second pass, the better implementation order is:

1. **Piper tuning first** because we already have it, its Python API exposes useful controls, and this can improve voice feel immediately.
2. **Kokoro second** because it is small/open/local and likely offers the fastest quality jump without huge hardware risk.
3. **Zonos third** because it is the best match for explicit emotion controls, but heavier and likely non-interactive on CPU.
4. **Orpheus/Dia as specialty experiments**, not default voice candidates yet.

This avoids getting trapped in a large model setup before we have a clean evaluation harness.

## Proposed first implementation step

When Kevin is back: implement Phase 1 only.

Exact low-risk change:

- edit `voice/speak.py` to expose Piper `SynthesisConfig` options and style presets;
- add `voice/tests/VOICE-TEST-SCRIPT.txt` with a fixed sample script;
- generate a small WAV comparison set with `--no-play --wav-out`;
- update `voice/README.md` with the selected style once Kevin chooses.

No packages, no service changes, no OpenClaw internals, no model downloads.

## Source notes

- Piper current local helper: `voice/speak.py`, `voice/say-last.py`, `voice/README.md`.
- Piper API inspection: local `.venv-voice` showed `SynthesisConfig(speaker_id, length_scale, noise_scale, noise_w_scale, normalize_audio, volume)`.
- Kokoro: https://github.com/hexgrad/kokoro
- Zonos: https://github.com/Zyphra/Zonos and `CONDITIONING_README.md`
- F5-TTS: https://github.com/SWivid/F5-TTS
- Orpheus: https://github.com/canopyai/Orpheus-TTS
- Dia: https://github.com/nari-labs/dia
- StyleTTS2: https://github.com/yl4579/StyleTTS2
