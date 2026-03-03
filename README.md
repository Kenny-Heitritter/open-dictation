# open-dictation

Local speech-to-text dictation and voice assistant for Linux and macOS. All processing runs on-device -- no data leaves the machine.

## Features

- **Push-to-talk**: Hold Menu key (Linux) or Right Option (macOS) to record, release to transcribe and type at cursor
- **Wake word**: Say "Computer" to activate hands-free dictation (custom openWakeWord model)
- **Voice commands**: Spoken phrases trigger shell commands (defined in YAML config)
- **Real-time preview**: Partial transcriptions shown as you speak (tiny.en model)
- **Final transcription**: High-accuracy output from Whisper large-v2 on GPU
- **Agent mode**: Voice-to-voice conversation with a local LLM (Ollama) and TTS (Qwen3-TTS)
- **Voice cloning**: Clone your voice for the agent's TTS responses
- **Whisper fine-tuning**: Collect your speech data and fine-tune Whisper to recognize your vocabulary

## Requirements

**Linux:**
- Ubuntu 22.04+ with X11 (not Wayland)
- NVIDIA GPU with CUDA support
- System packages: `xdotool`, `xmodmap` (x11-xserver-utils), `portaudio19-dev`

**macOS:**
- macOS 12+ (Apple Silicon recommended)
- Homebrew for system dependencies

**Common:**
- Python 3.10+ (required for agent mode; dictation-only works with 3.9+)

## Installation

```bash
# Linux: install system dependencies
sudo apt install python3.10 python3.10-venv portaudio19-dev xdotool x11-xserver-utils

# Clone and set up
git clone https://github.com/Kenny-Heitritter/open-dictation.git
cd open-dictation
./setup.sh
```

The setup script creates a venv, installs PyTorch (with CUDA on Linux), RealtimeSTT, faster-whisper, openWakeWord, and other dependencies. It also patches RealtimeSTT's signal handling for thread safety and installs config files.

For agent mode, create a separate Python 3.10 venv with the additional dependencies:

```bash
python3.10 -m venv ~/dictation-env
~/dictation-env/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
~/dictation-env/bin/pip install -r requirements.txt
```

## Usage

### Dictation mode (default)

Transcribed speech is typed at the cursor position.

```bash
python -u dictation.py
```

### Agent mode

Speech goes to a local LLM, and the response is spoken back through your speakers.

```bash
python -u dictation.py --agent
```

Requires [Ollama](https://ollama.ai) running locally. Pull a model first:

```bash
ollama pull qwen3.5:9b    # or any model you prefer
```

### Voice cloning

Clone your voice so the agent responds in your voice:

```bash
python clone-voice.py                # full flow: record -> extract -> launch
python clone-voice.py --launch-only  # reuse existing voice clone
python clone-voice.py --skip-record  # re-extract from existing recording
```

Voice data is saved to `~/.dictation-voice/` (recording, transcript, cached clone prompt).

### Push-to-talk

- **Linux:** Hold the **Menu key** (right of Right Alt). Text is typed via `xdotool`. The Menu key is automatically remapped to F24 to prevent context menus.
- **macOS:** Hold the **Right Option** key. Text is pasted via clipboard and `osascript`. Requires Accessibility permissions in System Settings.

### Wake word

Say **"Computer"**, then speak naturally. Recording stops after 1.2 seconds of silence. The wake word timeout is 30 seconds.

### Voice commands

Defined in `~/.dictation-commands.yaml`. When a transcription matches a command pattern, the shell command runs instead of typing text. Matching is fuzzy keyword-based: all words in a pattern must appear, longest match wins.

**Commands are disabled by default.** Use long, unambiguous patterns to avoid false triggers:

```yaml
commands:
  - name: "Open GitHub"
    patterns:
      - "hey computer open github"
    action: "google-chrome 'https://github.com' &"
    respond: "Opening GitHub"
```

Voice commands take priority over LLM routing in agent mode.

## Whisper Fine-Tuning

The system automatically collects audio + transcript pairs during normal use (saved to `~/.dictation-training/`). You can review and correct these, then fine-tune a personal Whisper model that adapts to your voice and vocabulary.

### 1. Collect data

Just use the dictation system normally. Every transcription is silently saved.

### 2. Correct transcriptions

```bash
python correct-transcripts.py           # review uncorrected samples
python correct-transcripts.py --stats   # check how many samples you have
python correct-transcripts.py --all     # re-review previously corrected samples
```

Plays each audio clip, shows what Whisper heard, and lets you accept or type the correction. Aim for 50+ corrected samples (100+ recommended).

### 3. Fine-tune

```bash
pip install peft datasets              # one-time dependency install
python finetune-whisper.py             # LoRA fine-tune + convert to CTranslate2
python finetune-whisper.py --dry-run   # show stats without training
```

This trains a LoRA adapter on Whisper large-v2, merges it, and converts to CTranslate2 format at `~/.dictation-models/whisper-finetuned/`. The fine-tuned model is automatically detected on next startup.

To use a specific model path: `WHISPER_MODEL=/path/to/model python -u dictation.py`

## Configuration

### Vocabulary corrections: `~/.dictation-vocabulary.txt`

Post-transcription word replacements for proper nouns Whisper consistently gets wrong:

```
cubraid -> qBraid
cube raid -> qBraid
```

This is a fallback. The more effective approach is the `INITIAL_PROMPT` in `dictation/conversation.py`, which biases Whisper toward correct spellings at transcription time. The best approach is fine-tuning (see above).

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `llama3.2` | Ollama model name for agent mode |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `GROQ_API_KEY` | (none) | Use Groq cloud for LLM instead of Ollama |
| `OPENAI_API_KEY` | (none) | Use OpenAI for LLM instead of Ollama |
| `ELEVENLABS_API_KEY` | (none) | Use ElevenLabs for TTS instead of Qwen3-TTS |
| `TTS_CLONE_PROMPT` | (none) | Path to cached voice clone prompt (.pt file) |
| `TTS_VOICE_SAMPLE` | (none) | Path to voice reference WAV for cloning |
| `TTS_VOICE_SAMPLE_TRANSCRIPT` | (none) | Transcript of the voice reference recording |
| `WHISPER_MODEL` | (auto) | Path to a CTranslate2 Whisper model, or a model name like `large-v2` |
| `DICTATION_TRAINING_DIR` | `~/.dictation-training` | Where training audio/transcript pairs are saved |

### Custom wake word

The included `models/computer.onnx` was trained as a custom openWakeWord model. To train your own, see [WAKE_WORD_TRAINING.md](WAKE_WORD_TRAINING.md).

## Architecture

```
dictation.py                Thin entry point (--agent flag)
clone-voice.py              Voice cloning setup (record -> extract -> launch)
correct-transcripts.py      Review and correct transcriptions for fine-tuning
finetune-whisper.py         LoRA fine-tune Whisper + convert to CTranslate2

dictation/
  types.py                  Immutable state, events, actions (dataclasses)
  state.py                  Pure state machine: (State, Event) -> (State, Actions)
  conversation.py           Event loop: queue events, run state machine, dispatch I/O
  tracer.py                 Per-turn latency instrumentation (saves to ~/.dictation-traces/)
  agent.py                  Agent pipeline: LLM -> TTS -> speaker (streaming)
  services/
    text_output.py          Platform-aware text injection (xdotool / osascript)
    hotkey.py               Push-to-talk listener (Menu key / Right Option)
    commands.py             Voice commands + vocabulary corrections
    llm.py                  LLM service (Ollama native API / Groq / OpenAI)
    tts.py                  TTS service (Qwen3-TTS with voice cloning / ElevenLabs)
    training_data.py        Auto-saves audio + transcript pairs for fine-tuning

tests/
  test_state.py             Unit tests for the pure state machine

RealtimeSTT                 Manages VAD, streaming audio, wake words, Whisper
  +-- faster-whisper        large-v2 on GPU for final, tiny.en for realtime
  +-- openWakeWord          custom computer.onnx model
  +-- Silero VAD + WebRTC VAD
  +-- PyAudio               mic input
```

## Files

```
dictation.py                      Entry point
dictation/                        Core package (state machine + services)
clone-voice.py                    Voice cloning setup script
correct-transcripts.py            Transcription correction tool
finetune-whisper.py               Whisper fine-tuning script
tests/test_state.py               State machine unit tests
models/computer.onnx              Custom "Computer" wake word model
config/dictation-commands.yaml    Example voice commands (copied to ~/.dictation-commands.yaml)
config/dictation-vocabulary.txt   Example vocabulary (copied to ~/.dictation-vocabulary.txt)
config/dictation.desktop          GNOME autostart template
setup.sh                          Installation script
requirements.txt                  Python dependencies
WAKE_WORD_TRAINING.md             Guide for training a custom wake word
training/computer_model.yaml      openWakeWord training config
training/real-samples/            Voice recordings of "Computer" used for wake word training
```

## Tuning

Key parameters in `dictation/conversation.py`:

| Parameter | Default | Effect |
|---|---|---|
| `post_speech_silence_duration` | 1.2s | How long to wait after silence before finalizing. Lower = faster but may cut off pauses. |
| `wake_words_sensitivity` | 0.7 | Wake word detection threshold (0-1). Lower = more sensitive but more false triggers. |
| `wake_word_timeout` | 30.0s | Max recording duration after wake word activation. |
| `silero_sensitivity` | 0.4 | Voice activity detection sensitivity. |
| `MAIN_MODEL` | large-v2 | Final transcription model. Auto-detected fine-tuned model takes priority. |
| `REALTIME_MODEL` | tiny.en | Real-time preview model. Smaller = faster updates. |

## Known Issues

- **Linux X11 only**: Push-to-talk and text injection require X11 on Linux. Wayland is not supported.
- **macOS Accessibility**: The global hotkey requires Accessibility permissions for the Python process.
- **Menu key remap (Linux)**: `xmodmap` remaps don't survive logout. Re-applied on every startup.
- **ALSA/JACK warnings**: Harmless ALSA warnings appear on stderr during startup. These are cosmetic.
- **RealtimeSTT signal bug**: RealtimeSTT calls `signal.signal()` from worker threads. The setup script patches this.
- **Whisper proper nouns**: Whisper may mis-transcribe domain-specific words. Use `INITIAL_PROMPT`, vocabulary corrections, or fine-tuning to address this.

## License

MIT License. See [LICENSE](LICENSE) for details.
