# local-dictation

Local speech-to-text dictation for Linux and macOS. Hardware acceleration is supported via NVIDIA CUDA (Linux) and Apple Silicon MPS (macOS). All processing runs on-device, and no data leaves the machine.

## Features

- **Push-to-talk**: Hold the Menu key to record, release to transcribe and type at cursor
- **Wake word**: Say "Computer" to activate hands-free dictation (custom-trained openWakeWord model)
- **Voice commands**: Spoken phrases trigger shell commands (e.g., "open my email" opens Gmail in Chrome)
- **Real-time preview**: See partial transcriptions as you speak (tiny.en model)
- **Final transcription**: High-accuracy output from large-v2 model on GPU

## Requirements

**Linux:**
- Ubuntu 22.04 with X11 (not Wayland)
- NVIDIA GPU with CUDA support
- System packages: `xdotool`, `xmodmap` (x11-xserver-utils), `portaudio19-dev`

**macOS:**
- macOS 12+ (Apple Silicon M1/M2/M3 recommended for best performance)
- Homebrew (`brew`) for installing system dependencies

**Common:**
- Python 3.9+

## Installation

```bash
# On Linux: Install system dependencies
sudo apt install python3.9 python3.9-venv portaudio19-dev xdotool x11-xserver-utils

# On macOS: Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Clone and run setup
git clone <this-repo> ~/local-dictation
cd ~/local-dictation
./setup.sh
```

The setup script will:
1. Create a Python 3.9 venv at `./stt-env`
2. Install PyTorch with CUDA 12.1 support
3. Install RealtimeSTT, faster-whisper, openWakeWord, and other dependencies
4. Build PyAudio (with a workaround if `portaudio19-dev` headers aren't installed via apt)
5. Patch RealtimeSTT's signal handling for thread safety
6. Copy the wake word model into openwakeword's model directory
7. Install config files to `~/.dictation-commands.yaml` and `~/.dictation-vocabulary.txt`
8. Set up autostart (GNOME `.desktop` on Linux or `LaunchAgent` on macOS)

## Usage

```bash
# Dictation mode (default) - transcribed speech is typed at cursor
./stt-env/bin/python -u dictation.py

# Agent mode - transcribed speech goes to LLM, response is spoken back
./stt-env/bin/python -u dictation.py --agent

# Or it starts automatically on login (GNOME autostart or macOS LaunchAgent)
```

### Push-to-talk

- **Linux:** Hold the **Menu key** (right of Right Alt). The text is typed via `xdotool`. The Menu key is automatically remapped to F24 to prevent context menus.
- **macOS:** Hold the **Right Option** key. The text is pasted via clipboard and `osascript` keystrokes. You will need to grant Terminal/Python *Accessibility* permissions in System Settings for the global hotkey to work.

### Wake word

Say **"Computer"**, then speak naturally. Recording stops after 1.2 seconds of silence. The wake word timeout is 30 seconds, so you can pause and continue within that window.

### Voice commands

Voice commands are defined in `~/.dictation-commands.yaml`. When a transcription matches a command pattern, the associated shell command runs instead of typing text.

Matching uses fuzzy keyword matching: all words in a pattern must appear in the transcription, but order and extra words don't matter. The longest (most specific) pattern wins.

**Commands are disabled by default.** Short, generic patterns like "open email" or "open code" trigger too easily during normal speech. Enable only commands you actually need, and use long, unambiguous patterns:

```yaml
# Bad - triggers too easily
- "open code"

# Good - requires intentional phrasing
- "hey computer open vs code"
```

See `config/dictation-commands.yaml` for examples.

## Configuration

### Voice commands: `~/.dictation-commands.yaml`

```yaml
commands:
  - name: "Open GitHub"
    patterns:
      - "hey computer open github"   # long patterns = fewer false triggers
      - "computer go to github"
    action: "google-chrome 'https://github.com' &"
    respond: "Opening GitHub"
```

Each command has:
- `patterns`: List of phrases that trigger the command. All words in a pattern must appear in the transcription (fuzzy keyword match). Add multiple variations for how you might say the same thing, including common Whisper mis-transcriptions.
- `action`: Any shell command. Runs in the background.
- `respond`: Optional message printed to the dictation log.

**Tip**: Longer patterns are much safer. Single-word or two-word patterns will fire during normal dictation. Aim for 4+ words that you would only say intentionally.

### Vocabulary corrections: `~/.dictation-vocabulary.txt`

Post-transcription word replacements for proper nouns that Whisper consistently mis-transcribes:

```
cubraid -> qBraid
cube raid -> qBraid
```

This is a fallback. The more effective approach is the `INITIAL_PROMPT` variable in `dictation.py`, which biases Whisper toward correct spellings at transcription time rather than fixing them afterward. Edit that string to add your own domain-specific terms.

### Custom wake word

The included `models/computer.onnx` was trained on a custom openWakeWord model. To train your own wake word (different phrase, or retrained on your voice), see [WAKE_WORD_TRAINING.md](WAKE_WORD_TRAINING.md).

### Agent mode

Run with `--agent` to enable voice-to-voice conversation with a local LLM. Your speech is transcribed, sent to the LLM, and the response is spoken back through your speakers.

**Local-first (no API keys needed):**
- **LLM:** [Ollama](https://ollama.ai) running locally (default model: `llama3.2`)
- **TTS:** [Piper TTS](https://github.com/rhasspy/piper) for offline speech synthesis

**Cloud backends (optional, via environment variables):**
- `GROQ_API_KEY` - Use Groq for LLM inference
- `OPENAI_API_KEY` - Use OpenAI for LLM inference
- `ELEVENLABS_API_KEY` - Use ElevenLabs for TTS
- `LLM_MODEL` - Override the default model name

Voice commands still work in agent mode and take priority over LLM routing.

Barge-in is supported: press the PTT key while the agent is speaking to immediately interrupt it and start a new recording.

## Architecture

```
dictation.py                Thin entry point
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
    llm.py                  LLM service (Ollama local / Groq / OpenAI)
    tts.py                  TTS service (Piper local / ElevenLabs)
tests/
  test_state.py             23 unit tests for the pure state machine

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
tests/test_state.py               State machine unit tests
models/computer.onnx              Custom "Computer" wake word model (openWakeWord ONNX)
config/dictation-commands.yaml    Example voice commands (copied to ~/.dictation-commands.yaml)
config/dictation-vocabulary.txt   Example vocabulary (copied to ~/.dictation-vocabulary.txt)
config/dictation.desktop          GNOME autostart template
setup.sh                          Installation script
requirements.txt                  Pinned Python dependencies
WAKE_WORD_TRAINING.md             Guide for training a custom wake word
training/computer_model.yaml      openWakeWord training config (update paths before use)
training/real-samples/            30 real voice recordings of "Computer" used for training
```

## Known Issues

- **Linux X11 only**: On Linux, push-to-talk key detection and text injection require X11. Wayland is not supported.
- **macOS Accessibility**: On macOS, the global hotkey requires Accessibility permissions for the Python process.
- **Menu key remap (Linux)**: `xmodmap` remaps don't survive logout. The script re-applies the remap on every startup.
- **ALSA/JACK warnings**: PyAudio's device enumeration triggers harmless ALSA/JACK error messages. These are suppressed by temporarily redirecting C-level stderr during initialization.
- **RealtimeSTT signal bug**: RealtimeSTT calls `signal.signal()` from worker threads, which raises `ValueError` in Python. The setup script patches this automatically.
- **Whisper hallucination on proper nouns**: Whisper may transcribe "qBraid" as various phonetic approximations. The `initial_prompt` + vocabulary file handle most cases, but unusual transcriptions may still occur.

## Tuning

Key parameters in `dictation.py` that affect behavior:

| Parameter | Default | Effect |
|---|---|---|
| `post_speech_silence_duration` | 1.2s | How long to wait after silence before finalizing. Lower = faster but may cut off pauses. |
| `wake_words_sensitivity` | 0.7 | Wake word detection threshold (0-1). Lower = more sensitive but more false triggers. |
| `wake_word_timeout` | 30.0s | Max recording duration after wake word activation. |
| `silero_sensitivity` | 0.4 | Voice activity detection sensitivity. |
| `MAIN_MODEL` | large-v2 | Final transcription model. Options: tiny, base, small, medium, large-v2, large-v3. |
| `REALTIME_MODEL` | tiny.en | Real-time preview model. Smaller = faster updates. |

## License

Private repository. Not for redistribution.
