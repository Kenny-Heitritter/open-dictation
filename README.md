# local-dictation

Local speech-to-text dictation for Linux with GPU acceleration. All processing runs on-device via an NVIDIA GPU. No data leaves the machine.

## Features

- **Push-to-talk**: Hold the Menu key to record, release to transcribe and type at cursor
- **Wake word**: Say "Computer" to activate hands-free dictation (custom-trained openWakeWord model)
- **Voice commands**: Spoken phrases trigger shell commands (e.g., "open my email" opens Gmail in Chrome)
- **Real-time preview**: See partial transcriptions as you speak (tiny.en model)
- **Final transcription**: High-accuracy output from large-v2 model on GPU

## Requirements

- Ubuntu 22.04 with X11 (not Wayland)
- NVIDIA GPU with CUDA support (tested on RTX 3090)
- Python 3.9
- PulseAudio
- System packages: `xdotool`, `xmodmap` (x11-xserver-utils), `portaudio19-dev`

## Installation

```bash
# Install system dependencies (if you have sudo)
sudo apt install python3.9 python3.9-venv portaudio19-dev xdotool x11-xserver-utils

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
8. Set up GNOME autostart

## Usage

```bash
# Run manually
./stt-env/bin/python -u dictation.py

# Or it starts automatically on GNOME login via ~/.config/autostart/dictation.desktop
```

### Push-to-talk

Hold the **Menu key** (right of Right Alt on most keyboards) and speak. Release to stop recording. The transcribed text is typed at the current cursor position via `xdotool type`.

The Menu key is remapped to F24 on startup to prevent context menus from appearing. This remap persists until logout or manual `xmodmap` reset.

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

## Architecture

```
dictation.py          Main script (single process)
  |
  +-- RealtimeSTT     Manages VAD, streaming audio, wake words, Whisper
  |     +-- faster-whisper (large-v2 on GPU for final, tiny.en for realtime)
  |     +-- openWakeWord (custom computer.onnx model)
  |     +-- Silero VAD + WebRTC VAD
  |     +-- PyAudio (PulseAudio -> mic input)
  |
  +-- pynput           Keyboard listener for Menu key push-to-talk
  +-- xdotool          Text injection at cursor position
  +-- xmodmap          Menu key -> F24 remap
```

## Files

```
dictation.py                      Main script
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

- **X11 only**: Push-to-talk key detection (pynput) and text injection (xdotool) require X11. Wayland is not supported.
- **Menu key remap**: `xmodmap` remaps don't survive logout. The script re-applies the remap on every startup.
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
