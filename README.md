# open-dictation

Local speech-to-text dictation for Linux and macOS. Speak and it types at your cursor. All processing runs on-device -- no data leaves your machine.

## Quick Start

### macOS

```bash
git clone https://github.com/Kenny-Heitritter/open-dictation.git
cd open-dictation
./setup.sh
```

Then grant Accessibility permission so the app can type for you:

**System Settings > Privacy & Security > Accessibility** -- add your terminal app (Terminal, iTerm2, etc.)

Run it:

```bash
./dictation.sh run            # Interactive (Ctrl+C to stop)
./dictation.sh start          # Background daemon
./dictation.sh stop           # Stop daemon
./dictation.sh status         # Check if running
./dictation.sh log            # Tail the log
```

**Push-to-talk:** Hold **Right Option** (right of Right Cmd), speak, release. Text appears at your cursor.

**Wake word:** Say **"Computer"**, then speak naturally -- no key needed.

> Apple Silicon Macs use the Metal GPU via [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) for fast transcription. The first run downloads the model (~1.5 GB) which takes a minute; after that, transcriptions are under 1 second.

### Linux

```bash
sudo apt install python3.10 python3.10-venv portaudio19-dev xdotool x11-xserver-utils
git clone https://github.com/Kenny-Heitritter/open-dictation.git
cd open-dictation
./setup.sh
```

Run it:

```bash
./dictation.sh run            # Interactive (Ctrl+C to stop)
./dictation.sh start          # Background daemon (systemd)
```

**Push-to-talk:** Hold the **Menu key** (right of Right Alt), speak, release.

**Wake word:** Say **"Computer"**, then speak naturally -- no key needed.

Requires X11 and an NVIDIA GPU with CUDA.

---

## Features

- **Push-to-talk**: Hold Right Option (macOS) or Menu key (Linux), speak, release -- text appears at your cursor
- **Wake word**: Say "Computer", then speak naturally -- no key needed
- **Real-time preview**: See partial transcriptions as you speak
- **Voice commands**: Trigger shell commands with your voice (configurable)

## Agent Mode

Talk to a local LLM and hear it respond. Requires [Ollama](https://ollama.ai).

```bash
ollama pull qwen3.5:9b
./dictation.sh run --agent
```

### Voice cloning

Clone any voice so the agent responds with it:

```bash
python clone-voice.py                # full flow: record -> extract -> launch
python clone-voice.py --launch-only  # reuse existing voice clone
python clone-voice.py --skip-record  # re-extract from existing recording
```

Voice data is saved to `~/.dictation-voice/`.

## Voice Commands

Defined in `~/.dictation-commands.yaml`. Disabled by default. When a transcription matches a command pattern, the shell command runs instead of typing text.

```yaml
commands:
  - name: "Open GitHub"
    patterns:
      - "hey computer open github"
    action: "open 'https://github.com'"
    respond: "Opening GitHub"
```

Voice commands take priority over LLM routing in agent mode.

## Whisper Fine-Tuning

The system silently saves audio + transcript pairs during normal use. You can correct these and fine-tune Whisper to learn your voice and vocabulary.

```bash
# 1. Use dictation normally -- data is collected automatically

# 2. Review and correct transcriptions (aim for 50+ samples)
python correct-transcripts.py

# 3. Fine-tune
pip install peft datasets
python finetune-whisper.py
```

The fine-tuned model is saved to `~/.dictation-models/whisper-finetuned/` and automatically used on next startup.

## Configuration

### Vocabulary corrections: `~/.dictation-vocabulary.txt`

Post-transcription word replacements for proper nouns Whisper gets wrong:

```
cubraid -> qBraid
cube raid -> qBraid
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | (auto) | Whisper model name (`large-v2`, `small.en`) or MLX repo (`mlx-community/whisper-large-v3-turbo`) |
| `LLM_MODEL` | `llama3.2` | Ollama model name for agent mode |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `GROQ_API_KEY` | (none) | Use Groq cloud for LLM instead of Ollama |
| `OPENAI_API_KEY` | (none) | Use OpenAI for LLM instead of Ollama |
| `ELEVENLABS_API_KEY` | (none) | Use ElevenLabs for TTS instead of Qwen3-TTS |
| `TTS_CLONE_PROMPT` | (none) | Path to cached voice clone prompt (.pt file) |
| `TTS_VOICE_SAMPLE` | (none) | Path to voice reference WAV for cloning |
| `TTS_VOICE_SAMPLE_TRANSCRIPT` | (none) | Transcript of the voice reference recording |
| `DICTATION_TRAINING_DIR` | `~/.dictation-training` | Where training audio/transcript pairs are saved |

### Custom wake word

The included `models/computer.onnx` was trained as a custom openWakeWord model. To train your own, see [WAKE_WORD_TRAINING.md](WAKE_WORD_TRAINING.md).

## Tuning

Key parameters in `dictation/conversation.py`:

| Parameter | Default | Effect |
|---|---|---|
| `post_speech_silence_duration` | 1.2s | Silence before finalizing. Lower = faster but may cut off pauses. |
| `wake_words_sensitivity` | 0.7 | Wake word threshold (0-1). Lower = more sensitive, more false triggers. |
| `wake_word_timeout` | 30.0s | Max recording time after wake word. |
| `silero_sensitivity` | 0.4 | Voice activity detection sensitivity. |
| `MAIN_MODEL` | large-v2 | Final transcription model. On Apple Silicon, mapped to `whisper-large-v3-turbo` via MLX. |
| `REALTIME_MODEL` | tiny.en | Real-time preview model. |

## Running as a Daemon

Dictation can run in the background and auto-start on login.

```bash
./dictation.sh enable         # Enable auto-start on login
./dictation.sh disable        # Disable auto-start
```

On macOS this installs a LaunchAgent; on Linux it creates a systemd user service. `setup.sh` enables autostart automatically.

Logs: `~/Library/Logs/dictation/dictation.log` (macOS) or `journalctl --user -u dictation -f` (Linux).

## Architecture

```
dictation.py                Thin entry point (--agent flag)
dictation.sh                Daemon management (start/stop/restart/status/log/run)
clone-voice.py              Voice cloning setup (record -> extract -> launch)
correct-transcripts.py      Review and correct transcriptions for fine-tuning
finetune-whisper.py         LoRA fine-tune Whisper + convert to CTranslate2

dictation/
  types.py                  Immutable state, events, actions (dataclasses)
  state.py                  Pure state machine: (State, Event) -> (State, Actions)
  conversation.py           Event loop: queue events, run state machine, dispatch I/O
  tracer.py                 Per-turn latency instrumentation
  agent.py                  Agent pipeline: LLM -> TTS -> speaker (streaming)
  services/
    text_output.py          Platform-aware text injection (xdotool / CGEvent Cmd+V)
    hotkey.py               Push-to-talk listener (Menu key / Right Option)
    commands.py             Voice commands + vocabulary corrections
    llm.py                  LLM service (Ollama / Groq / OpenAI)
    tts.py                  TTS service (Qwen3-TTS / ElevenLabs)
    training_data.py        Auto-saves audio + transcript pairs for fine-tuning
    mlx_transcribe.py       macOS Metal GPU transcription via mlx-whisper
```

On Apple Silicon Macs, final transcription runs on the Metal GPU via [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper). RealtimeSTT handles audio capture, VAD, wake word detection, and realtime preview. On Linux, faster-whisper runs on CUDA.

## Known Issues

- **macOS Accessibility**: Push-to-talk and text pasting require Accessibility permission. The app warns at startup if missing.
- **Linux X11 only**: Push-to-talk and text injection require X11. Wayland is not supported.
- **ALSA/JACK warnings**: Harmless warnings on stderr during Linux startup. Cosmetic only.
- **Whisper proper nouns**: Use `INITIAL_PROMPT`, vocabulary corrections, or fine-tuning.

## License

MIT License. See [LICENSE](LICENSE) for details.
