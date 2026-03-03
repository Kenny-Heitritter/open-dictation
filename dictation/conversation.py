"""Main event loop for the dictation system.

Follows the architecture from shuo:
    while running:
        event = queue.get()
        state, actions = process_event(state, event)
        for action in actions: dispatch(action)

All I/O happens in dispatch(); the state machine is pure.
"""

from __future__ import annotations

import os
import sys
import platform
import threading
import time
import uuid
from queue import Queue, Empty
from typing import Optional

from dictation.types import (
    AppState, Phase,
    WakeWordDetectedEvent,
    PTTPressedEvent,
    PTTReleasedEvent,
    RecordingStartedEvent,
    RecordingStoppedEvent,
    RealtimeTranscriptionEvent,
    TranscriptionReadyEvent,
    AgentResponseDoneEvent,
    ShutdownEvent,
    PasteTextAction,
    RunCommandAction,
    SendEscapeKeyAction,
    StartRecordingAction,
    StopRecordingAction,
    StartAgentTurnAction,
    CancelAgentTurnAction,
    LogAction,
)
from dictation.state import process_event
from dictation.tracer import Tracer
from dictation.services.text_output import paste_text
from dictation.services.hotkey import (
    unbind_menu_key, send_escape_key, start_hotkey_listener,
)
from dictation.services.commands import (
    load_vocabulary, load_commands, apply_vocabulary,
    check_command, run_command, VOCABULARY, COMMANDS,
)
from dictation.services.training_data import save_sample

IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

# ── Configuration ────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)

WAKE_WORD_MODEL_PATHS = [
    os.path.join(REPO_DIR, "models", "computer.onnx"),
    # Fallback: find inside whichever venv is running this process
    os.path.join(
        os.path.dirname(os.path.dirname(sys.executable)),
        "lib", f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages", "openwakeword", "resources", "models", "computer.onnx",
    ),
]

def _resolve_main_model() -> str:
    """Use fine-tuned model if available, else stock large-v2."""
    override = os.environ.get("WHISPER_MODEL", "")
    if override:
        return override
    finetuned = os.path.expanduser("~/.dictation-models/whisper-finetuned")
    if os.path.isdir(finetuned) and os.path.exists(
        os.path.join(finetuned, "model.bin")
    ):
        return finetuned
    return "large-v2"


MAIN_MODEL = _resolve_main_model()
REALTIME_MODEL = "tiny.en"

INITIAL_PROMPT = (
    "qBraid, qBraid email, qBraid calendar, GitHub, VS Code, "
    "Slack, Discord, Spotify, NVIDIA, CUDA, Ubuntu, "
    ".cubitrc, OpenCode"
)


# ── Audio noise suppression (Linux only) ─────────────────────────────

def _suppress_alsa_noise():
    """Suppress ALSA/JACK C-level warnings during PyAudio init.

    NOTE: We only set environment variables here.  The old ctypes-based
    snd_lib_error_set_handler approach caused segfaults because the
    callback pointer could be garbage-collected before the forked
    multiprocessing workers (used by RealtimeSTT) tried to use it.
    The ALSA warnings are harmless — just cosmetic noise on stderr.
    """
    if not IS_LINUX:
        return
    os.environ["JACK_NO_START_SERVER"] = "1"


# ── Main conversation loop ───────────────────────────────────────────

def run(agent_mode: bool = False) -> None:
    """Run the dictation event loop.

    Parameters
    ----------
    agent_mode : bool
        If True, transcriptions are sent to the LLM agent instead of
        being pasted as text.  Requires Phase 2 services.
    """
    from RealtimeSTT import AudioToTextRecorder

    _suppress_alsa_noise()

    print("=" * 50)
    print("Dictation (RealtimeSTT + faster-whisper)")
    print("=" * 50)

    # Load config
    vocab = load_vocabulary()
    if vocab:
        print(f"Loaded {len(vocab)} vocabulary mappings")

    cmds = load_commands()
    if cmds:
        print(f"Loaded {len(cmds)} voice commands")
        for cmd in cmds:
            print(f"  - {cmd['name']}")

    # Event queue - all events flow through here
    event_queue: Queue = Queue()

    # Tracer for latency recording
    tracer = Tracer()
    session_id = uuid.uuid4().hex[:12]

    # Find wake word model
    wake_word_model = None
    for p in WAKE_WORD_MODEL_PATHS:
        if os.path.exists(p):
            wake_word_model = p
            break

    # ── Build RealtimeSTT config ─────────────────────────────────
    # macOS Apple Silicon: final transcription uses mlx-whisper on Metal GPU.
    #   RealtimeSTT still loads a tiny CPU model for its subprocess (required
    #   by its architecture) but we monkey-patch transcribe() to use MLX.
    # Linux with NVIDIA GPU: cuda + float16 via faster-whisper (fastest)
    # macOS Intel / other: cpu + int8 via faster-whisper
    _use_mlx = IS_MAC and platform.machine() == "arm64"

    if IS_LINUX:
        _device = "cuda"
        _compute = "float16"
        _main_model = MAIN_MODEL
    elif _use_mlx:
        # RealtimeSTT subprocess gets tiny model (unused -- MLX handles final)
        _device = "cpu"
        _compute = "int8"
        _main_model = "tiny.en"
    else:
        _device = "cpu"
        _compute = "int8"
        _main_model = MAIN_MODEL

    config = dict(
        model=_main_model,
        language="en",
        compute_type=_compute,
        device=_device,
        gpu_device_index=0,
        beam_size=1,
        initial_prompt=INITIAL_PROMPT,

        enable_realtime_transcription=True,
        realtime_model_type=REALTIME_MODEL,
        use_main_model_for_realtime=False,
        realtime_processing_pause=0.1,
        beam_size_realtime=1,

        silero_sensitivity=0.4,
        webrtc_sensitivity=3,
        post_speech_silence_duration=1.2,
        min_length_of_recording=0.3,
        min_gap_between_recordings=0,
        pre_recording_buffer_duration=0.5,
        early_transcription_on_silence=300,

        ensure_sentence_starting_uppercase=True,
        ensure_sentence_ends_with_period=False,

        # Wire callbacks into the event queue
        on_recording_start=lambda: event_queue.put(RecordingStartedEvent()),
        on_recording_stop=lambda: event_queue.put(RecordingStoppedEvent()),
        on_realtime_transcription_update=lambda t: (
            event_queue.put(RealtimeTranscriptionEvent(text=t))
            if t.strip() else None
        ),

        spinner=False,
        print_transcription_time=True,
        no_log_file=True,
    )

    # Wake word config
    if wake_word_model and os.path.exists(wake_word_model):
        config.update(
            wakeword_backend="oww",
            openwakeword_model_paths=wake_word_model,
            openwakeword_inference_framework="onnx",
            wake_words="computer",
            wake_words_sensitivity=0.7,
            wake_word_activation_delay=0.0,
            wake_word_timeout=30.0,
            on_wakeword_detected=lambda: event_queue.put(
                WakeWordDetectedEvent()
            ),
        )
        print(f"Wake word: 'Computer' (openWakeWord)")
    else:
        print("Wake word: disabled (model not found)")

    # macOS: check Accessibility permissions (needed for PTT and text paste)
    if IS_MAC:
        try:
            from ApplicationServices import AXIsProcessTrusted
            if not AXIsProcessTrusted():
                sys.stdout.write(
                    "\n*** Accessibility permission required ***\n"
                    "Push-to-talk and text pasting need Accessibility access.\n"
                    "Grant it in: System Settings > Privacy & Security > Accessibility\n"
                    "Add your terminal app (e.g. Terminal, iTerm2), then restart.\n\n"
                )
                sys.stdout.flush()
        except Exception:
            pass

    ptt_key = "Right Option" if IS_MAC else "Menu key"
    model_label = MAIN_MODEL
    if "/" in MAIN_MODEL or MAIN_MODEL.startswith(os.path.expanduser("~")):
        model_label = f"{MAIN_MODEL} (fine-tuned)"
    if _use_mlx:
        from dictation.services.mlx_transcribe import _resolve_mlx_model
        mlx_model_id = _resolve_mlx_model(MAIN_MODEL)
        print(f"Main model: {mlx_model_id.split('/')[-1]} (Metal GPU via mlx-whisper)")
    else:
        print(f"Main model: {model_label} ({_device} {_compute})")
    print(f"Realtime: {REALTIME_MODEL}")
    print(f"Push-to-talk: {ptt_key} (hold to record)")
    if agent_mode:
        print("Mode: AGENT (transcriptions -> LLM -> TTS)")
    else:
        print("Mode: DICTATION (transcriptions -> typed at cursor)")
    print("Ctrl+C to exit.")
    print("=" * 50)
    print("Loading models...", flush=True)

    # Create recorder (ALSA warnings are suppressed at the C level
    # by _suppress_alsa_noise above; no need to redirect stderr)
    recorder = AudioToTextRecorder(**config)

    # macOS Apple Silicon: patch recorder to use mlx-whisper on Metal GPU
    if _use_mlx:
        from dictation.services.mlx_transcribe import patch_recorder_for_mlx
        patch_recorder_for_mlx(recorder, MAIN_MODEL, INITIAL_PROMPT)

    # Set up PTT
    unbind_menu_key()
    hotkey_listener = start_hotkey_listener(
        on_press=lambda: event_queue.put(PTTPressedEvent()),
        on_release=lambda: event_queue.put(PTTReleasedEvent()),
    )

    # ── Ready ────────────────────────────────────────────────────
    sys.stdout.write("\n" + "=" * 50 + "\n")
    sys.stdout.write("READY\n")
    sys.stdout.write("  Wake word: say 'Computer' then speak\n")
    sys.stdout.write(f"  Push-to-talk: hold {ptt_key} and speak\n")
    if cmds:
        sys.stdout.write(f"  Voice commands: {len(cmds)} loaded\n")
    sys.stdout.write("=" * 50 + "\n\n")
    sys.stdout.flush()

    # ── Agent setup (Phase 2) ───────────────────────────────────
    agent = None
    if agent_mode:
        from dictation.agent import Agent
        agent = Agent(
            on_done=lambda: event_queue.put(AgentResponseDoneEvent()),
            tracer=tracer,
        )
        # Preload TTS model in background so first response is fast
        agent.preload_tts()

    # ── Event loop ───────────────────────────────────────────────
    state = AppState()
    running = True

    # Background thread: poll recorder.text() and push transcriptions
    def _recorder_loop():
        while running:
            try:
                sys.stdout.write("[Listening...]\n")
                sys.stdout.flush()
                text = recorder.text()
                if text and text.strip():
                    # Save audio + transcript for future fine-tuning
                    try:
                        audio_b64 = getattr(
                            recorder, "last_transcription_bytes_b64", ""
                        )
                        if audio_b64:
                            save_sample(audio_b64, text.strip())
                    except Exception:
                        pass  # never block dictation for training data

                    event_queue.put(
                        TranscriptionReadyEvent(text=text.strip())
                    )
                else:
                    event_queue.put(
                        TranscriptionReadyEvent(text="")
                    )
            except Exception as e:
                sys.stdout.write(f"Recorder error: {e}\n")
                sys.stdout.flush()

    recorder_thread = threading.Thread(target=_recorder_loop, daemon=True)
    recorder_thread.start()

    try:
        while running:
            try:
                event = event_queue.get(timeout=0.1)
            except Empty:
                continue

            if isinstance(event, ShutdownEvent):
                running = False
                break

            # Pure state transition
            new_state, actions = process_event(
                state, event,
                check_command=check_command,
                apply_vocabulary=apply_vocabulary,
                agent_mode=agent_mode,
            )
            state = new_state

            # Dispatch side-effects
            for action in actions:
                _dispatch(action, recorder, tracer, agent=agent)

    except KeyboardInterrupt:
        sys.stdout.write("\nShutting down...\n")
    finally:
        running = False
        hotkey_listener.stop()
        if agent:
            agent.cleanup()
        recorder.shutdown()

        # Save trace
        trace_path = tracer.save(session_id)
        if trace_path:
            sys.stdout.write(f"Trace saved to {trace_path}\n")
        summary = tracer.summary()
        if summary and summary != "No turns recorded.":
            sys.stdout.write(f"\n{summary}\n")
        sys.stdout.flush()


def _dispatch(action, recorder, tracer: Tracer, agent=None) -> None:
    """Execute a single side-effect action."""

    if isinstance(action, LogAction):
        sys.stdout.write(action.message + "\n")
        sys.stdout.flush()

    elif isinstance(action, PasteTextAction):
        paste_text(action.text)

    elif isinstance(action, RunCommandAction):
        run_command(action.command)

    elif isinstance(action, SendEscapeKeyAction):
        send_escape_key()

    elif isinstance(action, StartRecordingAction):
        recorder.start()

    elif isinstance(action, StopRecordingAction):
        recorder.stop()

    elif isinstance(action, StartAgentTurnAction):
        if agent:
            agent.start_turn(action.transcript)
        else:
            sys.stdout.write(
                f"[Agent not available: {action.transcript!r}]\n"
            )
            sys.stdout.flush()

    elif isinstance(action, CancelAgentTurnAction):
        if agent:
            agent.cancel_turn()
