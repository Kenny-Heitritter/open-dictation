#!/usr/bin/env python3
"""
Local speech-to-text dictation using RealtimeSTT.
All processing runs locally on GPU. No data leaves the machine.

Three modes:
  1. Wake word: Say "Computer", then speak naturally
  2. Push-to-talk: Hold Menu key, speak, release
  3. Voice commands: Say things like "open my qbraid email"
     Commands are defined in ~/.dictation-commands.yaml
"""

import subprocess
import os
import sys
import re
import time
import ctypes
import threading
import yaml

# ── Suppress ALSA/JACK C-level noise during PyAudio init ─────────────
os.environ['JACK_NO_START_SERVER'] = '1'
try:
    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
        None, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
    def _err_handler(filename, line, function, err, fmt):
        pass
    _c_err_handler = ERROR_HANDLER_FUNC(_err_handler)
    ctypes.cdll.LoadLibrary('libasound.so.2').snd_lib_error_set_handler(_c_err_handler)
except Exception:
    pass


def _suppress_stderr():
    """Redirect C-level stderr to /dev/null. Returns fd to restore later."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    return old_stderr


def _restore_stderr(old_stderr):
    """Restore C-level stderr."""
    os.dup2(old_stderr, 2)
    os.close(old_stderr)


# ── Configuration ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

VOCABULARY_FILE = os.path.expanduser("~/.dictation-vocabulary.txt")
COMMANDS_FILE = os.path.expanduser("~/.dictation-commands.yaml")

# Wake word model - looks in repo models/ dir first, then openwakeword package dir
WAKE_WORD_MODEL_PATHS = [
    os.path.join(SCRIPT_DIR, "models", "computer.onnx"),
    os.path.expanduser("~/stt-env/lib/python3.9/site-packages/openwakeword/resources/models/computer.onnx"),
]
WAKE_WORD_MODEL = None
for p in WAKE_WORD_MODEL_PATHS:
    if os.path.exists(p):
        WAKE_WORD_MODEL = p
        break

# Whisper models
MAIN_MODEL = "large-v2"
REALTIME_MODEL = "tiny.en"

# Initial prompt - biases Whisper toward correct spellings of proper nouns.
# Edit this string to add your own domain-specific words.
INITIAL_PROMPT = (
    "qBraid, qBraid email, qBraid calendar, GitHub, VS Code, "
    "Slack, Discord, Spotify, NVIDIA, CUDA, Ubuntu, "
    ".cubitrc, OpenCode"
)


# ── Vocabulary ─────────────────────────────────────────────────────────
VOCABULARY = {}


def load_vocabulary():
    """Load word replacement mappings from ~/.dictation-vocabulary.txt"""
    global VOCABULARY
    VOCABULARY = {}
    if not os.path.exists(VOCABULARY_FILE):
        return
    try:
        with open(VOCABULARY_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '->' in line:
                    wrong, correct = line.split('->', 1)
                    VOCABULARY[wrong.strip()] = correct.strip()
    except Exception as e:
        print(f"Warning: Could not load vocabulary: {e}")


def apply_vocabulary(text):
    """Apply word replacements to transcribed text."""
    for wrong, correct in VOCABULARY.items():
        pattern = r'\b' + re.escape(wrong) + r'\b'
        text = re.sub(pattern, correct, text, flags=re.IGNORECASE)
    return text


# ── Voice Commands ─────────────────────────────────────────────────────
COMMANDS = []


def load_commands():
    """Load voice commands from ~/.dictation-commands.yaml"""
    global COMMANDS
    COMMANDS = []
    if not os.path.exists(COMMANDS_FILE):
        return
    try:
        with open(COMMANDS_FILE, 'r') as f:
            data = yaml.safe_load(f)
        if data and 'commands' in data:
            for cmd in data['commands']:
                patterns = [_normalize(p) for p in cmd.get('patterns', [])]
                COMMANDS.append({
                    'name': cmd.get('name', 'unnamed'),
                    'patterns': patterns,
                    'action': cmd.get('action', ''),
                    'respond': cmd.get('respond', ''),
                })
    except Exception as e:
        sys.stdout.write(f"Warning: Could not load commands: {e}\n")
        sys.stdout.flush()


def _normalize(text):
    """Normalize text for command matching: lowercase, strip punctuation."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def check_command(text):
    """Check if transcribed text matches a voice command.

    Uses fuzzy keyword matching: all keywords in a pattern must appear
    in the transcribed text. Longest (most specific) pattern wins.
    """
    normalized = _normalize(text)
    words = set(normalized.split())

    best_match = None
    best_score = 0

    for cmd in COMMANDS:
        for pattern in cmd['patterns']:
            pattern_words = pattern.split()
            matched = sum(1 for pw in pattern_words if any(pw in w for w in words))
            if matched == len(pattern_words):
                score = len(pattern_words)
                if score > best_score:
                    best_score = score
                    best_match = cmd

    return best_match


def run_command(cmd):
    """Execute a voice command's shell action."""
    name = cmd['name']
    action = cmd['action']
    respond = cmd['respond']

    sys.stdout.write(f"  [CMD] {name}\n")
    if respond:
        sys.stdout.write(f"  {respond}\n")
    sys.stdout.flush()

    try:
        subprocess.Popen(
            action,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:
        sys.stdout.write(f"  Command failed: {e}\n")
        sys.stdout.flush()


# ── Text output ────────────────────────────────────────────────────────
def paste_text(text):
    """Insert text at cursor position using xdotool type.

    Falls back to clipboard paste if xdotool type fails.
    Uses xdotool type instead of clipboard to avoid pasting images
    in Electron apps.
    """
    try:
        subprocess.run(
            ['xdotool', 'type', '--clearmodifiers', '--delay', '0', '--', text],
            check=True, timeout=5
        )
    except Exception as e:
        sys.stdout.write(f"xdotool type failed ({e}), trying clipboard\n")
        sys.stdout.flush()
        try:
            subprocess.run(
                ['xclip', '-selection', 'clipboard'],
                input=text.encode('utf-8'),
                check=True, timeout=1
            )
            time.sleep(0.05)
            subprocess.run(['xdotool', 'key', 'ctrl+v'], check=True, timeout=2)
        except Exception as e2:
            sys.stdout.write(f"Clipboard paste also failed: {e2}\n")
            sys.stdout.flush()


# ── Push-to-talk (Menu key) ───────────────────────────────────────────
MENU_KEYCODE = 135  # X11 hardware keycode for the Menu key


def unbind_menu_key():
    """Remap Menu key to F24 so it doesn't trigger context menus.

    F24 is a valid keysym that no application or window manager binds to.
    The key still generates events that pynput can detect.
    """
    try:
        subprocess.run(
            ['xmodmap', '-e', 'keycode 135 = F24'],
            check=True, timeout=2,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        pass


def start_hotkey_listener(recorder):
    """Start a background keyboard listener for Menu key push-to-talk."""
    from pynput import keyboard
    from pynput.keyboard import Key, KeyCode

    menu_held = False

    def _is_menu_key(key):
        """Match the physical Menu key before and after xmodmap remap."""
        if key == Key.menu:
            return True
        # After remap to F24, pynput sees keysym 0xFFD5 = 65493
        if hasattr(key, 'vk') and key.vk == 0xFFD5:
            return True
        return False

    def on_press(key):
        nonlocal menu_held
        if _is_menu_key(key) and not menu_held:
            menu_held = True
            subprocess.Popen(['xdotool', 'key', 'Escape'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            sys.stdout.write("\n[PTT] Recording...\n")
            sys.stdout.flush()
            recorder.start()

    def on_release(key):
        nonlocal menu_held
        if _is_menu_key(key) and menu_held:
            menu_held = False
            recorder.stop()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()
    return listener


# ── Main ───────────────────────────────────────────────────────────────
def main():
    from RealtimeSTT import AudioToTextRecorder

    print("=" * 50)
    print("Dictation (RealtimeSTT + faster-whisper)")
    print("=" * 50)

    load_vocabulary()
    if VOCABULARY:
        print(f"Loaded {len(VOCABULARY)} vocabulary mappings")

    load_commands()
    if COMMANDS:
        print(f"Loaded {len(COMMANDS)} voice commands")
        for cmd in COMMANDS:
            print(f"  - {cmd['name']}")

    config = dict(
        # Whisper model config - GPU accelerated
        model=MAIN_MODEL,
        language="en",
        compute_type="float16",
        device="cuda",
        gpu_device_index=0,
        beam_size=1,
        initial_prompt=INITIAL_PROMPT,

        # Realtime preview (tiny model shows text as you speak)
        enable_realtime_transcription=True,
        realtime_model_type=REALTIME_MODEL,
        use_main_model_for_realtime=False,
        realtime_processing_pause=0.1,
        beam_size_realtime=1,

        # Voice activity detection
        silero_sensitivity=0.4,
        webrtc_sensitivity=3,
        post_speech_silence_duration=1.2,
        min_length_of_recording=0.3,
        min_gap_between_recordings=0,
        pre_recording_buffer_duration=0.5,
        early_transcription_on_silence=300,

        # Text formatting
        ensure_sentence_starting_uppercase=True,
        ensure_sentence_ends_with_period=False,

        # Callbacks
        on_recording_start=lambda: sys.stdout.write("\nRecording...\n") or sys.stdout.flush(),
        on_recording_stop=lambda: sys.stdout.write("Processing...\n") or sys.stdout.flush(),
        on_realtime_transcription_update=lambda t: (
            sys.stdout.write(f"\r  >> {t.strip()}" + " " * 20) or sys.stdout.flush()
        ) if t.strip() else None,

        spinner=False,
        print_transcription_time=True,
        no_log_file=True,
    )

    # Wake word config
    if WAKE_WORD_MODEL and os.path.exists(WAKE_WORD_MODEL):
        config.update(
            wakeword_backend="oww",
            openwakeword_model_paths=WAKE_WORD_MODEL,
            openwakeword_inference_framework="onnx",
            wake_words="computer",
            wake_words_sensitivity=0.7,
            wake_word_activation_delay=0.0,
            wake_word_timeout=30.0,
            on_wakeword_detected=lambda: sys.stdout.write("\nWake word detected!\n") or sys.stdout.flush(),
        )
        print(f"Wake word: 'Computer' (openWakeWord)")
    else:
        print("Wake word: disabled (model not found)")

    print(f"Main model: {MAIN_MODEL} (GPU float16)")
    print(f"Realtime: {REALTIME_MODEL}")
    print(f"Push-to-talk: Menu key (hold to record)")
    print("Ctrl+C to exit.")
    print("=" * 50)
    print("Loading models...", flush=True)

    # Suppress ALSA/JACK noise during PyAudio device enumeration
    old_stderr = _suppress_stderr()
    try:
        recorder = AudioToTextRecorder(**config)
    finally:
        _restore_stderr(old_stderr)

    # Unbind Menu key from context menu, start push-to-talk listener
    unbind_menu_key()
    hotkey_listener = start_hotkey_listener(recorder)

    sys.stdout.write("\n" + "=" * 50 + "\n")
    sys.stdout.write("READY\n")
    sys.stdout.write("  Wake word: say 'Computer' then speak\n")
    sys.stdout.write("  Push-to-talk: hold Menu key and speak\n")
    if COMMANDS:
        sys.stdout.write(f"  Voice commands: {len(COMMANDS)} loaded\n")
    sys.stdout.write("=" * 50 + "\n\n")
    sys.stdout.flush()

    try:
        while True:
            sys.stdout.write("[Listening...]\n")
            sys.stdout.flush()
            text = recorder.text()
            if text and text.strip():
                sys.stdout.write("\n")
                original = text.strip()
                corrected = apply_vocabulary(original)

                cmd = check_command(corrected)
                if cmd:
                    if corrected != original:
                        sys.stdout.write(f"  \"{original}\" -> \"{corrected}\"\n")
                    else:
                        sys.stdout.write(f"  \"{corrected}\"\n")
                    run_command(cmd)
                else:
                    if corrected != original:
                        sys.stdout.write(f"  {original} -> {corrected}\n")
                    else:
                        sys.stdout.write(f"  {corrected}\n")
                    sys.stdout.flush()
                    paste_text(corrected + " ")
            else:
                sys.stdout.write("[Empty transcription, restarting...]\n")
                sys.stdout.flush()
    except KeyboardInterrupt:
        sys.stdout.write("\nShutting down...\n")
    finally:
        hotkey_listener.stop()
        recorder.shutdown()


if __name__ == '__main__':
    main()
