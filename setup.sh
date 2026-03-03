#!/usr/bin/env bash
# setup.sh - Install local-dictation and its dependencies
#
# Usage:
#   ./setup.sh              # Full install (venv + deps + config + autostart)
#   ./setup.sh --deps-only  # Only install Python dependencies into existing venv
#
# Requirements:
#   - Ubuntu 22.04 (X11 session, not Wayland)
#   - Python 3.9+
#   - NVIDIA GPU with CUDA support
#   - portaudio19-dev (or see workaround below if you can't sudo)
#
# This script will:
#   1. Create a Python venv at ./stt-env
#   2. Install pinned Python dependencies (GPU-accelerated)
#   3. Build PyAudio (handling missing portaudio headers if needed)
#   4. Patch RealtimeSTT for thread-safe signal handling
#   5. Copy wake word model to openwakeword's model directory
#   6. Install config files to ~/.dictation-{commands,vocabulary}.txt
#   7. Set up GNOME autostart

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${REPO_DIR}/stt-env"
if [ -z "${PYTHON:-}" ]; then
    if command -v python3.9 &>/dev/null; then
        PYTHON="python3.9"
    elif command -v python3.10 &>/dev/null; then
        PYTHON="python3.10"
    elif command -v python3.11 &>/dev/null; then
        PYTHON="python3.11"
    else
        PYTHON="python3"
    fi
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Parse arguments ──────────────────────────────────────────────────
DEPS_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --deps-only) DEPS_ONLY=true ;;
        --help|-h)
            echo "Usage: $0 [--deps-only]"
            echo "  --deps-only  Only install Python deps into existing venv"
            exit 0
            ;;
        *) error "Unknown argument: $arg" ;;
    esac
done

# ── Pre-flight checks ───────────────────────────────────────────────
info "Checking prerequisites..."

OS="$(uname -s)"
IS_MAC=false
IS_LINUX=false
if [ "$OS" = "Darwin" ]; then
    IS_MAC=true
elif [ "$OS" = "Linux" ]; then
    IS_LINUX=true
fi


if ! command -v "$PYTHON" &>/dev/null; then
    if [ "$IS_LINUX" = true ]; then
        error "Python not found. Install python3: sudo apt install python3 python3-venv"
    else
        error "Python not found. Install python3: brew install python3"
    fi
fi

PY_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
info "Python version: ${PY_VERSION}"

if [ "$IS_LINUX" = true ]; then
    if ! command -v xdotool &>/dev/null; then
        warn "xdotool not found. Install with: sudo apt install xdotool"
    fi
    if ! command -v xmodmap &>/dev/null; then
        warn "xmodmap not found. Install with: sudo apt install x11-xserver-utils"
    fi
    if ! nvidia-smi &>/dev/null; then
        warn "nvidia-smi not found. GPU acceleration may not work."
    fi
elif [ "$IS_MAC" = true ]; then
    if ! command -v brew &>/dev/null; then
        warn "Homebrew not found. You may need it to install portaudio."
    fi
fi

# ── Create venv ──────────────────────────────────────────────────────
if [ "$DEPS_ONLY" = false ]; then
    if [ -d "$VENV_DIR" ]; then
        warn "Venv already exists at ${VENV_DIR}"
        read -rp "Delete and recreate? [y/N] " yn
        case "$yn" in
            [Yy]*) rm -rf "$VENV_DIR" ;;
            *) info "Keeping existing venv" ;;
        esac
    fi

    if [ ! -d "$VENV_DIR" ]; then
        info "Creating virtual environment at ${VENV_DIR}..."
        "$PYTHON" -m venv "$VENV_DIR"
    fi
fi

if [ ! -f "${VENV_DIR}/bin/pip" ]; then
    error "Venv not found at ${VENV_DIR}. Run without --deps-only first."
fi

PIP="${VENV_DIR}/bin/pip"
VENV_PYTHON="${VENV_DIR}/bin/python"

info "Upgrading pip..."
"$PIP" install --upgrade pip "setuptools<81" wheel -q

# ── Install PyTorch (CUDA) ──────────────────────────────────────────
if [ "$IS_LINUX" = true ]; then
    info "Installing PyTorch with CUDA support..."
    "$PIP" install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 -q
else
    info "Installing PyTorch..."
    "$PIP" install torch==2.3.1 torchaudio==2.3.1 -q
fi

# ── Install PyAudio (needs portaudio headers) ────────────────────────
info "Installing PyAudio..."

# Check if portaudio headers are available
if [ "$IS_MAC" = true ]; then
    if ! brew list portaudio &>/dev/null; then
        info "Installing portaudio via brew..."
        brew install portaudio || warn "Failed to install portaudio"
    fi
    "$PIP" install PyAudio==0.2.14 -q
elif pkg-config --exists portaudio-2.0 2>/dev/null || [ -f /usr/include/portaudio.h ]; then
    "$PIP" install PyAudio==0.2.14 -q
else
    warn "portaudio19-dev not found. Attempting workaround..."
    warn "For a clean install, run: sudo apt install portaudio19-dev"

    # Try to extract headers from the .deb package
    TMPDIR=$(mktemp -d)
    PORTAUDIO_DEV_DEB="${TMPDIR}/portaudio19-dev.deb"
    PORTAUDIO_INCLUDE="${TMPDIR}/portaudio-include"

    if apt-cache show portaudio19-dev &>/dev/null; then
        apt-get download portaudio19-dev -o Dir::Cache::archives="$TMPDIR" 2>/dev/null || true
        DEB_FILE=$(find "$TMPDIR" -name 'portaudio19-dev*.deb' 2>/dev/null | head -1)

        if [ -n "$DEB_FILE" ]; then
            mkdir -p "$PORTAUDIO_INCLUDE"
            dpkg-deb -x "$DEB_FILE" "$PORTAUDIO_INCLUDE"
            INCLUDE_DIR="${PORTAUDIO_INCLUDE}/usr/include"

            # Find libportaudio.so on the system
            PORTAUDIO_LIB=$(find /usr/lib -name 'libportaudio.so*' 2>/dev/null | head -1)
            if [ -n "$PORTAUDIO_LIB" ]; then
                LIB_DIR="${TMPDIR}/lib"
                mkdir -p "$LIB_DIR"
                ln -sf "$PORTAUDIO_LIB" "${LIB_DIR}/libportaudio.so"
                C_INCLUDE_PATH="$INCLUDE_DIR" LIBRARY_PATH="$LIB_DIR" "$PIP" install PyAudio==0.2.14 -q
            else
                error "Could not find libportaudio.so on system. Install portaudio19-dev."
            fi
        else
            error "Could not download portaudio19-dev .deb. Install it manually: sudo apt install portaudio19-dev"
        fi
    else
        error "portaudio19-dev not available in apt cache. Install it manually."
    fi

    rm -rf "$TMPDIR"
fi

# ── Install remaining Python dependencies ────────────────────────────
info "Installing Python dependencies..."
DEPS=(
    RealtimeSTT==0.3.0
    faster-whisper==1.0.3
    openwakeword==0.6.0
    onnxruntime==1.19.2
    webrtcvad==2.0.10
    pynput==1.8.1
    PyYAML==6.0.3
    "openai>=1.0.0"
)

# tflite-runtime has no macOS ARM64 wheels; openwakeword uses onnxruntime instead
if [ "$IS_LINUX" = true ]; then
    DEPS+=(tflite-runtime==2.14.0)
fi

"$PIP" install "${DEPS[@]}" -q

# macOS Apple Silicon: install mlx-whisper for Metal GPU acceleration
if [ "$IS_MAC" = true ] && [ "$(uname -m)" = "arm64" ]; then
    info "Installing mlx-whisper for Metal GPU acceleration..."
    "$PIP" install mlx-whisper -q
fi

# ── Install agent mode dependencies (optional) ──────────────────────
info "Installing agent mode dependencies..."
if command -v piper &>/dev/null; then
    info "Piper TTS already installed"
else
    info "Piper TTS not found. For local TTS in agent mode, install piper-tts:"
    if [ "$IS_MAC" = true ]; then
        warn "  pip install piper-tts  (or brew install piper)"
    else
        warn "  pip install piper-tts"
    fi
fi

info "Checking for Ollama (local LLM)..."
if command -v ollama &>/dev/null; then
    info "Ollama found. Agent mode will use local LLM by default."
else
    warn "Ollama not found. For local LLM in agent mode, install from: https://ollama.ai"
    warn "  Then run: ollama pull llama3.2"
fi

# ── Patch RealtimeSTT signal handling ────────────────────────────────
info "Patching RealtimeSTT for thread-safe signal handling..."
AUDIO_RECORDER=$(find "$VENV_DIR" -name "audio_recorder.py" -path "*/RealtimeSTT/*" | head -1)

if [ -n "$AUDIO_RECORDER" ]; then
    # Patch: wrap signal.signal() calls in try/except ValueError
    # RealtimeSTT calls signal.signal(SIGINT, SIG_IGN) from worker threads,
    # which raises ValueError in Python. This patch makes it safe.
    "$VENV_PYTHON" -c "
import re

with open('${AUDIO_RECORDER}', 'r') as f:
    content = f.read()

# Pattern: bare signal.signal(...SIG_IGN) lines that aren't already wrapped
old = '            system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)'
new = '''        try:
            system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)
        except ValueError:
            pass  # not in main thread'''

# Only patch if not already patched
if 'except ValueError:' not in content:
    content = content.replace(old, new)
    with open('${AUDIO_RECORDER}', 'w') as f:
        f.write(content)
    print('Patched audio_recorder.py')
else:
    print('Already patched')
"
else
    warn "Could not find RealtimeSTT audio_recorder.py to patch"
fi

# ── Copy wake word model ─────────────────────────────────────────────
info "Installing wake word model..."
OWW_MODEL_DIR=$(find "$VENV_DIR" -type d -name "models" -path "*/openwakeword/resources/*" | head -1)

if [ -n "$OWW_MODEL_DIR" ] && [ -f "${REPO_DIR}/models/computer.onnx" ]; then
    cp "${REPO_DIR}/models/computer.onnx" "${OWW_MODEL_DIR}/computer.onnx"
    info "Copied computer.onnx to ${OWW_MODEL_DIR}/"
else
    warn "Could not install wake word model. Copy models/computer.onnx manually."
fi

# ── Install config files ─────────────────────────────────────────────
if [ "$DEPS_ONLY" = false ]; then
    info "Installing config files..."

    if [ ! -f "$HOME/.dictation-commands.yaml" ]; then
        cp "${REPO_DIR}/config/dictation-commands.yaml" "$HOME/.dictation-commands.yaml"
        info "Installed ~/.dictation-commands.yaml"
    else
        info "~/.dictation-commands.yaml already exists, skipping"
    fi

    if [ ! -f "$HOME/.dictation-vocabulary.txt" ]; then
        cp "${REPO_DIR}/config/dictation-vocabulary.txt" "$HOME/.dictation-vocabulary.txt"
        info "Installed ~/.dictation-vocabulary.txt"
    else
        info "~/.dictation-vocabulary.txt already exists, skipping"
    fi

    # ── Set up autostart ─────────────────────────────────────────────
    if [ "$IS_MAC" = true ]; then
        info "Setting up macOS LaunchAgent..."
        LOG_DIR="$HOME/Library/Logs/dictation"
        mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"
        sed \
            -e "s|VENV_PATH|${VENV_DIR}|g" \
            -e "s|REPO_PATH|${REPO_DIR}|g" \
            -e "s|LOG_PATH|${LOG_DIR}|g" \
            "${REPO_DIR}/config/com.dictation.agent.plist" \
            > "$HOME/Library/LaunchAgents/com.dictation.agent.plist"
        info "Installed LaunchAgent (starts on login)"
        info "  Log: ${LOG_DIR}/dictation.log"
        info "  Manage: ./dictation.sh start|stop|restart|status|log"
    elif [ "$IS_LINUX" = true ]; then
        info "Setting up GNOME autostart..."
        AUTOSTART_DIR="$HOME/.config/autostart"
        mkdir -p "$AUTOSTART_DIR"
        sed \
            -e "s|VENV_PATH|${VENV_DIR}|g" \
            -e "s|REPO_PATH|${REPO_DIR}|g" \
            "${REPO_DIR}/config/dictation.desktop" > "${AUTOSTART_DIR}/dictation.desktop"
        info "Installed autostart entry at ${AUTOSTART_DIR}/dictation.desktop"
    fi
fi

# ── Set PulseAudio default source (Scarlett 2i2) ─────────────────────
if [ "$IS_LINUX" = true ]; then
    SCARLETT_SOURCE="alsa_input.usb-Focusrite_Scarlett_2i2_4th_Gen_S2W0760368328E-00.multichannel-input"
    if pactl list sources short 2>/dev/null | grep -q "$SCARLETT_SOURCE"; then
        pactl set-default-source "$SCARLETT_SOURCE"
        info "Set default audio source to Scarlett 2i2"
    else
        warn "Scarlett 2i2 not detected. Set your default PulseAudio source manually."
    fi
fi

# ── Done ─────────────────────────────────────────────────────────────
echo ""
info "Setup complete!"
echo ""
echo "  Quick start:"
echo "    ./dictation.sh start          Start in background"
echo "    ./dictation.sh run            Run interactively (Ctrl+C to stop)"
echo ""
echo "  Management:"
echo "    ./dictation.sh stop           Stop the daemon"
echo "    ./dictation.sh restart        Restart the daemon"
echo "    ./dictation.sh status         Check if running"
echo "    ./dictation.sh log            Tail the log file"
echo "    ./dictation.sh enable         Auto-start on login"
echo "    ./dictation.sh disable        Disable auto-start"
echo ""
echo "  Config files:"
echo "    ~/.dictation-commands.yaml    Voice commands"
echo "    ~/.dictation-vocabulary.txt   Word replacements"
if [ "$IS_MAC" = true ]; then
    echo ""
    echo "  macOS: Grant Accessibility permission to your terminal app:"
    echo "    System Settings > Privacy & Security > Accessibility"
fi
