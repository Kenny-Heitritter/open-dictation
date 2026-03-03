#!/usr/bin/env bash
# dictation.sh - Manage the dictation daemon
#
# Usage:
#   ./dictation.sh start     Start dictation (background daemon)
#   ./dictation.sh stop      Stop dictation
#   ./dictation.sh restart   Restart dictation
#   ./dictation.sh status    Check if dictation is running
#   ./dictation.sh log       Tail the log file
#   ./dictation.sh run       Run in foreground (interactive, Ctrl+C to stop)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${REPO_DIR}/stt-env"
PYTHON="${VENV_DIR}/bin/python"

OS="$(uname -s)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ── macOS (launchd) ─────────────────────────────────────────────────

PLIST_LABEL="com.dictation.agent"
PLIST_INSTALLED="$HOME/Library/LaunchAgents/${PLIST_LABEL}.plist"
LOG_DIR="$HOME/Library/Logs/dictation"
LOG_FILE="${LOG_DIR}/dictation.log"

# ── Linux (systemd user service, fallback to direct) ────────────────

SYSTEMD_SERVICE="dictation.service"
SYSTEMD_DIR="$HOME/.config/systemd/user"
SYSTEMD_FILE="${SYSTEMD_DIR}/${SYSTEMD_SERVICE}"

# ── Helpers ──────────────────────────────────────────────────────────

_check_venv() {
    if [ ! -f "$PYTHON" ]; then
        echo -e "${RED}Error:${NC} venv not found at ${VENV_DIR}"
        echo "Run ./setup.sh first."
        exit 1
    fi
}

_pid() {
    # Find dictation.py process
    pgrep -f "python.*dictation\.py" 2>/dev/null | head -1 || true
}

# ── macOS commands ───────────────────────────────────────────────────

_mac_install_plist() {
    # Ensure the LaunchAgent plist is installed with correct paths
    if [ ! -f "$PLIST_INSTALLED" ]; then
        mkdir -p "$HOME/Library/LaunchAgents"
        mkdir -p "$LOG_DIR"
        sed \
            -e "s|VENV_PATH|${VENV_DIR}|g" \
            -e "s|REPO_PATH|${REPO_DIR}|g" \
            -e "s|LOG_PATH|${LOG_DIR}|g" \
            "${REPO_DIR}/config/com.dictation.agent.plist" > "$PLIST_INSTALLED"
    fi
}

_mac_start() {
    _check_venv
    _mac_install_plist
    if launchctl list "$PLIST_LABEL" &>/dev/null; then
        echo -e "${YELLOW}Already running.${NC} Use './dictation.sh restart' to restart."
        return
    fi
    mkdir -p "$LOG_DIR"
    launchctl load "$PLIST_INSTALLED"
    sleep 1
    if launchctl list "$PLIST_LABEL" &>/dev/null; then
        echo -e "${GREEN}Dictation started.${NC}"
        echo "  Log: $LOG_FILE"
        echo "  Stop: ./dictation.sh stop"
    else
        echo -e "${RED}Failed to start.${NC} Check: ./dictation.sh log"
    fi
}

_mac_stop() {
    if ! launchctl list "$PLIST_LABEL" &>/dev/null; then
        echo "Not running."
        return
    fi
    launchctl unload "$PLIST_INSTALLED"
    echo -e "${GREEN}Dictation stopped.${NC}"
}

_mac_status() {
    if launchctl list "$PLIST_LABEL" &>/dev/null; then
        local pid
        pid=$(_pid)
        echo -e "${GREEN}Running${NC} (PID: ${pid:-unknown})"
        if [ -f "$LOG_FILE" ]; then
            echo "  Log: $LOG_FILE"
            echo "  Last line: $(tail -1 "$LOG_FILE" 2>/dev/null)"
        fi
    else
        echo -e "${YELLOW}Not running.${NC}"
    fi
}

_mac_log() {
    if [ ! -f "$LOG_FILE" ]; then
        echo "No log file yet. Start dictation first."
        return
    fi
    tail -f "$LOG_FILE"
}

# ── Linux commands ───────────────────────────────────────────────────

_linux_install_service() {
    if [ ! -f "$SYSTEMD_FILE" ]; then
        mkdir -p "$SYSTEMD_DIR"
        cat > "$SYSTEMD_FILE" << EOF
[Unit]
Description=Open Dictation - Local speech-to-text
After=graphical-session.target

[Service]
Type=simple
ExecStart=${PYTHON} -u ${REPO_DIR}/dictation.py
WorkingDirectory=${REPO_DIR}
Restart=on-failure
RestartSec=5
Environment=DISPLAY=:0

[Install]
WantedBy=default.target
EOF
        systemctl --user daemon-reload
    fi
}

_linux_start() {
    _check_venv
    _linux_install_service
    if systemctl --user is-active --quiet "$SYSTEMD_SERVICE" 2>/dev/null; then
        echo -e "${YELLOW}Already running.${NC} Use './dictation.sh restart' to restart."
        return
    fi
    systemctl --user start "$SYSTEMD_SERVICE"
    sleep 1
    if systemctl --user is-active --quiet "$SYSTEMD_SERVICE"; then
        echo -e "${GREEN}Dictation started.${NC}"
        echo "  Log: journalctl --user -u $SYSTEMD_SERVICE -f"
        echo "  Stop: ./dictation.sh stop"
    else
        echo -e "${RED}Failed to start.${NC} Check: ./dictation.sh log"
    fi
}

_linux_stop() {
    if ! systemctl --user is-active --quiet "$SYSTEMD_SERVICE" 2>/dev/null; then
        echo "Not running."
        return
    fi
    systemctl --user stop "$SYSTEMD_SERVICE"
    echo -e "${GREEN}Dictation stopped.${NC}"
}

_linux_status() {
    systemctl --user status "$SYSTEMD_SERVICE" 2>/dev/null || echo -e "${YELLOW}Not running.${NC}"
}

_linux_log() {
    journalctl --user -u "$SYSTEMD_SERVICE" -f
}

# ── Autostart enable/disable ────────────────────────────────────────

_enable_autostart() {
    if [ "$OS" = "Darwin" ]; then
        _check_venv
        _mac_install_plist
        # RunAtLoad is already true in the plist
        echo -e "${GREEN}Autostart enabled.${NC} Dictation will start on login."
    else
        _check_venv
        _linux_install_service
        systemctl --user enable "$SYSTEMD_SERVICE"
        echo -e "${GREEN}Autostart enabled.${NC} Dictation will start on login."
    fi
}

_disable_autostart() {
    if [ "$OS" = "Darwin" ]; then
        if launchctl list "$PLIST_LABEL" &>/dev/null; then
            launchctl unload "$PLIST_INSTALLED"
        fi
        rm -f "$PLIST_INSTALLED"
        echo -e "${GREEN}Autostart disabled.${NC}"
    else
        systemctl --user disable "$SYSTEMD_SERVICE" 2>/dev/null || true
        echo -e "${GREEN}Autostart disabled.${NC}"
    fi
}

# ── Main ─────────────────────────────────────────────────────────────

_usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  start       Start dictation in the background"
    echo "  stop        Stop dictation"
    echo "  restart     Restart dictation"
    echo "  status      Check if dictation is running"
    echo "  log         Tail the log file"
    echo "  run         Run in foreground (interactive, Ctrl+C to stop)"
    echo "  enable      Enable auto-start on login"
    echo "  disable     Disable auto-start on login"
}

CMD="${1:-}"
shift || true
case "$CMD" in
    start)
        if [ "$OS" = "Darwin" ]; then _mac_start; else _linux_start; fi
        ;;
    stop)
        if [ "$OS" = "Darwin" ]; then _mac_stop; else _linux_stop; fi
        ;;
    restart)
        if [ "$OS" = "Darwin" ]; then _mac_stop; sleep 1; _mac_start
        else _linux_stop; sleep 1; _linux_start; fi
        ;;
    status)
        if [ "$OS" = "Darwin" ]; then _mac_status; else _linux_status; fi
        ;;
    log)
        if [ "$OS" = "Darwin" ]; then _mac_log; else _linux_log; fi
        ;;
    run)
        _check_venv
        exec "$PYTHON" -u "${REPO_DIR}/dictation.py" "$@"
        ;;
    enable)
        _enable_autostart
        ;;
    disable)
        _disable_autostart
        ;;
    -h|--help|help|"")
        _usage
        ;;
    *)
        echo -e "${RED}Unknown command:${NC} $CMD"
        _usage
        exit 1
        ;;
esac
