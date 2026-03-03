"""Platform-aware text injection at cursor position.

Linux: xdotool type, fallback to xclip + xdotool key ctrl+v
macOS: pbcopy + osascript Cmd+V
"""

import subprocess
import sys
import time
import platform

IS_MAC = platform.system() == "Darwin"


def paste_text(text: str) -> None:
    """Insert text at the current cursor position."""
    if IS_MAC:
        _paste_text_mac(text)
    else:
        _paste_text_linux(text)


def _paste_text_mac(text: str) -> None:
    """macOS: copy to clipboard via pbcopy, then Cmd+V via osascript."""
    try:
        subprocess.run(
            ["pbcopy"], input=text.encode("utf-8"), check=True, timeout=1
        )
        time.sleep(0.05)
        subprocess.run(
            [
                "osascript", "-e",
                'tell application "System Events" to keystroke "v" using command down',
            ],
            check=True, timeout=2,
        )
    except Exception as e:
        sys.stdout.write(f"macOS paste failed: {e}\n")
        sys.stdout.flush()


def _paste_text_linux(text: str) -> None:
    """Linux/X11: xdotool type, fallback to xclip clipboard paste."""
    try:
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", "0", "--", text],
            check=True, timeout=5,
        )
    except Exception as e:
        sys.stdout.write(f"xdotool type failed ({e}), trying clipboard\n")
        sys.stdout.flush()
        try:
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text.encode("utf-8"),
                check=True, timeout=1,
            )
            time.sleep(0.05)
            subprocess.run(
                ["xdotool", "key", "ctrl+v"], check=True, timeout=2
            )
        except Exception as e2:
            sys.stdout.write(f"Clipboard paste also failed: {e2}\n")
            sys.stdout.flush()
