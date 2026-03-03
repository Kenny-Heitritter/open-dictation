"""Platform-aware text injection at cursor position.

Linux: xdotool type, fallback to xclip + xdotool key ctrl+v
macOS: pbcopy + CGEvent Cmd+V (via pyobjc-framework-Quartz)
       Requires Accessibility permission for the terminal app.
"""

import subprocess
import sys
import time
import platform

IS_MAC = platform.system() == "Darwin"

# Track whether we've already warned about accessibility
_accessibility_warned = False


def paste_text(text: str) -> None:
    """Insert text at the current cursor position."""
    if IS_MAC:
        _paste_text_mac(text)
    else:
        _paste_text_linux(text)


def _check_accessibility() -> bool:
    """Check if the current process has Accessibility permission on macOS."""
    try:
        from ApplicationServices import AXIsProcessTrusted
        return AXIsProcessTrusted()
    except Exception:
        return True  # assume OK if we can't check


def _paste_text_mac(text: str) -> None:
    """macOS: copy to clipboard via pbcopy, then Cmd+V via CGEvent.

    Uses Quartz CGEvent API which requires Accessibility permission.
    Falls back to osascript if CGEvent doesn't work.
    """
    global _accessibility_warned

    if not _check_accessibility() and not _accessibility_warned:
        _accessibility_warned = True
        sys.stdout.write(
            "*** Text will be on clipboard but cannot paste without Accessibility. ***\n"
            "*** Grant access in: System Settings > Privacy & Security > Accessibility ***\n"
        )
        sys.stdout.flush()

    try:
        # Copy to clipboard
        subprocess.run(
            ["pbcopy"], input=text.encode("utf-8"), check=True, timeout=1
        )
        time.sleep(0.05)

        # Try CGEvent Cmd+V (preferred -- same permission as pynput)
        _cgevent_cmd_v()
    except Exception as e:
        sys.stdout.write(f"macOS paste failed: {e}\n")
        sys.stdout.flush()


def _cgevent_cmd_v() -> None:
    """Simulate Cmd+V keypress using Quartz CGEvent API."""
    from Quartz import (
        CGEventSourceCreate,
        kCGEventSourceStateCombinedSessionState,
        CGEventCreateKeyboardEvent,
        CGEventSetFlags,
        CGEventPost,
        kCGHIDEventTap,
        kCGEventFlagMaskCommand,
    )

    # Virtual keycode 9 = 'v' on macOS
    src = CGEventSourceCreate(kCGEventSourceStateCombinedSessionState)
    event_down = CGEventCreateKeyboardEvent(src, 9, True)
    CGEventSetFlags(event_down, kCGEventFlagMaskCommand)
    event_up = CGEventCreateKeyboardEvent(src, 9, False)
    CGEventSetFlags(event_up, kCGEventFlagMaskCommand)

    CGEventPost(kCGHIDEventTap, event_down)
    CGEventPost(kCGHIDEventTap, event_up)


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
