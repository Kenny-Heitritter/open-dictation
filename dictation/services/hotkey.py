"""Push-to-talk hotkey listener.

Linux: Menu key (remapped to F24 via xmodmap to suppress context menus).
macOS: Right Option key.
"""

import subprocess
import sys
import platform
from typing import Callable

IS_MAC = platform.system() == "Darwin"


def unbind_menu_key() -> None:
    """Remap Menu key to F24 so it doesn't trigger context menus.

    Only applies on Linux/X11.  F24 is a valid keysym that no application
    or window manager binds to.  The key still generates events that
    pynput can detect.
    """
    if IS_MAC:
        return
    try:
        subprocess.run(
            ["xmodmap", "-e", "keycode 135 = F24"],
            check=True, timeout=2,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def send_escape_key() -> None:
    """Send an Escape keypress to dismiss context menus on PTT activation."""
    try:
        if IS_MAC:
            _cgevent_escape()
        else:
            subprocess.Popen(
                ["xdotool", "key", "Escape"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
    except Exception:
        pass


def _cgevent_escape() -> None:
    """Simulate Escape keypress using Quartz CGEvent API."""
    from Quartz import (
        CGEventSourceCreate,
        kCGEventSourceStateCombinedSessionState,
        CGEventCreateKeyboardEvent,
        CGEventPost,
        kCGHIDEventTap,
    )

    # Virtual keycode 53 = Escape on macOS
    src = CGEventSourceCreate(kCGEventSourceStateCombinedSessionState)
    event_down = CGEventCreateKeyboardEvent(src, 53, True)
    event_up = CGEventCreateKeyboardEvent(src, 53, False)
    CGEventPost(kCGHIDEventTap, event_down)
    CGEventPost(kCGHIDEventTap, event_up)


def start_hotkey_listener(
    on_press: Callable[[], None],
    on_release: Callable[[], None],
):
    """Start a background keyboard listener for push-to-talk.

    Parameters
    ----------
    on_press : callable
        Called when the PTT key is pressed down.
    on_release : callable
        Called when the PTT key is released.

    Returns
    -------
    pynput.keyboard.Listener
        The running listener (daemon thread).  Call .stop() to clean up.
    """
    from pynput import keyboard
    from pynput.keyboard import Key

    held = False

    def _is_ptt_key(key) -> bool:
        if IS_MAC:
            return key == Key.alt_r
        if key == Key.menu:
            return True
        # After xmodmap remap to F24, pynput sees keysym 0xFFD5 = 65493
        if hasattr(key, "vk") and key.vk == 0xFFD5:
            return True
        return False

    def _on_press(key):
        nonlocal held
        if _is_ptt_key(key) and not held:
            held = True
            on_press()

    def _on_release(key):
        nonlocal held
        if _is_ptt_key(key) and held:
            held = False
            on_release()

    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.daemon = True
    listener.start()
    return listener
