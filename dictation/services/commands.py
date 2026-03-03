"""Voice commands and vocabulary correction.

Voice commands are loaded from ~/.dictation-commands.yaml.
Vocabulary replacements are loaded from ~/.dictation-vocabulary.txt.
"""

import os
import re
import subprocess
import sys
from typing import Optional

import yaml

VOCABULARY_FILE = os.path.expanduser("~/.dictation-vocabulary.txt")
COMMANDS_FILE = os.path.expanduser("~/.dictation-commands.yaml")

# Module-level state
VOCABULARY: dict[str, str] = {}
COMMANDS: list[dict] = []


# ── Vocabulary ────────────────────────────────────────────────────────

def load_vocabulary() -> dict[str, str]:
    """Load word replacement mappings from ~/.dictation-vocabulary.txt."""
    global VOCABULARY
    VOCABULARY = {}
    if not os.path.exists(VOCABULARY_FILE):
        return VOCABULARY
    try:
        with open(VOCABULARY_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "->" in line:
                    wrong, correct = line.split("->", 1)
                    VOCABULARY[wrong.strip()] = correct.strip()
    except Exception as e:
        print(f"Warning: Could not load vocabulary: {e}")
    return VOCABULARY


def apply_vocabulary(text: str) -> str:
    """Apply word replacements to transcribed text."""
    for wrong, correct in VOCABULARY.items():
        pattern = r"\b" + re.escape(wrong) + r"\b"
        text = re.sub(pattern, correct, text, flags=re.IGNORECASE)
    return text


# ── Voice Commands ────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Normalize text for command matching: lowercase, strip punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def load_commands() -> list[dict]:
    """Load voice commands from ~/.dictation-commands.yaml."""
    global COMMANDS
    COMMANDS = []
    if not os.path.exists(COMMANDS_FILE):
        return COMMANDS
    try:
        with open(COMMANDS_FILE, "r") as f:
            data = yaml.safe_load(f)
        if data and "commands" in data:
            for cmd in data["commands"]:
                patterns = [_normalize(p) for p in cmd.get("patterns", [])]
                COMMANDS.append({
                    "name": cmd.get("name", "unnamed"),
                    "patterns": patterns,
                    "action": cmd.get("action", ""),
                    "respond": cmd.get("respond", ""),
                })
    except Exception as e:
        sys.stdout.write(f"Warning: Could not load commands: {e}\n")
        sys.stdout.flush()
    return COMMANDS


def check_command(text: str) -> Optional[dict]:
    """Check if transcribed text matches a voice command.

    Uses fuzzy keyword matching: all keywords in a pattern must appear
    in the transcribed text.  Longest (most specific) pattern wins.
    """
    normalized = _normalize(text)
    words = set(normalized.split())

    best_match = None
    best_score = 0

    for cmd in COMMANDS:
        for pattern in cmd["patterns"]:
            pattern_words = pattern.split()
            matched = sum(
                1 for pw in pattern_words if any(pw in w for w in words)
            )
            if matched == len(pattern_words):
                score = len(pattern_words)
                if score > best_score:
                    best_score = score
                    best_match = cmd

    return best_match


def run_command(cmd: dict) -> None:
    """Execute a voice command's shell action."""
    name = cmd["name"]
    action = cmd["action"]
    respond = cmd["respond"]

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
