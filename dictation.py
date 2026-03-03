#!/usr/bin/env python3
"""
Local speech-to-text dictation using RealtimeSTT.
All processing runs locally on GPU. No data leaves the machine.

Three modes:
  1. Wake word: Say "Computer", then speak naturally
  2. Push-to-talk: Hold Menu key (Linux) or Right Option (macOS), speak, release
  3. Voice commands: Say things like "open my qbraid email"
     Commands are defined in ~/.dictation-commands.yaml

Usage:
  python -u dictation.py              # Normal dictation mode
  python -u dictation.py --agent      # Agent mode (Phase 2)
"""

import sys

from dictation.conversation import run


def main():
    agent_mode = "--agent" in sys.argv
    run(agent_mode=agent_mode)


if __name__ == "__main__":
    main()
