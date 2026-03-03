"""Immutable types for the dictation state machine.

Following the pattern from shuo: all state transitions are modeled as
(State, Event) -> (State, list[Action]).  Events are inputs from the
outside world; Actions are side-effects to dispatch.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional, Union


# ── Phases ────────────────────────────────────────────────────────────

class Phase(enum.Enum):
    """Top-level phase of the dictation system."""
    IDLE = "idle"                # Waiting for wake word or PTT
    LISTENING = "listening"      # Actively recording speech
    PROCESSING = "processing"   # Transcription in progress
    RESPONDING = "responding"   # Agent is generating / speaking (Phase 2)


# ── State ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AppState:
    phase: Phase = Phase.IDLE
    ptt_active: bool = False         # Is push-to-talk key held?
    wake_word_active: bool = False   # Was recording triggered by wake word?


# ── Events (inputs) ──────────────────────────────────────────────────

@dataclass(frozen=True)
class WakeWordDetectedEvent:
    """The wake word model fired."""
    pass


@dataclass(frozen=True)
class PTTPressedEvent:
    """Push-to-talk key was pressed down."""
    pass


@dataclass(frozen=True)
class PTTReleasedEvent:
    """Push-to-talk key was released."""
    pass


@dataclass(frozen=True)
class RecordingStartedEvent:
    """RealtimeSTT began recording audio."""
    pass


@dataclass(frozen=True)
class RecordingStoppedEvent:
    """RealtimeSTT stopped recording (silence detected or PTT released)."""
    pass


@dataclass(frozen=True)
class RealtimeTranscriptionEvent:
    """Partial transcription update from the realtime (tiny) model."""
    text: str


@dataclass(frozen=True)
class TranscriptionReadyEvent:
    """Final transcription from the main model is available."""
    text: str


@dataclass(frozen=True)
class AgentResponseDoneEvent:
    """Agent finished generating and playing its response (Phase 2)."""
    pass


@dataclass(frozen=True)
class ShutdownEvent:
    """Clean shutdown requested (Ctrl+C)."""
    pass


# Union of all event types
Event = Union[
    WakeWordDetectedEvent,
    PTTPressedEvent,
    PTTReleasedEvent,
    RecordingStartedEvent,
    RecordingStoppedEvent,
    RealtimeTranscriptionEvent,
    TranscriptionReadyEvent,
    AgentResponseDoneEvent,
    ShutdownEvent,
]


# ── Actions (outputs / side-effects) ─────────────────────────────────

@dataclass(frozen=True)
class PasteTextAction:
    """Inject text at the cursor position."""
    text: str


@dataclass(frozen=True)
class RunCommandAction:
    """Execute a matched voice command."""
    command: dict  # The command dict from commands.yaml


@dataclass(frozen=True)
class SendEscapeKeyAction:
    """Dismiss any open context menus (on PTT press)."""
    pass


@dataclass(frozen=True)
class StartAgentTurnAction:
    """Begin the LLM -> TTS -> speaker pipeline (Phase 2)."""
    transcript: str


@dataclass(frozen=True)
class CancelAgentTurnAction:
    """Interrupt the agent mid-response (Phase 2 barge-in)."""
    pass


@dataclass(frozen=True)
class StartRecordingAction:
    """Tell the recorder to start capturing audio (PTT press)."""
    pass


@dataclass(frozen=True)
class StopRecordingAction:
    """Tell the recorder to stop capturing audio (PTT release)."""
    pass


@dataclass(frozen=True)
class LogAction:
    """Write a message to stdout."""
    message: str


# Union of all action types
Action = Union[
    PasteTextAction,
    RunCommandAction,
    SendEscapeKeyAction,
    StartRecordingAction,
    StopRecordingAction,
    StartAgentTurnAction,
    CancelAgentTurnAction,
    LogAction,
]
