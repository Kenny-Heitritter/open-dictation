"""Pure state machine for the dictation system.

Every transition is a pure function:
    (AppState, Event) -> (AppState, list[Action])

No I/O, no imports beyond the type definitions.  Fully unit-testable.
"""

from __future__ import annotations

from dictation.types import (
    AppState, Phase, Event, Action,
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


def process_event(
    state: AppState,
    event: Event,
    *,
    check_command=None,
    apply_vocabulary=None,
    agent_mode: bool = False,
) -> tuple[AppState, list[Action]]:
    """Advance the state machine by one event.

    Parameters
    ----------
    state : AppState
        Current immutable state.
    event : Event
        The event to process.
    check_command : callable, optional
        Function that checks if text matches a voice command.
        Signature: (text: str) -> dict | None
    apply_vocabulary : callable, optional
        Function that applies vocabulary corrections.
        Signature: (text: str) -> str
    agent_mode : bool
        If True, transcriptions are routed to the agent pipeline
        instead of being pasted as text.

    Returns
    -------
    (new_state, actions) where actions is a list of side-effects to dispatch.
    """
    actions: list[Action] = []

    # ── Wake word ────────────────────────────────────────────────
    if isinstance(event, WakeWordDetectedEvent):
        if state.phase == Phase.IDLE:
            actions.append(LogAction("\nWake word detected!"))
            return (
                AppState(phase=Phase.LISTENING, wake_word_active=True),
                actions,
            )
        return (state, actions)

    # ── Push-to-talk pressed ─────────────────────────────────────
    if isinstance(event, PTTPressedEvent):
        if state.phase in (Phase.IDLE, Phase.LISTENING):
            actions.append(SendEscapeKeyAction())
            actions.append(StartRecordingAction())
            actions.append(LogAction("\n[PTT] Recording..."))
            return (
                AppState(phase=Phase.LISTENING, ptt_active=True),
                actions,
            )
        if state.phase == Phase.RESPONDING:
            # Barge-in: interrupt agent, start recording
            actions.append(CancelAgentTurnAction())
            actions.append(SendEscapeKeyAction())
            actions.append(StartRecordingAction())
            actions.append(LogAction("\n[PTT] Barge-in, recording..."))
            return (
                AppState(phase=Phase.LISTENING, ptt_active=True),
                actions,
            )
        return (state, actions)

    # ── Push-to-talk released ────────────────────────────────────
    if isinstance(event, PTTReleasedEvent):
        if state.ptt_active:
            actions.append(StopRecordingAction())
            return (
                AppState(phase=Phase.PROCESSING),
                actions,
            )
        return (state, actions)

    # ── Recording started (callback from RealtimeSTT) ────────────
    if isinstance(event, RecordingStartedEvent):
        if state.phase in (Phase.IDLE, Phase.LISTENING):
            actions.append(LogAction("\nRecording..."))
            return (
                AppState(
                    phase=Phase.LISTENING,
                    ptt_active=state.ptt_active,
                    wake_word_active=state.wake_word_active,
                ),
                actions,
            )
        return (state, actions)

    # ── Recording stopped ────────────────────────────────────────
    if isinstance(event, RecordingStoppedEvent):
        if state.phase == Phase.LISTENING:
            actions.append(LogAction("Processing..."))
            return (AppState(phase=Phase.PROCESSING), actions)
        return (state, actions)

    # ── Realtime transcription (preview) ─────────────────────────
    if isinstance(event, RealtimeTranscriptionEvent):
        if event.text.strip():
            actions.append(LogAction(f"\r  >> {event.text.strip()}" + " " * 20))
        return (state, actions)

    # ── Final transcription ──────────────────────────────────────
    if isinstance(event, TranscriptionReadyEvent):
        text = event.text.strip()
        if not text:
            actions.append(LogAction("[Empty transcription, restarting...]"))
            return (AppState(phase=Phase.IDLE), actions)

        # Apply vocabulary corrections
        corrected = text
        if apply_vocabulary:
            corrected = apply_vocabulary(text)

        # Check for voice commands first
        if check_command:
            cmd = check_command(corrected)
            if cmd:
                if corrected != text:
                    actions.append(LogAction(f'  "{text}" -> "{corrected}"'))
                else:
                    actions.append(LogAction(f'  "{corrected}"'))
                actions.append(RunCommandAction(command=cmd))
                return (AppState(phase=Phase.IDLE), actions)

        # Log the transcription
        if corrected != text:
            actions.append(LogAction(f"  {text} -> {corrected}"))
        else:
            actions.append(LogAction(f"  {corrected}"))

        # Route to agent or paste directly
        if agent_mode:
            return (
                AppState(phase=Phase.RESPONDING),
                actions + [StartAgentTurnAction(transcript=corrected)],
            )
        else:
            return (
                AppState(phase=Phase.IDLE),
                actions + [PasteTextAction(text=corrected + " ")],
            )

    # ── Agent response done (Phase 2) ────────────────────────────
    if isinstance(event, AgentResponseDoneEvent):
        if state.phase == Phase.RESPONDING:
            return (AppState(phase=Phase.IDLE), actions)
        return (state, actions)

    # ── Shutdown ─────────────────────────────────────────────────
    if isinstance(event, ShutdownEvent):
        actions.append(LogAction("\nShutting down..."))
        return (state, actions)

    return (state, actions)
