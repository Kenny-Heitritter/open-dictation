"""Unit tests for the pure state machine.

Following shuo's pattern: test every transition exhaustively.
The state machine has zero I/O, so these tests are fast and deterministic.
"""

import pytest

from dictation.types import (
    AppState, Phase,
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
from dictation.state import process_event


# ── Helpers ──────────────────────────────────────────────────────────

def idle() -> AppState:
    return AppState(phase=Phase.IDLE)


def listening(ptt=False, wake=False) -> AppState:
    return AppState(phase=Phase.LISTENING, ptt_active=ptt, wake_word_active=wake)


def processing() -> AppState:
    return AppState(phase=Phase.PROCESSING)


def responding() -> AppState:
    return AppState(phase=Phase.RESPONDING)


def _dummy_check_command(text):
    if "open github" in text.lower():
        return {"name": "Open GitHub", "action": "echo test", "respond": "Opening GitHub", "patterns": []}
    return None


def _dummy_vocabulary(text):
    return text.replace("cubraid", "qBraid")


# ── Wake word tests ──────────────────────────────────────────────────

class TestWakeWord:
    def test_wake_word_from_idle(self):
        state, actions = process_event(idle(), WakeWordDetectedEvent())
        assert state.phase == Phase.LISTENING
        assert state.wake_word_active is True
        assert any(isinstance(a, LogAction) for a in actions)

    def test_wake_word_ignored_when_listening(self):
        state, actions = process_event(listening(), WakeWordDetectedEvent())
        assert state.phase == Phase.LISTENING
        assert actions == []

    def test_wake_word_ignored_when_responding(self):
        state, actions = process_event(responding(), WakeWordDetectedEvent())
        assert state.phase == Phase.RESPONDING
        assert actions == []


# ── Push-to-talk tests ───────────────────────────────────────────────

class TestPTT:
    def test_ptt_press_from_idle(self):
        state, actions = process_event(idle(), PTTPressedEvent())
        assert state.phase == Phase.LISTENING
        assert state.ptt_active is True
        assert any(isinstance(a, SendEscapeKeyAction) for a in actions)
        assert any(isinstance(a, StartRecordingAction) for a in actions)

    def test_ptt_release_stops_recording(self):
        state, actions = process_event(
            listening(ptt=True), PTTReleasedEvent()
        )
        assert state.phase == Phase.PROCESSING
        assert any(isinstance(a, StopRecordingAction) for a in actions)

    def test_ptt_release_ignored_when_not_held(self):
        state, actions = process_event(listening(), PTTReleasedEvent())
        assert state.phase == Phase.LISTENING

    def test_ptt_bargein_cancels_agent(self):
        state, actions = process_event(responding(), PTTPressedEvent())
        assert state.phase == Phase.LISTENING
        assert state.ptt_active is True
        assert any(isinstance(a, CancelAgentTurnAction) for a in actions)
        assert any(isinstance(a, SendEscapeKeyAction) for a in actions)
        assert any(isinstance(a, StartRecordingAction) for a in actions)


# ── Recording lifecycle ──────────────────────────────────────────────

class TestRecording:
    def test_recording_started_from_idle(self):
        state, actions = process_event(idle(), RecordingStartedEvent())
        assert state.phase == Phase.LISTENING

    def test_recording_stopped_transitions_to_processing(self):
        state, actions = process_event(listening(), RecordingStoppedEvent())
        assert state.phase == Phase.PROCESSING

    def test_recording_stopped_ignored_when_idle(self):
        state, actions = process_event(idle(), RecordingStoppedEvent())
        assert state.phase == Phase.IDLE


# ── Transcription tests ─────────────────────────────────────────────

class TestTranscription:
    def test_realtime_transcription_logged(self):
        state, actions = process_event(
            listening(), RealtimeTranscriptionEvent(text="hello world")
        )
        assert any(isinstance(a, LogAction) and "hello world" in a.message for a in actions)

    def test_empty_realtime_ignored(self):
        state, actions = process_event(
            listening(), RealtimeTranscriptionEvent(text="   ")
        )
        assert not any(isinstance(a, LogAction) for a in actions)

    def test_final_transcription_pastes_text(self):
        state, actions = process_event(
            processing(), TranscriptionReadyEvent(text="hello world")
        )
        assert state.phase == Phase.IDLE
        paste_actions = [a for a in actions if isinstance(a, PasteTextAction)]
        assert len(paste_actions) == 1
        assert paste_actions[0].text == "hello world "

    def test_empty_transcription_resets_to_idle(self):
        state, actions = process_event(
            processing(), TranscriptionReadyEvent(text="")
        )
        assert state.phase == Phase.IDLE
        assert any(isinstance(a, LogAction) and "Empty" in a.message for a in actions)

    def test_vocabulary_applied(self):
        state, actions = process_event(
            processing(),
            TranscriptionReadyEvent(text="cubraid is great"),
            apply_vocabulary=_dummy_vocabulary,
        )
        paste_actions = [a for a in actions if isinstance(a, PasteTextAction)]
        assert paste_actions[0].text == "qBraid is great "

    def test_command_matched(self):
        state, actions = process_event(
            processing(),
            TranscriptionReadyEvent(text="open github"),
            check_command=_dummy_check_command,
        )
        assert state.phase == Phase.IDLE
        cmd_actions = [a for a in actions if isinstance(a, RunCommandAction)]
        assert len(cmd_actions) == 1
        assert cmd_actions[0].command["name"] == "Open GitHub"
        # No paste when command matches
        assert not any(isinstance(a, PasteTextAction) for a in actions)


# ── Agent mode tests ─────────────────────────────────────────────────

class TestAgentMode:
    def test_transcription_starts_agent_turn(self):
        state, actions = process_event(
            processing(),
            TranscriptionReadyEvent(text="what is the weather"),
            agent_mode=True,
        )
        assert state.phase == Phase.RESPONDING
        agent_actions = [a for a in actions if isinstance(a, StartAgentTurnAction)]
        assert len(agent_actions) == 1
        assert agent_actions[0].transcript == "what is the weather"

    def test_agent_done_returns_to_idle(self):
        state, actions = process_event(
            responding(), AgentResponseDoneEvent()
        )
        assert state.phase == Phase.IDLE

    def test_agent_done_ignored_when_not_responding(self):
        state, actions = process_event(idle(), AgentResponseDoneEvent())
        assert state.phase == Phase.IDLE

    def test_commands_take_priority_over_agent(self):
        """Voice commands should still work in agent mode."""
        state, actions = process_event(
            processing(),
            TranscriptionReadyEvent(text="open github"),
            check_command=_dummy_check_command,
            agent_mode=True,
        )
        assert state.phase == Phase.IDLE
        assert any(isinstance(a, RunCommandAction) for a in actions)
        assert not any(isinstance(a, StartAgentTurnAction) for a in actions)


# ── Immutability tests ───────────────────────────────────────────────

class TestImmutability:
    def test_state_not_mutated(self):
        original = idle()
        process_event(original, WakeWordDetectedEvent())
        assert original.phase == Phase.IDLE  # original unchanged

    def test_sequential_transitions(self):
        s1, _ = process_event(idle(), WakeWordDetectedEvent())
        assert s1.phase == Phase.LISTENING

        s2, _ = process_event(s1, RecordingStoppedEvent())
        assert s2.phase == Phase.PROCESSING

        s3, _ = process_event(s2, TranscriptionReadyEvent(text="hello"))
        assert s3.phase == Phase.IDLE

    def test_full_agent_cycle(self):
        s1, _ = process_event(idle(), PTTPressedEvent())
        assert s1.phase == Phase.LISTENING
        assert s1.ptt_active is True

        s2, _ = process_event(s1, PTTReleasedEvent())
        assert s2.phase == Phase.PROCESSING

        s3, a3 = process_event(
            s2, TranscriptionReadyEvent(text="ask about weather"),
            agent_mode=True,
        )
        assert s3.phase == Phase.RESPONDING
        assert any(isinstance(a, StartAgentTurnAction) for a in a3)

        s4, _ = process_event(s3, AgentResponseDoneEvent())
        assert s4.phase == Phase.IDLE
