"""Agent: orchestrates the LLM -> TTS -> speaker pipeline.

Following shuo's Agent pattern: each turn acquires services, chains
callbacks (LLM token -> TTS send -> audio play), and supports
cancellation at any point in the pipeline.
"""

from __future__ import annotations

import sys
import time
from typing import Callable, Optional

from dictation.services.llm import LLMService
from dictation.services.tts import TTSService
from dictation.tracer import Tracer


class Agent:
    """Manages the full agent response lifecycle.

    On each turn:
      1. User transcript -> LLM starts streaming tokens
      2. Each token -> TTS buffers and synthesizes on sentence boundaries
      3. TTS audio -> system speaker playback
      4. When playback finishes -> on_done callback fires

    Supports cancellation (barge-in) at any stage.
    """

    def __init__(
        self,
        on_done: Callable[[], None],
        tracer: Optional[Tracer] = None,
    ):
        self._on_done = on_done
        self._tracer = tracer

        self._llm: Optional[LLMService] = None
        self._tts: Optional[TTSService] = None
        self._current_turn: Optional[int] = None
        self._t0: float = 0

        # Build LLM service with the streaming callback chain
        self._llm = LLMService(
            on_token=self._on_llm_token,
            on_done=self._on_llm_done,
        )

    def preload_tts(self) -> None:
        """Preload the TTS model in background at startup.

        This avoids a multi-second delay on the first agent response.
        """
        self._tts = TTSService(on_done=self._on_playback_done)

    def start_turn(self, transcript: str) -> None:
        """Start a new agent response turn."""
        self._t0 = time.monotonic()

        if self._tracer:
            self._current_turn = self._tracer.begin_turn(transcript)
            self._tracer.begin(self._current_turn, "agent_turn")
            self._tracer.begin(self._current_turn, "llm")

        # Create fresh TTS for this turn (reuse preloaded model)
        self._tts = TTSService(on_done=self._on_playback_done)

        sys.stdout.write(f"\n[Agent] Generating response...\n")
        sys.stdout.flush()

        # Start LLM generation - tokens will flow via callbacks
        self._llm.start(transcript)

    def cancel_turn(self) -> None:
        """Cancel the current turn (barge-in).

        Cancels LLM -> TTS -> playback in order.
        Conversation history is preserved with partial response.
        """
        sys.stdout.write("[Agent] Turn cancelled (barge-in)\n")
        sys.stdout.flush()

        if self._llm:
            self._llm.cancel()
        if self._tts:
            self._tts.cancel()

        if self._tracer and self._current_turn is not None:
            self._tracer.cancel_turn(self._current_turn)
            self._current_turn = None

    def cleanup(self) -> None:
        """Final cleanup on shutdown."""
        if self._tts:
            self._tts.cancel()
        if self._llm:
            self._llm.cancel()

    # ── Callback chain (the streaming pipeline) ──────────────────

    def _on_llm_token(self, token: str) -> None:
        """Called for each LLM token.  Forwards to TTS."""
        if self._tracer and self._current_turn is not None:
            # Mark first token latency
            if not any(
                m.name == "llm_first_token"
                for m in self._tracer._turns[self._current_turn].markers
            ):
                self._tracer.mark(self._current_turn, "llm_first_token")
                elapsed = (time.monotonic() - self._t0) * 1000
                sys.stdout.write(f"[Agent] First token: {elapsed:.0f}ms\n")
                sys.stdout.flush()

        # Stream token to TTS
        sys.stdout.write(token)
        sys.stdout.flush()
        if self._tts:
            self._tts.send(token)

    def _on_llm_done(self) -> None:
        """Called when LLM finishes generating."""
        sys.stdout.write("\n")
        sys.stdout.flush()

        if self._tracer and self._current_turn is not None:
            self._tracer.end(self._current_turn, "llm")
            self._tracer.begin(self._current_turn, "tts_playback")

        # Flush remaining text to TTS
        if self._tts:
            self._tts.flush()

    def _on_playback_done(self) -> None:
        """Called when TTS playback finishes."""
        if self._tracer and self._current_turn is not None:
            self._tracer.end(self._current_turn, "tts_playback")
            self._tracer.end(self._current_turn, "agent_turn")

            # Log turn latency
            turn = self._tracer._turns[self._current_turn]
            for span in turn.spans:
                if span.name == "agent_turn":
                    sys.stdout.write(
                        f"[Agent] Turn complete: {span.duration_ms:.0f}ms\n"
                    )
                    sys.stdout.flush()

        self._current_turn = None
        self._on_done()
