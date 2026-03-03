"""Lightweight latency tracer for dictation turns.

Inspired by shuo's tracer.  Records spans (time ranges) and markers
(point-in-time events) for each dictation turn, enabling performance
analysis across different hardware setups.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


TRACE_DIR = Path.home() / ".dictation-traces"


@dataclass
class Span:
    name: str
    start_ms: float
    end_ms: float = 0.0

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass
class Marker:
    name: str
    time_ms: float


@dataclass
class Turn:
    turn_number: int
    transcript: str = ""
    spans: list[Span] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)
    cancelled: bool = False


class Tracer:
    """Records per-turn latency data for the dictation session."""

    def __init__(self):
        self._turns: list[Turn] = []
        self._t0 = time.monotonic()
        self._open_spans: dict[tuple[int, str], Span] = {}

    def _now_ms(self) -> float:
        return (time.monotonic() - self._t0) * 1000

    def begin_turn(self, transcript: str = "") -> int:
        """Start a new turn.  Returns the turn number."""
        turn_num = len(self._turns)
        self._turns.append(Turn(turn_number=turn_num, transcript=transcript))
        return turn_num

    def begin(self, turn: int, name: str) -> None:
        """Start a named span within a turn."""
        if turn < 0 or turn >= len(self._turns):
            return
        span = Span(name=name, start_ms=self._now_ms())
        self._turns[turn].spans.append(span)
        self._open_spans[(turn, name)] = span

    def end(self, turn: int, name: str) -> None:
        """End a named span within a turn."""
        key = (turn, name)
        span = self._open_spans.pop(key, None)
        if span:
            span.end_ms = self._now_ms()

    def mark(self, turn: int, name: str) -> None:
        """Record a point-in-time marker within a turn."""
        if turn < 0 or turn >= len(self._turns):
            return
        self._turns[turn].markers.append(
            Marker(name=name, time_ms=self._now_ms())
        )

    def cancel_turn(self, turn: int) -> None:
        """Mark a turn as cancelled and close all open spans."""
        if turn < 0 or turn >= len(self._turns):
            return
        self._turns[turn].cancelled = True
        now = self._now_ms()
        keys_to_close = [k for k in self._open_spans if k[0] == turn]
        for key in keys_to_close:
            self._open_spans[key].end_ms = now
            del self._open_spans[key]

    def save(self, session_id: str) -> Optional[Path]:
        """Save all turns to a JSON file.  Returns the path."""
        if not self._turns:
            return None
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        path = TRACE_DIR / f"{session_id}.json"
        data = []
        for turn in self._turns:
            data.append({
                "turn": turn.turn_number,
                "transcript": turn.transcript,
                "cancelled": turn.cancelled,
                "spans": [
                    {"name": s.name, "start_ms": round(s.start_ms, 1),
                     "end_ms": round(s.end_ms, 1),
                     "duration_ms": round(s.duration_ms, 1)}
                    for s in turn.spans
                ],
                "markers": [
                    {"name": m.name, "time_ms": round(m.time_ms, 1)}
                    for m in turn.markers
                ],
            })
        path.write_text(json.dumps(data, indent=2))
        return path

    def summary(self) -> str:
        """Return a human-readable summary of the session."""
        if not self._turns:
            return "No turns recorded."
        lines = []
        for turn in self._turns:
            status = " (cancelled)" if turn.cancelled else ""
            lines.append(f"Turn {turn.turn_number}{status}: {turn.transcript!r}")
            for span in turn.spans:
                lines.append(f"  {span.name}: {span.duration_ms:.0f}ms")
            for marker in turn.markers:
                lines.append(f"  @ {marker.name}: {marker.time_ms:.0f}ms")
        return "\n".join(lines)
