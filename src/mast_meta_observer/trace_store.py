from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from .models import TraceEvent


class TraceStore:
    """Append-only JSON trace writer with in-memory sliding window."""

    def __init__(self, trace_path: Path, window_size: int) -> None:
        self._trace_path = trace_path
        self._trace_jsonl_path = trace_path.with_suffix(".jsonl")
        self._window_size = window_size
        self._window: deque[TraceEvent] = deque(maxlen=window_size)
        self._events: list[TraceEvent] = []
        self._count = 0
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._trace_jsonl_path.write_text("", encoding="utf-8")

    @property
    def count(self) -> int:
        """Return the total number of events written."""
        return self._count

    def append(self, event: TraceEvent) -> None:
        """Persist one event and update the sliding window."""
        self._window.append(event)
        self._events.append(event)
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=True))
            handle.write("\n")
        self._count += 1

    def finalize(self) -> None:
        """Write compact final trace JSON once at end of run."""
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._trace_path.write_text(
            json.dumps(
                [item.to_dict() for item in self._events], indent=2, ensure_ascii=True
            ),
            encoding="utf-8",
        )

    def window(self) -> list[TraceEvent]:
        """Return a copy of the current event window."""
        return list(self._window)
