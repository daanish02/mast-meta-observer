from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from .config import (
    get_default_persistence,
    get_default_threshold,
    get_default_window,
)


class RunMode(StrEnum):
    """Execution mode for the wrapper."""

    OBSERVER = "observer"
    BASELINE = "baseline"


@dataclass(slots=True)
class ObserverConfig:
    """Configuration for observer detection and rollback policy.

    Args:
        window: Sliding window size for signature evaluation.
        threshold: Score threshold that counts as a failure window.
        persistence: Number of consecutive windows above threshold before rollback.
        max_rollbacks: Maximum rollbacks allowed in a run.
        context_token_limit: Token limit used by the context overload signature.
    """

    window: int = field(default_factory=get_default_window)
    threshold: float = field(default_factory=get_default_threshold)
    persistence: int = field(default_factory=get_default_persistence)
    max_rollbacks: int = 3
    context_token_limit: int = 80_000


@dataclass(slots=True)
class TraceEvent:
    """Normalized event used by the observer signatures."""

    ts: str
    kind: str
    role: str | None = None
    tool_name: str | None = None
    tool_input_hash: str | None = None
    is_error: bool = False
    file_write: bool = False
    malformed_output: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    on_task: bool | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(slots=True)
class SignatureResult:
    """Result of one MAST signature evaluation."""

    name: str
    score: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert signature result to dictionary."""
        return asdict(self)


@dataclass(slots=True)
class RollbackRecord:
    """Metadata for one rollback event."""

    ts: str
    reason: str
    signature: str
    score: float
    restored_checkpoint_id: str | None
    recovered_after_events: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert rollback record to dictionary."""
        return asdict(self)


@dataclass(slots=True)
class RunReport:
    """Final run summary persisted at `created-projects/<project>/.mast/report.json`."""

    project: str
    mode: str
    task: str
    model: str
    started_at: str
    completed_at: str | None = None
    success: bool = False
    status: str = "running"
    total_events: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    rollbacks: list[RollbackRecord] = field(default_factory=list)
    mttr_events: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        data = asdict(self)
        data["rollbacks"] = [item.to_dict() for item in self.rollbacks]
        return data


def utc_now_iso() -> str:
    """Return an RFC3339 UTC timestamp with seconds precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
