from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .models import SignatureResult, TraceEvent


def evaluate_signatures(
    events: list[TraceEvent],
    *,
    context_token_limit: int,
) -> list[SignatureResult]:
    """Evaluate all seven MAST signatures over a window.

    Args:
        events: Recent normalized events in the current sliding window.
        context_token_limit: Token threshold for context overload.

    Returns:
        Signature score list in fixed order.
    """
    return [
        _tool_use_loop(events),
        _repeated_invalid_action(events),
        _no_progress_stagnation(events),
        _context_overload(events, context_token_limit=context_token_limit),
        _malformed_tool_output(events),
        _role_disobedience(events),
        _instruction_drift(events),
    ]


def _tool_use_loop(events: list[TraceEvent]) -> SignatureResult:
    calls = [
        (event.tool_name, event.tool_input_hash)
        for event in events
        if event.kind == "tool_call" and event.tool_name and event.tool_input_hash
    ]
    if not calls:
        return SignatureResult(name="tool_use_loop", score=0.0)
    top_call, top_count = Counter(calls).most_common(1)[0]
    score = min(1.0, max(0.0, (top_count - 2) / 2))
    return SignatureResult(
        name="tool_use_loop",
        score=score,
        evidence={"tool": top_call[0], "count": top_count},
    )


def _repeated_invalid_action(events: list[TraceEvent]) -> SignatureResult:
    streak = _max_streak(
        event.is_error for event in events if event.kind == "tool_result"
    )
    score = 1.0 if streak >= 2 else 0.0
    return SignatureResult(
        name="repeated_invalid_action",
        score=score,
        evidence={"max_error_streak": streak},
    )


def _no_progress_stagnation(events: list[TraceEvent]) -> SignatureResult:
    if len(events) < 8:
        return SignatureResult(name="no_progress_stagnation", score=0.0)

    last_window = events[-min(12, len(events)) :]
    tool_results = [event for event in last_window if event.kind == "tool_result"]
    if len(tool_results) < 3:
        return SignatureResult(
            name="no_progress_stagnation",
            score=0.0,
            evidence={
                "checked_steps": len(last_window),
                "reason": "insufficient_tool_results",
            },
        )

    has_write = any(event.file_write for event in last_window)
    successes = sum(
        1 for event in tool_results if not event.is_error and not event.malformed_output
    )
    errors = sum(1 for event in tool_results if event.is_error)
    error_ratio = errors / max(1, len(tool_results))

    if has_write or successes >= 2:
        score = 0.0
    elif successes == 0 and error_ratio >= 0.8:
        score = 1.0
    elif error_ratio >= 0.6:
        score = 0.8
    else:
        score = 0.3

    return SignatureResult(
        name="no_progress_stagnation",
        score=score,
        evidence={
            "checked_steps": len(last_window),
            "tool_results": len(tool_results),
            "has_file_write": has_write,
            "successful_tool_results": successes,
            "error_ratio": round(error_ratio, 2),
        },
    )


def _context_overload(
    events: list[TraceEvent],
    *,
    context_token_limit: int,
) -> SignatureResult:
    total_input_tokens = sum(event.input_tokens for event in events)
    score = min(1.0, total_input_tokens / max(1, context_token_limit))
    return SignatureResult(
        name="context_overload",
        score=score,
        evidence={"input_tokens": total_input_tokens, "limit": context_token_limit},
    )


def _malformed_tool_output(events: list[TraceEvent]) -> SignatureResult:
    malformed = [
        event
        for event in events
        if event.kind == "tool_result" and event.malformed_output
    ]
    score = min(1.0, len(malformed) / 2)
    return SignatureResult(
        name="malformed_tool_output",
        score=score,
        evidence={"count": len(malformed)},
    )


def _role_disobedience(events: list[TraceEvent]) -> SignatureResult:
    forbidden = {
        "write_file",
        "edit_file",
        "execute",
        "run_in_terminal",
        "bash",
        "python",
    }
    violations = [
        event
        for event in events
        if event.kind == "tool_call"
        and event.tool_name in forbidden
        and event.role
        and any(tag in event.role.lower() for tag in ("planner", "reviewer"))
    ]
    score = 1.0 if violations else 0.0
    return SignatureResult(
        name="role_disobedience",
        score=score,
        evidence={"count": len(violations)},
    )


def _instruction_drift(events: list[TraceEvent]) -> SignatureResult:
    task_labeled = [
        event
        for event in events
        if event.kind == "tool_call" and event.on_task is not None
    ]
    if not task_labeled:
        return SignatureResult(name="instruction_drift", score=0.0)
    off_task = sum(1 for event in task_labeled if event.on_task is False)
    on_task = sum(1 for event in task_labeled if event.on_task is True)
    total = off_task + on_task
    if total < 10:
        score = 0.0
    else:
        score = 1.0 if off_task > (on_task * 2) else 0.0
    return SignatureResult(
        name="instruction_drift",
        score=score,
        evidence={"on_task": on_task, "off_task": off_task},
    )


def _max_streak(values: Iterable[bool]) -> int:
    best = 0
    current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best
