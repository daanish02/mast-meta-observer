from __future__ import annotations

from mast_meta_observer.models import TraceEvent, utc_now_iso
from mast_meta_observer.signatures import evaluate_signatures


def _event(**kwargs):
    base = {
        "ts": utc_now_iso(),
        "kind": "tool_call",
    }
    base.update(kwargs)
    return TraceEvent(**base)


def test_tool_use_loop_signature_triggers() -> None:
    events = [
        _event(kind="tool_call", tool_name="read_file", tool_input_hash="abc"),
        _event(kind="tool_call", tool_name="read_file", tool_input_hash="abc"),
        _event(kind="tool_call", tool_name="read_file", tool_input_hash="abc"),
    ]
    scores = {
        item.name: item.score
        for item in evaluate_signatures(events, context_token_limit=80_000)
    }
    assert scores["tool_use_loop"] >= 0.5


def test_stagnation_signature_triggers_without_file_writes() -> None:
    events = [
        _event(kind="tool_result", file_write=False),
        _event(kind="tool_result", file_write=False),
        _event(kind="tool_result", file_write=False),
        _event(kind="tool_result", file_write=False),
    ]
    scores = {
        item.name: item.score
        for item in evaluate_signatures(events, context_token_limit=80_000)
    }
    assert scores["no_progress_stagnation"] == 1.0


def test_context_overload_hits_threshold() -> None:
    events = [
        _event(kind="model_text", input_tokens=30_000),
        _event(kind="model_text", input_tokens=30_000),
        _event(kind="model_text", input_tokens=20_000),
    ]
    scores = {
        item.name: item.score
        for item in evaluate_signatures(events, context_token_limit=80_000)
    }
    assert scores["context_overload"] == 1.0


def test_instruction_drift_when_off_task_dominates() -> None:
    events = [
        _event(kind="tool_call", on_task=False),
        _event(kind="tool_call", on_task=False),
        _event(kind="tool_call", on_task=True),
    ]
    scores = {
        item.name: item.score
        for item in evaluate_signatures(events, context_token_limit=80_000)
    }
    assert scores["instruction_drift"] == 1.0
