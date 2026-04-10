from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import uuid4

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from .checkpoints import CheckpointManager
from .config import (
    get_default_model_spec,
    get_default_persistence,
    get_default_reasoning_effort,
    get_default_threshold,
    get_default_window,
    resolve_runtime_model,
)
from .models import (
    ObserverConfig,
    RollbackRecord,
    RunMode,
    RunReport,
    TraceEvent,
    utc_now_iso,
)
from .observer import ObserverEngine
from .paths import ensure_output_paths
from .trace_store import TraceStore

FILE_WRITE_TOOLS = {
    "write_file",
    "edit_file",
    "create_file",
    "mkdir",
    "cp",
    "mv",
}

STOPWORDS = {
    "build",
    "with",
    "using",
    "into",
    "from",
    "that",
    "this",
    "your",
    "have",
    "will",
    "the",
    "and",
    "for",
}


async def run_task(
    task: str,
    *,
    project: str,
    run_id: str,
    mode: RunMode,
    model: str,
    reasoning_effort: str,
    observer_config: ObserverConfig,
    root: Path,
) -> RunReport:
    """Execute one task via DeepAgents with optional observer supervision.

    Args:
        task: User task prompt.
        project: Project name under `created-projects`.
        mode: Observer or baseline mode.
        model: LangChain model identifier.
        observer_config: Observer policy options.
        root: Root folder of this wrapper project.

    Returns:
        Completed run report.
    """
    outputs = ensure_output_paths(root, project, run_id)
    trace_store = TraceStore(outputs.trace_path, window_size=observer_config.window)

    report = RunReport(
        project=project,
        mode=mode.value,
        task=task,
        model=model,
        notes=[f"run_id={run_id}"],
        started_at=utc_now_iso(),
    )
    report.notes.append(f"reasoning_effort={reasoning_effort}")

    if not _has_model_auth():
        _write_offline_scaffold(outputs.project_dir, task)
        _write_offline_trace_and_report(
            trace_store=trace_store,
            report=report,
            task=task,
            project_dir=outputs.project_dir,
        )
        report.completed_at = utc_now_iso()
        report.success = True
        report.status = "offline-demo"
        trace_store.finalize()
        outputs.report_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return report

    backend = FilesystemBackend(root_dir=str(outputs.project_dir), virtual_mode=True)
    checkpointer = MemorySaver() if mode == RunMode.OBSERVER else None

    runtime_model = resolve_runtime_model(model, reasoning_effort)
    agent = create_deep_agent(
        model=runtime_model,
        backend=backend,
        checkpointer=checkpointer,
        system_prompt=_build_builder_system_prompt(task),
    )

    task_keywords = _extract_keywords(task)
    base_config: dict[str, Any] = {"configurable": {"thread_id": str(uuid4())}}

    observer: ObserverEngine | None = None
    checkpoint_manager: CheckpointManager | None = None
    if mode == RunMode.OBSERVER and checkpointer is not None:
        observer = ObserverEngine(observer_config)
        checkpoint_manager = CheckpointManager(
            checkpointer=checkpointer,
            index_path=outputs.checkpoint_index_path,
            base_config=base_config,
        )

    rollback_recovery_events: list[int] = []
    recovery_hint: str | None = None
    attempts = 0
    active_config = dict(base_config)
    minimum_rollback_events = max(observer_config.window * 4, 60)
    has_written_file = False
    warned_no_checkpoint = False

    while True:
        attempts += 1
        user_prompt = task
        if recovery_hint:
            user_prompt = (
                f"{task}\n\n"
                f"Recovery hint: {recovery_hint} "
                "Use a different strategy than the previous failed attempt."
            )

        stream_input = {"messages": [{"role": "user", "content": user_prompt}]}
        rollback_decision: tuple[str, float, str] | None = None
        previous_usage_input = 0
        previous_usage_output = 0

        async for chunk in agent.astream(
            stream_input,
            stream_mode=["messages", "updates"],
            subgraphs=True,
            config=active_config,
            durability="exit",
        ):
            (
                usage_input,
                usage_output,
                _has_tool_call,
            ) = _token_usage_from_chunk(chunk)
            (
                input_tokens,
                output_tokens,
                previous_usage_input,
                previous_usage_output,
            ) = _usage_deltas(
                usage_input,
                usage_output,
                previous_usage_input,
                previous_usage_output,
            )

            report.total_input_tokens += input_tokens
            report.total_output_tokens += output_tokens
            report.total_tokens += input_tokens + output_tokens

            for event in _events_from_chunk(chunk, task_keywords=task_keywords):
                trace_store.append(event)
                report.total_events += 1
                has_written_file = has_written_file or event.file_write

                if observer is None:
                    continue

                if event.kind == "model_text":
                    continue

                if report.total_events < observer_config.window:
                    continue

                if not has_written_file:
                    continue

                decision = observer.evaluate(trace_store.window())
                if (
                    observer.should_mark_stable(decision.signature_scores)
                    and checkpoint_manager is not None
                ):
                    await checkpoint_manager.mark_stable(
                        reason="low-risk-window",
                        event_count=report.total_events,
                    )

                if report.total_events < minimum_rollback_events:
                    continue

                if decision.trigger_rollback and decision.signature is not None:
                    if (
                        checkpoint_manager is not None
                        and not checkpoint_manager.has_stable_checkpoint()
                    ):
                        if not warned_no_checkpoint:
                            report.notes.append(
                                "Observer trigger skipped because no stable checkpoint was available yet."
                            )
                            warned_no_checkpoint = True
                        continue
                    rollback_decision = (
                        decision.signature.name,
                        decision.signature.score,
                        decision.reason or "policy-trigger",
                    )
                    break
            if rollback_decision is not None:
                break

        if mode == RunMode.BASELINE:
            report.success = True
            report.status = "completed"
            break

        if rollback_decision is None:
            report.success = True
            report.status = "completed"
            break

        signature_name, signature_score, reason = rollback_decision
        restored_id = None
        if checkpoint_manager is not None:
            restored_id = checkpoint_manager.record_restore(
                signature=signature_name,
                score=signature_score,
                reason=reason,
            )
        report.rollbacks.append(
            RollbackRecord(
                ts=utc_now_iso(),
                reason=reason,
                signature=signature_name,
                score=signature_score,
                restored_checkpoint_id=restored_id,
            )
        )

        if checkpoint_manager is not None:
            restore_config = checkpoint_manager.build_restore_config()
            if restore_config is not None:
                active_config = restore_config

        rollback_recovery_events.append(report.total_events)
        recovery_hint = (
            f"Detected '{signature_name}' (score={signature_score:.2f}). "
            "Avoid repeating failing tool calls and verify each step before continuing."
        )

        report.notes.append(
            f"Healing hint injected after {signature_name}; continuing build."
        )

        if len(report.rollbacks) >= observer_config.max_rollbacks:
            report.notes.append(
                "Maximum rollback interventions reached; continuing without further observer-triggered resets."
            )
            observer = None

    report.completed_at = utc_now_iso()
    if report.rollbacks and rollback_recovery_events:
        report.mttr_events = _compute_mttr_events(
            rollback_start_events=rollback_recovery_events,
            end_event_count=report.total_events,
        )

    trace_store.finalize()
    outputs.report_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return report


async def run_cli() -> int:
    """CLI entrypoint for `python run.py ...`."""
    parser = argparse.ArgumentParser(description="MAST Meta-Observer for DeepAgents")
    parser.add_argument("task", type=str, help="Task prompt to execute")
    parser.add_argument(
        "--project",
        type=str,
        default="my-project",
        help="Folder under created-projects/",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[mode.value for mode in RunMode],
        default=RunMode.BASELINE.value,
        help="Execution mode: baseline or observer",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=get_default_model_spec(),
        help="LangChain model string",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=get_default_reasoning_effort(),
        help="Reasoning effort for supported providers (e.g., low|medium|high)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=get_default_window(),
        help="Sliding window size",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=get_default_threshold(),
        help="Failure threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--persistence",
        type=int,
        default=get_default_persistence(),
        help="Consecutive windows above threshold before rollback",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Unique run snapshot folder under created-projects/<project>/runs/",
    )
    args = parser.parse_args()

    run_id = args.run_id or _generate_run_id()

    report = await run_task(
        args.task,
        project=args.project,
        run_id=run_id,
        mode=RunMode(args.mode),
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        observer_config=ObserverConfig(
            window=args.window,
            threshold=args.threshold,
            persistence=args.persistence,
        ),
        root=Path(__file__).resolve().parents[2],
    )

    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=True))
    return 0 if report.success else 1


async def run_benchmark_cli() -> int:
    """CLI entrypoint for `python benchmark.py ...`."""
    parser = argparse.ArgumentParser(description="Benchmark observer vs baseline")
    parser.add_argument("task", type=str, help="Task prompt to execute")
    parser.add_argument(
        "--project",
        type=str,
        default="benchmark",
        help="Benchmark project prefix",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=get_default_model_spec(),
        help="LangChain model string",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=get_default_reasoning_effort(),
        help="Reasoning effort for supported providers (e.g., low|medium|high)",
    )
    parser.add_argument("--window", type=int, default=get_default_window())
    parser.add_argument("--threshold", type=float, default=get_default_threshold())
    parser.add_argument("--persistence", type=int, default=get_default_persistence())
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional shared benchmark run id prefix",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    benchmark_run_id = args.run_id or _generate_run_id()
    config = ObserverConfig(
        window=args.window,
        threshold=args.threshold,
        persistence=args.persistence,
    )

    observer_report = await run_task(
        args.task,
        project=args.project,
        run_id=f"{benchmark_run_id}-observer",
        mode=RunMode.OBSERVER,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        observer_config=config,
        root=root,
    )

    baseline_report = await run_task(
        args.task,
        project=args.project,
        run_id=f"{benchmark_run_id}-baseline",
        mode=RunMode.BASELINE,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        observer_config=config,
        root=root,
    )

    benchmark_dir = (
        root / "created-projects" / args.project / ".mast" / benchmark_run_id
    )
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    comparison = {
        "task": args.task,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "observer": observer_report.to_dict(),
        "baseline": baseline_report.to_dict(),
        "delta": {
            "events": observer_report.total_events - baseline_report.total_events,
            "input_tokens": observer_report.total_input_tokens
            - baseline_report.total_input_tokens,
            "output_tokens": observer_report.total_output_tokens
            - baseline_report.total_output_tokens,
            "total_tokens": observer_report.total_tokens - baseline_report.total_tokens,
            "rollbacks": len(observer_report.rollbacks),
            "observer_success": observer_report.success,
            "baseline_success": baseline_report.success,
        },
    }
    (benchmark_dir / "benchmark_report.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2, ensure_ascii=True))
    return 0


def _events_from_chunk(
    chunk: object,
    *,
    task_keywords: set[str],
) -> Iterable[TraceEvent]:
    if not isinstance(chunk, tuple) or len(chunk) != 3:
        return []
    namespace, stream_mode, data = chunk
    role = _role_from_namespace(namespace)

    if stream_mode == "updates" and isinstance(data, dict):
        if "__interrupt__" in data:
            return [
                TraceEvent(
                    ts=utc_now_iso(),
                    kind="interrupt",
                    role=role,
                    details={"count": len(data.get("__interrupt__", []))},
                )
            ]
        return []

    if stream_mode != "messages":
        return []

    if not isinstance(data, tuple) or len(data) != 2:
        return []

    message_obj, _metadata = data
    events: list[TraceEvent] = []

    if isinstance(message_obj, AIMessage):
        usage = getattr(message_obj, "usage_metadata", None)
        input_tokens = 0
        output_tokens = 0
        if isinstance(usage, dict):
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
        blocks = getattr(message_obj, "content_blocks", []) or []

        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_call":
                tool_name = block.get("name")
                if not tool_name:
                    continue
                tool_args = block.get("args")
                text_for_task = f"{tool_name} {tool_args}"
                events.append(
                    TraceEvent(
                        ts=utc_now_iso(),
                        kind="tool_call",
                        role=role,
                        tool_name=str(tool_name) if tool_name else None,
                        tool_input_hash=_hash_payload(tool_args),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        on_task=_is_on_task(text_for_task, task_keywords),
                        details={"args": _safe_json(tool_args)},
                    )
                )
            elif block_type == "text":
                text = str(block.get("text") or "")
                if text:
                    events.append(
                        TraceEvent(
                            ts=utc_now_iso(),
                            kind="model_text",
                            role=role,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            details={"preview": text[:200]},
                        )
                    )

    elif isinstance(message_obj, ToolMessage):
        content = _content_to_text(message_obj.content)
        tool_name = getattr(message_obj, "name", None)
        events.append(
            TraceEvent(
                ts=utc_now_iso(),
                kind="tool_result",
                role=role,
                tool_name=str(tool_name) if tool_name else None,
                is_error=_is_error_text(content),
                file_write=(str(tool_name) in FILE_WRITE_TOOLS),
                malformed_output=_is_malformed_output(content),
                on_task=_is_on_task(f"{tool_name} {content[:120]}", task_keywords),
                details={"preview": content[:300]},
            )
        )

    return events


def _extract_keywords(task: str) -> set[str]:
    words = [token.lower().strip(".,:;!?()[]{}\"'") for token in task.split()]
    return {
        token
        for token in words
        if len(token) >= 4 and token.isascii() and token not in STOPWORDS
    }


def _is_on_task(text: str, task_keywords: set[str]) -> bool:
    if not task_keywords:
        return True
    haystack = text.lower()
    return any(keyword in haystack for keyword in task_keywords)


def _hash_payload(payload: Any) -> str:
    raw = _safe_json(payload)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return digest[:16]


def _safe_json(payload: Any) -> str:
    try:
        return json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    except TypeError:
        return repr(payload)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part) for part in content)
    return str(content)


def _is_error_text(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered for token in ("error", "exception", "traceback", "failed")
    )


def _is_malformed_output(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return True
    if cleaned.lower() == "none":
        return True
    if cleaned.startswith("{") or cleaned.startswith("["):
        try:
            json.loads(cleaned)
        except json.JSONDecodeError:
            return True
    return False


def _role_from_namespace(namespace: Any) -> str:
    if not namespace:
        return "main"
    if isinstance(namespace, tuple):
        return "/".join(str(item) for item in namespace)
    if isinstance(namespace, list):
        return "/".join(str(item) for item in namespace)
    return str(namespace)


def _compute_mttr_events(
    *,
    rollback_start_events: list[int],
    end_event_count: int,
) -> float:
    if not rollback_start_events:
        return 0.0
    recoveries: list[int] = []
    for start in rollback_start_events:
        recoveries.append(max(0, end_event_count - start))
    return sum(recoveries) / len(recoveries)


def _generate_run_id() -> str:
    """Generate a unique run identifier for snapshotting outputs."""
    return f"run-{uuid4().hex[:8]}"


def _build_builder_system_prompt(task: str) -> str:
    """Build a filesystem-first instruction prompt for DeepAgents."""
    return (
        "You are a small autonomous software builder. "
        "Your first job is to create a working codebase for the user's task. "
        "Write all generated files in the current run workspace using virtual paths like /app/main.py and /README.md. "
        "Never use Windows absolute paths like C:/... . "
        "Use the available filesystem tools to create source files, templates, and documentation before you do anything else. "
        "Favor the smallest complete implementation that satisfies the request. "
        "If a dependency or file is missing, create it directly in the workspace. "
        "After the codebase exists, you may refine and extend it. "
        f"User task: {task}"
    )


def _has_model_auth() -> bool:
    """Return whether a supported model API key is available."""
    return any(
        os.environ.get(name)
        for name in (
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "ANTHROPIC_API_KEY",
        )
    )


def _write_offline_trace_and_report(
    *,
    trace_store: TraceStore,
    report: RunReport,
    task: str,
    project_dir: Path,
) -> None:
    """Record a deterministic offline run when no model key is configured."""
    trace_store.append(
        TraceEvent(
            ts=utc_now_iso(),
            kind="offline_start",
            details={"task": task},
        )
    )
    for path in sorted(project_dir.rglob("*")):
        if path.is_file():
            trace_store.append(
                TraceEvent(
                    ts=utc_now_iso(),
                    kind="file_write",
                    file_write=True,
                    details={
                        "path": str(path.relative_to(project_dir)).replace("\\", "/")
                    },
                )
            )
    report.total_events = trace_store.count
    report.total_input_tokens = 0
    report.total_output_tokens = 0
    report.total_tokens = 0
    report.notes.append(
        "No model API key was detected; generated an offline scaffold instead of running DeepAgents."
    )


def _token_usage_from_chunk(chunk: object) -> tuple[int, int, bool]:
    """Extract input/output token counts from a streamed chunk.

    Returns (input_tokens, output_tokens, has_tool_call).
    """
    if not isinstance(chunk, tuple) or len(chunk) != 3:
        return (0, 0, False)

    _namespace, stream_mode, data = chunk
    if stream_mode != "messages":
        return (0, 0, False)

    if not isinstance(data, tuple) or len(data) != 2:
        return (0, 0, False)

    message_obj, _metadata = data
    if not isinstance(message_obj, AIMessage):
        return (0, 0, False)

    usage = getattr(message_obj, "usage_metadata", None)
    if not isinstance(usage, dict):
        return (0, 0, _ai_message_has_tool_call(message_obj))

    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    has_tool_call = _ai_message_has_tool_call(message_obj)
    return (input_tokens, output_tokens, has_tool_call)


def _usage_deltas(
    current_input: int,
    current_output: int,
    previous_input: int,
    previous_output: int,
) -> tuple[int, int, int, int]:
    """Convert potentially cumulative usage values to safe deltas."""
    if current_input == 0 and current_output == 0:
        return (0, 0, previous_input, previous_output)

    if current_input >= previous_input and current_output >= previous_output:
        delta_input = current_input - previous_input
        delta_output = current_output - previous_output
        return (delta_input, delta_output, current_input, current_output)

    # Provider likely emitted non-cumulative usage values for this chunk.
    return (
        max(0, current_input),
        max(0, current_output),
        max(0, current_input),
        max(0, current_output),
    )


def _ai_message_has_tool_call(message_obj: AIMessage) -> bool:
    """Return whether an AI message includes at least one tool call block."""
    blocks = getattr(message_obj, "content_blocks", []) or []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") in {
            "tool_call",
            "tool_call_chunk",
        }:
            return True
    return False


def _write_offline_scaffold(project_dir: Path, task: str) -> None:
    """Create a small starter project without calling a remote model."""
    task_lower = task.lower()
    if "flask" in task_lower:
        _write_file(
            project_dir / "app.py",
            """from flask import Flask, render_template\n\napp = Flask(__name__)\n\n\n@app.get('/')\ndef index():\n    return render_template('index.html')\n\n\nif __name__ == '__main__':\n    app.run(debug=True)\n""",
        )
        _write_file(
            project_dir / "models.py",
            """from dataclasses import dataclass\n\n\n@dataclass\nclass TodoItem:\n    id: int\n    title: str\n    completed: bool = False\n""",
        )
        _write_file(
            project_dir / "README.md",
            """# Offline scaffold\n\nThis project was generated without a connected model because no API key was available.\n""",
        )
        templates_dir = project_dir / "templates"
        templates_dir.mkdir(parents=True, exist_ok=True)
        _write_file(
            templates_dir / "index.html",
            """<!doctype html>\n<html>\n  <head><meta charset='utf-8'><title>Todo App</title></head>\n  <body><h1>Todo App</h1></body>\n</html>\n""",
        )
        static_dir = project_dir / "static"
        static_dir.mkdir(parents=True, exist_ok=True)
        _write_file(
            static_dir / "app.js",
            "console.log('offline scaffold');\n",
        )
        return

    _write_file(
        project_dir / "README.md",
        """# Offline scaffold\n\nThis project was generated without a connected model because no API key was available.\n""",
    )
    _write_file(
        project_dir / "app.py",
        """def main():\n    print('offline scaffold')\n\n\nif __name__ == '__main__':\n    main()\n""",
    )


def _write_file(path: Path, content: str) -> None:
    """Write text content to a file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
