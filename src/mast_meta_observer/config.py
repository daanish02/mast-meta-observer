from __future__ import annotations

import os
from typing import Any

DEFAULT_MODEL_SPEC = "openai:gpt-4.1-mini"
DEFAULT_REASONING_EFFORT = "none"
DEFAULT_WINDOW = 3
DEFAULT_THRESHOLD = 0.75
DEFAULT_PERSISTENCE = 2


def get_default_model_spec() -> str:
    """Return the default model spec with optional env override."""
    value = os.environ.get("MAST_MODEL", DEFAULT_MODEL_SPEC).strip()
    return value or DEFAULT_MODEL_SPEC


def get_default_reasoning_effort() -> str:
    """Return default reasoning effort with optional env override."""
    value = os.environ.get("MAST_REASONING_EFFORT", DEFAULT_REASONING_EFFORT).strip()
    if not value:
        return DEFAULT_REASONING_EFFORT
    return value.lower()


def get_default_window() -> int:
    """Return default observer window with optional env override."""
    raw = os.environ.get("MAST_WINDOW")
    if raw is None:
        return DEFAULT_WINDOW
    try:
        value = int(raw)
        return value if value > 0 else DEFAULT_WINDOW
    except ValueError:
        return DEFAULT_WINDOW


def get_default_threshold() -> float:
    """Return default observer threshold with optional env override."""
    raw = os.environ.get("MAST_THRESHOLD")
    if raw is None:
        return DEFAULT_THRESHOLD
    try:
        value = float(raw)
        return value if 0.0 <= value <= 1.0 else DEFAULT_THRESHOLD
    except ValueError:
        return DEFAULT_THRESHOLD


def get_default_persistence() -> int:
    """Return default observer persistence with optional env override."""
    raw = os.environ.get("MAST_PERSISTENCE")
    if raw is None:
        return DEFAULT_PERSISTENCE
    try:
        value = int(raw)
        return value if value > 0 else DEFAULT_PERSISTENCE
    except ValueError:
        return DEFAULT_PERSISTENCE


def resolve_runtime_model(model_spec: str, reasoning_effort: str) -> str | Any:
    """Build a runtime model object when extra OpenAI reasoning settings are needed.

    Falls back to the plain model spec string to preserve compatibility.
    """
    if not model_spec.startswith("openai:"):
        return model_spec

    normalized_effort = reasoning_effort.strip().lower()
    if normalized_effort in {"", "none", "off", "disabled"}:
        return model_spec

    from langchain.chat_models import init_chat_model

    try:
        return init_chat_model(
            model_spec,
            use_responses_api=True,
            reasoning={"effort": normalized_effort},
        )
    except TypeError:
        try:
            return init_chat_model(
                model_spec,
                use_responses_api=True,
                reasoning_effort=normalized_effort,
            )
        except TypeError:
            return model_spec
