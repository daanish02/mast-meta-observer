from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from .models import utc_now_iso


class CheckpointManager:
    """Track stable checkpoints and provide rollback configs."""

    def __init__(
        self,
        checkpointer: Any,
        index_path: Path,
        base_config: dict[str, Any],
    ) -> None:
        self._checkpointer = checkpointer
        self._index_path = index_path
        self._base_config = base_config
        self._stable_checkpoints: list[dict[str, Any]] = []
        self._restores: list[dict[str, Any]] = []
        self._load()

    async def mark_stable(self, reason: str, event_count: int) -> str | None:
        """Capture and persist the latest stable checkpoint id."""
        checkpoint_id = await self._latest_checkpoint_id(self._base_config)
        if not checkpoint_id:
            return None
        if (
            self._stable_checkpoints
            and self._stable_checkpoints[-1]["checkpoint_id"] == checkpoint_id
        ):
            return checkpoint_id
        entry = {
            "checkpoint_id": checkpoint_id,
            "ts": utc_now_iso(),
            "reason": reason,
            "event_count": event_count,
        }
        self._stable_checkpoints.append(entry)
        self._save()
        return checkpoint_id

    def has_stable_checkpoint(self) -> bool:
        """Return whether at least one stable checkpoint is available."""
        return bool(self._stable_checkpoints)

    def build_restore_config(self) -> dict[str, Any] | None:
        """Build a config that restores the latest stable checkpoint."""
        if not self._stable_checkpoints:
            return dict(self._base_config)
        checkpoint_id = self._stable_checkpoints[-1]["checkpoint_id"]
        configurable = dict(self._base_config.get("configurable", {}))
        configurable["checkpoint_id"] = checkpoint_id
        return {"configurable": configurable}

    def record_restore(self, signature: str, score: float, reason: str) -> str | None:
        """Record a rollback event and return the restored checkpoint id."""
        checkpoint_id = (
            self._stable_checkpoints[-1]["checkpoint_id"]
            if self._stable_checkpoints
            else None
        )
        self._restores.append(
            {
                "ts": utc_now_iso(),
                "checkpoint_id": checkpoint_id,
                "signature": signature,
                "score": score,
                "reason": reason,
            }
        )
        self._save()
        return checkpoint_id

    async def _latest_checkpoint_id(self, config: dict[str, Any]) -> str | None:
        getter = getattr(self._checkpointer, "get_tuple", None)
        if getter is not None:
            result = getter(config)
            if inspect.isawaitable(result):
                result = await result
            return self._extract_checkpoint_id(result)

        async_getter = getattr(self._checkpointer, "aget_tuple", None)
        if async_getter is not None:
            result = async_getter(config)
            if inspect.isawaitable(result):
                result = await result
            return self._extract_checkpoint_id(result)
        return None

    @staticmethod
    def _extract_checkpoint_id(checkpoint_tuple: Any) -> str | None:
        if checkpoint_tuple is None:
            return None

        config = getattr(checkpoint_tuple, "config", None)
        if isinstance(config, dict):
            configurable = config.get("configurable", {})
            if isinstance(configurable, dict):
                checkpoint_id = configurable.get("checkpoint_id")
                if isinstance(checkpoint_id, str) and checkpoint_id:
                    return checkpoint_id

        checkpoint = getattr(checkpoint_tuple, "checkpoint", None)
        if isinstance(checkpoint, dict):
            checkpoint_id = checkpoint.get("id")
            if isinstance(checkpoint_id, str) and checkpoint_id:
                return checkpoint_id
        return None

    def _load(self) -> None:
        if not self._index_path.exists():
            return
        payload = json.loads(self._index_path.read_text(encoding="utf-8"))
        self._stable_checkpoints = list(payload.get("stable_checkpoints", []))
        self._restores = list(payload.get("restores", []))

    def _save(self) -> None:
        payload = {
            "stable_checkpoints": self._stable_checkpoints,
            "restores": self._restores,
        }
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
