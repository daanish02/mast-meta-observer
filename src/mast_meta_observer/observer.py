from __future__ import annotations

from dataclasses import dataclass

from .models import ObserverConfig, SignatureResult, TraceEvent
from .signatures import evaluate_signatures


@dataclass(slots=True)
class ObservationDecision:
    """Observer decision emitted after evaluating one window."""

    trigger_rollback: bool
    reason: str | None
    signature: SignatureResult | None
    signature_scores: list[SignatureResult]


class ObserverEngine:
    """Stateful MAST policy evaluator with persistence counters."""

    def __init__(self, config: ObserverConfig) -> None:
        self._config = config
        self._persistence: dict[str, int] = {}

    def evaluate(self, window: list[TraceEvent]) -> ObservationDecision:
        """Evaluate the sliding window and decide whether to rollback.

        Args:
            window: Recent events for signature scoring.

        Returns:
            ObservationDecision with rollback trigger and evidence.
        """
        scores = evaluate_signatures(
            window,
            context_token_limit=self._config.context_token_limit,
        )

        trigger: SignatureResult | None = None
        for result in scores:
            if result.score >= self._config.threshold:
                self._persistence[result.name] = (
                    self._persistence.get(result.name, 0) + 1
                )
                if self._persistence[result.name] >= self._config.persistence:
                    if trigger is None or result.score > trigger.score:
                        trigger = result
            else:
                self._persistence[result.name] = 0

        if trigger is None:
            return ObservationDecision(
                trigger_rollback=False,
                reason=None,
                signature=None,
                signature_scores=scores,
            )

        # Cool down persistence counters after a trigger to avoid immediate retriggers.
        for name in list(self._persistence):
            self._persistence[name] = 0

        reason = (
            f"Signature '{trigger.name}' exceeded threshold "
            f"{self._config.threshold:.2f} for {self._config.persistence} windows"
        )
        return ObservationDecision(
            trigger_rollback=True,
            reason=reason,
            signature=trigger,
            signature_scores=scores,
        )

    def should_mark_stable(self, score_snapshot: list[SignatureResult]) -> bool:
        """Determine whether current state can be considered stable."""
        if not score_snapshot:
            return False
        peak = max(item.score for item in score_snapshot)
        return peak < self._config.threshold
