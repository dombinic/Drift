"""Action sequence anomaly detection.

Builds a transition probability matrix from observed tool/LLM call sequences
and flags when the agent takes a path that has never or rarely been seen.
This catches agents going "off-script" — calling tools in unexpected orders,
skipping required steps, or entering novel execution paths.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from drift.detectors import BaseDetector
from drift.models import AgentEvent, Anomaly, AnomalyType, EventType, Severity


@dataclass
class SequenceDetectorConfig:
    """Configuration for the sequence anomaly detector."""
    min_observations: int = 10       # Min transitions before detection activates
    novel_severity: Severity = Severity.HIGH    # Severity for never-seen transitions
    rare_threshold: float = 0.02     # Transitions below this probability are flagged
    rare_severity: Severity = Severity.MEDIUM   # Severity for rare transitions


class SequenceDetector(BaseDetector):
    """Detects anomalous action sequences by tracking transition probabilities.
    
    Maintains a first-order Markov transition matrix over tool/LLM call names.
    When an agent takes a transition that's never been observed or is very rare
    relative to the baseline, it fires an anomaly.
    
    Example: If your agent always calls search → parse → respond, and suddenly
    calls search → delete → respond, the search→delete transition gets flagged.
    """

    def __init__(self, config: SequenceDetectorConfig | None = None):
        self.config = config or SequenceDetectorConfig()
        # transition_counts[from_action][to_action] = count
        self._transition_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._total_from: dict[str, int] = defaultdict(int)
        self._last_action: str | None = None
        self._total_transitions: int = 0

    @property
    def name(self) -> str:
        return "sequence_detector"

    def ingest(self, event: AgentEvent) -> list[Anomaly]:
        anomalies: list[Anomaly] = []

        # Only track start events (they represent the agent's decision to act)
        if event.event_type not in (
            EventType.TOOL_START, EventType.LLM_START, EventType.CHAIN_START
        ):
            return anomalies

        current_action = event.name or "unknown"

        if self._last_action is not None and self._total_transitions >= self.config.min_observations:
            anomaly = self._check_transition(self._last_action, current_action, event)
            if anomaly:
                anomalies.append(anomaly)

        # Update transition matrix
        if self._last_action is not None:
            self._transition_counts[self._last_action][current_action] += 1
            self._total_from[self._last_action] += 1
            self._total_transitions += 1

        self._last_action = current_action
        return anomalies

    def _check_transition(
        self, from_action: str, to_action: str, event: AgentEvent
    ) -> Anomaly | None:
        """Check if a transition is anomalous."""
        from_counts = self._transition_counts.get(from_action)

        # Case 1: We've never seen ANY transition from this action
        if from_counts is None or self._total_from.get(from_action, 0) == 0:
            return None  # Can't evaluate — not enough data for this source

        total_from = self._total_from[from_action]
        to_count = from_counts.get(to_action, 0)

        # Case 2: Never-seen transition from a known source
        if to_count == 0:
            seen_targets = list(from_counts.keys())
            return Anomaly(
                anomaly_type=AnomalyType.SEQUENCE_ANOMALY,
                severity=self.config.novel_severity,
                message=(
                    f"Novel transition: {from_action!r} → {to_action!r} "
                    f"(never observed; known transitions from {from_action!r}: "
                    f"{seen_targets})"
                ),
                event=event,
                detector_name=self.name,
                observed_value=0.0,
                context={
                    "from_action": from_action,
                    "to_action": to_action,
                    "known_transitions": dict(from_counts),
                    "total_from_count": total_from,
                },
            )

        # Case 3: Rare transition
        probability = to_count / total_from
        if probability < self.config.rare_threshold:
            return Anomaly(
                anomaly_type=AnomalyType.SEQUENCE_ANOMALY,
                severity=self.config.rare_severity,
                message=(
                    f"Rare transition: {from_action!r} → {to_action!r} "
                    f"(p={probability:.3f}, seen {to_count}/{total_from} times)"
                ),
                event=event,
                detector_name=self.name,
                observed_value=probability,
                expected_range=(self.config.rare_threshold, 1.0),
                context={
                    "from_action": from_action,
                    "to_action": to_action,
                    "probability": probability,
                    "count": to_count,
                    "total_from_count": total_from,
                },
            )

        return None

    def get_transition_matrix(self) -> dict[str, dict[str, float]]:
        """Return the current transition probability matrix (for debugging/viz)."""
        matrix: dict[str, dict[str, float]] = {}
        for from_action, targets in self._transition_counts.items():
            total = self._total_from[from_action]
            if total > 0:
                matrix[from_action] = {
                    to_action: count / total
                    for to_action, count in targets.items()
                }
        return matrix

    def reset(self) -> None:
        self._transition_counts.clear()
        self._total_from.clear()
        self._last_action = None
        self._total_transitions = 0
