"""Output drift detection.

Tracks statistical properties of agent outputs over time and flags when
the distribution shifts significantly. This catches hallucination drift,
prompt injection effects, and gradual quality degradation.

Uses lightweight heuristics (output length, vocabulary diversity, 
structural patterns) rather than embeddings for zero-dependency operation.
Embedding-based drift detection can be added as an optional enhancement.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from drift.detectors import BaseDetector
from drift.models import AgentEvent, Anomaly, AnomalyType, EventType, Severity


@dataclass
class OutputDriftConfig:
    """Configuration for the output drift detector."""
    window_size: int = 30            # Rolling window for baseline
    min_samples: int = 8             # Minimum samples before detection
    length_z_threshold: float = 3.0  # Z-score for output length anomaly
    vocab_z_threshold: float = 3.0   # Z-score for vocabulary diversity anomaly
    structure_change: bool = True    # Track structural pattern changes


class OutputDriftDetector(BaseDetector):
    """Detects drift in agent output characteristics.
    
    Tracks multiple lightweight signals per tool/model:
    - Output length distribution
    - Vocabulary diversity (unique words / total words)
    - Structural patterns (presence of code blocks, JSON, lists, etc.)
    
    All signals are zero-dependency (no embedding models required).
    """

    def __init__(self, config: OutputDriftConfig | None = None):
        self.config = config or OutputDriftConfig()
        self._length_windows: dict[str, list[float]] = defaultdict(list)
        self._vocab_windows: dict[str, list[float]] = defaultdict(list)
        self._structure_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._total_by_key: dict[str, int] = defaultdict(int)

    @property
    def name(self) -> str:
        return "output_drift_detector"

    def ingest(self, event: AgentEvent) -> list[Anomaly]:
        anomalies: list[Anomaly] = []

        if event.event_type not in (EventType.LLM_END, EventType.TOOL_END):
            return anomalies

        if not event.output_text:
            return anomalies

        key = event.name or "unknown"
        text = event.output_text

        # --- Length anomaly ---
        length = float(len(text))
        length_anomaly = self._check_zscore(
            value=length,
            window=self._length_windows[key],
            metric="output_length",
            unit="chars",
            event=event,
            z_threshold=self.config.length_z_threshold,
        )
        if length_anomaly:
            anomalies.append(length_anomaly)
        self._update_window(self._length_windows[key], length)

        # --- Vocabulary diversity ---
        words = text.lower().split()
        if len(words) > 5:
            diversity = len(set(words)) / len(words)
            vocab_anomaly = self._check_zscore(
                value=diversity,
                window=self._vocab_windows[key],
                metric="vocab_diversity",
                unit="ratio",
                event=event,
                z_threshold=self.config.vocab_z_threshold,
            )
            if vocab_anomaly:
                anomalies.append(vocab_anomaly)
            self._update_window(self._vocab_windows[key], diversity)

        # --- Structural pattern change ---
        if self.config.structure_change:
            structure = self._extract_structure(text)
            struct_anomaly = self._check_structure(key, structure, event)
            if struct_anomaly:
                anomalies.append(struct_anomaly)

        return anomalies

    def _check_zscore(
        self,
        value: float,
        window: list[float],
        metric: str,
        unit: str,
        event: AgentEvent,
        z_threshold: float,
    ) -> Anomaly | None:
        """Generic z-score check against a rolling window."""
        if len(window) < self.config.min_samples:
            return None

        arr = np.array(window)
        mean = float(np.mean(arr))
        std = float(np.std(arr))

        if std < 1e-9:
            return None

        z = (value - mean) / std
        if abs(z) < z_threshold:
            return None

        severity = Severity.HIGH if abs(z) > z_threshold + 2 else Severity.MEDIUM
        direction = "above" if z > 0 else "below"

        return Anomaly(
            anomaly_type=AnomalyType.OUTPUT_DRIFT,
            severity=severity,
            message=(
                f"{event.name!r} {metric} is {value:.1f} {unit}, "
                f"{abs(z):.1f}σ {direction} baseline ({mean:.1f} ± {std:.1f})"
            ),
            event=event,
            detector_name=self.name,
            observed_value=value,
            expected_range=(mean - z_threshold * std, mean + z_threshold * std),
            z_score=z,
        )

    def _extract_structure(self, text: str) -> str:
        """Classify the structural pattern of an output."""
        patterns: list[str] = []
        if re.search(r"```", text):
            patterns.append("code_block")
        if re.search(r"[{\[].*[}\]]", text, re.DOTALL):
            patterns.append("json_like")
        if re.search(r"^\s*[-*]\s", text, re.MULTILINE):
            patterns.append("bullet_list")
        if re.search(r"^\s*\d+\.\s", text, re.MULTILINE):
            patterns.append("numbered_list")
        if re.search(r"^#+\s", text, re.MULTILINE):
            patterns.append("markdown_headers")
        if len(text.strip()) == 0:
            patterns.append("empty")
        return "|".join(sorted(patterns)) if patterns else "plain_text"

    def _check_structure(
        self, key: str, structure: str, event: AgentEvent
    ) -> Anomaly | None:
        """Flag if the output structure is novel for this tool/model."""
        self._total_by_key[key] += 1
        counts = self._structure_counts[key]

        total = self._total_by_key[key]
        if total < self.config.min_samples:
            counts[structure] += 1
            return None

        # Check if this structure has been seen before
        if structure not in counts:
            anomaly = Anomaly(
                anomaly_type=AnomalyType.OUTPUT_DRIFT,
                severity=Severity.MEDIUM,
                message=(
                    f"{event.name!r} produced a novel output structure: {structure!r} "
                    f"(previously seen: {list(counts.keys())})"
                ),
                event=event,
                detector_name=self.name,
                context={
                    "novel_structure": structure,
                    "known_structures": dict(counts),
                    "total_observations": total,
                },
            )
            counts[structure] += 1
            return anomaly

        counts[structure] += 1
        return None

    def _update_window(self, window: list[float], value: float) -> None:
        """Append value and trim to window size."""
        window.append(value)
        if len(window) > self.config.window_size:
            window.pop(0)

    def reset(self) -> None:
        self._length_windows.clear()
        self._vocab_windows.clear()
        self._structure_counts.clear()
        self._total_by_key.clear()
