"""Latency and token count anomaly detection via statistical process control.

Uses a rolling window to compute mean and standard deviation, then flags
events whose latency or token count falls outside a configurable z-score
threshold. This is the simplest and most immediately useful detector —
it catches hung API calls, runaway generation, and upstream provider issues.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from drift.detectors import BaseDetector
from drift.models import AgentEvent, Anomaly, AnomalyType, EventType, Severity


@dataclass
class LatencyDetectorConfig:
    """Configuration for the latency/token detector."""
    window_size: int = 50          # Rolling window for baseline stats
    z_threshold: float = 3.0       # Standard deviations before flagging
    min_samples: int = 5           # Minimum events before detection activates
    critical_z: float = 5.0        # Z-score threshold for CRITICAL severity
    track_tokens: bool = True      # Also monitor token counts


class LatencyDetector(BaseDetector):
    """Detects anomalous latency and token counts using rolling z-scores.
    
    Maintains per-tool and per-model baselines, so a slow tool won't
    pollute the baseline for a fast one.
    """

    def __init__(self, config: LatencyDetectorConfig | None = None):
        self.config = config or LatencyDetectorConfig()
        # Keyed by event name (tool name or model name)
        self._latency_windows: dict[str, list[float]] = defaultdict(list)
        self._token_windows: dict[str, list[float]] = defaultdict(list)

    @property
    def name(self) -> str:
        return "latency_detector"

    def ingest(self, event: AgentEvent) -> list[Anomaly]:
        anomalies: list[Anomaly] = []

        # Only analyze completion events (they have latency data)
        if event.event_type not in (EventType.LLM_END, EventType.TOOL_END):
            return anomalies

        key = event.name or "unknown"

        # --- Latency check ---
        if event.latency_ms is not None:
            window = self._latency_windows[key]
            anomaly = self._check_value(
                value=event.latency_ms,
                window=window,
                metric_name="latency",
                unit="ms",
                anomaly_type=AnomalyType.LATENCY_SPIKE,
                event=event,
            )
            if anomaly:
                anomalies.append(anomaly)

            # Update window
            window.append(event.latency_ms)
            if len(window) > self.config.window_size:
                window.pop(0)

        # --- Token count check ---
        if self.config.track_tokens and event.token_count is not None:
            window = self._token_windows[key]
            anomaly = self._check_value(
                value=float(event.token_count),
                window=window,
                metric_name="token_count",
                unit="tokens",
                anomaly_type=AnomalyType.TOKEN_ANOMALY,
                event=event,
            )
            if anomaly:
                anomalies.append(anomaly)

            window.append(float(event.token_count))
            if len(window) > self.config.window_size:
                window.pop(0)

        return anomalies

    def _check_value(
        self,
        value: float,
        window: list[float],
        metric_name: str,
        unit: str,
        anomaly_type: AnomalyType,
        event: AgentEvent,
    ) -> Anomaly | None:
        """Check a single metric against its rolling baseline."""
        if len(window) < self.config.min_samples:
            return None

        arr = np.array(window)
        mean = float(np.mean(arr))
        std = float(np.std(arr))

        if std < 1e-9:  # Constant values — can't compute z-score meaningfully
            return None

        z = (value - mean) / std

        if abs(z) < self.config.z_threshold:
            return None

        # Determine severity
        if abs(z) >= self.config.critical_z:
            severity = Severity.CRITICAL
        elif abs(z) >= self.config.z_threshold + 1:
            severity = Severity.HIGH
        else:
            severity = Severity.MEDIUM

        direction = "above" if z > 0 else "below"

        return Anomaly(
            anomaly_type=anomaly_type,
            severity=severity,
            message=(
                f"{event.name!r} {metric_name} is {value:.1f}{unit}, "
                f"{abs(z):.1f}σ {direction} mean ({mean:.1f}{unit} ± {std:.1f})"
            ),
            event=event,
            detector_name=self.name,
            observed_value=value,
            expected_range=(mean - self.config.z_threshold * std, 
                          mean + self.config.z_threshold * std),
            z_score=z,
            context={"window_size": len(window), "mean": mean, "std": std},
        )

    def reset(self) -> None:
        self._latency_windows.clear()
        self._token_windows.clear()
