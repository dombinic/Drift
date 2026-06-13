"""Core Drift engine — orchestrates detectors and manages the event stream."""

from __future__ import annotations

import threading
from typing import Callable, Optional

from drift.detectors import BaseDetector
from drift.detectors.latency import LatencyDetector
from drift.detectors.output_drift import OutputDriftDetector
from drift.detectors.sequence import SequenceDetector
from drift.models import AgentEvent, Anomaly, Severity


# Type alias for anomaly callbacks
AnomalyCallback = Callable[[Anomaly], None]


class DriftGuard:
    """Main entry point for Drift anomaly detection.
    
    Manages a set of detectors, ingests agent events, collects anomalies,
    and optionally fires callbacks when anomalies are detected.
    
    Usage:
        guard = DriftGuard()
        
        # Use with LangChain:
        from drift.callbacks.langchain import DriftCallbackHandler
        handler = DriftCallbackHandler(guard)
        agent.run("query", callbacks=[handler])
        
        # Or ingest events directly:
        guard.ingest(event)
        
        # Check results:
        guard.report()
    """

    def __init__(
        self,
        detectors: list[BaseDetector] | None = None,
        on_anomaly: AnomalyCallback | None = None,
        min_severity: Severity = Severity.LOW,
    ):
        """Initialize DriftGuard.
        
        Args:
            detectors: Custom list of detectors. If None, uses all defaults.
            on_anomaly: Callback fired for each anomaly (e.g., log, alert, raise).
            min_severity: Minimum severity to record/report. 
        """
        if detectors is None:
            self.detectors: list[BaseDetector] = [
                LatencyDetector(),
                SequenceDetector(),
                OutputDriftDetector(),
            ]
        else:
            self.detectors = detectors

        self.on_anomaly = on_anomaly
        self.min_severity = min_severity
        self._anomalies: list[Anomaly] = []
        self._events: list[AgentEvent] = []
        self._lock = threading.Lock()

    def ingest(self, event: AgentEvent) -> list[Anomaly]:
        """Process an event through all detectors.
        
        Args:
            event: The agent event to analyze.
            
        Returns:
            List of anomalies detected from this event.
        """
        new_anomalies: list[Anomaly] = []

        with self._lock:
            self._events.append(event)

            for detector in self.detectors:
                try:
                    detected = detector.ingest(event)
                    for anomaly in detected:
                        if self._severity_value(anomaly.severity) >= self._severity_value(
                            self.min_severity
                        ):
                            new_anomalies.append(anomaly)
                            self._anomalies.append(anomaly)
                except Exception as e:
                    # Detectors should never crash the agent
                    import sys
                    print(
                        f"[drift] Detector {detector.name!r} error: {e}",
                        file=sys.stderr,
                    )

        # Fire callbacks outside the lock
        if self.on_anomaly:
            for anomaly in new_anomalies:
                try:
                    self.on_anomaly(anomaly)
                except Exception:
                    pass

        return new_anomalies

    @property
    def anomalies(self) -> list[Anomaly]:
        """All anomalies detected so far."""
        return list(self._anomalies)

    @property
    def events(self) -> list[AgentEvent]:
        """All events ingested so far."""
        return list(self._events)

    @property
    def anomaly_count(self) -> int:
        return len(self._anomalies)

    @property
    def event_count(self) -> int:
        return len(self._events)

    def report(self, verbose: bool = False) -> str:
        """Generate a human-readable anomaly report.
        
        Args:
            verbose: If True, include statistical context for each anomaly.
            
        Returns:
            Formatted report string (also printed to stdout).
        """
        lines: list[str] = []
        lines.append("")
        lines.append("=" * 60)
        lines.append("  DRIFT ANOMALY REPORT")
        lines.append("=" * 60)
        lines.append(f"  Events processed: {self.event_count}")
        lines.append(f"  Anomalies found:  {self.anomaly_count}")

        if not self._anomalies:
            lines.append("")
            lines.append("  ✓ No anomalies detected.")
        else:
            # Group by severity
            by_severity: dict[Severity, list[Anomaly]] = {}
            for a in self._anomalies:
                by_severity.setdefault(a.severity, []).append(a)

            for sev in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]:
                group = by_severity.get(sev, [])
                if not group:
                    continue
                lines.append("")
                lines.append(f"  [{sev.value.upper()}] ({len(group)})")
                for a in group:
                    lines.append(f"    • {a.anomaly_type.value}: {a.message}")
                    if verbose and a.z_score is not None:
                        lines.append(f"      z-score: {a.z_score:.2f}")
                    if verbose and a.expected_range is not None:
                        lo, hi = a.expected_range
                        lines.append(f"      expected range: [{lo:.1f}, {hi:.1f}]")

        lines.append("")
        lines.append("=" * 60)
        lines.append("")

        report_text = "\n".join(lines)
        print(report_text)
        return report_text

    def reset(self) -> None:
        """Reset all state — detectors, events, anomalies."""
        with self._lock:
            for detector in self.detectors:
                detector.reset()
            self._anomalies.clear()
            self._events.clear()

    @staticmethod
    def _severity_value(severity: Severity) -> int:
        return {
            Severity.LOW: 0,
            Severity.MEDIUM: 1,
            Severity.HIGH: 2,
            Severity.CRITICAL: 3,
        }[severity]
