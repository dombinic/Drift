"""Base detector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from drift.models import AgentEvent, Anomaly


class BaseDetector(ABC):
    """Abstract base class for all anomaly detectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable detector name."""
        ...

    @abstractmethod
    def ingest(self, event: AgentEvent) -> list[Anomaly]:
        """Process an event and return any anomalies detected.
        
        Args:
            event: The agent event to analyze.
            
        Returns:
            List of anomalies (empty if none detected).
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset detector state (clear baselines, history, etc.)."""
        ...
