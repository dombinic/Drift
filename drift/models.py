"""Data models for Drift agent events and anomalies."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class EventType(Enum):
    """Types of agent events we track."""
    LLM_START = "llm_start"
    LLM_END = "llm_end"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    CHAIN_START = "chain_start"
    CHAIN_END = "chain_end"
    ERROR = "error"


class AnomalyType(Enum):
    """Categories of detected anomalies."""
    LATENCY_SPIKE = "latency_spike"
    TOKEN_ANOMALY = "token_anomaly"
    SEQUENCE_ANOMALY = "sequence_anomaly"
    OUTPUT_DRIFT = "output_drift"
    ERROR_RATE = "error_rate"
    COST_SPIKE = "cost_spike"


class Severity(Enum):
    """Anomaly severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AgentEvent:
    """A single event in an agent's execution trace."""
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    run_id: Optional[str] = None

    # What happened
    name: str = ""  # tool name, llm model, chain name
    inputs: Optional[dict[str, Any]] = None
    outputs: Optional[dict[str, Any]] = None

    # Measurements
    latency_ms: Optional[float] = None
    token_count: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None

    # Raw content for drift detection
    output_text: Optional[str] = None
    error: Optional[str] = None

    # Parent event for nesting
    parent_id: Optional[str] = None

    def __repr__(self) -> str:
        parts = [f"{self.event_type.value}({self.name!r}"]
        if self.latency_ms is not None:
            parts.append(f", latency={self.latency_ms:.0f}ms")
        if self.token_count is not None:
            parts.append(f", tokens={self.token_count}")
        if self.error:
            parts.append(f", error={self.error!r}")
        return "".join(parts) + ")"


@dataclass
class Anomaly:
    """A detected anomaly in agent behavior."""
    anomaly_type: AnomalyType
    severity: Severity
    message: str
    timestamp: float = field(default_factory=time.time)

    # What triggered it
    event: Optional[AgentEvent] = None
    detector_name: str = ""

    # Statistical context
    observed_value: Optional[float] = None
    expected_range: Optional[tuple[float, float]] = None
    z_score: Optional[float] = None

    # The raw data window that informed the detection
    context: Optional[dict[str, Any]] = None

    def __repr__(self) -> str:
        sev = self.severity.value.upper()
        return f"[{sev}] {self.anomaly_type.value}: {self.message}"
