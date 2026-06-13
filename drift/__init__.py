"""Drift — Statistical anomaly detection for AI agent workflows.

Catch silent failures, hallucination drift, and off-script behavior
in your LangChain, CrewAI, and custom AI agents.

Quickstart:
    from drift import DriftGuard
    from drift.callbacks.langchain import DriftCallbackHandler

    guard = DriftGuard()
    handler = DriftCallbackHandler(guard)
    agent.run("your query", callbacks=[handler])
    guard.report()
"""

from drift.core import DriftGuard
from drift.models import AgentEvent, Anomaly, AnomalyType, EventType, Severity

__version__ = "0.1.0"

__all__ = [
    "DriftGuard",
    "AgentEvent",
    "Anomaly",
    "AnomalyType",
    "EventType",
    "Severity",
]
