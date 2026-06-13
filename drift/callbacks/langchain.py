"""LangChain callback handler for Drift.

Integrates with LangChain's callback system to automatically capture
agent events and feed them into the DriftGuard detection engine.

Usage:
    from drift import DriftGuard
    from drift.callbacks.langchain import DriftCallbackHandler
    
    guard = DriftGuard()
    handler = DriftCallbackHandler(guard)
    
    # Use with any LangChain agent/chain:
    agent.run("your query", callbacks=[handler])
    
    # Check results:
    guard.report()
"""

from __future__ import annotations

import time
from typing import Any, Optional
from uuid import UUID

from drift.core import DriftGuard
from drift.models import AgentEvent, EventType

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    raise ImportError(
        "LangChain integration requires langchain-core. "
        "Install it with: pip install driftguard[langchain]"
    )


class DriftCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that feeds events into DriftGuard.
    
    Captures LLM calls, tool calls, and chain executions with timing
    data and feeds them into the anomaly detection pipeline.
    """

    def __init__(self, guard: DriftGuard):
        self.guard = guard
        self._start_times: dict[str, float] = {}
        self._event_names: dict[str, str] = {}
        self._event_inputs: dict[str, dict[str, Any]] = {}

    def _run_key(self, run_id: UUID) -> str:
        return str(run_id)

    # ---- LLM events ----

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        key = self._run_key(run_id)
        self._start_times[key] = time.time()

        name = serialized.get("id", ["unknown"])
        if isinstance(name, list):
            name = name[-1] if name else "unknown"
        self._event_names[key] = str(name)

        self.guard.ingest(AgentEvent(
            event_type=EventType.LLM_START,
            name=str(name),
            run_id=key,
            parent_id=self._run_key(parent_run_id) if parent_run_id else None,
            inputs={"prompts": prompts[:1]},  # Only first prompt to save memory
        ))

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        key = self._run_key(run_id)
        start = self._start_times.pop(key, None)
        latency_ms = (time.time() - start) * 1000 if start else None
        name = self._event_names.pop(key, "unknown")

        # Extract output text and token usage
        output_text = ""
        token_count = None
        input_tokens = None
        output_tokens = None

        if hasattr(response, "generations") and response.generations:
            gen = response.generations[0]
            if gen:
                output_text = gen[0].text if hasattr(gen[0], "text") else str(gen[0])

        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
            token_count = usage.get("total_tokens")

        self.guard.ingest(AgentEvent(
            event_type=EventType.LLM_END,
            name=name,
            run_id=key,
            parent_id=self._run_key(parent_run_id) if parent_run_id else None,
            latency_ms=latency_ms,
            output_text=output_text,
            token_count=token_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ))

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        key = self._run_key(run_id)
        start = self._start_times.pop(key, None)
        latency_ms = (time.time() - start) * 1000 if start else None
        name = self._event_names.pop(key, "unknown")

        self.guard.ingest(AgentEvent(
            event_type=EventType.ERROR,
            name=name,
            run_id=key,
            latency_ms=latency_ms,
            error=str(error),
        ))

    # ---- Tool events ----

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        key = self._run_key(run_id)
        self._start_times[key] = time.time()

        name = serialized.get("name", "unknown_tool")
        self._event_names[key] = name

        self.guard.ingest(AgentEvent(
            event_type=EventType.TOOL_START,
            name=name,
            run_id=key,
            parent_id=self._run_key(parent_run_id) if parent_run_id else None,
            inputs={"input": input_str[:500]},  # Truncate large inputs
        ))

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        key = self._run_key(run_id)
        start = self._start_times.pop(key, None)
        latency_ms = (time.time() - start) * 1000 if start else None
        name = self._event_names.pop(key, "unknown_tool")

        output_text = str(output)[:2000] if output else ""

        self.guard.ingest(AgentEvent(
            event_type=EventType.TOOL_END,
            name=name,
            run_id=key,
            parent_id=self._run_key(parent_run_id) if parent_run_id else None,
            latency_ms=latency_ms,
            output_text=output_text,
        ))

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        key = self._run_key(run_id)
        start = self._start_times.pop(key, None)
        latency_ms = (time.time() - start) * 1000 if start else None
        name = self._event_names.pop(key, "unknown_tool")

        self.guard.ingest(AgentEvent(
            event_type=EventType.ERROR,
            name=name,
            run_id=key,
            latency_ms=latency_ms,
            error=str(error),
        ))

    # ---- Chain events ----

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        key = self._run_key(run_id)
        self._start_times[key] = time.time()

        name = serialized.get("id", ["unknown"])
        if isinstance(name, list):
            name = name[-1] if name else "unknown"
        self._event_names[key] = str(name)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        key = self._run_key(run_id)
        self._start_times.pop(key, None)
        self._event_names.pop(key, None)
