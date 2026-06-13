"""Drift standalone demo — no LangChain required.

Simulates a series of agent events (normal operation, then anomalies)
to demonstrate each detector catching real failure modes.

Run with:
    python examples/demo.py
"""

import random
import time

from drift import DriftGuard, AgentEvent, EventType


def simulate_normal_events(guard: DriftGuard, n: int = 20) -> None:
    """Simulate normal agent operation to build baselines."""
    tools = ["search_web", "parse_document", "write_response"]

    for i in range(n):
        for tool in tools:
            # Normal tool start
            guard.ingest(AgentEvent(
                event_type=EventType.TOOL_START,
                name=tool,
            ))

            # Normal tool end with realistic latency and output
            latency = random.gauss(200, 30)  # ~200ms ± 30ms
            output_len = random.randint(100, 300)

            guard.ingest(AgentEvent(
                event_type=EventType.TOOL_END,
                name=tool,
                latency_ms=max(50, latency),
                token_count=random.randint(50, 150),
                output_text="Normal output " * (output_len // 14),
            ))


def simulate_latency_spike(guard: DriftGuard) -> None:
    """Simulate a tool that suddenly takes 10x longer (upstream API issue)."""
    print("\n--- Injecting: Latency spike on 'search_web' ---")
    guard.ingest(AgentEvent(
        event_type=EventType.TOOL_START,
        name="search_web",
    ))
    guard.ingest(AgentEvent(
        event_type=EventType.TOOL_END,
        name="search_web",
        latency_ms=2500.0,  # 10x normal
        token_count=120,
        output_text="Normal output " * 15,
    ))


def simulate_sequence_anomaly(guard: DriftGuard) -> None:
    """Simulate agent calling tools in a never-seen order."""
    print("\n--- Injecting: Novel tool sequence (search_web → delete_file) ---")
    guard.ingest(AgentEvent(
        event_type=EventType.TOOL_START,
        name="search_web",
    ))
    guard.ingest(AgentEvent(
        event_type=EventType.TOOL_END,
        name="search_web",
        latency_ms=190.0,
        output_text="Found results",
    ))
    # This transition has never been seen
    guard.ingest(AgentEvent(
        event_type=EventType.TOOL_START,
        name="delete_file",  # Agent went off-script!
    ))
    guard.ingest(AgentEvent(
        event_type=EventType.TOOL_END,
        name="delete_file",
        latency_ms=50.0,
        output_text="File deleted",
    ))


def simulate_output_drift(guard: DriftGuard) -> None:
    """Simulate output that suddenly changes character (hallucination drift)."""
    print("\n--- Injecting: Output drift on 'write_response' ---")
    guard.ingest(AgentEvent(
        event_type=EventType.TOOL_START,
        name="write_response",
    ))
    # Massively different output length + structure
    guard.ingest(AgentEvent(
        event_type=EventType.TOOL_END,
        name="write_response",
        latency_ms=210.0,
        token_count=2000,  # Way more tokens than normal
        output_text="```python\n" + "x = 1\n" * 500 + "```\n",  # Code block (novel structure)
    ))


def main():
    print("=" * 60)
    print("  DRIFT DEMO — Agent Anomaly Detection")
    print("=" * 60)

    # Set up with a callback that prints anomalies in real-time
    def on_anomaly(anomaly):
        print(f"  🚨 {anomaly}")

    guard = DriftGuard(on_anomaly=on_anomaly)

    # Phase 1: Build baseline
    print("\n[Phase 1] Building baseline with 20 normal agent runs...")
    simulate_normal_events(guard, n=20)
    print(f"  ✓ {guard.event_count} events processed, {guard.anomaly_count} anomalies")

    # Phase 2: Inject anomalies
    print("\n[Phase 2] Injecting anomalies...\n")

    simulate_latency_spike(guard)
    simulate_sequence_anomaly(guard)
    simulate_output_drift(guard)

    # Phase 3: Report
    guard.report(verbose=True)


if __name__ == "__main__":
    main()
