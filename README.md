# Drift

**Statistical anomaly detection for AI agent workflows.**

Catch silent failures, hallucination drift, and off-script behavior before they corrupt your data.

---

Your AI agents are failing silently. A tool call takes 10x longer than usual. The agent calls `delete_file` when it's never done that before. Output quality gradually degrades over hundreds of runs. Traditional monitoring tools weren't built for non-deterministic systems — Drift is.

## What it does

Drift hooks into your agent's execution and applies statistical anomaly detection to the event stream:

- **Latency & token SPC** — Flags when a tool call or LLM response takes significantly longer or uses significantly more tokens than its rolling baseline. Catches hung API calls, runaway generation, and upstream provider issues.

- **Sequence anomaly detection** — Builds a transition matrix of tool-call sequences and flags when the agent takes a path that's never or rarely been seen. Catches agents going off-script, skipping required steps, or entering novel execution paths.

- **Output drift detection** — Tracks output length, vocabulary diversity, and structural patterns over time. Flags when outputs shift significantly from baseline. Catches hallucination drift, prompt injection effects, and gradual quality degradation.

## Install

```bash
pip install driftguard
```

With LangChain support:
```bash
pip install driftguard[langchain]
```

## Quickstart

### With LangChain

```python
from drift import DriftGuard
from drift.callbacks.langchain import DriftCallbackHandler

guard = DriftGuard(on_anomaly=lambda a: print(f"🚨 {a}"))
handler = DriftCallbackHandler(guard)

# Use with any LangChain agent, chain, or LLM
agent.run("your query", callbacks=[handler])

# See what happened
guard.report()
```

### Standalone (no framework required)

```python
from drift import DriftGuard, AgentEvent, EventType

guard = DriftGuard(on_anomaly=lambda a: print(f"🚨 {a}"))

# Feed events from any source
guard.ingest(AgentEvent(
    event_type=EventType.TOOL_END,
    name="search_web",
    latency_ms=150.0,
    token_count=85,
    output_text="Found 3 results for query...",
))

guard.report()
```

### Run the demo

```bash
python examples/demo.py
```

This simulates normal agent operation, builds baselines, then injects latency spikes, sequence anomalies, and output drift — showing each detector catching real failure modes.

## Architecture

```
drift/
├── core.py              # DriftGuard engine — orchestrates detectors
├── models.py            # AgentEvent, Anomaly, Severity data models
├── detectors/
│   ├── latency.py       # Statistical process control on latency/tokens
│   ├── sequence.py      # Action transition probability anomalies
│   └── output_drift.py  # Output distribution shift detection
└── callbacks/
    └── langchain.py     # LangChain callback integration
```

**Design principles:**

1. **Zero-overhead default** — Detectors use numpy for fast rolling statistics. No embedding models, no external services, no network calls.
2. **Per-tool baselines** — Each tool and model gets its own statistical baseline, so a slow tool won't pollute the baseline for a fast one.
3. **Framework-agnostic core** — The detection engine works with raw `AgentEvent` objects. Framework integrations (LangChain, CrewAI, etc.) are thin adapters that translate framework callbacks into events.
4. **Non-blocking** — Drift never throws exceptions that would crash your agent. Detector errors are caught and logged to stderr.

## Detectors

| Detector | What it catches | Method |
|----------|----------------|--------|
| `LatencyDetector` | Hung calls, slow APIs, runaway generation | Rolling z-score on latency and token counts |
| `SequenceDetector` | Off-script behavior, unexpected tool calls | First-order Markov transition probabilities |
| `OutputDriftDetector` | Hallucination drift, prompt injection, quality degradation | Output length, vocab diversity, structural pattern tracking |

## Configuration

Each detector is independently configurable:

```python
from drift import DriftGuard
from drift.detectors.latency import LatencyDetector, LatencyDetectorConfig
from drift.detectors.sequence import SequenceDetector, SequenceDetectorConfig

guard = DriftGuard(detectors=[
    LatencyDetector(LatencyDetectorConfig(
        window_size=100,     # Longer baseline window
        z_threshold=2.5,     # More sensitive
        min_samples=10,      # Require more data before alerting
    )),
    SequenceDetector(SequenceDetectorConfig(
        min_observations=20, # Require more transitions before flagging
    )),
])
```

## Roadmap

- [ ] CrewAI callback handler
- [ ] OpenAI Agents SDK integration
- [ ] Slack / PagerDuty alerting
- [ ] Persistent baselines (save/load detector state)
- [ ] Embedding-based output drift (optional dependency)
- [ ] Web dashboard
- [ ] Cost anomaly detection (track spend per run)

## Contributing

Issues and PRs welcome. Run tests with:

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
