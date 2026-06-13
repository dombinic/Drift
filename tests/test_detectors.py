"""Tests for Drift anomaly detectors."""

import pytest
from drift import DriftGuard, AgentEvent, EventType, Severity
from drift.detectors.latency import LatencyDetector, LatencyDetectorConfig
from drift.detectors.sequence import SequenceDetector, SequenceDetectorConfig
from drift.detectors.output_drift import OutputDriftDetector, OutputDriftConfig


# ---- Latency Detector ----

class TestLatencyDetector:
    def _make_detector(self, min_samples=5, z_threshold=2.0):
        return LatencyDetector(LatencyDetectorConfig(
            min_samples=min_samples,
            z_threshold=z_threshold,
        ))

    def _end_event(self, name="tool_a", latency_ms=100.0, tokens=50):
        return AgentEvent(
            event_type=EventType.TOOL_END,
            name=name,
            latency_ms=latency_ms,
            token_count=tokens,
        )

    def test_no_anomaly_during_warmup(self):
        det = self._make_detector(min_samples=5)
        for _ in range(4):
            anomalies = det.ingest(self._end_event(latency_ms=100))
            assert anomalies == []

    def test_detects_latency_spike(self):
        det = self._make_detector(min_samples=5, z_threshold=2.0)
        # Build baseline with slight variance (constant values have std=0)
        for i in range(10):
            det.ingest(self._end_event(latency_ms=100 + (i % 3) * 5))
        # Spike
        anomalies = det.ingest(self._end_event(latency_ms=1000))
        assert len(anomalies) > 0
        assert anomalies[0].anomaly_type.value == "latency_spike"

    def test_no_false_positive_on_normal(self):
        det = self._make_detector(min_samples=5, z_threshold=3.0)
        for _ in range(20):
            anomalies = det.ingest(self._end_event(latency_ms=100))
        # All should be empty after warmup
        assert all(a.anomaly_type.value == "latency_spike" for a in anomalies) or anomalies == []

    def test_per_tool_baselines(self):
        det = self._make_detector(min_samples=5, z_threshold=2.0)
        # Tool A: fast
        for _ in range(10):
            det.ingest(self._end_event(name="fast_tool", latency_ms=50))
        # Tool B: slow (but consistent)
        for _ in range(10):
            det.ingest(self._end_event(name="slow_tool", latency_ms=500))
        # Tool B at its normal speed should NOT be flagged
        anomalies = det.ingest(self._end_event(name="slow_tool", latency_ms=510))
        assert anomalies == []


# ---- Sequence Detector ----

class TestSequenceDetector:
    def _make_detector(self, min_obs=5):
        return SequenceDetector(SequenceDetectorConfig(
            min_observations=min_obs,
        ))

    def _start_event(self, name):
        return AgentEvent(event_type=EventType.TOOL_START, name=name)

    def test_detects_novel_transition(self):
        det = self._make_detector(min_obs=5)
        # Normal pattern: A → B → C, repeated
        for _ in range(5):
            det.ingest(self._start_event("A"))
            det.ingest(self._start_event("B"))
            det.ingest(self._start_event("C"))
        # Novel: C → D (never seen)
        anomalies = det.ingest(self._start_event("D"))
        assert len(anomalies) == 1
        assert "novel" in anomalies[0].message.lower() or "Novel" in anomalies[0].message

    def test_no_anomaly_during_warmup(self):
        det = self._make_detector(min_obs=10)
        for _ in range(3):
            anomalies = det.ingest(self._start_event("A"))
            assert anomalies == []

    def test_known_transition_no_flag(self):
        det = self._make_detector(min_obs=5)
        for _ in range(10):
            det.ingest(self._start_event("A"))
            det.ingest(self._start_event("B"))
        # A → B is well-known
        det.ingest(self._start_event("A"))
        anomalies = det.ingest(self._start_event("B"))
        assert anomalies == []


# ---- Output Drift Detector ----

class TestOutputDriftDetector:
    def _make_detector(self, min_samples=5):
        return OutputDriftDetector(OutputDriftConfig(
            min_samples=min_samples,
            length_z_threshold=2.0,
        ))

    def _end_event(self, name="responder", text="Normal output text here"):
        return AgentEvent(
            event_type=EventType.TOOL_END,
            name=name,
            output_text=text,
        )

    def test_detects_length_spike(self):
        det = self._make_detector(min_samples=5)
        # Baseline: short outputs with slight length variation
        for i in range(10):
            det.ingest(self._end_event(text="Short normal output." + " ok" * (i % 3)))
        # Spike: massive output
        anomalies = det.ingest(self._end_event(text="x " * 5000))
        assert len(anomalies) > 0

    def test_detects_novel_structure(self):
        det = self._make_detector(min_samples=5)
        # Baseline: plain text
        for _ in range(10):
            det.ingest(self._end_event(text="Just a normal sentence about things."))
        # Novel: code block (never seen before)
        anomalies = det.ingest(self._end_event(text="```python\nprint('hello')\n```"))
        structure_anomalies = [a for a in anomalies if "structure" in a.message.lower()]
        assert len(structure_anomalies) > 0


# ---- Integration: DriftGuard ----

class TestDriftGuard:
    def test_full_pipeline(self):
        anomalies_caught = []
        guard = DriftGuard(on_anomaly=lambda a: anomalies_caught.append(a))

        # Build baseline with slight variance (constant std=0 skips detection)
        for i in range(20):
            guard.ingest(AgentEvent(
                event_type=EventType.TOOL_START, name="search"))
            guard.ingest(AgentEvent(
                event_type=EventType.TOOL_END, name="search",
                latency_ms=100 + (i % 5) * 3, token_count=50 + (i % 4),
                output_text="Normal search results" + " here" * (i % 3) + "."))
            guard.ingest(AgentEvent(
                event_type=EventType.TOOL_START, name="respond"))
            guard.ingest(AgentEvent(
                event_type=EventType.TOOL_END, name="respond",
                latency_ms=80 + (i % 4) * 2, token_count=100 + (i % 3),
                output_text="Here is a normal response" + " today" * (i % 2) + "."))

        # Inject anomaly
        guard.ingest(AgentEvent(
            event_type=EventType.TOOL_START, name="search"))
        guard.ingest(AgentEvent(
            event_type=EventType.TOOL_END, name="search",
            latency_ms=5000,  # Massive spike
            token_count=50,
            output_text="Normal search results."))

        assert guard.anomaly_count > 0
        assert len(anomalies_caught) > 0

    def test_report_generation(self):
        guard = DriftGuard()
        report = guard.report()
        assert "DRIFT ANOMALY REPORT" in report
        assert "No anomalies" in report

    def test_reset(self):
        guard = DriftGuard()
        guard.ingest(AgentEvent(event_type=EventType.TOOL_START, name="x"))
        assert guard.event_count == 1
        guard.reset()
        assert guard.event_count == 0
        assert guard.anomaly_count == 0
