"""Tests for the append-only evidence log."""

import json
import threading

import pytest

from opsaudit.evidence import EvidenceLog
from opsaudit.targets import RagTarget, Target


class _Target(Target):
    def describe(self):
        return {"target_type": "test", "name": "t"}


def _log(tmp_path, target=None):
    return EvidenceLog(tmp_path / "audit.jsonl", target=target)


class TestEvidenceLog:
    def test_record_stamps_seq_ts_and_target(self, tmp_path):
        log = _log(tmp_path, target=_Target())
        rec = log.record({"kind": "probe_batch", "n": 4})
        assert rec["seq"] == 0
        assert "ts" in rec and rec["ts"].endswith("+00:00")
        assert rec["target"] == {"target_type": "test", "name": "t"}
        assert rec["kind"] == "probe_batch"
        assert rec["n"] == 4

    def test_sequence_is_monotonic(self, tmp_path):
        log = _log(tmp_path)
        seqs = [log.record({"kind": "note"})["seq"] for _ in range(5)]
        assert seqs == [0, 1, 2, 3, 4]

    def test_read_round_trip(self, tmp_path):
        log = _log(tmp_path)
        log.record({"kind": "probe_batch", "prompts": ["a", "b"]})
        log.record({"kind": "response_batch", "responses": ["x", "y"]})
        records = log.read()
        assert len(records) == 2
        assert records[0]["kind"] == "probe_batch"
        assert records[1]["responses"] == ["x", "y"]

    def test_seq_continues_after_reopen(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        EvidenceLog(path).record({"kind": "note", "i": 0})
        EvidenceLog(path).record({"kind": "note", "i": 1})
        log = EvidenceLog(path)
        rec = log.record({"kind": "note", "i": 2})
        assert rec["seq"] == 2
        assert [r["seq"] for r in log.read()] == [0, 1, 2]

    def test_read_missing_file_returns_empty(self, tmp_path):
        assert _log(tmp_path).read() == []
        assert len(_log(tmp_path)) == 0

    def test_len_counts_records(self, tmp_path):
        log = _log(tmp_path)
        assert len(log) == 0
        log.record({"kind": "note"})
        log.record({"kind": "note"})
        assert len(log) == 2

    def test_rejects_non_dict_event(self, tmp_path):
        with pytest.raises(ValueError, match="must be a dict"):
            _log(tmp_path).record(["not", "a", "dict"])

    def test_target_descriptor_snapshot_at_creation(self, tmp_path):
        log = _log(tmp_path, target=RagTarget(lambda q: q, name="snap"))
        rec = log.record({"kind": "note"})
        assert rec["target"]["target_type"] == "rag"
        assert rec["target"]["name"] == "snap"

    def test_no_target_gives_empty_descriptor(self, tmp_path):
        rec = _log(tmp_path).record({"kind": "note"})
        assert rec["target"] == {}

    def test_file_is_valid_jsonl(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = EvidenceLog(path)
        log.record({"kind": "note", "text": "héllo ✓"})
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["text"] == "héllo ✓"

    def test_concurrent_writes_keep_unique_seqs(self, tmp_path):
        log = _log(tmp_path)
        n_threads, n_each = 8, 25

        def worker(k):
            for i in range(n_each):
                log.record({"kind": "note", "thread": k, "i": i})

        threads = [threading.Thread(target=worker, args=(k,))
                   for k in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        records = log.read()
        assert len(records) == n_threads * n_each
        seqs = sorted(r["seq"] for r in records)
        assert seqs == list(range(n_threads * n_each))

    def test_arbitrary_event_payloads(self, tmp_path):
        log = _log(tmp_path)
        payload = {
            "kind": "metric_computed",
            "metric": "disparate_impact",
            "value": 0.73,
            "groups": {"a": 10, "b": 12},
            "nested": [{"x": 1}],
        }
        rec = log.record(payload)
        assert log.read()[0]["groups"] == {"a": 10, "b": 12}
        assert rec["nested"] == [{"x": 1}]
