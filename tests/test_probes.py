"""Tests for the Phase 2 probe generators.

The property that matters most is counterfactual symmetry: a pair that
drifts in anything but the protected attribute is a broken audit
instrument. Several tests fuzz that property with random attributes and
values.
"""

from __future__ import annotations

import json
import random

import pandas as pd
import pytest

from opsaudit import EvidenceLog, audit_disparities
from opsaudit.probes import (
    BUILTIN_RELATIONS,
    MetamorphicRelation,
    Probe,
    ProbeBatch,
    check_equal_output,
    check_symmetry,
    evaluate_metamorphic,
    generate_adversarial,
    generate_counterfactuals,
    generate_llm_assisted,
    generate_metamorphic,
)
from opsaudit.targets import TabularTarget, Target

# ----------------------------------------------------------------------
# Probe / ProbeBatch basics
# ----------------------------------------------------------------------


def _probe(pid, kind="counterfactual", payload=None, attributes=None):
    return Probe(
        id=pid,
        kind=kind,
        payload=payload if payload is not None else {},
        attributes=attributes or {},
        invariant="inv",
    )


def test_batch_filter_and_add():
    b1 = ProbeBatch(
        [_probe("a", attributes={"g": "x"}), _probe("b", kind="adversarial")],
        name="one",
    )
    b2 = ProbeBatch([_probe("c", attributes={"g": "y"})], name="two")
    assert len(b1 + b2) == 3
    assert (b1 + b2).name == "one+two"
    assert len(b1.filter(kind="adversarial")) == 1
    assert len(b1.filter(attribute="g")) == 1
    assert len((b1 + b2).filter(value="y")) == 1


def test_text_prompts_rejects_mixed_payloads():
    batch = ProbeBatch([_probe("a", payload="hello"), _probe("b", payload={})])
    with pytest.raises(TypeError, match="non-text payload"):
        batch.text_prompts()
    assert ProbeBatch([_probe("a", payload="hello")]).text_prompts() == [
        "hello"
    ]


def test_rows_rejects_mixed_payloads():
    batch = ProbeBatch([_probe("a", payload={"x": 1}), _probe("b", payload="s")])
    with pytest.raises(TypeError, match="non-row payload"):
        batch.rows()
    assert ProbeBatch([_probe("a", payload={"x": 1})]).rows() == [{"x": 1}]


def test_rows_returns_copies():
    batch = ProbeBatch([_probe("a", payload={"x": 1})])
    rows = batch.rows()
    rows[0]["x"] = 999
    assert batch.probes[0].payload == {"x": 1}


def test_to_dataframe_column_order():
    batch = ProbeBatch(
        [_probe("a", payload={"b": 2, "a": 1}), _probe("b", payload={"b": 4, "a": 3})]
    )
    frame = batch.to_dataframe(columns=["a", "b"])
    assert list(frame.columns) == ["a", "b"]
    assert frame["a"].tolist() == [1, 3]
    with pytest.raises(ValueError, match="not present"):
        batch.to_dataframe(columns=["nope"])


def test_to_event_shape():
    event = ProbeBatch([_probe("a")], name="n").to_event()
    assert event["kind"] == "probe_batch"
    assert event["n"] == 1
    assert event["probes"][0]["id"] == "a"


# ----------------------------------------------------------------------
# Counterfactuals — tabular
# ----------------------------------------------------------------------


def test_counterfactual_tabular_symmetry():
    base = {"tenure_months": 12, "shifts_requested": 4, "group": "FT"}
    batch = generate_counterfactuals(base, {"group": ["FT", "PT", "temp"]})
    # 3 values -> C(3,2) = 3 pairs -> 6 probes
    assert len(batch) == 6
    ok, problems = check_symmetry(batch)
    assert ok, problems
    # every payload equals base except the group
    for probe in batch:
        assert probe.payload["tenure_months"] == 12
        assert probe.payload["shifts_requested"] == 4
        assert probe.payload["group"] == probe.attributes["group"]


def test_counterfactual_does_not_mutate_base():
    base = {"a": 1, "group": "x"}
    snapshot = dict(base)
    generate_counterfactuals(base, {"group": ["x", "y"]})
    assert base == snapshot


def test_counterfactual_multiple_attributes_varied_one_at_a_time():
    base = {"age": 30, "group": "a", "region": "north"}
    batch = generate_counterfactuals(
        base, {"group": ["a", "b"], "region": ["north", "south"]}
    )
    ok, problems = check_symmetry(batch)
    assert ok, problems
    for members in batch.pair_groups().values():
        attr = members[0].meta["attribute"]
        for probe in members:
            for key in ("age", "group", "region"):
                if key != attr:
                    assert probe.payload[key] == base[key]


def test_counterfactual_validation_errors():
    base = {"group": "x"}
    with pytest.raises(ValueError, match="non-empty dict"):
        generate_counterfactuals(base, {})
    with pytest.raises(ValueError, match="at least two distinct values"):
        generate_counterfactuals(base, {"group": ["x", "x"]})
    with pytest.raises(ValueError, match="not a key of base"):
        generate_counterfactuals(base, {"missing": ["a", "b"]})
    with pytest.raises(ValueError, match="must be a dict"):
        generate_counterfactuals(["not", "a", "dict"], {"g": ["a", "b"]})


def test_counterfactual_symmetry_fuzz():
    rng = random.Random(20260922)
    for trial in range(50):
        n_attrs = rng.randint(1, 3)
        base, attributes = {"f0": rng.randint(0, 99)}, {}
        for a in range(n_attrs):
            name = f"attr{a}"
            values = rng.sample(
                [f"v{i}" for i in range(6)], k=rng.randint(2, 4)
            )
            base[name] = values[0]
            attributes[name] = values
        batch = generate_counterfactuals(base, attributes)
        ok, problems = check_symmetry(batch)
        assert ok, f"trial {trial}: {problems}"
        # pair count = sum over attributes of C(k,2)*2
        expected = sum(
            len(values) * (len(values) - 1) for values in attributes.values()
        )
        assert len(batch) == expected, f"trial {trial}"


def test_check_symmetry_catches_drift():
    base = {"age": 30, "group": "a"}
    batch = generate_counterfactuals(base, {"group": ["a", "b"]})
    # Corrupt one mirror: drift a non-attribute field.
    bad = batch.probes[0]
    corrupted = Probe(
        id=bad.id,
        kind=bad.kind,
        payload={**bad.payload, "age": 99},
        attributes=bad.attributes,
        invariant=bad.invariant,
        meta=bad.meta,
    )
    broken = ProbeBatch(
        [corrupted if p.id == bad.id else p for p in batch.probes]
    )
    ok, problems = check_symmetry(broken)
    assert not ok
    assert any("drifted" in p for p in problems)


def test_check_symmetry_catches_orphan():
    probe = Probe(
        id="solo",
        kind="counterfactual",
        payload={"g": "a"},
        attributes={"g": "a"},
        meta={"pair_id": "p-1", "attribute": "g", "values": ["a", "b"],
              "mode": "tabular"},
    )
    ok, problems = check_symmetry(ProbeBatch([probe]))
    assert not ok
    assert any("expected 2 mirrors" in p for p in problems)


# ----------------------------------------------------------------------
# Counterfactuals — text mode
# ----------------------------------------------------------------------


def test_counterfactual_text_mode():
    base = {
        "prompt": "Should we approve the loan for this {group} applicant?",
        "group": "male",
    }
    batch = generate_counterfactuals(
        base, {"group": ["male", "female"]}, text_field="prompt"
    )
    assert len(batch) == 2
    prompts = sorted(batch.text_prompts())
    assert prompts[0] == (
        "Should we approve the loan for this female applicant?"
    )
    assert prompts[1] == (
        "Should we approve the loan for this male applicant?"
    )
    ok, problems = check_symmetry(batch)
    assert ok, problems


def test_counterfactual_text_missing_placeholder():
    base = {"prompt": "no placeholders here", "group": "x"}
    with pytest.raises(ValueError, match="placeholder"):
        generate_counterfactuals(
            base, {"group": ["x", "y"]}, text_field="prompt"
        )


def test_check_symmetry_catches_text_drift():
    base = {"prompt": "Hi {group}", "group": "a"}
    batch = generate_counterfactuals(
        base, {"group": ["a", "b"]}, text_field="prompt"
    )
    bad = batch.probes[0]
    corrupted = Probe(
        id=bad.id,
        kind=bad.kind,
        payload=bad.payload + " EXTRA",
        attributes=bad.attributes,
        invariant=bad.invariant,
        meta=bad.meta,
    )
    broken = ProbeBatch(
        [corrupted if p.id == bad.id else p for p in batch.probes]
    )
    ok, _problems = check_symmetry(broken)
    assert not ok


# ----------------------------------------------------------------------
# Adversarial
# ----------------------------------------------------------------------


def test_adversarial_tabular_coverage():
    base = {"age": 30, "income": 50000.0, "group": "a", "flag": True}
    batch = generate_adversarial(target_type="tabular", base_row=base)
    strategies = {p.meta["strategy"] for p in batch}
    assert {"extreme_value", "unseen_category", "null_row"} <= strategies
    extremes = [
        p.payload["income"]
        for p in batch
        if p.meta.get("strategy") == "extreme_value"
        and p.meta.get("field") == "income"
    ]
    assert 0.0 in extremes and 1e12 in extremes and -1.0 in extremes
    # non-targeted fields untouched
    for probe in batch.filter():
        if probe.meta.get("strategy") == "extreme_value":
            f = probe.meta["field"]
            for key, value in base.items():
                if key != f:
                    assert probe.payload[key] == value
    nulls = [p for p in batch if p.meta["strategy"] == "null_row"]
    assert len(nulls) == 1 and all(v is None for v in nulls[0].payload.values())


def test_adversarial_text_coverage():
    batch = generate_adversarial(target_type="text")
    strategies = {p.meta["strategy"] for p in batch}
    assert {
        "empty_input",
        "whitespace_input",
        "very_long_input",
        "special_characters",
        "ambiguous_phrasing",
        "contradictory_instruction",
    } <= strategies
    prompts = batch.text_prompts()
    assert "" in prompts
    assert any(len(p) > 1000 for p in prompts)


def test_adversarial_validation():
    with pytest.raises(ValueError, match="tabular.*text"):
        generate_adversarial(target_type="weird")
    with pytest.raises(ValueError, match="base_row is required"):
        generate_adversarial(target_type="tabular")


# ----------------------------------------------------------------------
# Metamorphic
# ----------------------------------------------------------------------


def test_metamorphic_pairs_and_evaluate_pass():
    inputs = [
        {"age": 30, "note": "Applied on 2026-01-05", "group": "a"},
        "Decide the case from 2026-03-01.",
    ]
    batch = generate_metamorphic(inputs)
    assert len(batch) > 0
    # every follow-up has a source in the same batch
    by_id = {p.id: p for p in batch}
    for probe in batch:
        if probe.meta.get("role") == "followup":
            assert probe.meta["source_id"] in by_id
    # identity target: outputs equal -> all equal_output verdicts pass
    outputs = {}
    for probe in batch:
        outputs[probe.id] = "APPROVE"
    verdicts = evaluate_metamorphic(batch, outputs)
    assert verdicts
    assert all(v["passed"] for v in verdicts)


def test_metamorphic_evaluate_catches_violation():
    batch = generate_metamorphic(
        ["Hello World"],
        relations=[
            r for r in BUILTIN_RELATIONS if r.name == "case_fold"
        ],
    )
    outputs = {}
    for probe in batch:
        # a case-sensitive target: different outputs -> violation
        outputs[probe.id] = (
            "OUT-1" if probe.meta["role"] == "source" else "OUT-2"
        )
    verdicts = evaluate_metamorphic(batch, outputs)
    assert len(verdicts) == 1
    assert verdicts[0]["passed"] is False
    assert verdicts[0]["relation"] == "case_fold"


def test_metamorphic_builtin_transforms():
    assert _apply_date("x 2026-01-05 y") == "x Jan 5, 2026 y"
    assert _apply_date("not a date 2026-13-99") == "not a date 2026-13-99"


def _apply_date(text):
    rel = next(r for r in BUILTIN_RELATIONS if r.name == "date_format")
    return rel.transform(text)


def test_metamorphic_relation_validation():
    with pytest.raises(ValueError, match="expected must be"):
        MetamorphicRelation(
            name="x", transform=lambda p: p, expected="bogus"
        )
    with pytest.raises(ValueError, match="inputs must be a non-empty"):
        generate_metamorphic([])


def test_metamorphic_evaluate_missing_outputs():
    batch = generate_metamorphic([{"a": 1}])
    verdicts = evaluate_metamorphic(batch, {})
    assert verdicts and all(v["passed"] is False for v in verdicts)


def test_check_equal_output_nan():
    assert check_equal_output(float("nan"), float("nan"))
    assert check_equal_output("a", "a")
    assert not check_equal_output("a", "b")


# ----------------------------------------------------------------------
# LLM-assisted
# ----------------------------------------------------------------------


class _FakeTextTarget(Target):
    """Canned-response text target: no network, fully deterministic."""

    name = "fake-llm"

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def generate(self, prompts):
        self.calls += 1
        idx = min(self.calls - 1, len(self._responses) - 1)
        return [self._responses[idx] for _ in prompts]

    def describe(self):
        return {"target_type": "fake", "name": self.name}


def _cand(kind, payload, attributes, invariant="inv", **extra):
    return {
        "kind": kind,
        "payload": payload,
        "attributes": attributes,
        "invariant": invariant,
        **extra,
    }


def test_llm_assisted_keeps_valid_pairs_discards_orphans():
    candidates = [
        _cand(
            "counterfactual",
            {"age": 30, "group": "a"},
            {"group": "a"},
            "group should not matter",
        ),
        _cand(
            "counterfactual",
            {"age": 30, "group": "b"},
            {"group": "b"},
            "group should not matter",
        ),
        _cand(  # orphan: no mirror
            "counterfactual",
            {"age": 40, "group": "a"},
            {"group": "a"},
        ),
        _cand(  # text counterfactual: cannot be validated
            "counterfactual", "hire the {group} one", {"group": "a"}
        ),
        _cand("adversarial", {"age": -5}, {}, "must not crash"),
        {"kind": "bogus", "payload": {}, "attributes": {}, "invariant": "x"},
    ]
    target = _FakeTextTarget([json.dumps(candidates)])
    batch, report = generate_llm_assisted(target, brief="audit a loan model")
    assert report["requested"] == 10  # default n
    assert report["kept"] == 3  # 1 pair + 1 adversarial
    assert report["discarded"] == 3
    assert report["parse_ok"] is True
    assert all(p.kind == "llm_assisted" for p in batch)
    ok, problems = check_symmetry(batch)
    assert ok, problems  # the kept pair passes the symmetry check


def test_llm_assisted_bad_json_reports_not_crashes():
    target = _FakeTextTarget(["this is not json at all"])
    batch, report = generate_llm_assisted(target, brief="x")
    assert len(batch) == 0
    assert report["parse_ok"] is False
    assert report["discarded"] == 1


def test_llm_assisted_retries_empty_response_once():
    good = json.dumps(
        [_cand("adversarial", {"x": 1}, {}, "must not crash")]
    )
    target = _FakeTextTarget(["", good])
    _batch, report = generate_llm_assisted(target, brief="x")
    assert target.calls == 2
    assert report["kept"] == 1


def test_llm_assisted_requires_text_target():
    with pytest.raises(RuntimeError, match="requires a text Target, got None"):
        generate_llm_assisted(None, brief="x")

    class _TabularOnly(Target):
        name = "tab"

        def predict(self, X):
            return []

        def describe(self):
            return {"target_type": "tab", "name": self.name}

    with pytest.raises(RuntimeError, match="does not support generate"):
        generate_llm_assisted(_TabularOnly(), brief="x")
    with pytest.raises(ValueError, match="non-empty string"):
        generate_llm_assisted(_FakeTextTarget(["[]"]), brief="  ")


# ----------------------------------------------------------------------
# End-to-end: probes -> Phase 1 tabular adapter -> evidence -> metrics
# ----------------------------------------------------------------------


class _BiasedGrants:
    """Toy classifier with a planted group disparity (temp workers denied)."""

    def predict(self, X):
        out = []
        for _, row in X.iterrows():
            if row["group"] == "temp" and row["tenure_months"] < 24:
                out.append(0)
            else:
                out.append(1)
        return out


def test_end_to_end_counterfactual_audit_finds_planted_disparity(tmp_path):
    base = {"tenure_months": 6, "shifts_requested": 4, "group": "FT"}
    batch = generate_counterfactuals(base, {"group": ["FT", "PT", "temp"]})
    ok, problems = check_symmetry(batch)
    assert ok, problems

    target = TabularTarget(_BiasedGrants(), name="biased-grants")
    frame = batch.to_dataframe(
        columns=["tenure_months", "shifts_requested", "group"]
    )
    preds = target.predict(frame)

    log = EvidenceLog(tmp_path / "e2e.jsonl", target=target)
    log.record(batch.to_event())
    log.record(
        {"kind": "response_batch", "n": len(preds), "predictions": preds}
    )

    y_true = [1] * len(preds)  # everyone "deserved" the shifts
    groups = [p.attributes["group"] for p in batch]
    result = audit_disparities(y_true, preds, groups)
    log.record(
        {
            "kind": "metric_computed",
            "tpr_gap": result.tpr_gap,
            "flags": [f.get("code", f) if isinstance(f, dict) else f for f in result.flags],
        }
    )

    # planted disparity: temp denied (tpr 0) while FT/PT granted (tpr 1)
    assert result.tpr_gap == pytest.approx(1.0)
    assert result.flags, "expected the gate to flag the disparity"
    records = log.read()
    assert [r["kind"] for r in records] == [
        "probe_batch",
        "response_batch",
        "metric_computed",
    ]
    assert all(r["target"]["name"] == "biased-grants" for r in records)


def test_end_to_end_adversarial_tabular_no_crash():
    base = {"tenure_months": 6, "shifts_requested": 4, "group": "FT"}
    batch = generate_adversarial(target_type="tabular", base_row=base)
    target = TabularTarget(_BiasedGrants(), name="robust-check")
    frame = batch.to_dataframe(
        columns=["tenure_months", "shifts_requested", "group"]
    )
    preds = target.predict(frame)  # must not raise
    assert len(preds) == len(batch)
    assert set(preds) <= {0, 1}


def test_end_to_end_metamorphic_reorder_stable():
    base = {"tenure_months": 6, "shifts_requested": 4, "group": "FT"}
    batch = generate_metamorphic(
        [base],
        relations=[r for r in BUILTIN_RELATIONS if r.name == "reorder_fields"],
    )
    target = TabularTarget(_BiasedGrants(), name="mm-check")
    outputs = {}
    for probe in batch:
        row = pd.DataFrame([probe.payload])[
            ["tenure_months", "shifts_requested", "group"]
        ]
        outputs[probe.id] = target.predict(row)[0]
    verdicts = evaluate_metamorphic(batch, outputs)
    assert len(verdicts) == 1
    assert verdicts[0]["passed"] is True
