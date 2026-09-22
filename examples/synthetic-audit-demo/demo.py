"""Demo: audit synthetic data derived from the UCI Adult dataset.

Downloads the UCI Adult census data (if missing), builds two synthetic
variants — a decent one and a deliberately bad one (biased + memorizing) —
and runs `opsaudit audit-synthetic` on both.

Usage:
    python demo.py [--n 4000] [--data-dir ./data]

Outputs (in the data dir): source.csv, synthetic_good.csv,
synthetic_bad.csv, config.yml, report_good.json, report_bad.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ADULT_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]
NUMERIC = ["age", "education-num", "hours-per-week", "capital-gain"]
CATEGORICAL = ["workclass", "education", "marital-status", "occupation",
               "relationship", "race", "sex"]


def fetch_adult(data_dir: Path) -> Path:
    """Download and extract the UCI Adult dataset if it is not cached.

    Honors the ADULT_DATA environment variable (set by .devcontainer) as a
    direct path to adult.data, so Codespaces with the pre-fetched dataset
    skip downloading.
    """
    env_path = os.environ.get("ADULT_DATA")
    if env_path and Path(env_path).is_file():
        print(f"using pre-fetched dataset at {env_path}")
        return Path(env_path)
    target = data_dir / "adult.data"
    if target.exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "adult.zip"
    print(f"downloading {ADULT_URL} ...")
    urllib.request.urlretrieve(ADULT_URL, archive)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extract("adult.data", data_dir)
    return target


def load_source(data_path: Path, n: int, seed: int) -> pd.DataFrame:
    """Load a deterministic sample of Adult with stripped string values."""
    frame = pd.read_csv(data_path, header=None, names=ADULT_COLUMNS,
                        na_values=" ?", skipinitialspace=True)
    frame = frame.dropna().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    sample = frame.iloc[rng.choice(len(frame), size=n, replace=False)]
    return sample.reset_index(drop=True).copy()


def approval_rule(frame: pd.DataFrame, bias_against_female: float,
                  seed: int) -> pd.Series:
    """Simulated approval decision: a fair-ish score rule, optionally biased.

    ``bias_against_female`` scales the approval probability for female rows;
    1.0 is unbiased, lower is more biased.
    """
    rng = np.random.default_rng(seed)
    score = (
        0.03 * frame["age"]
        + 0.25 * frame["education-num"]
        + 0.02 * frame["hours-per-week"]
        - 4.2
        + rng.normal(0, 0.6, size=len(frame))
    )
    prob = 1.0 / (1.0 + np.exp(-score))
    prob = np.where(frame["sex"] == "Female", prob * bias_against_female, prob)
    approved = (rng.random(len(frame)) < np.clip(prob, 0.01, 0.99)).astype(int)
    return pd.Series(approved, index=frame.index)


def synthesize_good(source: pd.DataFrame, seed: int) -> pd.DataFrame:
    """A decent synthesizer: independent per-column resampling.

    Every column is resampled independently from the source's empirical
    marginal, so each synthetic row is a novel combination that exists in
    no real record — the privacy-safe naive baseline. Marginals are
    preserved (fidelity); joint structure is not (the report surfaces it
    as correlation drift). The approval decision is recomputed from the
    synthetic features with the unbiased rule.
    """
    rng = np.random.default_rng(seed)
    n = len(source)
    synth = pd.DataFrame({
        column: rng.choice(source[column].to_numpy(), size=n)
        for column in ADULT_COLUMNS
    })
    synth["income_gt50k"] = (synth["income"] == ">50K").astype(int)
    synth["approved"] = approval_rule(synth, bias_against_female=1.0, seed=seed + 1)
    return synth


def synthesize_bad(source: pd.DataFrame, seed: int) -> pd.DataFrame:
    """A bad synthesizer: near-verbatim copies + a biased approval rule."""
    rng = np.random.default_rng(seed)
    synth = source.copy()
    for column in NUMERIC:  # tiny jitter: rows stay suspiciously close
        synth[column] = synth[column] + rng.integers(-1, 2, size=len(synth))
    synth["approved"] = approval_rule(synth, bias_against_female=0.45, seed=seed + 1)
    return synth


def write_config(path: Path) -> None:
    """Write the auditor config for this demo."""
    path.write_text(
        """\
truth: income_gt50k
pred: approved
groups: [sex]
numeric: [age, education-num, hours-per-week, capital-gain]
categorical: [workclass, education, marital-status, occupation, relationship, race, sex]
exclude: [fnlwgt]
seed: 42
""",
        encoding="utf-8",
    )


def run_audit(source_csv: Path, synth_csv: Path, config: Path, out: Path) -> dict:
    """Run the opsaudit CLI and return the parsed JSON report."""
    completed = subprocess.run(
        ["opsaudit", "audit-synthetic",
         "--source", str(source_csv), "--synthetic", str(synth_csv),
         "--config", str(config), "--out", str(out)],
        capture_output=True, text=True, check=False,
    )
    print(completed.stdout)
    if completed.returncode not in (0, 1, 2):
        print(completed.stderr, file=sys.stderr)
        raise RuntimeError(f"audit-synthetic failed: {completed.stderr[:500]}")
    return json.loads(out.read_text(encoding="utf-8"))


def summarize(name: str, report: dict) -> None:
    """Print a one-screen summary of a report."""
    print(f"\n===== {name}: {report['verdict'].upper()} =====")
    for axis in ("fidelity", "privacy", "bias"):
        axis_report = report["axes"][axis]
        print(f"[{axis_report['verdict'].upper()}] {axis}")
        for finding in axis_report["findings"]:
            print(f"  - {finding['code']}: {finding['message']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    adult_path = fetch_adult(data_dir)
    source = load_source(adult_path, args.n, seed=20260921)
    source["income_gt50k"] = (source["income"] == ">50K").astype(int)
    source["approved"] = approval_rule(source, bias_against_female=1.0, seed=7)

    good = synthesize_good(source, seed=11)
    bad = synthesize_bad(source, seed=13)

    source_csv = data_dir / "source.csv"
    good_csv = data_dir / "synthetic_good.csv"
    bad_csv = data_dir / "synthetic_bad.csv"
    config_path = data_dir / "config.yml"
    source.to_csv(source_csv, index=False)
    good.to_csv(good_csv, index=False)
    bad.to_csv(bad_csv, index=False)
    write_config(config_path)

    good_report = run_audit(source_csv, good_csv, config_path,
                            data_dir / "report_good.json")
    bad_report = run_audit(source_csv, bad_csv, config_path,
                           data_dir / "report_bad.json")
    summarize("GOOD synthetic", good_report)
    summarize("BAD synthetic", bad_report)


if __name__ == "__main__":
    main()
