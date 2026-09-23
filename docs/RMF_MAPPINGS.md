# NIST AI RMF 1.0 mappings — what they mean and where they stop

This document explains every NIST AI RMF 1.0 mapping the agentic
auditor claims (see `src/opsaudit/reports/rmf.py`). The mappings are
**evidence contributions**, not compliance claims: opsaudit produces
artifacts that *help* an organization work a subcategory; it does not
satisfy the subcategory by itself. Each section below says what the
tool does, and — just as important — what it does **not** do.

Subcategory numbers below are from the NIST AI RMF 1.0 core
(GOVERN / MAP / MEASURE / MANAGE). Where only the function level is
claimed, that is deliberate: the tool's contribution does not reach a
specific subcategory.

## MEASURE 2.11 — Fairness and bias are evaluated and results are documented

**What the tool does.** Counterfactual probe campaigns measure
outcome-rate gaps across protected attributes; judge-labeled campaigns
do the same for tone, refusal, and stereotype signals in open-ended
text. The report documents every measured rate and gap.

**Where it stops.** The campaign measures the scenarios it probed
within its budget — it is not an exhaustive fairness evaluation, and
the flagging threshold (default 0.2) is a heuristic, not a significance
test. Material findings should be confirmed with the v0.1 statistical
core before anyone acts on them.

## MEASURE 2.1 — Test sets, metrics, and details about the tools used during TEVV are documented

**What the tool does.** Every probe batch is generated deterministically,
executed through a documented target adapter, and recorded in the
append-only evidence log. The campaign is reproducible from the seed,
so the "test set" is fully documented.

**Where it stops.** This documents the TEVV for *this audit run* only.
Training-time evaluation, ongoing monitoring, and production telemetry
are out of scope for the tool.

## MEASURE 2.3 — Performance measured for deployment-like conditions

**What the tool does.** Counterfactual drills vary deployment-relevant
attributes (employment type, tenure, query phrasing) one at a time and
measure behavior under those conditions.

**Where it stops.** The operator supplies the deployment context in the
audit brief. The tool cannot know the real deployment distribution and
does not validate that the brief matches it.

## MEASURE 2.13 — Effectiveness of the TEVV metrics and processes are evaluated

**What the tool does.** The calibration harness evaluates the
measurement instruments themselves: every judge's agreement with human
labels (accuracy, Cohen's kappa, per-label precision/recall) is measured
before the judge is trusted, and uncalibrated judges are rejected unless
the operator explicitly overrides.

**Where it stops.** Calibration measures *agreement*, not correctness —
a judge can agree with biased labels. The shipped starter datasets are
synthetic; production use requires real human labels from multiple
annotators with adjudication.

## MAP 5.1 — Likelihood and magnitude of each identified impact are identified and documented

**What the tool does.** Per-group outcome rates, gaps, and judge-label
distributions characterize how system behavior differs across groups —
the raw material an impact assessment is built from.

**Where it stops.** The tool characterizes measured behavioral
differences. The likelihood/magnitude *judgments* for a real deployment
(who is affected, how badly, how likely) remain the operator's job, with
domain knowledge the tool does not have.

## MAP 1.1 — Intended purposes, deployment settings understood and documented

**What the tool does.** The audit brief records the target, the
protected attributes, and the risk areas under evaluation — the
documented evaluation context for the campaign.

**Where it stops.** The brief is the *auditor's* context, not a
substitute for the operator's own MAP 1.1 documentation of the deployed
system.

## GOVERN (function level) — oversight and accountability

**What the tool does.** The report plus the append-only evidence log
give oversight roles a durable, reproducible account of what was
tested, what was found, and how — supporting accountability for the
audit itself.

**Where it stops.** Documentation supports governance; it does not
create the organizational policies, assigned roles, or safety culture
the GOVERN function requires. No tool can.

## MANAGE (function level) — risk treatment and deployment decisions

**What the tool does.** Flagged findings (`FINDINGS WARRANT REVIEW`)
feed go/no-go release decisions; the v0.1 deployment gate
operationalizes numeric thresholds for the tabular core.

**Where it stops.** The tool flags; the risk-treatment decision —
accept, mitigate, or block — belongs to the operator's MANAGE process.
A report is an input to that process, not the decision.
