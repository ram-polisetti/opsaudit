# Operational AI Audit Playbook

Use this playbook before deploying or materially changing an operational ML decision system. It is designed for an operator, analyst, risk reviewer, or decision owner—not only a model developer.

## Purpose

An `opsaudit` result is evidence for a release decision. It does not certify fairness or identify a single cause. The goal is to make an accountable decision with a record of what was checked, what was found, and what happened next.

## The operating loop

| Step | Owner question | Evidence to retain |
| --- | --- | --- |
| 1. Define the decision | What action does the system allocate, recommend, approve, or deny? | Intended use and model/version |
| 2. Define groups | Which groups should be compared, and why is that comparison legitimate? | Group definitions and access controls |
| 3. Validate inputs | Are decision, outcome, and group fields complete and correctly defined? | Data period, row count, validation notes |
| 4. Set policy | What thresholds and minimum group size apply, and who approved them? | Versioned threshold policy |
| 5. Run the audit | What do allocation and error-rate metrics show? | Markdown, HTML, and JSON report |
| 6. Interpret evidence | Is the result a PASS, FAIL, or REVIEW? | Review rationale and any uncertainty notes |
| 7. Decide action | Release, hold, remediate, or escalate? | Decision owner and disposition |
| 8. Retain the record | Can a reviewer reconstruct the decision later? | Context manifest, reports, remediation evidence |

## What to do with each gate result

### PASS — exit 0

The configured metrics passed and no evidence-quality review condition was recorded. Record the report, policy version, reviewer, and deployment decision. A pass is not a permanent or universal fairness certification.

### FAIL — exit 1

One or more configured metrics crossed the policy threshold. Hold the release or use the organization’s approved exception process. Confirm the inputs, inspect affected groups and operational conditions, identify a plausible cause, document the decision, and rerun after remediation.

### REVIEW — exit 2

The tool cannot make a reliable release recommendation. Common causes include a group below the configured minimum size, no positive decisions, a missing prediction column, or a confidence interval crossing a policy boundary. Treat review as blocking unless a documented human exception process says otherwise.

## A practical dispatch example

The synthetic dispatch scenario asks: *Are premium routes allocated and calibrated comparably across delivery zones?*

1. Generate a clean scenario and expect the gate to pass.
2. Generate a biased scenario and expect a fail.
3. Do not stop at the ratio. Check group counts, data definitions, delivery-zone differences, and the allocation rule.
4. Decide whether the cause calls for model changes, policy changes, operational-process changes, better data, or a documented hold.

## Context-manifest checklist

Keep the manifest short and non-sensitive. Useful fields are system name, model version, intended use, decision owner, audit owner, decision period, group definition, threshold-policy version, and review disposition. Do not store raw rows, credentials, employee identifiers, customer data, or unnecessary personal information.

## The sentence to use in an interview

> I designed opsaudit to turn group-level disparity metrics into a documented operational release decision. The tool measures the signal, records evidence quality and context, and makes clear when a human review—not an automated pass/fail—is required.
