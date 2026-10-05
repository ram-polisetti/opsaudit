# EU AI Act cross-check (`opsaudit.eu_ai_act`)

## What it is

A version-pinned encoding of the EU AI Act's system-classification logic as
testable code, mirroring what the European Commission's official AI Act
Compliance Checker questionnaire does:

- **Article 5** — prohibited practices (social scoring, real-time remote
  biometric ID in public spaces, emotion inference at work/education,
  untargeted facial scraping, predictive policing by profiling, …)
- **Article 6 / Annex III** — high-risk use cases (recruitment, credit
  scoring, education admissions, critical infrastructure, law enforcement,
  migration, justice, …)
- **Article 50** — transparency obligations (chatbots, emotion recognition,
  AI-generated content/deepfakes)
- **Chapter V** — general-purpose AI duties (Art. 53) and systemic-risk
  duties (Art. 55)

The official Compliance Checker is a web questionnaire with no API, so this
module does not call it. Instead it encodes the published decision logic and
is validated against the official materials. Cross-checks are therefore
reproducible. Encoding version: `2026-10-05`.

**This is a research instrument, not legal advice.** The Commission itself
notes the official Checker does not replace legal counsel.

## API

```python
from opsaudit.eu_ai_act import classify_system, cross_check_gate, eu_ai_act_section

# 1. Classify the system
c = classify_system({
    "use_cases": ["employment_recruitment"],
    "is_gpai": False,
})
c.tier            # 'high_risk'
c.obligations     # conformity assessment, human oversight, logging, ...
c.references      # ['Art. 6 / Annex III(4)']

# 2. Cross-check against the deployment gate verdict
from opsaudit.gate import evaluate_gate_status
status, _ = evaluate_gate_status(audit_result)   # 'pass' | 'fail' | 'review'
check = cross_check_gate(status, c)
check.verdict     # 'hard_stop' | 'conditional_pass' | 'blocked' | 'pass'

# 3. Render the report section
print(eu_ai_act_section(c, check))
```

### Cross-check verdicts

| Gate | EU tier | Verdict | Meaning |
|------|---------|---------|---------|
| any | prohibited | `hard_stop` | Art. 5: gate verdict irrelevant, do not deploy |
| pass | high_risk | `conditional_pass` | Stats bar met; conformity assessment, oversight, registration still required |
| fail/review | high_risk | `blocked` | Both bars block deployment |
| pass | transparency/minimal | `pass` | Gate governs (disclosures may still apply) |
| fail/review | transparency/minimal | `blocked` | Gate governs |

The stricter of the two bars always wins.

### Controlled vocabulary

`USE_CASES` maps every supported use-case key to its tier, article
reference, and label. Unknown keys raise `ValueError` — the vocabulary is
deliberately closed so classifications stay auditable.

## Research protocol: the cross-check study

This module exists to support an empirical study (candidate: EAI RAIDS 2027):

1. **Sample** N operational decision systems (hiring, lending, dispatch,
   admissions, …) with documented intended purposes.
2. **Three classifications per system:**
   a. opsaudit gate verdict (statistical: PASS/FAIL/REVIEW),
   b. `eu_ai_act.classify_system` tier (encoded logic),
   c. the official Compliance Checker questionnaire (manual, recorded).
3. **Agreement analysis:** gate-vs-encoded, encoded-vs-official,
   gate-vs-official. Where do they diverge, and in which direction?
   (Hypothesis: the gate is stricter on statistical disparity; the Act is
   stricter on process duties for high-risk systems that pass statistically.)
4. **Report** per-system `eu_ai_act_section` outputs as the evidence trail.

Re-run the official Checker (c) whenever the encoding version is bumped;
record the Checker date alongside the encoding version.
