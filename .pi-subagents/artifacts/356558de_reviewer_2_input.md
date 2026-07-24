# Task for reviewer

[Read from: /home/spencer/Repositories/predictability-aware-training/plan.md, /home/spencer/Repositories/predictability-aware-training/progress.md]

RED-TEAM ROUND 3 — publication readiness of /home/spencer/Repositories/predictability-aware-training. Read README.md, RESEARCH.md (novelty claim + prior-art tables), RESULTS.md (all experiments), PLAN.md. The evidence arc: toy-scale from-scratch mechanism (3 seeds, robust), structure-not-sharpening, StickyMoE defused, grows with training, sim economics, and a clean boundary result (LoRA fine-tune on OLMoE fails to transfer — pretraining-time only). Assess: (1) What is the strongest honest claim set for a workshop paper (e.g., ICLR/MLSys workshop)? Write the 3-4 bullet claims that survive and the single figure/table that carries each. (2) What would kill this paper at review — the top 3 reviewer objections after all remediation so far, and which (if any) require new experiments vs just careful writing. (3) Is the Tier C negative result a strength (boundary mapping) or a weakness (method doesn't apply to existing models)? How to frame it. (4) The novelty claim wording in RESEARCH.md — final stress test against the prior-art table (StickyMoE, Oracle-MoE, ReMoE, Pre-gated MoE, Halfway SD): any remaining vulnerability? (5) What goes in the paper vs stays in the repo (e.g., the two sim blockers, the label double-shift bug — disclose or omit?). READ-ONLY. Deliver: a concrete paper skeleton (sections, claims, figures) calibrated to the actual evidence.

## Acceptance Contract
Acceptance level: attested
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Return concrete findings with file paths and severity when applicable

Required evidence: review-findings, residual-risks

Finish with a fenced JSON block tagged `acceptance-report` in this shape:
Use empty arrays when no items apply; array fields contain strings unless object entries are shown.
`criteriaSatisfied[].status` must be exactly one of: satisfied, not-satisfied, not-applicable.
`commandsRun[].result` must be exactly one of: passed, failed, not-run.
`manualNotes` and `notes` are optional strings; an empty string means no note and does not satisfy `manual-notes` evidence.
```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "specific proof"
    }
  ],
  "changedFiles": [
    "src/file.ts"
  ],
  "testsAddedOrUpdated": [
    "test/file.test.ts"
  ],
  "commandsRun": [
    {
      "command": "command",
      "result": "passed",
      "summary": "short result"
    }
  ],
  "validationOutput": [
    "validation output or concise summary"
  ],
  "residualRisks": [
    "none"
  ],
  "noStagedFiles": true,
  "diffSummary": "short description of the diff",
  "reviewFindings": [
    "blocker: file.ts:12 - issue found, or no blockers"
  ],
  "manualNotes": "anything else the parent should know"
}
```