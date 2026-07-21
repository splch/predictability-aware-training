# Task for reviewer

[Read from: /home/spencer/Repositories/predictability-aware-training/plan.md, /home/spencer/Repositories/predictability-aware-training/progress.md]

RED-TEAM the experimental design and scientific claims in /home/spencer/Repositories/predictability-aware-training — read RESEARCH.md, PLAN.md, RESULTS.md, then model.py/train.py as needed. The project claims: folding expert-prediction accuracy into an MoE's training loss makes the model more predictable (higher expert hit@k) at zero quality cost, enabling disk-expert prefetching. Attack the methodology: (1) Confounds — is the posthoc control fairly matched (predictor capacity, training steps 2000 vs 6000, same data stream position)? (2) Metric validity — is hit@k (top-k set overlap) the right proxy for prefetch success, or should it be e.g. weighted by expert size, per-token cache-miss cost, or recall@k under a cache budget? (3) The 'zero quality cost' claim — is val LM loss on a continuation of the training stream a valid quality measure, or is it just train loss by another name? What would contamination do to the claim? (4) Statistical rigor — single seed, eval variance across stream chunks; are +2.5pt differences meaningful? (5) Does the λ=0.03≈posthoc observation actually isolate 'training-induced predictability' as claimed? Alternative explanations? (6) Threats to the end-to-end thesis: even with perfect prediction, does prefetching actually pay at 2MB expert size and modern NVMe latency vs. just caching hot experts (Colibri's learned hot-store)? When would prediction NOT help? (7) Scale threats: why might Tier B (8 experts, 94M) results fail to transfer to 256-expert production MoEs? READ-ONLY: do not modify files, do not run training. Return a numbered list of the strongest attacks, each with severity and a concrete experimental fix or additional measurement that would address it.

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