# Task for reviewer

[Read from: /home/spencer/Repositories/predictability-aware-training/plan.md, /home/spencer/Repositories/predictability-aware-training/progress.md]

RED-TEAM ROUND 2 — methodology & statistical claims of a research project at /home/spencer/Repositories/predictability-aware-training. Read RESULTS.md (all 6 experiments), PLAN.md, and the code (model.py, train.py) as needed. Since round 1, the project added: fixed held-out eval, routing diagnostics, a backbone-isolation test (posthoc predictor on the frozen JOINT backbone recovers co-trained accuracy: 0.890 vs 0.887), a 3-seed replication (joint advantage +6.2-6.8 pts hit@k over posthoc control, seed std ~0.4; quality cost +0.023+-0.009 nats), and corrected the 'zero cost' claim to 'small but real'. Attack what's LEFT standing: (1) The isolation test — does posthoc-on-joint recovering accuracy really prove 'deployability'? What does it NOT prove (e.g., predictor cost at inference, per-layer heads, the engine seeing only hidden states not our training pipeline)? (2) The quality-cost measurement — is val LM loss at 25M tokens of training sensitive enough to detect the relevant quality damage? What about the fact that BOTH models are ~75x undertrained (5.77 nats ~= garbage quality) — can predictability-vs-quality tradeoffs at this loss level say ANYTHING about trained-to-convergence models? (3) The entropy/persistence diagnostics — are they sufficient to rule out degenerate predictability? What's a cheaper degenerate solution these diagnostics would MISS (e.g., prediction via making hidden states more similar across tokens — hurting nothing we measure but harming downstream capability)? (4) The +6pt effect size — hit@k overlap of top-2 sets: is part of the 'advantage' an artifact of the joint model's router putting more mass on fewer experts (entropy -9%)? i.e., is the predictor just having an easier job rather than the model being more 'predictable' in the sense an engine cares about? Propose the decisive metric/control. (5) Data: single FineWeb-edu stream, GPT-2 BPE, no EOS between docs — what does this threaten? (6) What is the SINGLE most important missing experiment before Tier C (OLMoE fine-tune)? READ-ONLY (quick CPU checks OK, no GPU jobs). Ranked findings with severity and concrete fixes.

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