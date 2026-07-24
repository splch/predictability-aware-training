# Task for reviewer

[Read from: /home/spencer/Repositories/predictability-aware-training/plan.md, /home/spencer/Repositories/predictability-aware-training/progress.md]

RED-TEAM ROUND 3 — full-arc audit of /home/spencer/Repositories/predictability-aware-training before writeup. Read RESULTS.md (all 10 experiments + 2 red-team sections), README.md, and check the run logs (run_*.log) for consistency with the published tables. The project is about to write up: (a) joint predictability training works from scratch (+6.2pts vs linear, +3-4pts vs ranking control, 3 seeds), (b) backbone property via isolation test at Tier A, (c) structure not sharpening, (d) StickyMoE defused, (e) effect grows at 4x tokens, (f) systems conversion +3.2% tok/s / -31% waste, (g) TIER C NEGATIVE: LoRA fine-tune on OLMoE does NOT transfer predictability to the backbone (posthoc-on-joint 0.800 == posthoc-on-base 0.799). Attack specifically: (1) Is the Tier C negative result itself underpowered? Single seed, single lambda (0.1), single LoRA rank (16), 25M tokens, one model — the claim 'pretraining-time only' rests on one configuration of one method of light fine-tuning. What alternative explanations exist (lambda too weak for a 4.16-nat-entropy router? LoRA capacity? predictor loss scale vs LM loss 2.1)? Which controls would a reviewer demand before accepting the negative? (2) Cross-protocol consistency: Exp 4-6 used the old stream, Exp 7+ used shuffled+EOS — are any published comparisons crossing protocols? (3) The 'co-adaptation gap is not deployable value' claim in Exp 10 — is that fully justified, or could co-adapted predictors be deployed (engine uses the exact predictor from training)? (4) Any numbers in tables that don't match logs. (5) The isolation-test logic: posthoc-on-joint ≈ posthoc-on-base is taken as 'backbone unchanged' — could it instead mean the posthoc predictor saturates a ceiling (both at 0.80, ceiling ~0.85?)? What measurement distinguishes 'backbone unchanged' from 'predictor ceiling'? READ-ONLY, quick CPU checks OK. Ranked findings with severity + fixes.

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