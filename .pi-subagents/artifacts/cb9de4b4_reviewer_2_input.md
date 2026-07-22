# Task for reviewer

[Read from: /home/spencer/Repositories/predictability-aware-training/plan.md, /home/spencer/Repositories/predictability-aware-training/progress.md]

RED-TEAM ROUND 2 — strategy, positioning, and next steps of /home/spencer/Repositories/predictability-aware-training. Read RESEARCH.md (especially the re-scoped novelty claim + training-time prior art table), RESULTS.md (all 6 experiments + red-team sections), PLAN.md. The project plans Tier C next: fine-tune OLMoE-1B-7B (real pretrained MoE, Colibri-supported) with the joint predictability loss. Attack the strategy: (1) Is Tier C the right next move, or is something cheaper/more decisive being skipped? Consider: the unresolved StickyMoE baseline (lambda_sticky=0.1 was 'too weak', inconclusive, and left hanging), the missing ranking-aware SOTA posthoc control (2511.10676 protocol), a longer Tier A run (the 75x-undertrained confound hangs over EVERY claim — would 10x more tokens change the answer?), or TTFT measurement. Rank what you would actually do next and why. (2) Tier C design critique: OLMoE fine-tune with router+predictor-only trainable vs full fine-tune — what does each prove? Is 1B-active/7B-total with 64 experts the right target? What does success/failure look like, and can this box (31 TFLOPS bf16 iGPU, 128GB) execute it in reasonable time? (3) Publication posture: given StickyMoE/Oracle-MoE/ReMoE/Pre-gated MoE, what is the strongest honest framing — a workshop paper? a systems co-design paper? What experiments would a skeptical MLSys/ICLR reviewer demand that are missing? (4) The throughput reframing in Exp 5 finding 5 ('accuracy multiplier on the prefetch component', 'TTFT may be the headline') — is this a graceful retreat or the truth? What measurement would settle it? (5) Anything in the git history/docs that overclaims? Check RESEARCH.md/RESULTS.md/PLAN.md wording against the actual evidence. READ-ONLY. Ranked findings with severity and concrete actions.

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