# Task for reviewer

[Read from: /home/spencer/Repositories/predictability-aware-training/plan.md, /home/spencer/Repositories/predictability-aware-training/progress.md]

RED-TEAM the code in /home/spencer/Repositories/predictability-aware-training (model.py, train.py). This is a research harness training a GPT-style MoE with a jointly-trained expert-activation predictor; the scientific claim rests on this code being correct. Adversarially hunt for bugs that could invalidate results, specifically: (1) the MoE dispatch math (per-expert gather/scatter, top-k weight renormalization, the `w.sum(dim=-1)` weighting); (2) the Switch-style load-balancing loss formulation; (3) the predictability loss — target construction (soft scatter of realized top-k), whether gradients flow to the backbone as intended in joint mode, and whether posthoc mode truly freezes everything else; (4) hit@k metric computation (does it measure top-k set overlap correctly?); (5) train/eval contamination — the FineWeb streaming generator is shared between training steps and evaluate() calls; is eval actually held out?; (6) LR schedule, autocast bf16 correctness, checkpoint/load issues (posthoc loads a full state dict then re-initializes predictor? check ParameterDict init and load ordering); (7) anything about horizons indexing (layer_hiddens[l] predicts topks[l+h]). READ-ONLY: do not modify files, do not run training (GPU is busy with a run). You may run quick CPU python snippets with the .venv-rocm or .venv interpreters to verify suspicions. Report each finding with severity (critical/major/minor), file:line, and a concrete fix.

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