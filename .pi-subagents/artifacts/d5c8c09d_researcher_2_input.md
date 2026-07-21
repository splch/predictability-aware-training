# Task for researcher

RED-TEAM the novelty claim of the 'predictability-aware training' project (context in /home/spencer/Repositories/predictability-aware-training/RESEARCH.md — read it first). The claim: no prior work folds expert-activation prediction accuracy into an MoE LLM's OWN training loss so the big model is optimized to BE predictable (all prior work trains predictors post-hoc against frozen models). Adversarially search for prior art that would invalidate or scope this claim: (1) any paper jointly training a router-prediction/lookahead module WITH the main model where gradients shape the main model's routing (check Pre-gated MoE 2205.10034 carefully — how exactly is its pre-gate trained? does anything shape the backbone?); (2) 'routing regularization', 'router smoothing', 'temporal consistency of expert routing', 'predictable routing', 'cache-aware routing training', 'locality-aware MoE training'; (3) speculative-decoding-style co-training of drafter+target applied to MoE experts; (4) industry/blog precedents (e.g., anything from DeepSeek/Qwen/Mistral tech reports about training for inference-friendly routing); (5) also verify our SOTA-control choice: arXiv 2511.10676 (pre-attention linear predictor, ranking-aware loss) — is that really the strongest published post-hoc predictor protocol, or is there something stronger we should compare against? Return: verdict on novelty (stands / weakened / dead), the closest 3-5 works with one-paragraph 'how it differs', and any stronger baseline we should implement.

---
Update progress at: /home/spencer/Repositories/predictability-aware-training/.pi-subagents/artifacts/progress/d5c8c09d/progress.md

---
**Output:**
Write your findings to exactly this path: /home/spencer/Repositories/predictability-aware-training/.pi-subagents/artifacts/outputs/d5c8c09d/research.md
This path is authoritative for this run.
Ignore any other output filename or output path mentioned elsewhere, including output destinations in the base agent prompt, system prompt, or task instructions.

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