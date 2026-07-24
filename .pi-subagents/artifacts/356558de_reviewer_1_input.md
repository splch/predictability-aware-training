# Task for reviewer

[Read from: /home/spencer/Repositories/predictability-aware-training/plan.md, /home/spencer/Repositories/predictability-aware-training/progress.md]

RED-TEAM ROUND 3 — systems claims and the planned real-engine demo at /home/spencer/Repositories/predictability-aware-training. Read RESULTS.md Exp 5/8 (cache sim, TTFT, sensitivity), cache_sim.py, and RESEARCH.md's Colibri section. The project plans to wire a trained predictor into Colibri's PILOT path (github.com/JustVugg/colibri, cloned at /tmp/pi-github-repos/JustVugg/colibri if present, else consider its README architecture: c/olmoe.c backend, PILOT=1 router-lookahead thread, per-layer LRU, async I/O pool, batch-union). Attack: (1) Are the sim's final corrected numbers (+3.2% tok/s joint-vs-base, -31% waste, +21-24% prediction-general) internally consistent and correctly computed? Re-verify key rows by rerunning cache_sim.py yourself (CPU). (2) The sim's policy (one fetch per idle window, demand-priority FIFO) vs Colibri's real PILOT (router-lookahead thread prefetching next layer's experts into an async pool) — will the demo likely show LARGER or SMALLER effects than the sim? Sign and magnitude reasoning. (3) The demo plan: OLMoE-1B-7B via c/olmoe.c + trained linear predictor heads (2048x64 per layer-horizon) exported and used for prefetch instead of/in addition to PILOT's router lookahead. What are the top 3 technical risks (weight conversion, hidden-state access inside the C engine, comparator fairness)? Propose the cheapest experimental design that would produce ONE defensible end-to-end number. (4) What could make the demo look good but be misleading (warm page cache, batch-union effects, measurement windows)? (5) Is OLMoE-on-Colibri even the right demo given Tier C showed the OLMoE backbone didn't change — should the demo instead use the Tier A toy model with a purpose-built small engine? READ-ONLY. Ranked findings + concrete design recommendation.

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