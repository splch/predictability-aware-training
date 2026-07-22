# Task for reviewer

[Read from: /home/spencer/Repositories/predictability-aware-training/plan.md, /home/spencer/Repositories/predictability-aware-training/progress.md]

RED-TEAM ROUND 2 — the systems/cache-simulator claims at /home/spencer/Repositories/predictability-aware-training. Read cache_sim.py carefully, plus the Experiment 5 section of RESULTS.md and traces (traces_base.npz, traces_joint.npz via numpy if useful). The simulator claims: honest single-disk FIFO queue, demand-priority, prefetch only into idle compute windows (one fetch per window); conclusions: bandwidth conservation dominates, thesis ordering (demand<base<joint<oracle) holds in slack regimes, joint cuts misprefetch waste 7-10%, effect sizes +0.5-3% tok/s. Attack: (1) Timing model realism — 5GB/s sequential + 0.1ms latency for 2.88MB reads: is this right for pread of random 2.88MB chunks on modern NVMe (QLC vs TLC, queue depth, page cache, readahead)? What about io_uring/batched reads (Colibri uses an async pool + batch-union single pread for 3 matrices)? (2) The 'one prefetch per idle window' policy — is this the BEST engine policy or a strawman? What would Colibri actually do (it has an async I/O pool, PIPE=1, and overlaps I/O with compute of resident experts)? Does the conclusion 'prediction barely helps' survive a better engine model? (3) The synthetic Colibri-geometry scenario — Zipf+Markov(0.3) routing: how defensible is this? Real routing has cluster/topic structure (see Colibri's expert atlas). What does the synthetic choice bias toward/against? (4) Missing dimensions: TTFT/prefill (where prediction might help most), batch-union at 256 experts (union << E*k?), hot-store pinning interaction, PCIe-offload (GPU boxes) vs NVMe. Which omission most threatens the conclusions? (5) The waste-reduction claim (63.8 vs 68.3 MB/tok): is this robust or an artifact of the synthetic predictor model (independent per-expert Bernoulli errors vs correlated real mispredictions)? (6) Verify the simulator code for remaining bugs (queue recompute correctness, LRU+waste accounting, oracle tl fix, demand-priority starvation). READ-ONLY, quick CPU numpy checks OK. Ranked findings with severity and concrete fixes.

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