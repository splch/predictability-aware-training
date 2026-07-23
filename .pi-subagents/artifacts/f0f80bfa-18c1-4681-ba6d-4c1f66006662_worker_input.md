# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Build the Tier C harness in /home/spencer/Repositories/predictability-aware-training — a NEW file `train_olmoe.py` (do NOT modify model.py, train.py, or any existing file; the GPU is busy with a run, so ALL your testing must be CPU-only with tiny configs).

GOAL: fine-tune the pretrained OLMoE-1B-7B MoE (allenai/OLMoE-1B-7B-0924, downloading in the background to the HF cache) with the project's joint predictability loss, plus the controls the project's methodology requires. Use HuggingFace transformers (already installed in .venv-rocm) — do not port the architecture by hand.

CONTEXT (read first): RESEARCH.md (proposal + prior art), RESULTS.md (Experiments 4-7 — the methodology you must replicate: joint vs baseline vs posthoc-on-frozen-backbone isolation test, hit@k metric, entropy diagnostics), model.py (reference implementation of the loss/metrics — mirror its semantics), PLAN.md Tier C row.

REQUIREMENTS for train_olmoe.py:
1. Load OlmoeForCausalLM via transformers. Add a predictor module: per-(layer, horizon) linear heads from the pre-MoE hidden state at layer l to router targets at layer l+h (horizons configurable, default 1,2,4). OLMoE: 16 layers, 64 experts, top-8. Get hidden states and per-layer router logits via output_hidden_states=True and model hooks on the MoE blocks (OlmoeSparseMoeBlock / its gate) — verify actual module names by introspecting the tiny model.
2. Loss = LM loss + lambda_bal * (OLMoE's own aux loss if exposed) + lambda_pred * predictability loss: soft cross-entropy of predictor logits against the realized top-8 routing set (uniform 1/8 mass over realized top-8), exactly mirroring model.py. Gradients must flow into the backbone through the hidden states.
3. LoRA on attention and MLP (NOT router-only — the mechanism under test is representation-shaping). Implement LoRA manually (small wrapper module, freeze base, train A/B) — do NOT add the peft dependency. Trainable: LoRA params + all router/gate weights + predictor. Backbone base weights frozen.
4. Modes mirroring train.py: --mode joint (lambda_pred configurable), baseline (lambda_pred=0), and --mode posthoc --ckpt X (freeze everything except predictor, re-initialize predictor, train predictor only). --horizons, --seed, --lambda-pred, --steps, --batch, --lr, --eval-every, --device, --save args.
5. Data: reuse the project's fineweb_stream pattern (shuffled 1000-doc buffer, EOS) but with OLMoE's own tokenizer (allenai/OLMoE-1B-7B-0924) — import the stream function pattern from train.py and parameterize the tokenizer. ctx 512.
6. Eval: fixed 16-batch held-out set drawn once before training; per-eval metrics: LM loss, pred loss, hit@k per horizon (top-8 set overlap |pred∩true|/8), mean router entropy, per-layer adjacent-persistence, util_max — mirror model.py's diag semantics so numbers are comparable.
7. bf16 autocast, AdamW, cosine schedule with warmup, grad clip 1.0. Save/load state_dicts (load must tolerate predictor-key mismatch across modes: strict=False for backbone, predictor always re-initialized in posthoc).
8. CPU smoke test PROOF (this is your acceptance test — run it, capture output): create a tiny OlmoeConfig (e.g., 4 layers, hidden 128, 8 experts, top-2, vocab 1000) with random weights, and show 20 training steps on random tokens in EACH mode (joint with lambda-pred 0.1, posthoc from the saved ckpt), printing losses and eval line. Also verify with a gradient probe that in joint mode loss_pred puts nonzero gradient on a backbone LoRA param AND on a router weight (through hidden states), and that in posthoc mode all non-predictor grads are None/zero.
9. OLMoE-1B-7B specifics to handle: its router/gate is per-block; MoE blocks are every block (no shared experts); it uses QK-norm and clipped logits — don't fight the HF implementation, just hook it. If the weights download is still running, develop entirely against the tiny config.
10. Write a short header comment in the file: usage examples for the three modes, mirroring train.py's style.

Do NOT run anything on GPU. Do NOT start long downloads. Do not modify other files. Report: the file path, the smoke-test output proving gradient flow in both modes, and any OLMoE-specific gotchas discovered (exact module attr names for gate/experts).

## Acceptance Contract
Acceptance level: reviewed
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Implement the requested change without widening scope
- criterion-2: Return evidence sufficient for an independent acceptance review

Required evidence: changed-files, tests-added, commands-run, validation-output, residual-risks, no-staged-files

Review gate: required by reviewer.

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
    },
    {
      "id": "criterion-2",
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