# State — localization-convergent-validity

## Core Value
A training-free, per-instance signal — cross-method disagreement `D(x)` — that flags which inputs an aggressive KV-compression method will break, grounded in a causal oracle so the result is more than "the tools are noisy."

## Current Focus
CPU foundation complete. The entire torch-free core — data contracts, agreement metrics (§7), substrates (§6), correctness (§9.2), leakage-safe flip model (§9.3), corruption (§8.2), NIAH data (§10), tokenization (§3.3/11.1e) — plus the §11 phase-gate suite (phase0–4) is implemented and green in CI (132 CPU tests pass). GPU instrumentation is stubbed behind typed contracts with every §11 acceptance threshold pinned; model-bound gates auto-skip without CUDA/weights. Remaining work is GPU-bound and waits on the rented H100.

## Milestone
**M0 — Convergent-Validity Result** (v0.1.0) · status: in_progress

## Phase
**Phase 0 — Environment & Parity** · status: in_progress (CPU plumbing + §11 gate scaffolding done; GPU parity gates 11.0a/b/c/d pending on the H100)

## Loop Position
```
[CPU foundation built ✓] → [provision H100] → run GPU gates phase-by-phase
```
PLAN → BUILD → groom → human checkpoint (per §13) → next phase.

## Recently Completed
- SEED ideation → graduation (project brief + design docs in repo).
- PAUL initialized: PROJECT.md, ROADMAP.md (Phase 0–4), STATE.md, paul.json.
- **Full CPU foundation (12-task ledger, all committed + pushed to origin/main):**
  contracts, agreement (§7), substrates (§6), correctness (§9.2), flip_model (§9.3),
  corruption (§8.2), niah (§10), tokenization (§3.3/11.1e), GPU-module stubs (§12),
  and §11 phase0–4 acceptance gates (CPU cores real, GPU/needs_model marked).
- CI green: ruff (lint+format) + 132 CPU tests pass; 12 GPU + 3 needs_model gates
  collected and auto-skipped where CUDA/weights are absent.

## Blockers / Watch-outs
- **Provably-correct gate:** a phase is green only when it reproduces a published number, not when it runs. The failure mode is clean-but-wrong code.
- **Do NOT** `pip install nightdessert/Retrieval_Head` (pins transformers 4.37.2; can't load Llama-3.1) — port the Wu algorithm.
- Attention-reading passes must use **eager attention** (FlashAttention returns no weights).
- Leakage rule: `D(x)` is computed from the **full-cache pass only** — structurally isolated from the compressed run.
- Compute is **rented per phase** (SF Compute H100); plumbing/tests run locally on CPU first.

## Next Action
Provision the SF Compute H100, install the `gpu` extra, and run the model-bound gates phase-by-phase (`pytest -m gpu`), starting with Phase 0 parity (11.0a/b/c/d). Before 11.1a, fill `PUBLISHED_WU_HEADS` in `tests/phase1.py` from the released Llama-3.1-8B `head_score` set.

---
*STATE.md — the live cursor. Updated every loop step.*
*Last updated: 2026-06-01*
