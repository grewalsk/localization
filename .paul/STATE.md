# State — localization-convergent-validity

## Core Value
A training-free, per-instance signal — cross-method disagreement `D(x)` — that flags which inputs an aggressive KV-compression method will break, grounded in a causal oracle so the result is more than "the tools are noisy."

## Current Focus
CPU foundation complete. The entire torch-free core — data contracts, agreement metrics (§7), substrates (§6), correctness (§9.2), leakage-safe flip model (§9.3), corruption (§8.2), NIAH data (§10), tokenization (§3.3/11.1e), and the three natural-text loaders (HotpotQA/LongBench/TriviaQA, §10) over a shared gold-mapping core — plus the §11 phase-gate suite (phase0–4) is implemented and green in CI (151 CPU tests pass). GPU instrumentation is stubbed behind typed contracts with every §11 acceptance threshold pinned; model-bound gates auto-skip without CUDA/weights. Remaining work is GPU-bound and waits on the rented H100.

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
- **Fixed the Gemma replication pairing** (`cef7832`): `gemma-2-9b-it` is paired with
  the *original* Gemma Scope (SAE-only), not Gemma Scope 2; transcoder signals dropped
  in replication as a finding (§13.6). Corrected at the root in the binding spec/addendum
  as dated errata.
- **Wrote the three natural-text loaders** (`lcv.data.{hotpotqa,longbench,triviaqa}`)
  over a shared `assembly.assemble_text_instance` core: pure `dict -> Instance` build
  functions (no `datasets` dep, fully CPU-testable) + lazily-guarded `load_*` HF wrappers.
  19 new tests on synthetic columnar fixtures pin gate 11.1e on loader output (gold
  located + decodes back ≥0.9, question excluded from the content mask, schema-drift
  raises); real HF downloads gated behind `LCV_RUN_DATA_DOWNLOADS=1`. **Verified all
  three HF schemas live via /browse first** — caught that all three dataset IDs were
  renamed (`hotpot_qa`→`hotpotqa/hotpot_qa`, `THUDM/LongBench`→`zai-org/LongBench`,
  `trivia_qa`→`mandarjoshi/trivia_qa`); fixed at the root as dated errata in addendum
  §10/§12 + spec §5.7. LongBench needs `trust_remote_code=True` (script-based).

## Blockers / Watch-outs
- **Provably-correct gate:** a phase is green only when it reproduces a published number, not when it runs. The failure mode is clean-but-wrong code.
- **TL-primary is a memory bet — run the 11.0b parity gate at FULL 4k context on the actual H100 BEFORE fanning out all six signal implementations against TL hooks.** TransformerLens is ergonomic but materializes a lot; at 4k context on an 8B model with attribution-patching gradients + caching it may OOM where the spec leaned toward nnsight-primary. The nnsight fallback + parity gate is the hedge. **Keep every signal backbone-portable** (read patterns/activations through a thin accessor, not TL-specific globals) so an OOM at 4k means swapping the backbone, not rewriting signals. If TL OOMs at 4k, learn it first (§13.1 checkpoint) while the signal code is still portable.
- **Replication transcoders dropped on Gemma 2.** Gemma 2 has only the original Gemma Scope (SAE-only); no transcoder suite exists for it (transcoders are Gemma-3/Gemma Scope 2 only). On the `gemma-2-9b-it` rerun, do **not** attempt to load Gemma transcoders — the transcoder-based signals are omitted and reported as a finding (§13.6, gate 11.2c); RQ1–RQ3 replicate over the attention-based signals.
- **Do NOT** `pip install nightdessert/Retrieval_Head` (pins transformers 4.37.2; can't load Llama-3.1) — port the Wu algorithm.
- Attention-reading passes must use **eager attention** (FlashAttention returns no weights).
- Leakage rule: `D(x)` is computed from the **full-cache pass only** — structurally isolated from the compressed run.
- Compute is **rented per phase** (SF Compute H100); plumbing/tests run locally on CPU first.
- **Dataset loaders are network-gated.** `load_{hotpotqa,longbench,triviaqa}` download via the `data` extra; the real-load tests only run under `LCV_RUN_DATA_DOWNLOADS=1`. `zai-org/LongBench` is script-based — it needs `trust_remote_code=True` (already passed by `load_longbench`). HF IDs are pinned as module constants (renamed from the originals; see addendum §10 errata) — do not revert to `hotpot_qa`/`THUDM/LongBench`/`trivia_qa`, they 404.

## Next Action
Provision the SF Compute H100, install the `gpu` extra, and run the model-bound gates phase-by-phase (`pytest -m gpu`), starting with Phase 0 parity (11.0a/b/c/d). **Run 11.0b at the full 4k context length first** (not a short prompt) to settle the TL-vs-nnsight memory question before any signal fan-out; if TL OOMs at 4k, take the §13.1 nnsight checkpoint while the signal code is still portable. Before 11.1a, fill `PUBLISHED_WU_HEADS` in `tests/phase1.py` from the released Llama-3.1-8B `head_score` set.

---
*STATE.md — the live cursor. Updated every loop step.*
*Last updated: 2026-06-02*
