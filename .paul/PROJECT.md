# localization-convergent-validity

## What This Is

A research harness that tests whether the localization signals deployed in LLM inference (KV heavy-hitters, retrieval heads, transcoder attribution, internal-confidence probes) agree per-instance on *where information lives*, validates each against a per-instance causal oracle, and tests whether their disagreement predicts KV-compression failure. Imported from a SEED-graduated PLANNING.md backed by two binding design docs (`docs/`).

## Core Value

A training-free, per-instance signal — cross-method disagreement `D(x)` — that flags which inputs an aggressive KV-compression method will break, grounded in a causal oracle so the result is more than "the tools are noisy."

## Current State

| Attribute | Value |
|-----------|-------|
| Type | Application (research pipeline) |
| Version | 0.0.0 |
| Status | Initializing |
| Last Updated | 2026-06-01 |

## Requirements

### Core Features

- **Localization signals** — accumulated-attention, Wu retrieval heads, QRHead, transcoder attribution, ITI heads, Orgad answer-token probe (each emits a contract-conformant per-token vector or per-head score).
- **Agreement engine** — within-substrate Spearman/Jaccard/RBO + per-instance disagreement `D(x)` + permutation/FDR (substrates never merged).
- **Causal oracle** — true token/span ablation `E(t)` + validated attribution-patching approximation.
- **Compression flip-test** — H2O/SnapKV budget sweep + leakage-safe logistic model predicting flips from `D(x)` beyond difficulty confounds.
- **Replication** — the whole pipeline on a second model family (`gemma-2-9b-it` + the original Gemma Scope, SAE-only). Gemma 2 has no transcoder suite, so the transcoder-based signals are dropped in replication and reported as a finding (§13.6, gate 11.2c); the attention-based signals carry RQ1–RQ3.

### Validated (Shipped)
None yet.

### Active (In Progress)
None yet.

### Planned (Next)
- Phase 0 — Environment & Parity → Phase 1 Signals & Agreement → Phase 2 Causal Oracle → Phase 3 Flip-Test Payoff → Phase 4 Replication. See ROADMAP.md.

### Out of Scope
- Head-level eviction (DuoAttention) and its GQA query→KV-group pooling — deferred to future work; the flip test uses token-level eviction only.
- LLM-judge correctness labeling in the core — optional robustness pass only.
- 40K+ context — the oracle-validated core runs at 1K–4K.

## Target Users

**Primary:** mechanistic-interpretability researchers (the project owner) studying localization, KV compression, and SAEs.
- Fluent in TransformerLens / SAELens / activation patching and the relevant literature.
- Needs results that withstand the MIB / Kantamneni / Hase reviewer challenges.

## Context

**Technical Context:** single 80GB H100 rented via SF Compute; ephemeral nodes make the uv lockfile + model/SAE download scripts load-bearing. The binding implementation contract is `docs/…engineering-addendum.md` (§11 acceptance tests, §12 module layout, §13 human checkpoints).

## Constraints

### Technical Constraints
- Single 80GB GPU (H100); 1K–4K context core; sentence/span-granularity oracle; attribution patching for the full sweep with subsample verification.
- Attention-reading passes must use eager attention (FlashAttention returns no weights).
- Do **not** pip-install `nightdessert/Retrieval_Head` (pins transformers 4.37.2, can't load Llama-3.1) — port the algorithm.
- Llama Scope SAEs are base-trained but applied to Instruct — guarded by the captured-variance control.
- Attention-output SAEs (`LXA`) forbidden (authors' dead-feature warning).

### Business Constraints
- Single-researcher, single-GPU footing; rented compute booked per phase.

## Key Decisions

| Decision | Rationale | Date | Status |
|----------|-----------|------|--------|
| TransformerLens primary, nnsight fallback | Unambiguous hooks + turnkey patching; drift closed by gate 11.0b | 2026-06-01 | Active |
| Instruct model + base-trained SAEs + variance control | Deployment realism is the point; variance control guards interpretation | 2026-06-01 | Active |
| Transcoders over residual SAEs (8x/32K) | Built for clean attribution | 2026-06-01 | Active |
| Wu *and* QRHead both included | Turns a live scientific fork into a headline disagreement number | 2026-06-01 | Active |
| Token ablation as the oracle | Same operation as KV eviction → oracle measures what the flip test stresses | 2026-06-01 | Active |
| Token-level eviction (H2O, SnapKV) | Sidesteps the GQA query/KV-head granularity issue | 2026-06-01 | Active |
| Deterministic correctness (F1≥0.5; no LLM judge in core) | Reproducibility | 2026-06-01 | Active |
| Corruption = token-swap (never Gaussian) | Stable, semantically controlled | 2026-06-01 | Active |
| Full Phase 0–4 scope incl. Gemma replication | Cross-model evidence | 2026-06-01 | Active |

## Success Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Phase 0 parity (11.0b) | TL↔HF KL < 1e-3 | - | Not started |
| Signal reproduction (11.1a/b/c) | Wu heads reproduce; ablation collapses NIAH; QRHead ≥ Wu | - | Not started |
| Oracle fidelity (11.2a) | Spearman(E_hat, E) ≥ 0.8 | - | Not started |
| Flip prediction (RQ3) | D(x) adds AUROC over confound-only model (LRT significant) | - | Not started |
| Replication (Phase 4) | Disagreement structure + D(x) result replicate on Gemma | - | Not started |

## Tech Stack / Tools

| Layer | Technology | Notes |
|-------|------------|-------|
| Instrumentation | TransformerLens (primary), nnsight (fallback) | string hook points; parity-gated |
| SAE / transcoder | SAELens + Llama Scope (transcoders) / original Gemma Scope (SAE-only) | `LXTC` 8x transcoders on Llama; **no Gemma-2 transcoders** → transcoder signals dropped in replication (§13.6) |
| KV compression | KVPress | H2O / SnapKV under HF generate |
| Stats | scipy, scikit-learn, statsmodels | agreement + logistic flip model |
| Models | Llama-3.1-8B-Instruct, gemma-2-9b-it | primary + replication |
| Runtime | Python 3.10+, torch 2.1+, transformers 4.44+ | uv lockfile |
| Compute | single 80GB H100 | SF Compute (rented per phase) |

## Links

| Resource | URL |
|----------|-----|
| Spec | docs/localization-convergent-validity-spec (1).md |
| Engineering addendum | docs/localization-convergent-validity-engineering-addendum.md |
| Ideation plan | ~/projects/localization-convergent-validity/PLANNING.md |
| Origin repo | https://github.com/grewalsk/localization |

---
*PROJECT.md — Updated when requirements or context change*
*Last updated: 2026-06-01*
