# Roadmap — localization-convergent-validity

The phases mirror the engineering addendum's §11 acceptance-gate numbering (Phase 0 ↔ §11.0, … Phase 4 ↔ §11.4). A phase is **green only when the instrumentation is provably correct** (reproduces a published number), not when it runs. Human signs off at every phase boundary (§13).

**Substrates are never merged.** Agreement is measured within {token-attribution, component, answer-position} only. The leakage rule is structural: `D(x)` is computed from the full-cache pass exclusively.

---

## Milestone: M0 — Convergent-Validity Result (v0.1.0)

**Goal:** ship the end-to-end result — per-instance agreement matrices, oracle-validated method fidelity, and a leakage-safe test of whether `D(x)` predicts KV-compression flips beyond difficulty confounds — replicated on a second model family.

**Status:** not_started

---

### Phase 0 — Environment & Parity
- **Status:** not_started
- **Research flag:** infrastructure (no scientific claim; trustworthy instrumentation is the deliverable)
- **Depends on:** —
- **Goal:** stand up reproducible instrumentation whose outputs are numerically trustworthy.
- **Scope (build):** uv lockfile + `env/` model/SAE download scripts; TransformerLens `HookedTransformer` load (+ nnsight fallback path); SAELens transcoder loader (Llama Scope `LXTC`); chat-template content-token mask from tokenizer offsets.
- **Exit criteria (§11.0):**
  - coherence: model generates sensible text on held-out prompts
  - **TL↔HF KL < 1e-3** on matched forward passes (parity gate 11.0b)
  - SAE/transcoder reconstruction in published range + L0 sparsity in range (11.0c)
  - attention tensor shape `[b, 32, q, k]`, rows sum to 1 under eager attention
- **Plans:** `00-01` env+lockfile+download, `00-02` TL/HF parity, `00-03` SAE loader + reconstruction check, `00-04` chat-template content-mask.
- **Decision point:** if 11.0b fails → fall back to nnsight (§13.1).

### Phase 1 — Signals & Within-Substrate Agreement
- **Status:** not_started
- **Research flag:** science (RQ1 — do signals agree per-instance?)
- **Depends on:** Phase 0
- **Goal:** emit every contract-conformant signal and the within-substrate agreement matrices + `D(x)`.
- **Scope (build):** accumulated-attention; **ported** Wu retrieval heads (do NOT install `nightdessert/Retrieval_Head`); QRHead; ITI heads; Orgad answer-token probe; `substrates.py` projections + per-instance normalization; `agreement.py` (Spearman/Jaccard/RBO + `D(x)` + permutation + BH-FDR); gold-span→token mapping.
- **Exit criteria (§11.1):**
  - Wu reproduces published retrieval heads (11.1a)
  - top-N Wu ablation collapses NIAH vs random (11.1b)
  - top-32 QRHead ≥ top-32 Wu on retrieval (11.1c)
  - accumulated-attention ranks needle > random (11.1d)
  - gold-span decode ≥ 0.9, content-mask clean (11.1e)
- **Outcome:** agreement matrices + gold-span precision (RQ1); **Wu-vs-QRHead disagreement = headline number**.

### Phase 2 — Causal Oracle
- **Status:** not_started
- **Research flag:** science (RQ2 — does each signal track a causal oracle?)
- **Depends on:** Phase 1
- **Goal:** establish the per-instance causal ground truth and classify each method against it.
- **Scope (build):** true token-ablation `E(t)`; attribution-patching `E_hat` + fidelity gate; corruption module (symmetric token-swap, never Gaussian); method-vs-oracle fidelity.
- **Exit criteria (§11.2):**
  - **Spearman(E_hat, E) ≥ 0.8** else fall back to true ablation (11.2a)
  - corruption stability across two corruptions (11.2b)
  - transcoder gate: gold AUROC > chance & oracle-Spearman > ~0.3, else **drop transcoder token substrate and report as a finding** (11.2c, §13.6)
  - gold-span ablation → large logit drop
- **Outcome:** outcome classification — noisy tools / plural localization / method ranking (RQ2).

### Phase 3 — Compression Flip-Test (the payoff)
- **Status:** not_started
- **Research flag:** science (RQ3 — does `D(x)` predict compression failure?)
- **Depends on:** Phase 2
- **Goal:** test whether pre-measured `D(x)` predicts H2O/SnapKV flips beyond difficulty proxies.
- **Scope (build):** KVPress H2O + SnapKV budget sweep; pre-registered correctness (F1 ≥ 0.5 primary, EM robustness); logistic `flip ~ D(x)` vs `+confounds` (length, depth, confidence); 5-fold CV; LRT; calibration; **leakage-safe pipeline** (`D(x)` from full-cache run only).
- **Exit criteria (§11.3):**
  - compression reproduces paper LongBench within a few points (11.3a)
  - flip rate lands in the 5–40% band (11.3b)
  - confound-only AUROC reported **before** `D(x)`; leakage check passes (11.3c)
- **Outcome:** does `D(x)` add AUROC over the confound-only model (LRT significant) → training-free compression-fragility flag (RQ3).

### Phase 4 — Replication
- **Status:** not_started
- **Research flag:** science (cross-model evidence)
- **Depends on:** Phase 3
- **Goal:** re-run Phase 0–3 on a second model family.
- **Scope (build):** `gemma-2-9b-it` backbone + Gemma Scope 2 hook points/transcoders; re-execute the §11.0–11.3 gates and the full sweep.
- **Exit criteria (§11.4):** disagreement structure + the `D(x)` result replicate (or don't — either is reportable).
- **Outcome:** cross-family generalization claim.

---

## Open Questions (carried from ideation)
1. SF Compute single-node SKU + live H100 pricing (verify before booking).
2. Parity fallback to nnsight if 11.0b fails (§13.1).
3. Drop transcoder token substrate if 11.2c fails — report as a finding (§13.6).
4. Flip-test budget points tuned to the 5–40% band (§13.7).
5. Probing datasets: ITI needs a labeled true/false set distinct from QA evals; QRHead needs real long-context QA examples.
6. Gemma-2-9B architecture facts + Gemma Scope 2 hook points (Phase 4).

---
*ROADMAP.md — the phase plan. Updated by `/paul:plan` and at phase boundaries.*
*Last updated: 2026-06-01*
