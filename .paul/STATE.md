# State — localization-convergent-validity

## Core Value
A training-free, per-instance signal — cross-method disagreement `D(x)` — that flags which inputs an aggressive KV-compression method will break, grounded in a causal oracle so the result is more than "the tools are noisy."

## Current Focus
**CPU foundation COMPLETE — all 23 CPU/non-GPU tasks done, green, committed, pushed.** The entire torch-free core — data contracts, agreement metrics (§7), substrates (§6), correctness (§9.2), leakage-safe flip model (§9.3), corruption (§8.2), masking-primary oracle (§8.1), NIAH data (§10), tokenization (§3.3/11.1e), the three natural-text loaders (HotpotQA/LongBench/TriviaQA, §10), the two probing/detection loaders (ITI/TruthfulQA §5.4, QRHead/LongMemEval §5.3), the faithful QRscore detector core (arXiv:2506.09944 Eq. 1-3, §5.3), the transcoder input-gradient + firing-position reductions (§4.4), and the §11.2a adjudication-subset fallback — all over shared backbone-portable cores (`*_from_patterns`, tensor-in) — plus the §11 phase-gate suite (phase0–4) is implemented and green (**265 CPU tests pass**, 5 download-gated skips, 13 GPU deselected). The **pre-experiment audit is fully closed**: all 22 findings (1 BLOCKER + 11 MAJORs + 10 MINORs) are fixed across 8 commits, each verified against source first, all pushed to origin/main. GPU instrumentation is stubbed behind typed contracts with every §11 acceptance threshold pinned; model-bound gates auto-skip without CUDA/weights. A separate **CPU-only smoke path** (`smoke` extra: gpt2 + HF eager attention) runs the full pipeline spine end-to-end on a laptop. Plumbing only — gpt2 has no localization ground truth, so passing means "the wiring runs", not "the method is right". Remaining science work is GPU-bound and waits on the rented H100.

## Milestone
**M0 — Convergent-Validity Result** (v0.1.0) · status: in_progress

## Phase
**Phase 0 — Environment & Parity** · status: in_progress (CPU plumbing + §11 gate scaffolding done; GPU parity gates 11.0a/b/c/d pending on the H100)

## Loop Position
```
[CPU foundation built ✓ — ALL CPU work done] → [provision H100 ← NEXT] → run GPU gates phase-by-phase
```
PLAN → BUILD → groom → human checkpoint (per §13) → next phase.

## Recently Completed
- **Pre-experiment audit fully closed — 22 findings (1 BLOCKER + 11 MAJORs + 10
  MINORs) across 8 commits, all pushed to origin/main, each verified against source
  before changing:**
  - `4311716` B1+M11 — content mask drops in-text template specials by id + clips to
    the verbatim context char range (the §3.3 leakage trap).
  - `d5e6d99` M5+M6 — exact non-rectangular head-pair selection + per-layer attention
    streaming (4k-OOM avoidance); streamed == whole-tensor.
  - `4856ac1` M8+M9+m5 — Ê pinned to attention-pattern AtP, seeded adjudication
    subset, attribution-patching credited to Nanda 2022 / Kramár 2024 (not Syed).
  - `194a8bd` M1–M4 — NaN-guard degenerate rankings, undefined permutation tests, and
    flat D(x); Benjamini-Yekutieli fallback under dependence.
  - `c1d9c64` M10+M7 — two-factor Llama-Scope transcoder fold (nsf_out/nsf_in ≈18.9×
    decoder inflation at L8) + ≥4k-token TL/HF parity gate.
  - `cc7f9fd` m1+m2+m4 — stable ascending-index tie-break, CPU Wu head-set selector,
    component-bundle single-instance guard.
  - `a569dfd` m3+m6+m8 — single normalization core, locate-all-text, n_effective_pairs
    recorded on D(x).
  - `2289da1` m7+m9+m10 — permutation-null doc reconciled to the paper's "shuffle one
    ranking", ITI K=20-vs-48 note, pre-wired llama-2-7b-80k Wu fixture (33 heads
    reproduced directly from Wu's released head_score at mean≥0.1).
- **Paper-revision CPU tasks 17–23 (all committed + pushed to origin/main):**
  - `b51d760` — masking-primary oracle (`oracle/masking.py`, §8.1 Definition 2) +
    transcoder input-gradient/firing-position reductions (§4.4) + §11.2a
    adjudication-subset fallback (`ADJUDICATION_SUBSET_SIZE=150`).
  - `a2caba5` — faithful QRscore head detection (`signals/retrieval_qr.py`,
    arXiv:2506.09944 Eq. 1-3): question-tokens→gold-document-tokens attention, /|q|,
    averaged over the detection set; backbone-portable `qr_score_from_patterns` core
    + 7 CPU tests. Verified it is a real QRscore port, not hand-rolled (task 21/#3).
  - `4c53d41` — ITI (`data/iti.py`, TruthfulQA §5.4) + QRHead (`data/qrhead_qa.py`,
    LongMemEval §5.3) probing/detection loaders, pure `dict -> pairs/Instance` cores,
    held out from the §10 QA eval sets; 15 CPU tests + 2 download-gated. Verified both
    HF schemas live first (TruthfulQA mc1/mc2; LongMemEval cleaned repo, raw-JSON
    because `answer` is sometimes scalar).
  - `d3de2f1` — verified Llama-Scope checkpoint provenance (task 23/#7) + documented
    the Wu-head gap (task 23/#4). See Blockers for the two findings.
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
- **CPU-only end-to-end smoke on gpt2** (`48e0086`): new `lcv.model.hf_backbone`
  reads HF eager attention as plain numpy `[L,H,q,k]` + the §3.3 content mask, with
  torch/transformers imported lazily so `import lcv` stays torch-free (verified). It
  feeds the *same* signal logic as the GPU path via a new backbone-portable core,
  `accumulated_attention_from_patterns` (tensor-in, no model object). `scripts/smoke_cpu.py`
  + `tests/test_cpu_smoke.py` (`needs_model`) prove the spine on gpt2: attention rows
  sum to 1 (gate 11.0d analog), gold needle maps back through the real tokenizer (gate
  11.1e), importance well-formed, two query regions → finite D(x). Added the `smoke`
  extra (torch+transformers only; **not** the TL/CUDA science stack — GPU/TL stubs
  untouched). Also hardened two latent `needs_model` tests this extra exercised for the
  first time (transformers was never installed before, so they always auto-skipped):
  `test_import_is_torch_free` now checks in a fresh subprocess (process-global
  `sys.modules` is polluted once any sibling test imports torch), and phase1 11.1e
  enforces specials-exclusion only when the template actually emits specials (the CI
  stand-in `hf-internal-testing/llama-tokenizer` is the Llama-2 `[INST]` template whose
  role tags are ordinary text — nothing to drop; the real Llama-3.1 template has true
  specials). Gold-maps-back is still asserted every run; specials-dropping stays pinned
  model-free by the synthetic CPU-core test.

## Blockers / Watch-outs
- **Provably-correct gate:** a phase is green only when it reproduces a published number, not when it runs. The failure mode is clean-but-wrong code.
- **TL-primary is a memory bet — run the 11.0b parity gate at FULL 4k context on the actual H100 BEFORE fanning out all six signal implementations against TL hooks.** TransformerLens is ergonomic but materializes a lot; at 4k context on an 8B model with attribution-patching gradients + caching it may OOM where the spec leaned toward nnsight-primary. The nnsight fallback + parity gate is the hedge. **Keep every signal backbone-portable** (read patterns/activations through a thin accessor, not TL-specific globals) so an OOM at 4k means swapping the backbone, not rewriting signals. If TL OOMs at 4k, learn it first (§13.1 checkpoint) while the signal code is still portable.
- **Replication transcoders dropped on Gemma 2.** Gemma 2 has only the original Gemma Scope (SAE-only); no transcoder suite exists for it (transcoders are Gemma-3/Gemma Scope 2 only). On the `gemma-2-9b-it` rerun, do **not** attempt to load Gemma transcoders — the transcoder-based signals are omitted and reported as a finding (§13.6, gate 11.2c); RQ1–RQ3 replicate over the attention-based signals.
- **Do NOT** `pip install nightdessert/Retrieval_Head` (pins transformers 4.37.2; can't load Llama-3.1) — port the Wu algorithm.
- **Llama-Scope transcoders are NOT in the SAELens registry** (verified 2026-06-07 against `sae_lens/pretrained_saes.yaml`: only `llama_scope_lx{r,m,a}_{8,32}x` exist — R/M/A, zero TC entries). The transcoder `final.safetensors` DO exist on HF (`fnlp/Llama3_1-8B-Base-LXTC-{8,32}x` → canonical `OpenMOSS-Team/*`, per-layer `Llama3_1-8B-Base-L{n}TC-{8,32}x/checkpoints/final.safetensors`). So on the GPU: R/M load via `SAE.from_pretrained(release, sae_id)` (resolved by `sae_loader.llama_scope_sae_lens_release`); **TC must load DIRECTLY from the per-layer safetensors** (`sae_loader.llama_scope_checkpoint_ref` gives `(repo, subdir)`). `SAE.from_pretrained("llama_scope_lxtc_8x", …)` would miss the registry — a clean-but-wrong trap, now blocked by a CPU test (phase0).
- **No Wu-released Llama-3.1-8B retrieval-head set exists** (verified 2026-06-07). Wu's `head_score/` ships only llama-2-7b-80k, llama-2-13b-64k, Mistral-7B-v0.2, Mixtral-8x7B, Qwen1.5-14B(-Chat), Yi-6B-200K — not Llama-3.1-8B (our PRIMARY_MODEL); the QRHead repo (`princeton-pli/QRHead`) ships QR-head sets for Llama-3.1-8B-Instruct but recomputes Wu/RetHead on the fly and stores no Wu list. So `PUBLISHED_WU_HEADS` stays `frozenset()` **by design, not omission** — do NOT fabricate it from a secondary source. To run gate 11.1a as a reproduce-published-numbers check, detect on a model Wu DID release: head score = mean of the per-example list, retrieval head iff ≥0.1; **llama-2-7b-80k yields 33 heads** (top 16-19, 11-15, 8-26). Wiring 11.1a to load llama-2-7b-80k is a GPU-phase design choice to confirm with the user.
- Attention-reading passes must use **eager attention** (FlashAttention returns no weights).
- **Attention-sink confound in accumulated-attention (§5.1), surfaced by the CPU smoke** (see `docs/notes/2026-06-05-cpu-smoke-gpt2.md`). On gpt2 the first content token absorbed **51.8%** of all content attention mass (raw 38.3 vs ≤1.18); the NIAH needle drew 5.2% and ranked at chance (AUROC 0.600, permutation p=0.20, indistinguishable from 0.5). gpt2 has no BOS, so the sink fell on a *content* token. On Llama-3.1 + chat template the sink mass should land on BOS/special tokens already dropped by the §3.3 content mask — **verify explicitly: after specials are excluded, does the sink jump to the first *content* token?** If yes, the accumulated-attention signal needs explicit sink exclusion (and/or rank-normalization before minmax, which a 38× outlier otherwise dominates). The smoke's job is exactly this: surface methodology issues cheaply, not validate numbers.
- Leakage rule: `D(x)` is computed from the **full-cache pass only** — structurally isolated from the compressed run.
- Compute is **rented per phase** (SF Compute H100); plumbing/tests run locally on CPU first.
- **Dataset loaders are network-gated.** `load_{hotpotqa,longbench,triviaqa}` download via the `data` extra; the real-load tests only run under `LCV_RUN_DATA_DOWNLOADS=1`. `zai-org/LongBench` is script-based — it needs `trust_remote_code=True` (already passed by `load_longbench`). HF IDs are pinned as module constants (renamed from the originals; see addendum §10 errata) — do not revert to `hotpot_qa`/`THUDM/LongBench`/`trivia_qa`, they 404.

## Next Action
**All CPU work is green and pushed (origin/main @ `d3de2f1`); the gate to provision the H100 is satisfied.** Next: provision the SF Compute H100, install the `gpu` extra, and run the model-bound gates phase-by-phase (`pytest -m gpu`), starting with Phase 0 parity (11.0a/b/c/d). **Run 11.0b at the full 4k context length first** (not a short prompt) to settle the TL-vs-nnsight memory question before any signal fan-out; if TL OOMs at 4k, take the §13.1 nnsight checkpoint while the signal code is still portable. Gate 11.1a cannot use a Llama-3.1-8B Wu set (none released — see Blockers); enable it by detecting on a Wu-released model (llama-2-7b-80k, 33 heads at ≥0.1) — confirm this scope with the user first. (Provisioning spends real money and needs the user's SF Compute account, so it is a human-checkpoint handoff, not an autonomous step.)

---
*STATE.md — the live cursor. Updated every loop step.*
*Last updated: 2026-06-07*
