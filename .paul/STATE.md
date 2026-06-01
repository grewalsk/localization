# State — localization-convergent-validity

## Core Value
A training-free, per-instance signal — cross-method disagreement `D(x)` — that flags which inputs an aggressive KV-compression method will break, grounded in a causal oracle so the result is more than "the tools are noisy."

## Current Focus
Project initialized from SEED graduation. Instrumentation not yet built. Ready for planning.

## Milestone
**M0 — Convergent-Validity Result** (v0.1.0) · status: not_started

## Phase
**Phase 0 — Environment & Parity** · status: not_started

## Loop Position
```
[Ready for first PLAN]
```
PLAN → BUILD → groom → human checkpoint (per §13) → next phase.

## Recently Completed
- SEED ideation → graduation (project brief + design docs in repo).
- PAUL initialized: PROJECT.md, ROADMAP.md (Phase 0–4), STATE.md, paul.json.

## Blockers / Watch-outs
- **Provably-correct gate:** a phase is green only when it reproduces a published number, not when it runs. The failure mode is clean-but-wrong code.
- **Do NOT** `pip install nightdessert/Retrieval_Head` (pins transformers 4.37.2; can't load Llama-3.1) — port the Wu algorithm.
- Attention-reading passes must use **eager attention** (FlashAttention returns no weights).
- Leakage rule: `D(x)` is computed from the **full-cache pass only** — structurally isolated from the compressed run.
- Compute is **rented per phase** (SF Compute H100); plumbing/tests run locally on CPU first.

## Next Action
Run `/paul:plan` to plan Phase 0 (Environment & Parity).

---
*STATE.md — the live cursor. Updated every loop step.*
*Last updated: 2026-06-01*
