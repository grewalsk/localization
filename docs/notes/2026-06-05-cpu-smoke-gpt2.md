# CPU smoke on gpt2: pipeline spine validation + evaluation

*Dated note, 2026-06-05. Commits `48e0086` (smoke) / `56bfd0c` (STATE).*

## Goal

Prove the localization pipeline runs end to end on a real transformer, on a
laptop, without touching the GPU/TransformerLens science path. This validates
**plumbing only**. gpt2 is a 124M base model with no localization ground truth,
so a passing run means "the wiring runs", never "the method is right". The
validated-number path stays on TransformerLens + Llama-3.1-8B (spec/addendum
§11).

## Setup

- Model: `gpt2` (124M, 12 layers x 12 heads), CPU, **eager** attention (only
  eager exposes attention weights; same rule as the GPU path).
- Data: one synthetic NIAH instance, needle `"The magic number is 4821."` at
  depth 0.5, 6 filler sentences, seed 0.
- Spine exercised: tokenize -> §3.3 content mask -> eager attention ->
  accumulated-attention importance (§5.1) -> agreement metrics (§7).
- Reproduce: `uv run --extra smoke python scripts/smoke_cpu.py`

## Result 1: plumbing passed

Every structural gate held on a real model:

| Check | Result |
|---|---|
| Attention tensor shape | `[12, 12, 76, 76]` |
| Attention rows sum to 1 (gate 11.0d analog) | True |
| Gold needle -> real tokens -> decodes back >= 0.9 (gate 11.1e) | True (7 tokens) |
| Importance vector well-formed | len = 68 content tokens, finite, minmax in [0,1] |
| Two query regions -> finite internal D(x) | True |

The accumulated-attention signal was factored into a backbone-portable core,
`accumulated_attention_from_patterns(attentions, content_mask, ...)`, that takes
attention **tensors** (not a model object). The CPU path (HF eager) and the
future GPU path (TL `hook_pattern`) call the same function, so moving to the
H100 means rewriting one accessor, not the signals.

## Result 2: the science numbers are a clean null (and that is correct)

**The `'The' = 1.000` spike is the attention sink, not the needle** (verified):

- Top-importance token is seq-idx 0, the first content token, **not in gold**.
  The needle lives at seq 32-38 (` The magic number is 4821.`).
- Raw accumulated attention on token 0: **38.3**; everything else <= 1.18.
- Token 0 holds **51.8%** of all content attention mass; the entire 7-token
  needle holds **5.2%**.
- gpt2 has no BOS by default, so the first real token becomes the sink (the
  StreamingLLM / massive-activations phenomenon).

**The needle is not surfaced above chance:**

- AUROC = **0.600**.
- Permutation null (20k label shuffles): mean 0.500, sd 0.116,
  95% CI [0.276, 0.724].
- One-sided **p = 0.204**: indistinguishable from chance.
- Removing the sink barely moves it (0.600 -> 0.610), so the sink is not even
  what props up the (non-significant) AUROC.
- Needle token ranks among 68 content tokens: `[19, 20, 21, 22, 24, 34, 59]`.
  The lexical words ("magic/number/is") land top-third; the answer payload
  `'48' '21' '.'` rank 34/59.

This null is the right outcome. gpt2-small has no NIAH competence and no
retrieval-head circuitry. A confident AUROC here would mean the harness
manufactures signal from noise; instead it honestly reports "no localization",
which is the trust precondition before Llama-3.1.

## Implications for the H100 run

1. **Attention-sink handling in §5.1.** Here the sink fell on a *content* token
   (no BOS). On Llama-3.1 + chat template the sink mass should land on
   BOS/special tokens that the §3.3 content mask already drops. Verify
   explicitly: after specials are excluded, does the sink jump to the first
   *content* token? If yes, the accumulated-attention signal needs explicit sink
   exclusion.
2. **Normalization under a heavy outlier.** A 38x outlier under minmax crushes
   the rest to ~0.02. It hid no signal here, but consider rank-normalization or
   sink-exclusion before minmax on the real model.
3. **Per-instance AUROC is noisy** (null sd ~= 0.12 with 7 positives). Single
   instances cannot distinguish 0.6 from chance; power comes from the
   multi-instance aggregation in gate 11.1d and the D(x) design.

## Test status

- CI stays torch-free and green: **151 pass** (`import lcv` provably does not
  pull torch; `needs_model` auto-skips when transformers is absent).
- With `--extra smoke` installed: **161 pass** (adds the 7 gpt2 smoke checks and
  the previously-skipped 11.1e `needs_model` tests).
- GPU/TransformerLens stubs untouched.
