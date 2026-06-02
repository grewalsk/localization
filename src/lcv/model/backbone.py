"""Model backbone: TransformerLens load (+ nnsight fallback) and eager attention.

Addendum §2 (instrumentation stack), §3.1/§3.2 (models, GQA facts), §1.2 (eager
attention is mandatory for any attention-reading signal). Serves parity gate
11.0b (TL vs HF-eager logits, next-token ``KL < 1e-3``) and the attention-shape
gate 11.0d (``[batch, 32, q, k]`` per layer, rows summing to 1).

GPU phase only: every entry point needs model weights on a CUDA device, so torch
/ TransformerLens / nnsight are imported lazily (under ``TYPE_CHECKING`` for type
hints only) and the package stays CPU-importable. Each call raises
``NotImplementedError`` until the GPU phase wires it up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # hints only; never imported at runtime on the CPU path
    import torch
    from transformer_lens import HookedTransformer

# Models (§3.1).
PRIMARY_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# Replication backbone. Gemma 2 has only the *original* Gemma Scope (SAE-only); no
# transcoder suite exists for it -- Gemma Scope 2 / transcoders are Gemma-3-only.
# So the transcoder-based signals (transcoder_attr in the token-attribution
# substrate, transcoder_loadings in the component substrate) are dropped from this
# replication and the omission is reported as a finding (§13.6, gate 11.2c); RQ1-RQ3
# replicate over the attention-based signals on Gemma 2.
REPLICATION_MODEL = "google/gemma-2-9b-it"

# Llama-3.1-8B architecture facts (§3.2). ``hook_pattern`` returns 32 *query*
# heads per layer (TL expands the 8 KV heads internally); token-level eviction in
# Phase 3 is unaffected by the GQA grouping, so do not over-engineer for it.
N_LAYERS = 32
N_QUERY_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
KV_GROUP_SIZE = N_QUERY_HEADS // N_KV_HEADS  # kv_group(q) = q // 4


def load_backbone(
    model_name: str = PRIMARY_MODEL,
    *,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> HookedTransformer:
    """Load a TransformerLens ``HookedTransformer`` with **eager** attention (§2).

    Eager attention is mandatory: FlashAttention does not expose the attention
    pattern, and every attention-reading signal (§5.1-5.3) reads ``hook_pattern``.
    This is the primary instrumentation backbone; if the logit-parity gate 11.0b
    fails, switch to :func:`load_nnsight_fallback` (a §13 human checkpoint).
    """
    raise NotImplementedError("requires GPU phase")


def load_nnsight_fallback(model_name: str = PRIMARY_MODEL, *, device: str = "cuda"):
    """Load the nnsight parity-exact fallback backbone (§2; checkpoint after 11.0b).

    Used only when TransformerLens fails the logit-parity gate: nnsight runs the
    HF model directly so logits match HF exactly, at the cost of TL's turnkey hook
    points. Swapping backbones is a §13 checkpoint surfaced to the human.
    """
    raise NotImplementedError("requires GPU phase")


def forward_logits(model: HookedTransformer, tokens: torch.Tensor) -> torch.Tensor:
    """Next-token logits from the TL backbone (the TL side of parity gate 11.0b)."""
    raise NotImplementedError("requires GPU phase")


def hf_reference_logits(
    model_name: str, tokens: torch.Tensor, *, device: str = "cuda"
) -> torch.Tensor:
    """HF-eager next-token logits: the reference distribution for parity gate 11.0b.

    Gate 11.0b requires next-token ``KL(TL || HF) < 1e-3`` (or a documented
    max-abs-logit bound) on a fixed 10-prompt set; failure routes to the nnsight
    fallback. The KL computation itself is pure-CPU and lives in the phase-0 gate.
    """
    raise NotImplementedError("requires GPU phase")


def eager_attention_patterns(model: HookedTransformer, tokens: torch.Tensor) -> torch.Tensor:
    """Stacked eager attention, shape ``[batch, n_layers, 32, q_pos, k_pos]``.

    Each per-layer slice is the ``[batch, 32, q, k]`` of gate 11.0d (32 query
    heads, §3.2). The gate checks that shape and that every attention row sums to
    1.0 within fp tolerance: a ``None`` means FlashAttention slipped in, a bad row
    sum means a softmax-axis error.
    """
    raise NotImplementedError("requires GPU phase")
