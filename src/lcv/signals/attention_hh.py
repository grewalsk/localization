"""Accumulated-attention importance (§5.1; H2O / Scissorhands operationalization).

``importance(t) = mean over selected (layer, head) of sum over query positions
q in Q of A[layer, head, q, t]``, restricted to content tokens and normalized per
instance. Two query regions:

* **primary** (``query_region="answer"``): ``Q`` = question + generated answer
  tokens (what the model attended to *while answering*);
* **robustness** (``query_region="all"``): ``Q`` = all positions (the cumulative
  heavy-hitter definition closest to published H2O/Scissorhands).

This is a token-attribution member; it shares the attention substrate with the
retrieval-head signals, so its agreement with them is partly mechanical (§5.1,
§6). Gate 11.1d: needle tokens out-rank random context tokens (AUROC > chance).
GPU phase only — torch/TL are imported lazily and the body is stubbed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..contracts import ImportanceVector, Instance, Method

if TYPE_CHECKING:  # hints only
    import torch
    from transformer_lens import HookedTransformer


def _normalize_importance(values: np.ndarray, normalization: str) -> np.ndarray:
    """Per-instance normalization (``minmax`` | ``zscore``), uniform per run (§5)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v
    if normalization == "minmax":
        lo, hi = float(v.min()), float(v.max())
        return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    if normalization == "zscore":
        mu, sd = float(v.mean()), float(v.std())
        return (v - mu) / sd if sd > 0 else np.zeros_like(v)
    raise ValueError(f"unknown normalization {normalization!r}")


def accumulated_attention_from_patterns(
    attentions: np.ndarray,
    content_mask: np.ndarray,
    *,
    instance_id: str,
    query_positions: Sequence[int] | None = None,
    layers: Sequence[int] | None = None,
    heads: Sequence[int] | None = None,
    method: Method = Method.ACCUMULATED_ATTENTION,
    normalization: str = "minmax",
) -> ImportanceVector:
    """Accumulated-attention importance from a pre-extracted attention tensor (§5.1).

    This is the **backbone-portable core**: it takes attention already pulled off
    the model as ``attentions[layer, head, q_pos, k_pos]`` (numpy) and a boolean
    ``content_mask`` over key positions, sums the attention paid by the query
    region ``Q`` onto each key, averages over the selected heads/layers, restricts
    to content tokens, and normalizes. The CPU smoke path (HF eager) and the GPU
    path (TL ``hook_pattern``) both feed this same function, so the signal logic is
    identical regardless of how the attention was obtained.
    """
    a = np.asarray(attentions, dtype=float)
    if a.ndim != 4:
        raise ValueError(f"attentions must be [n_layers, n_heads, q, k], got {a.shape}")
    n_layers, n_heads, _n_q, n_k = a.shape
    mask = np.asarray(content_mask, dtype=bool)
    if mask.shape[0] != n_k:
        raise ValueError(f"content_mask length {mask.shape[0]} != key positions {n_k}")

    lyr = np.arange(n_layers) if layers is None else np.asarray(list(layers), dtype=int)
    hd = np.arange(n_heads) if heads is None else np.asarray(list(heads), dtype=int)
    sel = a[lyr][:, hd]  # [L, H, q, k]
    if query_positions is not None:
        sel = sel[:, :, np.asarray(list(query_positions), dtype=int), :]  # [L, H, Q, k]

    per_key = sel.sum(axis=2).mean(axis=(0, 1))  # sum over query, mean over heads/layers
    values = _normalize_importance(per_key[mask], normalization)
    return ImportanceVector(
        instance_id=instance_id, method=method, values=values, normalization=normalization
    )


def accumulated_attention_importance(
    model: HookedTransformer,
    instance: Instance,
    *,
    query_region: str = "answer",
    answer_token_ids: Sequence[int] | None = None,
    layers: Sequence[int] | None = None,
    heads: Sequence[int] | None = None,
    normalization: str = "minmax",
) -> ImportanceVector:
    """Per-content-token accumulated-attention importance (§5.1), as ACCUMULATED_ATTENTION.

    Reads eager ``hook_pattern`` over the selected ``(layer, head)`` set, sums
    attention paid by the query region ``Q`` onto each key position, averages over
    heads/layers, restricts to ``instance.content_token_mask``, and normalizes
    (``minmax`` | ``zscore``, uniform per run). ``layers``/``heads`` default to
    all 32x32 query heads (§3.2).

    GPU phase: extract ``hook_pattern`` into ``[n_layers, n_heads, q, k]`` and call
    :func:`accumulated_attention_from_patterns` (the shared, backbone-portable
    core that the CPU smoke path also uses).
    """
    raise NotImplementedError("requires GPU phase")


def needle_attention_auroc(importance: ImportanceVector, gold_mask: torch.Tensor) -> float:
    """AUROC of importance vs needle/gold-token membership (gate 11.1d helper).

    The ranking-vs-membership AUROC itself is pure-CPU
    (:func:`lcv.agreement.auroc_vs_gold`); this wrapper exists so the signal and
    its gate live together. Implemented in the phase-1 gate, not here.
    """
    raise NotImplementedError("requires GPU phase")
