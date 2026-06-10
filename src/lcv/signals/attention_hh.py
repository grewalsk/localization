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

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..contracts import ImportanceVector, Instance, Method
from .normalization import normalize_importance

if TYPE_CHECKING:  # hints only
    import torch
    from transformer_lens import HookedTransformer


def _normalize_importance(values: np.ndarray, normalization: str) -> np.ndarray:
    """Per-instance normalization (``minmax`` | ``zscore``); see ``normalization``.

    Thin alias kept for back-compat; delegates to the shared
    :func:`lcv.signals.normalization.normalize_importance` so the attention signal
    and the transcoder reduction normalize identically.
    """
    return normalize_importance(values, normalization)


def select_head_mean(
    reduced: np.ndarray,
    *,
    layers: Sequence[int] | None = None,
    heads: Sequence[int] | None = None,
    head_pairs: Sequence[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Mean over a selected ``(layer, head)`` set of a query-reduced ``[L, H, k]`` tensor: ``[k]``.

    The head-selection half of :func:`summed_attention_per_key`, split out so the
    full-tensor path and the streaming path (:func:`accumulate_query_key_sums`)
    share *one* selection rule. ``reduced[layer, head, k]`` is attention already
    summed over the query region. Two **mutually exclusive** modes:

    * ``head_pairs`` -- an explicit set of ``(layer, head)`` pairs (e.g. a
      :class:`RetrievalHeadSet`'s ``head_ids``). Pairs are selected *exactly* via
      element-wise advanced indexing, so a **non-rectangular** set never pulls in
      out-of-set heads. This is the M5 fix: the Wu/QR head sets are not grids, so
      projecting them through ``layers`` x ``heads`` (below) silently averages in
      the Cartesian product's off-set heads.
    * ``layers`` / ``heads`` -- a rectangular cross-product over the given axes
      (each defaults to all). Correct only when the selection truly is a full grid
      (e.g. the default all-32x32 accumulated-attention signal).
    """
    r = np.asarray(reduced, dtype=float)
    if r.ndim != 3:
        raise ValueError(f"reduced must be [n_layers, n_heads, k], got {r.shape}")
    n_layers, n_heads, _n_k = r.shape

    if head_pairs is not None:
        if layers is not None or heads is not None:
            raise ValueError("pass head_pairs OR layers/heads, not both")
        pairs = np.asarray(list(head_pairs), dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.shape[0] == 0:
            raise ValueError("head_pairs must be a non-empty sequence of (layer, head) pairs")
        ll, hh = pairs[:, 0], pairs[:, 1]
        if np.any((ll < 0) | (ll >= n_layers)):
            raise ValueError(f"layer in head_pairs out of range [0, {n_layers})")
        if np.any((hh < 0) | (hh >= n_heads)):
            raise ValueError(f"head in head_pairs out of range [0, {n_heads})")
        return r[ll, hh].mean(axis=0)  # exact pairs, NO Cartesian product -> [k]

    lyr = np.arange(n_layers) if layers is None else np.asarray(list(layers), dtype=int)
    hd = np.arange(n_heads) if heads is None else np.asarray(list(heads), dtype=int)
    return r[lyr][:, hd].mean(axis=(0, 1))  # mean over heads/layers -> [k]


def summed_attention_per_key(
    attentions: np.ndarray,
    *,
    query_positions: Sequence[int] | None = None,
    layers: Sequence[int] | None = None,
    heads: Sequence[int] | None = None,
    head_pairs: Sequence[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Attention paid by the query region onto each key, averaged over a head set: ``[k]``.

    Sums ``A[layer, head, q, k]`` over the query positions ``Q`` (all if ``None``)
    into a ``[L, H, k]`` tensor, then averages over a selected ``(layer, head)`` set
    via :func:`select_head_mean`. Two **mutually exclusive** head-selection modes
    (see :func:`select_head_mean`): ``head_pairs`` for an exact non-rectangular set
    (the M5 fix), or ``layers`` / ``heads`` for a rectangular cross-product.

    This is the **whole-tensor** path; for 4K+ contexts where the full
    ``[L, H, q, k]`` tensor would OOM, stream per layer through
    :func:`accumulate_query_key_sums` + :func:`select_head_mean` instead (M6), which
    produces the identical ``[k]`` vector. The caller applies the content mask, any
    ``|q|`` normalization, and the per-instance importance normalization.
    """
    a = np.asarray(attentions, dtype=float)
    if a.ndim != 4:
        raise ValueError(f"attentions must be [n_layers, n_heads, q, k], got {a.shape}")
    if query_positions is not None:
        a = a[:, :, np.asarray(list(query_positions), dtype=int), :]
    reduced = a.sum(axis=2)  # [L, H, k] -- sum over the query region
    return select_head_mean(reduced, layers=layers, heads=heads, head_pairs=head_pairs)


def accumulate_query_key_sums(
    layer_patterns: Iterable[np.ndarray],
    *,
    query_positions: Sequence[int] | None = None,
) -> np.ndarray:
    """Stream per-layer ``[n_heads, q, k]`` patterns into the ``[L, H, k]`` query-sum (M6).

    Memory-bounded core for the 4K+ GPU path: the full ``[L, H, q, k]`` attention
    tensor is 68.7 GB fp32 at 4K context (137 GB once upcast to fp64), so the GPU
    backend yields **one layer at a time** (see
    :func:`lcv.model.backbone.stream_eager_attention_patterns`) and this folds each
    ``[n_heads, q, k]`` layer down to ``[n_heads, k]`` immediately, holding only the
    growing ``[L, H, k]`` accumulator (~33 MB at 4K) plus the current layer in RAM.

    Sums each layer over the query region ``Q`` (all positions if
    ``query_positions`` is ``None``) into an fp64 accumulator -- matching the fp64
    reduction in :func:`summed_attention_per_key`, so feeding the per-layer slices
    of a tensor here and calling :func:`select_head_mean` on the result reproduces
    ``summed_attention_per_key`` *exactly*. Returns the ``[L, H, k]`` tensor; pass it
    to :func:`select_head_mean` for the head-selected ``[k]`` vector.
    """
    qpos = None if query_positions is None else np.asarray(list(query_positions), dtype=int)
    reduced_layers: list[np.ndarray] = []
    for layer_idx, layer in enumerate(layer_patterns):
        la = np.asarray(layer)
        if la.ndim != 3:
            raise ValueError(f"layer {layer_idx} pattern must be [n_heads, q, k], got {la.shape}")
        if qpos is not None:
            la = la[:, qpos, :]
        reduced_layers.append(la.sum(axis=1, dtype=np.float64))  # [H, k] -- fp64 query-sum
    if not reduced_layers:
        raise ValueError("layer_patterns is empty; need >= 1 layer")
    return np.stack(reduced_layers, axis=0)  # [L, H, k]


def accumulated_attention_from_patterns(
    attentions: np.ndarray,
    content_mask: np.ndarray,
    *,
    instance_id: str,
    query_positions: Sequence[int] | None = None,
    layers: Sequence[int] | None = None,
    heads: Sequence[int] | None = None,
    head_pairs: Sequence[tuple[int, int]] | None = None,
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

    Head selection (see :func:`summed_attention_per_key`): default is all 32x32
    query heads; pass ``head_pairs`` to project an exact ``(layer, head)`` set (the
    Wu/QR head projections route through this so a non-rectangular set never pulls in
    out-of-set heads, M5). ``head_pairs`` is mutually exclusive with ``layers`` /
    ``heads``.
    """
    a = np.asarray(attentions, dtype=float)
    if a.ndim != 4:
        raise ValueError(f"attentions must be [n_layers, n_heads, q, k], got {a.shape}")
    n_k = a.shape[3]
    mask = np.asarray(content_mask, dtype=bool)
    if mask.shape[0] != n_k:
        raise ValueError(f"content_mask length {mask.shape[0]} != key positions {n_k}")

    per_key = summed_attention_per_key(
        a, query_positions=query_positions, layers=layers, heads=heads, head_pairs=head_pairs
    )
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
