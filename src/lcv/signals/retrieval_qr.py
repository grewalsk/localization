"""QRHead detection (§5.3; Query-Focused Retrieval Heads, EMNLP 2025 / 2506.09944).

The live scientific fork the project turns into a result: QRHead identifies
retrieval heads by aggregating attention onto **query-relevant** tokens using a
handful of *real-task* long-context QA examples, rather than synthetic NIAH
copy-paste. On Llama-3.1-8B, masking the top-32 QRHeads degrades NIAH at least as
much as masking the top-32 Wu heads (gate 11.1c). Wu-vs-QRHead mutual agreement
is one of the **headline disagreement numbers** (§5.3, §6) -- the paper itself
reports only 8/32 top-head overlap with the original retrieval head -- so the top
set count is matched to the Wu set for a fair Jaccard.

**QRscore, ported faithfully** (Zhang, Yin, Yen, Chen, Ye 2025, §3.1, Eq. 1-3).
For a head ``h`` whose post-softmax attention over the prompt ``{D, q}`` is
``A_h``, the per-query score aggregates the attention paid *by the query
(question) tokens* *onto the annotated gold-document tokens* ``D*``::

    QRscore_h(q) = (1/|q|) * sum_{t_q in q} sum_{t_d in D*} A_h^{t_q -> t_d}

and the detection-set score averages that over the example set ``T`` (Eq. 3)::

    QRscore_{h,T} = (1/|T|) * sum_{(q,D,D*) in T} QRscore_h(q)

Heads are ranked by ``QRscore_{h,T}`` and the top set is taken. Two things that
distinguish this from the Wu copy-paste rule and are easy to get wrong: the query
region is the **question tokens** (not the generated answer), and the keys are
restricted to the **gold-document tokens** (annotation, not the answer span). It
is a plain attention sum -- the null-query calibration in the paper enters only at
the downstream QRRetriever passage score, not at head detection.

The QRscore *mechanics* are model-independent and live here as pure-numpy,
CPU-tested cores (:func:`qr_score_from_patterns`, :func:`aggregate_qr_scores`,
:func:`qr_token_importance_from_patterns`); only the attention extraction needs
the GPU. Read **eager** attention (FlashAttention exposes no weights).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..contracts import (
    HeadScore,
    ImportanceVector,
    Instance,
    Method,
    RetrievalHeadSet,
    RetrievalSource,
)
from .normalization import normalize_importance
from .retrieval_wu import DEFAULT_TOP_K

if TYPE_CHECKING:  # hints only
    from transformer_lens import HookedTransformer

# Gate 11.1c masks the top-32 QRHeads (paper Figure 1 / §3.3); the Wu-vs-QR
# Jaccard uses DEFAULT_TOP_K (20), matched across detectors for a fair overlap.
QR_NIAH_MASK_K = 32


# --------------------------------------------------------------------------- #
# Backbone-portable CPU cores (pure numpy; the faithful QRscore mechanics)
# --------------------------------------------------------------------------- #


def qr_score_from_patterns(
    attentions: np.ndarray,
    query_positions: Sequence[int],
    gold_key_positions: Sequence[int],
) -> np.ndarray:
    """Per-head ``QRscore_h(q)`` from a pre-extracted attention tensor (Eq. 1-2, §3.1).

    Faithful port of arXiv:2506.09944 Eq. (1)-(2)::

        QRscore_h(q) = (1/|q|) * sum_{t_q in q} sum_{t_d in D*} A_h^{t_q -> t_d}

    i.e. the post-softmax attention paid *by* the query (question) tokens
    ``query_positions`` *onto* the gold-document tokens ``gold_key_positions``,
    summed over both and divided by the query length ``|q|``. ``attentions`` is
    ``[n_layers, n_heads, q, k]`` (post-softmax, eager). Returns the
    ``[n_layers, n_heads]`` score for this single ``(q, D*)`` example; average over
    the detection set ``T`` with :func:`aggregate_qr_scores` (Eq. 3).
    """
    a = np.asarray(attentions, dtype=float)
    if a.ndim != 4:
        raise ValueError(f"attentions must be [n_layers, n_heads, q, k], got {a.shape}")
    _n_layers, _n_heads, n_q, n_k = a.shape
    qpos = np.asarray(list(query_positions), dtype=int)
    gpos = np.asarray(list(gold_key_positions), dtype=int)
    if qpos.size == 0:
        raise ValueError("need >= 1 query token (|q| > 0)")
    if gpos.size == 0:
        raise ValueError("need >= 1 gold-document key token")
    if np.any((qpos < 0) | (qpos >= n_q)):
        raise ValueError(f"query position out of range [0, {n_q})")
    if np.any((gpos < 0) | (gpos >= n_k)):
        raise ValueError(f"gold key position out of range [0, {n_k})")
    sel = a[:, :, qpos][:, :, :, gpos]  # [L, H, |q|, |gold|]
    return sel.sum(axis=(2, 3)) / qpos.size  # [L, H]


def aggregate_qr_scores(per_example_scores: Sequence[np.ndarray]) -> np.ndarray:
    """Average per-example QRscore matrices over the detection set ``T`` (Eq. 3, §3.2).

    ``per_example_scores`` is a sequence of ``[n_layers, n_heads]`` matrices from
    :func:`qr_score_from_patterns`, one per detection example ``(q, D, D*)``. Returns
    their mean ``QRscore_{h,T}`` -- the matrix heads are ranked by to select QRHead.
    """
    if len(per_example_scores) == 0:
        raise ValueError("need >= 1 detection example")
    stack = np.stack([np.asarray(s, dtype=float) for s in per_example_scores])
    if stack.ndim != 3:
        raise ValueError("each example score must be 2-D [n_layers, n_heads]")
    return stack.mean(axis=0)


def qr_token_importance_from_patterns(
    attentions: np.ndarray,
    content_mask: np.ndarray,
    query_positions: Sequence[int],
    *,
    instance_id: str,
    layers: Sequence[int] | None = None,
    heads: Sequence[int] | None = None,
    normalization: str = "minmax",
) -> ImportanceVector:
    """Project the selected QRHead set onto content tokens (§6), the QR_ATTENTION member.

    The token-substrate analogue of QRscore: for each content key token ``t_d``,
    ``(1/|q|) * sum_{t_q in q} A_h^{t_q -> t_d}`` averaged over the selected QR
    ``(layer, head)`` set, restricted to ``content_mask`` and normalized. Unlike
    head detection there is no gold restriction -- every content token is a
    retrieval candidate being ranked. Same attention family as accumulated
    attention, so the load-bearing comparison is attention vs transcoder vs oracle,
    not Wu vs QR (§5.1, §6).
    """
    a = np.asarray(attentions, dtype=float)
    if a.ndim != 4:
        raise ValueError(f"attentions must be [n_layers, n_heads, q, k], got {a.shape}")
    n_layers, n_heads, n_q, n_k = a.shape
    mask = np.asarray(content_mask, dtype=bool)
    if mask.shape[0] != n_k:
        raise ValueError(f"content_mask length {mask.shape[0]} != key positions {n_k}")
    qpos = np.asarray(list(query_positions), dtype=int)
    if qpos.size == 0:
        raise ValueError("need >= 1 query token (|q| > 0)")
    if np.any((qpos < 0) | (qpos >= n_q)):
        raise ValueError(f"query position out of range [0, {n_q})")
    lyr = np.arange(n_layers) if layers is None else np.asarray(list(layers), dtype=int)
    hd = np.arange(n_heads) if heads is None else np.asarray(list(heads), dtype=int)
    sel = a[lyr][:, hd][:, :, qpos, :]  # [L, H, |q|, k]
    per_key = sel.sum(axis=2).mean(axis=(0, 1)) / qpos.size  # [k]
    values = normalize_importance(per_key[mask], normalization)
    return ImportanceVector(
        instance_id=instance_id,
        method=Method.QR_ATTENTION,
        values=values,
        normalization=normalization,
    )


def qr_head_set(score: HeadScore, k: int = DEFAULT_TOP_K) -> RetrievalHeadSet:
    """Top-``k`` QRHeads as a :class:`RetrievalHeadSet` (pure-CPU; the Wu-vs-QR Jaccard input).

    Ranks the ``QRscore_{h,T}`` matrix and wraps the top ``k`` ``(layer, head)``
    pairs. Pure-CPU: it only needs the already-computed component-substrate
    :class:`HeadScore`, so the QR-vs-Wu overlap is testable without the model.
    """
    pairs = score.top_k_heads(k)
    scores = {(layer, head): float(score.scores[layer, head]) for (layer, head) in pairs}
    return RetrievalHeadSet(source=RetrievalSource.QRHEAD, head_ids=tuple(pairs), scores=scores)


# --------------------------------------------------------------------------- #
# GPU entry points (extract eager attention, then call the CPU cores)
# --------------------------------------------------------------------------- #


def detect_qr_heads(
    model: HookedTransformer,
    qa_examples: Sequence[Instance],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> HeadScore:
    """Detect QRHeads from real-task QA attention (§5.3), as the QR_HEADS component score.

    For each example, extract eager ``[n_layers, n_heads, q, k]`` attention, score
    every head with :func:`qr_score_from_patterns` (query=question tokens,
    keys=gold-document tokens), then average over the examples with
    :func:`aggregate_qr_scores` and wrap as a ``[n_layers, n_heads]``
    :class:`HeadScore`. ``top_k`` is matched to the Wu count so the Wu-vs-QR Jaccard
    is fair. Only the attention extraction needs the GPU; the scoring is the
    CPU-tested core.
    """
    raise NotImplementedError("requires GPU phase")


def qr_token_attention(
    model: HookedTransformer,
    instance: Instance,
    head_set: RetrievalHeadSet,
    *,
    normalization: str = "minmax",
) -> ImportanceVector:
    """Project the QRHead set onto tokens (§6), as the QR_ATTENTION token member.

    GPU phase: extract eager attention for ``instance``, then call
    :func:`qr_token_importance_from_patterns` with the question tokens as the query
    region and ``head_set`` as the selected ``(layer, head)`` set. Like the Wu
    projection it is attention-family, so the load-bearing comparison is attention
    vs transcoder vs oracle, not Wu vs QR (§5.1, §6).
    """
    raise NotImplementedError("requires GPU phase")
