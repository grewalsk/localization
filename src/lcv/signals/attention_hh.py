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

from ..contracts import ImportanceVector, Instance

if TYPE_CHECKING:  # hints only
    import torch
    from transformer_lens import HookedTransformer


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
    """
    raise NotImplementedError("requires GPU phase")


def needle_attention_auroc(importance: ImportanceVector, gold_mask: torch.Tensor) -> float:
    """AUROC of importance vs needle/gold-token membership (gate 11.1d helper).

    The ranking-vs-membership AUROC itself is pure-CPU
    (:func:`lcv.agreement.auroc_vs_gold`); this wrapper exists so the signal and
    its gate live together. Implemented in the phase-1 gate, not here.
    """
    raise NotImplementedError("requires GPU phase")
