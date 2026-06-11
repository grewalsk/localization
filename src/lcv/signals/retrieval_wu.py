"""Wu retrieval-head detection (§5.2; port of nightdessert/Retrieval_Head).

Decoding-time copy-paste rule, stated cleanly:

    Decode the answer autoregressively. At each step where the generated token g
    is part of the needle answer, for each query head h let
    ``j = argmax_k A[layer, h, current_pos, k]``; if the token at key position j
    equals g AND j lies inside the needle span, increment head h's hit count.
    retrieval_score(h) = hits(h) / (needle tokens generated), averaged over many
    (needle, position) samples.

Implementation hazards that cause silent errors (§5.2): read **eager** attention;
score only steps where the generated token is in the needle (not all steps); the
"inside the needle span" check requires the needle's token indices in the full
sequence (reuse the content-token mask machinery). Output, matching the reference
for cross-checkability: ``head_score/*.json`` as ``{"layer-head": [per-sample
scores]}``. Sample budget 20-50 (needle, position) pairs; 4K-8K context suffices.

Reproduces gates 11.1a (overlap with published Llama-3.1-8B heads) and 11.1b
(ablating top-N collapses NIAH, random N does not). GPU phase only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..contracts import HeadScore, ImportanceVector, Instance, RetrievalHeadSet, RetrievalSource

if TYPE_CHECKING:  # hints only
    from transformer_lens import HookedTransformer

# (min, max) (needle, position) samples; few samples stably surface top heads (§5.2).
WU_SAMPLE_BUDGET = (20, 50)
WU_DETECTION_CONTEXT = 4096  # 4K-8K is sufficient and faster than 50K (§5.2)
DEFAULT_TOP_K = 20  # matched to QR/ITI sets for fair Jaccard (§5.3/§5.4)


def detect_wu_retrieval_heads(
    model: HookedTransformer,
    niah_instances: Sequence[Instance],
    *,
    max_samples: int = WU_SAMPLE_BUDGET[1],
    context_len: int = WU_DETECTION_CONTEXT,
) -> HeadScore:
    """Detect Wu copy-paste retrieval heads (§5.2), as the WU_HEADS component score.

    Runs the decoding-time scoring rule over ``niah_instances`` and returns a
    ``[n_layers, n_heads]`` :class:`HeadScore` of mean copy-paste rates. Persist
    per-sample scores as ``head_score/*.json`` for cross-checking against the
    released set (gate 11.1a). A wrong score here fails 11.1a/11.1b.
    """
    raise NotImplementedError("requires GPU phase")


def wu_retrieval_head_set(score: HeadScore, k: int = DEFAULT_TOP_K) -> RetrievalHeadSet:
    """Top-``k`` Wu heads as a :class:`RetrievalHeadSet` (pure-CPU; the Wu-vs-QR Jaccard input).

    Ranks the detection :class:`HeadScore` and wraps the top ``k`` ``(layer, head)``
    pairs. Pure-CPU like :func:`lcv.signals.retrieval_qr.qr_head_set`: it only needs the
    already-computed component-substrate score, so the Wu-vs-QR overlap (gate 11.1a) is
    testable without the model. The stable tie-break in
    :meth:`~lcv.contracts.HeadScore.top_k_heads` keeps the selected set deterministic (M1).
    """
    pairs = score.top_k_heads(k)
    scores = {(layer, head): float(score.scores[layer, head]) for (layer, head) in pairs}
    return RetrievalHeadSet(source=RetrievalSource.WU, head_ids=tuple(pairs), scores=scores)


def wu_token_attention(
    model: HookedTransformer,
    instance: Instance,
    head_set: RetrievalHeadSet,
    *,
    normalization: str = "minmax",
) -> ImportanceVector:
    """Project the Wu head set onto tokens (§6), as the WU_ATTENTION token member.

    Sums the top-Wu-heads' attention onto each content token and normalizes,
    giving the token-attribution-substrate vector (distinct from the component
    WU_HEADS score). Shares the attention family with accumulated attention and
    QRHead, so their mutual agreement is partly mechanical (§5.1, §6).

    GPU phase: extract eager attention for ``instance``, then call
    :func:`lcv.signals.attention_hh.accumulated_attention_from_patterns` with
    ``head_pairs=head_set.head_ids`` and ``method=Method.WU_ATTENTION``. Routing the
    head set through ``head_pairs`` selects the exact ``(layer, head)`` pairs; the Wu
    set is non-rectangular, so a ``layers`` x ``heads`` projection would average in
    out-of-set heads (M5).
    """
    raise NotImplementedError("requires GPU phase")
