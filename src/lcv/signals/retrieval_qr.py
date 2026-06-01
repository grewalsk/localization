"""QRHead detection (§5.3; Query-Focused Retrieval Heads, EMNLP 2025 / 2506.09944).

The live scientific fork the project turns into a result: QRHead identifies
retrieval heads by aggregating attention onto **query-relevant** tokens using a
handful of *real-task* long-context QA examples, rather than synthetic NIAH
copy-paste. On Llama-3.1-8B, masking the top-32 QRHeads degrades NIAH at least as
much as masking the top-32 Wu heads (gate 11.1c). Wu-vs-QRHead mutual agreement
is one of the **headline disagreement numbers** (§5.3, §6), so the top set count
is matched to the Wu set for a fair Jaccard.

Algorithm (§5.3): for a small set of real long-context QA examples, accumulate
each head's attention mass onto the query-relevant tokens, rank heads by that
accumulated mass, take the top set. GPU phase only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..contracts import HeadScore, ImportanceVector, Instance, RetrievalHeadSet
from .retrieval_wu import DEFAULT_TOP_K

if TYPE_CHECKING:  # hints only
    from transformer_lens import HookedTransformer


def detect_qr_heads(
    model: HookedTransformer,
    qa_examples: Sequence[Instance],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> HeadScore:
    """Detect QRHeads from real-task QA attention (§5.3), as the QR_HEADS component score.

    Accumulates each query head's attention mass onto query-relevant tokens over
    ``qa_examples`` and returns a ``[n_layers, n_heads]`` :class:`HeadScore`.
    ``top_k`` is matched to the Wu count so the Wu-vs-QR Jaccard is fair.
    """
    raise NotImplementedError("requires GPU phase")


def qr_head_set(score: HeadScore, k: int = DEFAULT_TOP_K) -> RetrievalHeadSet:
    """Top-``k`` QRHeads as a :class:`RetrievalHeadSet` (the Wu-vs-QR Jaccard input)."""
    raise NotImplementedError("requires GPU phase")


def qr_token_attention(
    model: HookedTransformer,
    instance: Instance,
    head_set: RetrievalHeadSet,
    *,
    normalization: str = "minmax",
) -> ImportanceVector:
    """Project the QRHead set onto tokens (§6), as the QR_ATTENTION token member.

    Sums the top-QR-heads' attention onto each content token and normalizes. Like
    the Wu projection it is attention-family, so the load-bearing comparison is
    attention vs transcoder vs oracle, not Wu vs QR (§5.1, §6).
    """
    raise NotImplementedError("requires GPU phase")
