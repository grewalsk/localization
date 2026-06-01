"""True token/span ablation oracle ``E(t)`` (§8.1).

The ground-truth per-token causal effect:

    E(t) = logP(answer | full context) - logP(answer | token t masked from all
                                                       attention)

This is an **ablation** (remove the token), deliberately the *same operation* as
the downstream failure mode (KV eviction removes tokens from the cache), so the
oracle measures exactly what the flip test stresses (§8.1). Cost is ``O(context
length)`` forward passes per instance, which §8.3 approximates and §8.4 mitigates
(sentence granularity where gold is sentence-level, 1K-4K context).

Ablation is distinct from patching: ablation removes a token, patching swaps a
token's activation from a corrupted run. Use ablation for ``E(t)``; see
:mod:`lcv.oracle.attr_patching` for the validated cheap approximation. GPU phase
only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..contracts import Granularity, Instance, OracleEffect

if TYPE_CHECKING:  # hints only
    import torch
    from transformer_lens import HookedTransformer


def answer_log_prob(
    model: HookedTransformer,
    tokens: torch.Tensor,
    answer_token_ids: Sequence[int],
    *,
    answer_start: int | None = None,
) -> float:
    """``logP(answer | context)`` under the model (the term ``E(t)`` differences).

    Sums the teacher-forced log-probs of the answer tokens. Gate 11.2d wiring
    depends on this being correct, so it is the one place to get the answer-metric
    plumbing right.
    """
    raise NotImplementedError("requires GPU phase")


def token_ablation_effect(
    model: HookedTransformer,
    instance: Instance,
    *,
    granularity: Granularity = Granularity.TOKEN,
    answer_token_ids: Sequence[int] | None = None,
) -> OracleEffect:
    """Compute the true ablation oracle ``E`` over content tokens/spans (§8.1).

    Masks each content token (or sentence span, §8.4) from all attention and
    measures the drop in answer log-prob, returning an :class:`OracleEffect` with
    ``E`` 1-D over the content units. Sentence granularity matches HotpotQA's
    sentence-level gold and cuts the sweep by ~10x.
    """
    raise NotImplementedError("requires GPU phase")


def gold_span_ablation_drop(model: HookedTransformer, instance: Instance) -> float:
    """Answer-logit drop from ablating the **entire** gold span (gate 11.2d).

    On instances the model gets right, ablating the gold span must produce a large
    drop (the span is causally necessary). A small drop means the answer-metric
    wiring is wrong, not that the span is unimportant.
    """
    raise NotImplementedError("requires GPU phase")
