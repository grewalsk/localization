"""Primary causal oracle: token masking ``E(t)`` (paper Definition 2, §8.1).

The per-token causal effect is the drop in the full-cache answer log-probability
when token ``t``'s key/value is masked from attention **at every layer**, in a
single counterfactual forward pass, so that no query at any position attends to
``t`` while every other cached representation is left in place::

    E(t) = logP(y(x) | x) - logP(y(x) | x^{\\t})

This is the operation earlier drafts called "token ablation"; masking is the
paper's term (Definition 2) and the canonical name here. :mod:`lcv.oracle.ablation`
is a thin alias re-exporting this module.

**Analogue, not identical** (paper §3). Masking is the closest single-token
analogue of KV eviction inside a clean causal-effect definition, but the oracle
reads one counterfactual prompt pass, not the autoregressive eviction trajectory.
We read ``E(t)`` as an *upper-bound proxy* for eviction's per-token information
loss; the empirical bridge to eviction is the flip test (§9).

The masking *mechanics* are model-independent and live here as pure-numpy,
CPU-tested cores (build the key-keep mask, renormalize an attention row after
masking, teacher-force the answer log-prob, difference clean vs masked). Only the
forward passes that produce logits need the GPU; those entry points call the same
cores and are stubbed until the GPU phase. Read **eager** attention (the same rule
as every attention-reading pass); FlashAttention exposes no weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..contracts import Granularity, Instance, OracleEffect

if TYPE_CHECKING:  # hints only; torch/TL live in the GPU phase
    import torch
    from transformer_lens import HookedTransformer


# --------------------------------------------------------------------------- #
# Backbone-portable CPU cores (pure numpy; the genuinely-new masking mechanics)
# --------------------------------------------------------------------------- #


def build_key_keep_mask(seq_len: int, removed_indices: Sequence[int] | int) -> np.ndarray:
    """Boolean keep-mask over key positions with ``removed_indices`` masked out.

    ``True`` = the key stays available to attention; ``False`` = its K/V is removed
    (Definition 2). Masking token ``t`` for *all* queries is exactly a one-hot
    ``False`` at column ``t`` in this mask, which the forward pass turns into a
    ``-inf`` additive bias before softmax.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    removed = (removed_indices,) if isinstance(removed_indices, int) else tuple(removed_indices)
    keep = np.ones(seq_len, dtype=bool)
    for i in removed:
        if not (0 <= int(i) < seq_len):
            raise ValueError(f"removed index {i} out of range [0, {seq_len})")
        keep[int(i)] = False
    return keep


def renormalize_after_masking(attn_row: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    """Zero out masked keys in one attention row and renormalize to sum 1.

    Removing a token's K/V is equivalent to setting its attention column to ``-inf``
    before softmax, i.e. zeroing it after softmax and renormalizing over the
    surviving keys. This is the position-level invariant the GPU masking must
    satisfy: a masked key receives exactly zero weight and the row still sums to 1.
    A row whose entire surviving mass is zero is returned as all-zeros (degenerate).
    """
    a = np.asarray(attn_row, dtype=float)
    keep = np.asarray(keep_mask, dtype=bool)
    if a.shape != keep.shape:
        raise ValueError(f"attn_row {a.shape} and keep_mask {keep.shape} must match")
    masked = np.where(keep, a, 0.0)
    total = float(masked.sum())
    if total <= 0.0:
        return np.zeros_like(masked)
    return masked / total


def teacher_forced_logprob(logit_rows: np.ndarray, answer_token_ids: Sequence[int]) -> float:
    """Sum of teacher-forced log-probs of the answer tokens (the term ``E(t)`` differences).

    ``logit_rows[i]`` is the next-token logit distribution that *predicts*
    ``answer_token_ids[i]`` (already aligned; see
    :func:`answer_log_prob_from_logits` for slicing full ``[seq, vocab]`` logits).
    Returns ``sum_i log softmax(logit_rows[i])[answer_token_ids[i]]``. This is the
    one place the answer-metric plumbing must be right (gate 11.2d).
    """
    rows = np.asarray(logit_rows, dtype=float)
    if rows.ndim != 2:
        raise ValueError(f"logit_rows must be 2-D [n_answer, vocab], got {rows.shape}")
    ids = np.asarray(list(answer_token_ids), dtype=int)
    if ids.shape[0] != rows.shape[0]:
        raise ValueError(f"need one logit row per answer token: {rows.shape[0]} vs {ids.shape[0]}")
    if ids.size == 0:
        raise ValueError("no answer tokens")
    # numerically stable log-softmax row-by-row
    m = rows.max(axis=1, keepdims=True)
    logZ = m[:, 0] + np.log(np.exp(rows - m).sum(axis=1))
    chosen = rows[np.arange(rows.shape[0]), ids]
    return float(np.sum(chosen - logZ))


def answer_log_prob_from_logits(
    logits: np.ndarray, answer_token_ids: Sequence[int], *, answer_start: int
) -> float:
    """``logP(answer | context)`` from full ``[seq, vocab]`` logits (teacher-forced).

    The answer tokens occupy sequence positions ``[answer_start, answer_start + L)``;
    under the standard convention ``logits[j]`` predicts position ``j + 1``, so the
    rows that predict the answer are ``[answer_start - 1, answer_start - 1 + L)``.
    Requires ``answer_start >= 1`` (there must be a position whose next-token
    prediction is the first answer token).
    """
    lg = np.asarray(logits, dtype=float)
    if lg.ndim != 2:
        raise ValueError(f"logits must be 2-D [seq, vocab], got {lg.shape}")
    ids = list(answer_token_ids)
    if answer_start < 1:
        raise ValueError("answer_start must be >= 1 (need a predicting row before the answer)")
    end = answer_start - 1 + len(ids)
    if end > lg.shape[0]:
        raise ValueError(f"answer rows [{answer_start - 1}, {end}) exceed seq len {lg.shape[0]}")
    rows = lg[answer_start - 1 : end]
    return teacher_forced_logprob(rows, ids)


def masking_effect(clean_logprob: float, masked_logprobs: np.ndarray) -> np.ndarray:
    """``E(t) = clean - masked[t]`` over content tokens (Definition 2).

    ``masked_logprobs[i]`` is ``logP(answer | x^{\\t_i})`` for the i-th content
    token. A positive ``E`` means masking that token *hurt* the answer (it carried
    causal content); a negative ``E`` means masking it *helped* (a distractor).
    """
    masked = np.asarray(masked_logprobs, dtype=float)
    if masked.ndim != 1:
        raise ValueError("masked_logprobs must be 1-D over content tokens")
    if masked.size == 0:
        raise ValueError("need >= 1 content token")
    return float(clean_logprob) - masked


# --------------------------------------------------------------------------- #
# GPU entry points (orchestrate the cores around real forward passes)
# --------------------------------------------------------------------------- #


def answer_log_prob(
    model: HookedTransformer,
    tokens: torch.Tensor,
    answer_token_ids: Sequence[int],
    *,
    answer_start: int | None = None,
) -> float:
    """``logP(answer | context)`` under the model (GPU phase).

    Runs one forward pass for ``tokens`` and reduces with
    :func:`answer_log_prob_from_logits` / :func:`teacher_forced_logprob`. The
    reduction is the CPU-tested core; only the forward pass needs the GPU.
    """
    raise NotImplementedError("requires GPU phase")


def token_masking_effect(
    model: HookedTransformer,
    instance: Instance,
    *,
    granularity: Granularity = Granularity.TOKEN,
    answer_token_ids: Sequence[int] | None = None,
) -> OracleEffect:
    """The true masking oracle ``E`` over content tokens/spans (Definition 2, §8.1).

    For each content token (or sentence span, §8.4) run the masked forward pass
    ``x^{\\t}`` -- token ``t``'s K/V removed at every layer via a ``-inf`` additive
    bias on its key column (see :func:`build_key_keep_mask` /
    :func:`renormalize_after_masking`) -- and difference against the clean answer
    log-prob with :func:`masking_effect`. Returns an :class:`OracleEffect` with
    ``E`` 1-D over the content units. Sentence granularity matches HotpotQA's
    sentence-level gold and cuts the O(content length) sweep by ~10x.
    """
    raise NotImplementedError("requires GPU phase")


def gold_span_masking_drop(model: HookedTransformer, instance: Instance) -> float:
    """Answer-logit drop from masking the **entire** gold span (gate 11.2d).

    On instances the model answers correctly, masking the whole gold span must
    produce a large drop (the span is causally necessary). A small drop means the
    answer-metric wiring is wrong, not that the span is unimportant.
    """
    raise NotImplementedError("requires GPU phase")
