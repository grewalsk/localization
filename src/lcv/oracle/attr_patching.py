"""Attribution-patching estimate of the masking oracle + gate 11.2a (§8.3).

The full oracle (:mod:`lcv.oracle.masking`, Definition 2) masks each content
token's K/V from attention and re-runs the forward pass -- one masked pass *per
token*, O(content length) forwards per instance. Attribution patching (Nanda
2022; AtP*, Kramár et al. 2024) collapses that sweep into a single forward + a
single backward pass by linearizing the **same masking intervention** the oracle
performs, so the estimate ``E_hat`` and the truth ``E`` measure the same quantity
-- the precondition for the 11.2a fidelity gate to mean anything. (The earlier
mean-ablation / corrupted-activation form linearized a *different* intervention
than masking; M9 re-pins it to the attention-pattern form below.)

**Attention-pattern AtP (what ``E_hat`` linearizes).** Masking key ``t`` zeroes
token ``t``'s column in every post-softmax attention pattern ``A[l, h, q, t]``
(then renormalizes the surviving keys). To first order, the change in the answer
metric ``m = logP(y | x)`` from removing key ``t`` is the clean attention onto
``t`` contracted with the metric's gradient w.r.t. those attention weights::

    E_hat(t) = sum_{l, h, q} A_clean[l, h, q, t] * (d m / d A[l, h, q, t])

``A_clean`` and ``m`` come from one forward pass, ``d m / d A`` from one backward
pass -- no corrupted reference is needed, because the masking perturbation's
first-order delta on column ``t`` is the analytic ``-A_clean[..., t]``. The sign
matches the masking oracle ``E(t) = m(x) - m(x^{\\t})``: a large positive
``E_hat(t)`` predicts that masking ``t`` drops the answer log-prob. The
renormalization of the surviving keys is the second-order term AtP* (Kramár et
al. 2024) corrects; whether the bare first-order estimate is good enough is
exactly what gate 11.2a certifies empirically.

**Mandatory gate 11.2a:** on a 5-10% held-out subsample, require ``Spearman(
E_hat, E) >= 0.8`` on token ranking with high sign-agreement (ranking, not
magnitude, because every downstream use ranks tokens). If unmet, drop ``E_hat``
and fall back to exact masking on the adjudication subset
(:mod:`lcv.oracle.adjudication`). The validation itself is pure-CPU and
implemented here; the ``E_hat`` computation is the GPU-phase stub.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..agreement import spearman
from ..contracts import Granularity, Instance, OracleEffect

if TYPE_CHECKING:  # hints only
    from transformer_lens import HookedTransformer

# Gate 11.2a: minimum Spearman(E_hat, E) on the held-out subsample (§8.3).
ATTR_PATCHING_SPEARMAN_GATE = 0.8
# Fraction of instances on which to run the expensive true-E validation.
VALIDATION_SUBSAMPLE = 0.1


def attribution_patching_effect(
    model: HookedTransformer,
    instance: Instance,
    *,
    granularity: Granularity = Granularity.TOKEN,
    answer_token_ids: Sequence[int] | None = None,
) -> OracleEffect:
    """First-order attention-pattern ``E_hat`` over content tokens/spans (§8.3).

    One forward pass for the clean attention patterns ``A_clean`` and the answer
    metric ``m = logP(y | x)``, one backward pass for ``d m / d A``, combined as
    ``E_hat(t) = sum_{l,h,q} A_clean[l,h,q,t] * (d m / d A[l,h,q,t])`` -- the
    linearization of the masking oracle. No corruption reference enters: the
    masking perturbation's first-order delta on column ``t`` is the analytic
    ``-A_clean[..., t]`` (M9). Returns an :class:`OracleEffect` carrying ``E_hat``;
    validate it against the true ``E`` with :func:`validate_attribution_patching`
    before trusting it downstream (gate 11.2a).
    """
    raise NotImplementedError("requires GPU phase")


def validate_attribution_patching(E: np.ndarray, E_hat: np.ndarray) -> tuple[float, float]:
    """``(spearman, sign_agreement)`` between true ``E`` and estimate ``E_hat`` (gate 11.2a).

    Pure-CPU. Spearman is on *ranking* (the quantity every downstream use cares
    about); sign agreement is the fraction of units where ``sign(E) ==
    sign(E_hat)``. Gate 11.2a passes when Spearman ``>=`` :data:`ATTR_PATCHING_
    SPEARMAN_GATE` with high sign-agreement.
    """
    e = np.asarray(E, dtype=float)
    e_hat = np.asarray(E_hat, dtype=float)
    if e.shape != e_hat.shape:
        raise ValueError(f"E and E_hat must share shape, got {e.shape} vs {e_hat.shape}")
    if e.size < 2:
        raise ValueError("need >= 2 units for a rank correlation")
    rho = spearman(e, e_hat)
    sign_agreement = float(np.mean(np.sign(e) == np.sign(e_hat)))
    return rho, sign_agreement


def passes_attr_patching_gate(
    E: np.ndarray,
    E_hat: np.ndarray,
    *,
    threshold: float = ATTR_PATCHING_SPEARMAN_GATE,
) -> bool:
    """Whether ``E_hat`` clears gate 11.2a (Spearman ``>= threshold``); pure-CPU.

    A NaN Spearman (degenerate ranking) fails the gate. If this returns False,
    fall back to subsampled true ablation rather than trusting ``E_hat`` (§8.3).
    """
    rho, _ = validate_attribution_patching(E, E_hat)
    return bool(rho >= threshold)
