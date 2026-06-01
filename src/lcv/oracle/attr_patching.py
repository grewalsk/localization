"""Attribution-patching approximation of the ablation oracle + gate 11.2a (§8.3).

Attribution patching (Syed et al.) approximates a patching/ablation effect with
one forward and one backward pass via a first-order expansion. For the
mean-ablation variant aligned with §8.1:

    E_hat(t) ~= ( a_mean(t) - a_clean(t) )^T . grad_{a(t)} [ logP(answer) ]

evaluated with the clean-run gradient, where ``a(t)`` is token ``t``'s
contribution (residual or KV) and ``a_mean(t)`` is the mean-ablation reference.
This collapses the per-token forward sweep into a single backward pass.

**Mandatory gate 11.2a:** on a 5-10% held-out subsample, require ``Spearman(E_hat,
E) >= 0.8`` on token ranking with high sign-agreement (ranking, not magnitude,
because every downstream use ranks tokens). If unmet, fall back to subsampled
true ablation. The validation itself is pure-CPU and implemented here; the
``E_hat`` computation is the GPU-phase stub.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..agreement import spearman
from ..contracts import CorruptionType, Granularity, Instance, OracleEffect

if TYPE_CHECKING:  # hints only
    from transformer_lens import HookedTransformer

# Gate 11.2a: minimum Spearman(E_hat, E) on the held-out subsample (§8.3).
ATTR_PATCHING_SPEARMAN_GATE = 0.8
# Fraction of instances on which to run the expensive true-E validation.
VALIDATION_SUBSAMPLE = 0.1


def attribution_patching_effect(
    model: HookedTransformer,
    instance: Instance,
    corruption: CorruptionType = CorruptionType.TOKEN_SWAP,
    *,
    granularity: Granularity = Granularity.TOKEN,
    answer_token_ids: Sequence[int] | None = None,
) -> OracleEffect:
    """First-order ``E_hat`` over content tokens/spans (§8.3).

    One forward + one backward pass with the mean-ablation reference; the
    corrupted reference for the activation delta uses ``corruption`` (token-swap
    default, never Gaussian, §8.2). Returns an :class:`OracleEffect` carrying
    ``E_hat`` (and ``corruption``); validate it against the true ``E`` with
    :func:`validate_attribution_patching` before trusting it downstream.
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
