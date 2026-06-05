"""§11.2a fallback: when attribution patching fails the fidelity gate (§8.3).

The full oracle sweep uses the cheap ``E_hat`` estimator (one backward pass,
independent of token count). Gate 11.2a guards it: on a verification subsample,
``Spearman(E_hat, E) >= 0.8`` against the **masking** oracle (Definition 2). If
the gate fails mid-run, we must *not* silently fall back to full-scale exact
masking -- that is one masked forward pass per content token over the whole
~1K-instance sweep, which would blow the paid GPU budget. Instead we run exact
masking on a fixed, small **adjudication subset** and report it as the oracle for
the downstream claims, with the patching estimate dropped.

This module pins that decision as an actual branch + config flag so the fallback
is automatic, not a thing someone has to remember at 3am on a rented H100. The
decision logic is pure-CPU and tested; the masking sweep it selects runs in the
GPU phase via :func:`lcv.oracle.masking.token_masking_effect`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .attr_patching import ATTR_PATCHING_SPEARMAN_GATE

# Pinned size of the exact-masking adjudication subset taken when 11.2a fails
# (§8.3 / §11.2a). Small enough to afford exact masking, large enough to adjudicate.
ADJUDICATION_SUBSET_SIZE = 150


@dataclass(frozen=True, slots=True)
class OraclePlan:
    """How the full oracle sweep will be computed for this run.

    ``use_attribution_patching`` True  -> run cheap ``E_hat`` over the full sweep.
    ``use_attribution_patching`` False -> run exact masking on the first
    ``adjudication_subset_size`` instances (the pinned 11.2a fallback).
    """

    use_attribution_patching: bool
    adjudication_subset_size: int
    reason: str


def decide_oracle_path(
    verification_spearman: float,
    *,
    gate: float = ATTR_PATCHING_SPEARMAN_GATE,
    subset_size: int = ADJUDICATION_SUBSET_SIZE,
    force_exact: bool = False,
) -> OraclePlan:
    """Pick the oracle path from the 11.2a verification result (pure-CPU).

    ``verification_spearman`` is ``Spearman(E_hat, E)`` measured on the verification
    subsample. If it clears ``gate`` (default 0.8) and ``force_exact`` is not set,
    the full sweep uses ``E_hat``; otherwise the run falls back to exact masking on
    a ``subset_size``-instance adjudication subset. A NaN Spearman (degenerate
    ranking) is treated as a gate failure -- never trust an undefined fidelity.

    ``force_exact`` is the config flag a human can set to bypass patching entirely
    (e.g. when budget allows exact masking, or to sanity-check a suspicious sweep).
    """
    if subset_size <= 0:
        raise ValueError("subset_size must be positive")
    if force_exact:
        return OraclePlan(
            use_attribution_patching=False,
            adjudication_subset_size=subset_size,
            reason=f"force_exact set: exact masking on {subset_size}-instance subset",
        )
    if verification_spearman is None or math.isnan(verification_spearman):
        return OraclePlan(
            use_attribution_patching=False,
            adjudication_subset_size=subset_size,
            reason=(
                "11.2a fidelity undefined (NaN Spearman); falling back to exact masking "
                f"on {subset_size}-instance subset"
            ),
        )
    if verification_spearman >= gate:
        return OraclePlan(
            use_attribution_patching=True,
            adjudication_subset_size=subset_size,
            reason=(
                f"11.2a passed (Spearman {verification_spearman:.3f} >= {gate}); "
                "E_hat over the full sweep"
            ),
        )
    return OraclePlan(
        use_attribution_patching=False,
        adjudication_subset_size=subset_size,
        reason=(
            f"11.2a failed (Spearman {verification_spearman:.3f} < {gate}); "
            f"exact masking on {subset_size}-instance adjudication subset"
        ),
    )
