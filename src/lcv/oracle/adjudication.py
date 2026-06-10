"""§11.2a fallback: when attribution patching fails the fidelity gate (§8.3).

The full oracle sweep uses the cheap ``E_hat`` estimator (one backward pass,
independent of token count). Gate 11.2a guards it: on a verification subsample,
``Spearman(E_hat, E) >= 0.8`` against the **masking** oracle (Definition 2). If
the gate fails mid-run, we must *not* silently fall back to full-scale exact
masking -- that is one masked forward pass per content token over the whole
~1K-instance sweep, which would blow the paid GPU budget. Instead we run exact
masking on a fixed, **seeded** adjudication subset (a reproducible random draw, not
the first N instances -- M8) and report it as the oracle for the downstream claims,
with the patching estimate dropped.

This module pins that decision as an actual branch + config flag so the fallback
is automatic, not a thing someone has to remember at 3am on a rented H100. The
decision logic is pure-CPU and tested; the masking sweep it selects runs in the
GPU phase via :func:`lcv.oracle.masking.token_masking_effect`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .attr_patching import ATTR_PATCHING_SPEARMAN_GATE

# Pinned size of the exact-masking adjudication subset taken when 11.2a fails
# (§8.3 / §11.2a). Small enough to afford exact masking, large enough to adjudicate.
ADJUDICATION_SUBSET_SIZE = 150
# Default RNG seed for the adjudication subsample; recorded on the OraclePlan so the
# subset is reproducible (M8). A seeded random draw, NOT the first N instances -- the
# first NIAH instances are short/shallow, a biased subset that would skew the oracle.
ADJUDICATION_SEED = 0


@dataclass(frozen=True, slots=True)
class OraclePlan:
    """How the full oracle sweep will be computed for this run.

    ``use_attribution_patching`` True  -> run cheap ``E_hat`` over the full sweep;
    ``adjudication_indices`` / ``adjudication_seed`` stay ``None`` (no subset needed).
    ``use_attribution_patching`` False -> run exact masking on the
    ``adjudication_subset_size`` instances at ``adjudication_indices`` -- a seeded
    random draw under ``adjudication_seed`` (recorded so the subset is reproducible,
    M8; the pinned 11.2a fallback). The indices are populated only when
    :func:`decide_oracle_path` is told ``n_instances``; otherwise the caller takes
    them later from :func:`select_adjudication_subset`.
    """

    use_attribution_patching: bool
    adjudication_subset_size: int
    reason: str
    adjudication_indices: tuple[int, ...] | None = None
    adjudication_seed: int | None = None


def select_adjudication_subset(
    n_instances: int,
    *,
    subset_size: int = ADJUDICATION_SUBSET_SIZE,
    seed: int = ADJUDICATION_SEED,
) -> np.ndarray:
    """Seeded, reproducible adjudication subset: sorted instance indices (M8).

    Draws ``min(subset_size, n_instances)`` instances **without replacement** via
    ``np.random.default_rng(seed)`` instead of taking the first ``subset_size`` (the
    first NIAH instances are short/shallow -- a biased subset that would skew the
    exact-masking oracle). Returns the chosen indices sorted ascending; ``seed`` is
    recorded on the :class:`OraclePlan` so the exact subset is reproducible.
    """
    if n_instances <= 0:
        raise ValueError("n_instances must be positive")
    if subset_size <= 0:
        raise ValueError("subset_size must be positive")
    k = min(subset_size, n_instances)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_instances, size=k, replace=False))


def decide_oracle_path(
    verification_spearman: float | None,
    *,
    gate: float = ATTR_PATCHING_SPEARMAN_GATE,
    subset_size: int = ADJUDICATION_SUBSET_SIZE,
    force_exact: bool = False,
    n_instances: int | None = None,
    seed: int = ADJUDICATION_SEED,
) -> OraclePlan:
    """Pick the oracle path from the 11.2a verification result (pure-CPU).

    ``verification_spearman`` is ``Spearman(E_hat, E)`` measured on the verification
    subsample. If it clears ``gate`` (default 0.8) and ``force_exact`` is not set,
    the full sweep uses ``E_hat``; otherwise the run falls back to exact masking on a
    ``subset_size``-instance adjudication subset. A NaN Spearman (degenerate ranking)
    is treated as a gate failure -- never trust an undefined fidelity.

    When the fallback fires and ``n_instances`` is given, the plan also carries the
    seeded :func:`select_adjudication_subset` indices and ``seed`` (M8), so the exact
    subset is fixed and reproducible. ``force_exact`` is the config flag a human can
    set to bypass patching entirely (e.g. when budget allows exact masking, or to
    sanity-check a suspicious sweep).
    """
    if subset_size <= 0:
        raise ValueError("subset_size must be positive")

    def _exact(reason: str) -> OraclePlan:
        indices: tuple[int, ...] | None = None
        used_seed: int | None = None
        if n_instances is not None:
            chosen = select_adjudication_subset(n_instances, subset_size=subset_size, seed=seed)
            indices = tuple(int(i) for i in chosen)
            used_seed = seed
        return OraclePlan(
            use_attribution_patching=False,
            adjudication_subset_size=subset_size,
            reason=reason,
            adjudication_indices=indices,
            adjudication_seed=used_seed,
        )

    if force_exact:
        return _exact(f"force_exact set: exact masking on {subset_size}-instance subset")
    if verification_spearman is None or math.isnan(verification_spearman):
        return _exact(
            "11.2a fidelity undefined (NaN Spearman); falling back to exact masking "
            f"on {subset_size}-instance subset"
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
    return _exact(
        f"11.2a failed (Spearman {verification_spearman:.3f} < {gate}); "
        f"exact masking on {subset_size}-instance adjudication subset"
    )
