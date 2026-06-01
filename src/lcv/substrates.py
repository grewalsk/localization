"""Substrate projections and per-instance normalization (addendum §6).

Agreement is defined only *within* a substrate. This module turns raw signal
outputs into aligned, normalized comparison vectors (a ``SubstrateBundle``) and
refuses to bundle members of different substrates, so "substrates are never
merged" holds structurally before anything reaches ``agreement``.

Normalization is uniform per run: pick ``minmax`` or ``zscore`` and apply it to
every member (§5). Constant vectors map to all-zeros (no spurious scale).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .agreement import (
    build_disagreement_score,
    disagreement,
    pairwise_spearman_matrix,
)
from .contracts import (
    CONDITIONAL_MEMBERS,
    SUBSTRATE_MEMBERS,
    DisagreementScore,
    HeadScore,
    ImportanceVector,
    Method,
    Substrate,
    assert_same_substrate,
)

# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


def minmax(values: Sequence[float] | np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo == 0.0:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def zscore(values: Sequence[float] | np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    sd = float(np.std(x))
    if sd == 0.0:
        return np.zeros_like(x)
    return (x - float(np.mean(x))) / sd


NORMALIZERS = {"minmax": minmax, "zscore": zscore}


def normalize(values: Sequence[float] | np.ndarray, method: str = "minmax") -> np.ndarray:
    try:
        fn = NORMALIZERS[method]
    except KeyError:
        raise ValueError(
            f"unknown normalization {method!r}; pick one of {list(NORMALIZERS)}"
        ) from None
    return fn(values)


def members(substrate: Substrate, *, include_conditional: bool = True) -> tuple[Method, ...]:
    """Registered members of a substrate (§6).

    ``include_conditional=False`` drops gate-dependent members (transcoder token
    attribution, kept only if gate 11.2c passes, §4.4).
    """
    ms = SUBSTRATE_MEMBERS[substrate]
    if include_conditional:
        return ms
    return tuple(m for m in ms if m not in CONDITIONAL_MEMBERS)


# --------------------------------------------------------------------------- #
# Bundles
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SubstrateBundle:
    """Aligned, normalized method vectors for one substrate, ready for §7.

    ``vectors`` is ``[m_methods, n_items]`` in ``methods`` order. For the
    token-attribution substrate ``n_items`` is the content-token count; for the
    component substrate it is the flattened head grid.
    """

    substrate: Substrate
    instance_id: str | None
    methods: tuple[Method, ...]
    vectors: np.ndarray
    normalization: str

    def pairwise_spearman(self) -> np.ndarray:
        return pairwise_spearman_matrix(list(self.vectors))

    def disagreement(self) -> float:
        return disagreement(list(self.vectors))

    def disagreement_score(self, *, from_full_cache: bool = True) -> DisagreementScore:
        if self.instance_id is None:
            raise ValueError("D(x) is per-instance; this bundle has no instance_id")
        return build_disagreement_score(
            self.instance_id,
            list(self.methods),
            list(self.vectors),
            substrate=self.substrate,
            from_full_cache=from_full_cache,
        )


def build_token_bundle(
    vectors: Sequence[ImportanceVector], *, normalization: str = "minmax"
) -> SubstrateBundle:
    """Bundle token-attribution ImportanceVectors for one instance (§6)."""
    if not vectors:
        raise ValueError("no vectors")
    ids = {v.instance_id for v in vectors}
    if len(ids) != 1:
        raise ValueError(f"vectors span multiple instances: {ids}")
    lengths = {v.values.shape[0] for v in vectors}
    if len(lengths) != 1:
        raise ValueError(f"vectors not aligned over content tokens (lengths {lengths})")
    methods = tuple(v.method for v in vectors)
    assert_same_substrate(methods)  # guaranteed token-attr by the type; guards anyway (§6)
    mat = np.stack([normalize(v.values, normalization) for v in vectors])
    return SubstrateBundle(
        substrate=Substrate.TOKEN_ATTRIBUTION,
        instance_id=vectors[0].instance_id,
        methods=methods,
        vectors=mat,
        normalization=normalization,
    )


def build_component_bundle(
    scores: Sequence[HeadScore], *, normalization: str = "minmax"
) -> SubstrateBundle:
    """Bundle component-substrate HeadScores (flattened head grids) (§6)."""
    if not scores:
        raise ValueError("no scores")
    shapes = {s.scores.shape for s in scores}
    if len(shapes) != 1:
        raise ValueError(f"head-score grids differ in shape: {shapes}")
    methods = tuple(s.method for s in scores)
    assert_same_substrate(methods)
    mat = np.stack([normalize(s.scores.ravel(), normalization) for s in scores])
    return SubstrateBundle(
        substrate=Substrate.COMPONENT,
        instance_id=scores[0].instance_id,
        methods=methods,
        vectors=mat,
        normalization=normalization,
    )
