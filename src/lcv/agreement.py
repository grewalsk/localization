"""Agreement metrics (addendum §7).

Per substrate, per instance, over the relevant ranking vectors:

* Spearman rank correlation (pairwise, with mean + bootstrap CI over instances),
* top-k Jaccard and rank-biased overlap (RBO) for a top-weighted view,
* gold-span agreement (precision@k, AUROC vs gold membership),
* the per-instance disagreement score ``D(x) = 1 - mean pairwise Spearman``,
* a permutation test for above-chance cross-method agreement,
* Benjamini-Hochberg FDR over the reported p-values (Benjamini-Yekutieli fallback
  when the per-instance tests violate positive dependence).

Pure numpy/scipy/sklearn/statsmodels — no GPU, no torch. Token rankings share
the content-token set, so RBO is the equal-length extrapolated variant
(identical rankings → 1.0).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from .contracts import (
    DisagreementScore,
    ImportanceVector,
    Method,
    Substrate,
    assert_same_substrate,
)

Vector = Sequence[float] | np.ndarray

# --------------------------------------------------------------------------- #
# Spearman
# --------------------------------------------------------------------------- #


def _is_constant(a: np.ndarray) -> bool:
    return bool(np.all(a == a.flat[0]))


def _rank_undefined(a: np.ndarray) -> bool:
    """True when a top-k / RBO ranking over ``a`` is undefined.

    A constant vector of 2+ elements has no rank information: ``argsort`` breaks the
    universal tie arbitrarily, so any top-k set, RBO, or precision@k derived from it
    is a coin flip dressed as a number (M2). Single/empty inputs carry no tie
    ambiguity and are left to the callers' own empty-set guards.
    """
    return a.size >= 2 and _is_constant(a)


def spearman(a: Vector, b: Vector) -> float:
    """Spearman rho. Returns NaN if either vector is constant (rho undefined)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    if a.size < 2 or _is_constant(a) or _is_constant(b):
        return float("nan")
    return float(spearmanr(a, b).statistic)


def pairwise_spearman_matrix(vectors: Sequence[Vector]) -> np.ndarray:
    """Symmetric ``[m, m]`` matrix of pairwise Spearman, unit diagonal."""
    m = len(vectors)
    mat = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            mat[i, j] = mat[j, i] = spearman(vectors[i], vectors[j])
    return mat


def disagreement(vectors: Sequence[Vector]) -> float:
    """``D(x) = 1 - mean pairwise Spearman`` over methods (§7).

    NaN pairs (a constant vector) are skipped; returns NaN only if every pair is
    undefined. Range is [0, 2]: 0 = perfect agreement, 2 = perfect anti-agreement.
    """
    m = len(vectors)
    if m < 2:
        raise ValueError("D(x) needs >= 2 methods to form a pair")
    rs = np.array(
        [spearman(vectors[i], vectors[j]) for i in range(m) for j in range(i + 1, m)],
        dtype=float,
    )
    if np.all(np.isnan(rs)):
        return float("nan")
    return float(1.0 - np.nanmean(rs))


def n_effective_pairs(vectors: Sequence[Vector]) -> int:
    """Count of method pairs with a defined (non-NaN) Spearman -- those that enter D(x).

    A constant vector makes every pair it is in NaN (M2), dropping it from D(x)'s mean;
    this is the count that actually contributed, so a D(x) resting on few comparisons is
    visible rather than masked by ``n_methods`` (M8).
    """
    m = len(vectors)
    rs = np.array(
        [spearman(vectors[i], vectors[j]) for i in range(m) for j in range(i + 1, m)],
        dtype=float,
    )
    return int(np.count_nonzero(~np.isnan(rs)))


def build_disagreement_score(
    instance_id: str,
    methods: Sequence[Method],
    vectors: Sequence[Vector],
    *,
    substrate: Substrate | None = None,
    from_full_cache: bool = True,
) -> DisagreementScore:
    """Compute ``D(x)`` and wrap it in the leakage-guarded contract.

    ``substrate`` is inferred from ``methods`` (raising if they span more than
    one, §6). ``from_full_cache`` flows into the type, which refuses to exist if
    it is False (§9.3 leakage rule).
    """
    if len(methods) != len(vectors):
        raise ValueError("methods and vectors must align")
    sub = substrate if substrate is not None else assert_same_substrate(methods)
    return DisagreementScore(
        instance_id=instance_id,
        substrate=sub,
        value=disagreement(vectors),
        n_methods=len(methods),
        from_full_cache=from_full_cache,
        n_effective_pairs=n_effective_pairs(vectors),
    )


def disagreement_from_vectors(vectors: Sequence[ImportanceVector]) -> DisagreementScore:
    """Convenience: build ``D(x)`` directly from ImportanceVector outputs."""
    if not vectors:
        raise ValueError("no vectors")
    instance_ids = {v.instance_id for v in vectors}
    if len(instance_ids) != 1:
        raise ValueError(f"vectors span multiple instances: {instance_ids}")
    return build_disagreement_score(
        instance_id=vectors[0].instance_id,
        methods=[v.method for v in vectors],
        vectors=[v.values for v in vectors],
    )


# --------------------------------------------------------------------------- #
# Aggregation over instances
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class AggregateAgreement:
    """Mean pairwise Spearman with a percentile bootstrap CI over instances."""

    mean: np.ndarray  # [m, m]
    ci_low: np.ndarray
    ci_high: np.ndarray
    n_instances: int


def aggregate_spearman(
    per_instance_vectors: Sequence[Sequence[Vector]],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> AggregateAgreement:
    """Mean pairwise-Spearman heatmap + bootstrap CI.

    ``per_instance_vectors[i]`` is the list of ``m`` aligned method vectors for
    instance ``i`` (same method order across instances).
    """
    if not per_instance_vectors:
        raise ValueError("no instances")
    mats = np.stack([pairwise_spearman_matrix(v) for v in per_instance_vectors])
    mean = np.nanmean(mats, axis=0)
    rng = np.random.default_rng(seed)
    n = mats.shape[0]
    boot = np.empty((n_boot, *mean.shape))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = np.nanmean(mats[idx], axis=0)
    lo = np.nanpercentile(boot, (1 - ci) / 2 * 100, axis=0)
    hi = np.nanpercentile(boot, (1 + ci) / 2 * 100, axis=0)
    return AggregateAgreement(mean=mean, ci_low=lo, ci_high=hi, n_instances=n)


# --------------------------------------------------------------------------- #
# Top-k Jaccard and RBO
# --------------------------------------------------------------------------- #


def _ranking(values: Vector) -> list[int]:
    """Item indices ordered by descending importance, ties broken by ascending index.

    ``argsort(-v, kind="stable")`` is deterministic *and* convention-consistent: equal
    scores keep their original (lowest-index-first) order. A plain ``argsort(v)[::-1]``
    would instead reverse ties into highest-index-first, so two callers ranking the same
    partially-tied vector could disagree on which item lands in the top-k (M1)."""
    return np.argsort(-np.asarray(values, dtype=float), kind="stable").tolist()


def top_k_set(values: Vector, k: int) -> set[int]:
    values = np.asarray(values, dtype=float)
    k = min(k, values.size)
    return set(_ranking(values)[:k])


def top_k_jaccard(a: Vector, b: Vector, k: int) -> float:
    """Jaccard overlap of the top-k index sets.

    NaN if both empty, or if either vector is constant -- its top-k set is an
    arbitrary argsort tie-break, so any overlap is spurious (M2).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if _rank_undefined(a) or _rank_undefined(b):
        return float("nan")
    sa, sb = top_k_set(a, k), top_k_set(b, k)
    if not sa and not sb:
        return float("nan")
    return len(sa & sb) / len(sa | sb)


def rbo(ranking_a: Sequence[int], ranking_b: Sequence[int], p: float = 0.9) -> float:
    """Extrapolated rank-biased overlap for equal-length rankings (§7).

    Identical rankings → 1.0; disjoint → 0.0. ``p`` is the persistence (top
    weight). Token rankings share the content-token set, so lengths are equal.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a, b = list(ranking_a), list(ranking_b)
    if len(a) != len(b):
        raise ValueError("equal-length rankings required (token rankings share the content set)")
    k = len(a)
    if k == 0:
        return float("nan")
    seen_a: set[int] = set()
    seen_b: set[int] = set()
    overlap = 0
    weighted = 0.0
    for d in range(1, k + 1):
        x, y = a[d - 1], b[d - 1]
        if x in seen_b and x not in seen_a:
            overlap += 1
        seen_a.add(x)
        if y in seen_a and y not in seen_b:
            overlap += 1
        seen_b.add(y)
        weighted += (overlap / d) * (p**d)
    return float((overlap / k) * (p**k) + (1.0 - p) / p * weighted)


def rbo_from_importance(a: Vector, b: Vector, p: float = 0.9) -> float:
    """Rank both vectors by importance, then RBO. NaN if either is constant --
    the ranking is an arbitrary tie-break, so RBO would be spurious (M2)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if _rank_undefined(a) or _rank_undefined(b):
        return float("nan")
    return rbo(_ranking(a), _ranking(b), p=p)


# --------------------------------------------------------------------------- #
# Gold-span agreement
# --------------------------------------------------------------------------- #


def precision_at_k(values: Vector, gold_mask: Vector, k: int) -> float:
    """Fraction of the top-k importance tokens that are gold-span members.

    NaN if the importance vector is constant -- its top-k set is an arbitrary
    argsort tie-break, so precision against the gold span would be spurious (M2).
    """
    values = np.asarray(values, dtype=float)
    gold = np.asarray(gold_mask, dtype=bool)
    if _rank_undefined(values):
        return float("nan")
    topk = top_k_set(values, k)
    if not topk:
        return float("nan")
    return sum(1 for i in topk if gold[i]) / len(topk)


def auroc_vs_gold(values: Vector, gold_mask: Vector) -> float:
    """AUROC of importance against gold-span membership. NaN if degenerate."""
    gold = np.asarray(gold_mask, dtype=bool)
    if gold.all() or (~gold).all():
        return float("nan")
    return float(roc_auc_score(gold, np.asarray(values, dtype=float)))


# --------------------------------------------------------------------------- #
# Permutation test + FDR
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class PermutationResult:
    observed: float
    p_value: float
    null: np.ndarray


def _mean_pairwise(vectors: Sequence[np.ndarray]) -> float:
    m = len(vectors)
    rs = np.asarray(
        [spearman(vectors[i], vectors[j]) for i in range(m) for j in range(i + 1, m)],
        dtype=float,
    )
    # Every pair undefined (e.g. all vectors constant) -> the mean agreement is
    # undefined. Return NaN explicitly rather than letting np.nanmean warn on an
    # all-NaN slice; permutation_test reads this NaN to refuse a spurious p-value.
    if rs.size == 0 or np.all(np.isnan(rs)):
        return float("nan")
    return float(np.nanmean(rs))


def permutation_test(
    vectors: Sequence[Vector], *, n_perm: int = 1000, seed: int = 0
) -> PermutationResult:
    """Test whether mean pairwise agreement exceeds chance under a permutation null (§7).

    The paper frames the null per pair as "shuffle one ranking"; this implementation
    shuffles every vector's token positions independently. The two are statistically
    equivalent: Spearman depends only on relative order, so for a pair the doubly
    shuffled correlation ``spearman(a[pi], b[rho])`` equals ``spearman(a, b[rho.pi^-1])``
    with ``rho.pi^-1`` again a uniform permutation -- fixing one ranking and shuffling
    the other draws from the same null. Shuffling all sides extends this to the
    >2-method mean without privileging a reference vector.

    p = (1 + #{null >= observed}) / (n_perm + 1). If the observed mean pairwise
    agreement is undefined (every pair constant -> NaN), the test is undefined: we
    return ``p_value=NaN`` and an all-NaN null rather than a spurious
    ``1/(n_perm+1)`` that ``null >= NaN`` (all False) would otherwise manufacture (M3).
    """
    m = len(vectors)
    if m < 2:
        raise ValueError("need >= 2 vectors")
    vecs = [np.asarray(v, dtype=float) for v in vectors]
    observed = _mean_pairwise(vecs)
    if np.isnan(observed):
        return PermutationResult(
            observed=float("nan"), p_value=float("nan"), null=np.full(n_perm, np.nan)
        )
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for t in range(n_perm):
        permuted = [v[rng.permutation(v.size)] for v in vecs]
        null[t] = _mean_pairwise(permuted)
    p = (1 + int(np.sum(null >= observed))) / (n_perm + 1)
    return PermutationResult(observed=observed, p_value=float(p), null=null)


def benjamini_hochberg(
    pvalues: Sequence[float], *, alpha: float = 0.05, dependent: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """FDR control over the reported p-values. Returns ``(reject, qvalues)`` (§7).

    ``dependent=False`` runs Benjamini-Hochberg (assumes positive dependence);
    ``dependent=True`` runs the Benjamini-Yekutieli procedure (``fdr_by``), the
    conservative fallback valid under arbitrary dependence -- use it when the
    per-instance tests violate the positive-dependence assumption (M4).

    NaN p-values (tests that were undefined, e.g. a permutation test on all-constant
    vectors -- M3) are carried through as *not rejected* with NaN q-value and are
    excluded from the correction, so an undefined test neither rejects nor inflates
    the denominator of the finite tests' FDR. ``multipletests`` would otherwise raise
    or silently propagate NaN across every q-value.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.size == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    method = "fdr_by" if dependent else "fdr_bh"
    reject = np.zeros(p.size, dtype=bool)
    qvalues = np.full(p.size, np.nan)
    finite = ~np.isnan(p)
    if not finite.any():
        return reject, qvalues
    rej, q, _, _ = multipletests(p[finite], alpha=alpha, method=method)
    reject[finite] = rej
    qvalues[finite] = q
    return reject, qvalues
