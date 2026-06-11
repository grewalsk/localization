"""Tests for the §7 agreement metrics."""

import numpy as np
import pytest

from lcv import agreement as ag
from lcv.contracts import ImportanceVector, Method, Substrate

# --- Spearman + D(x) ------------------------------------------------------- #


def test_spearman_perfect_anti_constant():
    x = np.arange(10.0)
    assert ag.spearman(x, x) == pytest.approx(1.0)
    assert ag.spearman(x, x[::-1]) == pytest.approx(-1.0)
    assert np.isnan(ag.spearman(x, np.ones(10)))  # constant -> undefined


def test_pairwise_matrix_symmetric_unit_diagonal():
    v = [np.arange(6.0), np.arange(6.0)[::-1], np.array([1.0, 3, 2, 6, 5, 4])]
    m = ag.pairwise_spearman_matrix(v)
    assert m.shape == (3, 3)
    np.testing.assert_allclose(np.diag(m), 1.0)
    np.testing.assert_allclose(m, m.T)


def test_disagreement_bounds():
    x = np.arange(8.0)
    assert ag.disagreement([x, x.copy()]) == pytest.approx(0.0)  # perfect agreement
    assert ag.disagreement([x, x[::-1]]) == pytest.approx(2.0)  # perfect anti
    with pytest.raises(ValueError, match=">= 2 methods"):
        ag.disagreement([x])


def test_build_disagreement_score_infers_substrate_and_guards():
    x = np.arange(5.0)
    ds = ag.build_disagreement_score(
        "i0", [Method.WU_ATTENTION, Method.QR_ATTENTION], [x, x.copy()]
    )
    assert ds.substrate is Substrate.TOKEN_ATTRIBUTION
    assert ds.value == pytest.approx(0.0)
    assert ds.n_methods == 2
    # cross-substrate methods are rejected (§6)
    with pytest.raises(ValueError, match="never merged"):
        ag.build_disagreement_score("i0", [Method.WU_ATTENTION, Method.WU_HEADS], [x, x])
    # leakage guard flows through the contract
    with pytest.raises(ValueError, match="full-cache"):
        ag.build_disagreement_score(
            "i0",
            [Method.WU_ATTENTION, Method.QR_ATTENTION],
            [x, x],
            from_full_cache=False,
        )


def test_disagreement_from_importance_vectors():
    x = np.array([0.1, 0.5, 0.9, 0.2])
    vs = [
        ImportanceVector("i0", Method.WU_ATTENTION, x),
        ImportanceVector("i0", Method.QR_ATTENTION, x.copy()),
    ]
    ds = ag.disagreement_from_vectors(vs)
    assert ds.instance_id == "i0"
    assert ds.value == pytest.approx(0.0)
    # different instances must not be mixed
    vs2 = [
        ImportanceVector("i0", Method.WU_ATTENTION, x),
        ImportanceVector("i1", Method.QR_ATTENTION, x),
    ]
    with pytest.raises(ValueError, match="multiple instances"):
        ag.disagreement_from_vectors(vs2)


def test_aggregate_spearman_shapes_and_ci():
    x = np.arange(7.0)
    per_instance = [[x, x.copy()] for _ in range(5)]
    res = ag.aggregate_spearman(per_instance, n_boot=100, seed=1)
    assert res.mean.shape == (2, 2)
    assert res.n_instances == 5
    assert res.mean[0, 1] == pytest.approx(1.0)
    assert np.all(res.ci_low <= res.mean + 1e-9)
    assert np.all(res.mean <= res.ci_high + 1e-9)


# --- Jaccard + RBO --------------------------------------------------------- #


def test_top_k_jaccard():
    a = np.array([0.9, 0.1, 0.8, 0.2])
    b = np.array([0.85, 0.2, 0.7, 0.1])
    assert ag.top_k_jaccard(a, a, 2) == pytest.approx(1.0)
    assert ag.top_k_jaccard(a, b, 2) == pytest.approx(1.0)  # both top2 = {0,2}
    assert ag.top_k_jaccard([1, 0, 0, 0], [0, 0, 0, 1], 1) == pytest.approx(0.0)


def test_top_k_jaccard_nan_on_constant():
    # A constant vector has no top-k set (argsort tie-break is arbitrary): the
    # overlap must be undefined, never a spurious 1.0 (M2).
    varied = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    const = np.ones(5)
    assert np.isnan(ag.top_k_jaccard(const, varied, 2))
    assert np.isnan(ag.top_k_jaccard(varied, const, 2))
    assert np.isnan(ag.top_k_jaccard(const, const, 2))
    # the all-zero oracle effect (every key evicted) is constant too -> undefined
    assert np.isnan(ag.top_k_jaccard(np.zeros(5), np.zeros(5), 3))


def test_rbo_identical_disjoint_and_validation():
    assert ag.rbo([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)
    assert ag.rbo([0, 1, 2, 3], [4, 5, 6, 7]) == pytest.approx(0.0)
    # top-weighting: agreeing on the head beats agreeing on the tail
    head = ag.rbo([0, 1, 9, 8], [0, 1, 7, 6])
    tail = ag.rbo([9, 8, 0, 1], [7, 6, 0, 1])
    assert head > tail
    with pytest.raises(ValueError, match="equal-length"):
        ag.rbo([0, 1], [0, 1, 2])
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        ag.rbo([0], [0], p=1.5)


def test_rbo_from_importance_ranks_then_compares():
    a = np.array([0.1, 0.2, 0.9, 0.8])  # ranking 2,3,1,0
    assert ag.rbo_from_importance(a, a) == pytest.approx(1.0)


def test_rbo_from_importance_nan_on_constant():
    const = np.full(4, 0.5)
    varied = np.array([0.1, 0.2, 0.9, 0.8])
    assert np.isnan(ag.rbo_from_importance(const, varied))  # no spurious 1.0 (M2)
    assert np.isnan(ag.rbo_from_importance(varied, const))
    assert np.isnan(ag.rbo_from_importance(const, const))


# --- Gold-span agreement --------------------------------------------------- #


def test_precision_at_k():
    values = [0.9, 0.1, 0.8, 0.2]
    gold = [1, 0, 1, 0]
    assert ag.precision_at_k(values, gold, 2) == pytest.approx(1.0)
    assert ag.precision_at_k(values, gold, 4) == pytest.approx(0.5)


def test_precision_at_k_nan_on_constant_importance():
    gold = [1, 0, 1, 0]
    # constant importance -> arbitrary top-k -> precision undefined, not lucky (M2)
    assert np.isnan(ag.precision_at_k(np.ones(4), gold, 2))
    assert np.isnan(ag.precision_at_k(np.zeros(4), gold, 2))
    # a single token has no tie ambiguity -> still defined
    assert ag.precision_at_k([0.9], [1], 1) == pytest.approx(1.0)


def test_auroc_vs_gold():
    values = [0.9, 0.8, 0.2, 0.1]
    assert ag.auroc_vs_gold(values, [1, 1, 0, 0]) == pytest.approx(1.0)
    assert np.isnan(ag.auroc_vs_gold(values, [1, 1, 1, 1]))  # degenerate


# --- Permutation + FDR ----------------------------------------------------- #


def test_permutation_test_detects_real_agreement():
    rng = np.random.default_rng(0)
    base = rng.normal(size=30)
    a = base
    b = base + 0.05 * rng.normal(size=30)
    res = ag.permutation_test([a, b], n_perm=200, seed=0)
    assert res.observed > 0.9
    assert res.p_value < 0.05
    assert res.null.shape == (200,)


def test_permutation_test_null_for_independent():
    rng = np.random.default_rng(1)
    a = rng.normal(size=40)
    b = rng.normal(size=40)
    res = ag.permutation_test([a, b], n_perm=300, seed=2)
    assert res.p_value > 0.05  # no real agreement


def test_benjamini_hochberg():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
    reject, q = ag.benjamini_hochberg(p, alpha=0.05)
    assert reject.shape == (8,)
    assert reject[0] is np.True_ or bool(reject[0]) is True
    assert bool(reject[-1]) is False
    assert np.all((q >= 0) & (q <= 1))
    assert ag.benjamini_hochberg([])[0].size == 0


def test_permutation_test_nan_when_all_vectors_constant():
    # every pair undefined -> observed mean agreement is NaN; the test must refuse
    # to report the spurious p = 1/(n_perm+1) that `null >= NaN` would yield (M3).
    res = ag.permutation_test([np.ones(10), np.full(10, 3.0)], n_perm=50, seed=0)
    assert np.isnan(res.observed)
    assert np.isnan(res.p_value)
    assert res.null.shape == (50,)
    assert np.all(np.isnan(res.null))


def test_benjamini_hochberg_carries_nan_through():
    # NaN p-values (undefined tests, M3) are excluded from the correction and
    # returned as not-rejected with NaN q -- they neither reject nor dilute the rest.
    p = [0.001, np.nan, 0.04, np.nan, 0.5]
    reject, q = ag.benjamini_hochberg(p, alpha=0.05)
    assert reject.shape == (5,)
    assert not reject[1] and not reject[3]
    assert np.isnan(q[1]) and np.isnan(q[3])
    assert bool(reject[0]) is True
    assert np.isfinite(q[0]) and np.isfinite(q[2]) and np.isfinite(q[4])
    # all-NaN input -> nothing rejected, every q NaN (no crash)
    rej2, q2 = ag.benjamini_hochberg([np.nan, np.nan])
    assert not rej2.any() and np.all(np.isnan(q2))


def test_benjamini_yekutieli_is_more_conservative_under_dependence():
    # BY scales BH q-values by the harmonic factor c(m) = sum_i 1/i > 1, so under
    # dependence it never rejects more and inflates the small q-values (M4).
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
    _, q_bh = ag.benjamini_hochberg(p, dependent=False)
    _, q_by = ag.benjamini_hochberg(p, dependent=True)
    assert np.all(q_by >= q_bh - 1e-12)
    assert np.any(q_by > q_bh + 1e-9)
