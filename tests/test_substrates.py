"""Tests for substrate projection, normalization, and bundling (§6).

The load-bearing invariant here is the same one from contracts: a bundle is
built only from members of a single substrate, and D(x) off a bundle is
per-instance and full-cache-only.
"""

import numpy as np
import pytest

from lcv import substrates as sub
from lcv.contracts import (
    DisagreementScore,
    HeadScore,
    ImportanceVector,
    Method,
    Substrate,
)

# --- Normalization --------------------------------------------------------- #


def test_minmax_maps_to_unit_interval():
    np.testing.assert_allclose(sub.minmax([1.0, 2.0, 3.0]), [0.0, 0.5, 1.0])


def test_minmax_constant_is_zeros_not_nan():
    np.testing.assert_array_equal(sub.minmax([4.0, 4.0, 4.0]), [0.0, 0.0, 0.0])


def test_zscore_centers_and_scales():
    z = sub.zscore([1.0, 2.0, 3.0])
    assert z.mean() == pytest.approx(0.0)
    assert z.std() == pytest.approx(1.0)


def test_zscore_constant_is_zeros():
    np.testing.assert_array_equal(sub.zscore([7.0, 7.0]), [0.0, 0.0])


def test_normalize_dispatch_and_unknown_raises():
    np.testing.assert_allclose(sub.normalize([1.0, 2.0, 3.0], "minmax"), [0.0, 0.5, 1.0])
    assert sub.normalize([1.0, 2.0, 3.0], "zscore").mean() == pytest.approx(0.0)
    with pytest.raises(ValueError, match="unknown normalization"):
        sub.normalize([1.0, 2.0], "softmax")


# --- Membership ------------------------------------------------------------ #


def test_members_includes_conditional_by_default():
    ms = sub.members(Substrate.TOKEN_ATTRIBUTION)
    assert Method.TRANSCODER_ATTR in ms


def test_members_can_drop_gate_conditional():
    ms = sub.members(Substrate.TOKEN_ATTRIBUTION, include_conditional=False)
    assert Method.TRANSCODER_ATTR not in ms
    assert Method.WU_ATTENTION in ms


# --- Token bundle ---------------------------------------------------------- #


def test_build_token_bundle_shapes_and_order():
    a = ImportanceVector("i0", Method.WU_ATTENTION, [0.1, 0.5, 0.9, 0.2])
    b = ImportanceVector("i0", Method.QR_ATTENTION, [0.3, 0.2, 0.8, 0.4])
    bundle = sub.build_token_bundle([a, b])
    assert bundle.substrate is Substrate.TOKEN_ATTRIBUTION
    assert bundle.instance_id == "i0"
    assert bundle.methods == (Method.WU_ATTENTION, Method.QR_ATTENTION)
    assert bundle.vectors.shape == (2, 4)
    # each row is minmax-normalized into [0, 1]
    assert bundle.vectors.min() == pytest.approx(0.0)
    assert bundle.vectors.max() == pytest.approx(1.0)


def test_build_token_bundle_disagreement_zero_for_identical():
    x = [0.1, 0.5, 0.9, 0.2]
    a = ImportanceVector("i0", Method.WU_ATTENTION, x)
    b = ImportanceVector("i0", Method.QR_ATTENTION, list(x))
    bundle = sub.build_token_bundle([a, b])
    assert bundle.disagreement() == pytest.approx(0.0)
    ds = bundle.disagreement_score()
    assert isinstance(ds, DisagreementScore)
    assert ds.instance_id == "i0"
    assert ds.substrate is Substrate.TOKEN_ATTRIBUTION
    assert ds.value == pytest.approx(0.0)
    assert ds.n_methods == 2


def test_build_token_bundle_pairwise_spearman_matrix():
    a = ImportanceVector("i0", Method.WU_ATTENTION, [0.1, 0.5, 0.9, 0.2])
    b = ImportanceVector("i0", Method.QR_ATTENTION, [0.9, 0.5, 0.1, 0.8])
    m = sub.build_token_bundle([a, b]).pairwise_spearman()
    assert m.shape == (2, 2)
    np.testing.assert_allclose(np.diag(m), 1.0)
    np.testing.assert_allclose(m, m.T)


def test_build_token_bundle_rejects_empty():
    with pytest.raises(ValueError, match="no vectors"):
        sub.build_token_bundle([])


def test_build_token_bundle_rejects_multiple_instances():
    a = ImportanceVector("i0", Method.WU_ATTENTION, [0.1, 0.2, 0.3])
    b = ImportanceVector("i1", Method.QR_ATTENTION, [0.3, 0.2, 0.1])
    with pytest.raises(ValueError, match="multiple instances"):
        sub.build_token_bundle([a, b])


def test_build_token_bundle_rejects_misaligned_lengths():
    a = ImportanceVector("i0", Method.WU_ATTENTION, [0.1, 0.2, 0.3, 0.4])
    b = ImportanceVector("i0", Method.QR_ATTENTION, [0.3, 0.2, 0.1])
    with pytest.raises(ValueError, match="not aligned"):
        sub.build_token_bundle([a, b])


# --- Component bundle ------------------------------------------------------ #


def test_build_component_bundle_flattens_head_grids():
    g1 = np.array([[0.1, 0.5, 0.2], [0.9, 0.3, 0.4]])
    g2 = np.array([[0.2, 0.4, 0.1], [0.8, 0.2, 0.5]])
    s1 = HeadScore(Method.WU_HEADS, g1, instance_id="i0")
    s2 = HeadScore(Method.QR_HEADS, g2, instance_id="i0")
    bundle = sub.build_component_bundle([s1, s2])
    assert bundle.substrate is Substrate.COMPONENT
    assert bundle.instance_id == "i0"
    assert bundle.methods == (Method.WU_HEADS, Method.QR_HEADS)
    assert bundle.vectors.shape == (2, 6)  # 2 layers * 3 heads, flattened


def test_build_component_bundle_rejects_shape_mismatch():
    s1 = HeadScore(Method.WU_HEADS, np.zeros((2, 3)), instance_id="i0")
    s2 = HeadScore(Method.QR_HEADS, np.zeros((2, 4)), instance_id="i0")
    with pytest.raises(ValueError, match="differ in shape"):
        sub.build_component_bundle([s1, s2])


def test_build_component_bundle_rejects_empty():
    with pytest.raises(ValueError, match="no scores"):
        sub.build_component_bundle([])


def test_build_component_bundle_rejects_mixed_instances():
    # a per-instance score bundled with a global (None) one would silently inherit
    # scores[0]'s id and mislabel D(x); require a single shared id (M4).
    s1 = HeadScore(Method.WU_HEADS, np.zeros((2, 2)), instance_id="i0")
    s2 = HeadScore(Method.QR_HEADS, np.zeros((2, 2)), instance_id="i1")
    with pytest.raises(ValueError, match="multiple instances"):
        sub.build_component_bundle([s1, s2])
    s3 = HeadScore(Method.QR_HEADS, np.zeros((2, 2)))  # instance_id None
    with pytest.raises(ValueError, match="multiple instances"):
        sub.build_component_bundle([s1, s3])


def test_component_bundle_without_instance_refuses_disagreement_score():
    # global head sets (instance_id None) have no per-instance D(x)
    s1 = HeadScore(Method.WU_HEADS, np.array([[0.1, 0.9], [0.2, 0.8]]))
    s2 = HeadScore(Method.QR_HEADS, np.array([[0.3, 0.7], [0.4, 0.6]]))
    bundle = sub.build_component_bundle([s1, s2])
    assert bundle.instance_id is None
    with pytest.raises(ValueError, match="per-instance"):
        bundle.disagreement_score()
