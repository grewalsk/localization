"""Tests for the core data contracts, especially the two baked-in invariants:
substrates never merge (§6) and D(x) is full-cache-only (§9.3)."""

import numpy as np
import pytest

from lcv.contracts import (
    CompressionMethod,
    Confounds,
    Dataset,
    DisagreementScore,
    FlipRecord,
    GoldSpan,
    HeadScore,
    ImportanceVector,
    Instance,
    Method,
    OracleEffect,
    RetrievalHeadSet,
    RetrievalSource,
    Substrate,
    assert_same_substrate,
    substrate_of,
)


def test_substrate_of_maps_each_member():
    assert substrate_of(Method.ACCUMULATED_ATTENTION) is Substrate.TOKEN_ATTRIBUTION
    assert substrate_of(Method.WU_HEADS) is Substrate.COMPONENT
    assert substrate_of(Method.ORGAD_TOKENS) is Substrate.ANSWER_POSITION


def test_assert_same_substrate_passes_within_and_raises_across():
    assert (
        assert_same_substrate([Method.WU_ATTENTION, Method.QR_ATTENTION])
        is Substrate.TOKEN_ATTRIBUTION
    )
    with pytest.raises(ValueError, match="never merged"):
        assert_same_substrate([Method.WU_ATTENTION, Method.WU_HEADS])
    with pytest.raises(ValueError, match="no methods"):
        assert_same_substrate([])


def test_importance_vector_rejects_wrong_substrate_and_shape():
    with pytest.raises(ValueError, match="token-attribution"):
        ImportanceVector("x", Method.WU_HEADS, np.zeros(3))
    with pytest.raises(ValueError, match="1-D"):
        ImportanceVector("x", Method.WU_ATTENTION, np.zeros((2, 2)))
    iv = ImportanceVector("x", Method.WU_ATTENTION, [0.1, 0.2, 0.3])
    assert iv.substrate is Substrate.TOKEN_ATTRIBUTION
    assert iv.values.dtype == float


def test_head_score_rejects_wrong_substrate_and_top_k():
    with pytest.raises(ValueError, match="component"):
        HeadScore(Method.WU_ATTENTION, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="2-D"):
        HeadScore(Method.WU_HEADS, np.zeros(4))
    hs = HeadScore(Method.WU_HEADS, np.array([[0.1, 0.5, 0.2], [0.9, 0.3, 0.4]]))
    assert hs.top_k_heads(2) == [(1, 0), (0, 1)]


def test_instance_mask_must_match_tokens():
    with pytest.raises(ValueError, match="content_token_mask"):
        Instance(
            instance_id="x",
            dataset=Dataset.NIAH,
            rendered_prompt="...",
            context_tokens=np.arange(5),
            content_token_mask=np.ones(4, dtype=bool),
            answer_string="a",
        )
    inst = Instance(
        instance_id="x",
        dataset=Dataset.NIAH,
        rendered_prompt="...",
        context_tokens=np.arange(5),
        content_token_mask=np.array([0, 1, 1, 1, 0], dtype=bool),
        answer_string="a",
        gold_spans=(GoldSpan(0, 3, "abc"),),
    )
    assert inst.n_content == 3
    assert inst.n_tokens == 5


def test_disagreement_score_enforces_leakage_rule():
    ds = DisagreementScore("x", Substrate.TOKEN_ATTRIBUTION, value=0.4, n_methods=3)
    assert ds.from_full_cache is True
    with pytest.raises(ValueError, match="full-cache"):
        DisagreementScore(
            "x", Substrate.TOKEN_ATTRIBUTION, value=0.4, n_methods=3, from_full_cache=False
        )
    with pytest.raises(ValueError, match=">= 2 methods"):
        DisagreementScore("x", Substrate.TOKEN_ATTRIBUTION, value=0.0, n_methods=1)


def test_flip_record_truth_table():
    assert FlipRecord("x", CompressionMethod.H2O, 128, True, False).flipped is True
    assert FlipRecord("x", CompressionMethod.H2O, 128, True, True).flipped is False
    # Not correct under full cache -> not a flip candidate.
    assert FlipRecord("x", CompressionMethod.SNAPKV, 128, False, False).flipped is False


def test_oracle_effect_e_hat_shape_guard():
    with pytest.raises(ValueError, match="E_hat must match"):
        OracleEffect("x", E=np.zeros(5), E_hat=np.zeros(4))
    oe = OracleEffect("x", E=np.zeros(5), E_hat=np.ones(5))
    assert oe.E.shape == oe.E_hat.shape


def test_retrieval_head_set_as_set():
    rhs = RetrievalHeadSet(RetrievalSource.WU, head_ids=((0, 1), (3, 4)))
    assert rhs.as_set() == frozenset({(0, 1), (3, 4)})


def test_confounds_as_array():
    c = Confounds(length=10.0, gold_depth=0.5, answer_entropy=1.2)
    np.testing.assert_allclose(c.as_array(), [10.0, 0.5, 1.2])
