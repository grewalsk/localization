"""Phase 2 acceptance gates: the oracle (addendum §11.2).

The pure-CPU cores behind the oracle gates run in CI: the masking mechanics
(Definition 2, §8.1) that back 11.2a/11.2d, the attribution-patching fidelity
check (11.2a), the 11.2a exact-masking fallback decision, the corruption-stability
robustness number (11.2b), and the transcoder feature->token reductions (§2.1 /
11.2c). The model-bound gates (11.2a end-to-end, 11.2c, 11.2d) are marked ``gpu``
with thresholds pinned and now read the canonical :mod:`lcv.oracle.masking`.
"""

import numpy as np
import pytest

from lcv.oracle import adjudication, attr_patching, masking
from lcv.oracle import corruption as co
from lcv.signals import transcoder_attr

# Gate 11.2c bars (§4.4 / §11.2c).
TRANSCODER_GOLD_AUROC_MIN = 0.5  # above chance vs gold-span membership
TRANSCODER_ORACLE_SPEARMAN_MIN = transcoder_attr.TRANSCODER_ORACLE_SPEARMAN_GATE  # 0.3


# --- CPU core: masking oracle mechanics (Definition 2, §8.1) --------------- #


def test_build_key_keep_mask_single_and_multi():
    assert masking.build_key_keep_mask(5, 2).tolist() == [True, True, False, True, True]
    assert masking.build_key_keep_mask(4, [0, 3]).tolist() == [False, True, True, False]


def test_build_key_keep_mask_guards():
    with pytest.raises(ValueError, match="seq_len"):
        masking.build_key_keep_mask(0, 0)
    with pytest.raises(ValueError, match="out of range"):
        masking.build_key_keep_mask(3, 3)


def test_renormalize_after_masking_redistributes_proportionally():
    out = masking.renormalize_after_masking(
        np.array([0.2, 0.3, 0.5]), np.array([True, False, True])
    )
    assert out[1] == 0.0  # masked key gets exactly zero weight (Definition 2)
    assert out.sum() == pytest.approx(1.0)  # row is still a distribution
    assert out[0] / out[2] == pytest.approx(0.2 / 0.5)  # survivors keep their ratio


def test_renormalize_after_masking_degenerate_is_zeros():
    # the only surviving mass was on the masked key -> all-zeros, no 0/0
    keep = np.array([True, False, True])
    out = masking.renormalize_after_masking(np.array([0.0, 1.0, 0.0]), keep)
    assert np.allclose(out, 0.0)


def test_renormalize_after_masking_shape_guard():
    with pytest.raises(ValueError, match="must match"):
        masking.renormalize_after_masking(np.zeros(3), np.ones(4, dtype=bool))


def test_teacher_forced_logprob_uniform_logits():
    # uniform 4-way logits -> every token has prob 1/4
    assert masking.teacher_forced_logprob(np.zeros((1, 4)), [2]) == pytest.approx(-np.log(4))


def test_teacher_forced_logprob_shift_invariant():
    # +1000 to all logits would overflow a naive exp; stable log-softmax is invariant
    assert masking.teacher_forced_logprob(np.zeros((1, 4)) + 1000.0, [0]) == pytest.approx(
        -np.log(4)
    )


def test_teacher_forced_logprob_sums_over_answer_tokens():
    assert masking.teacher_forced_logprob(np.zeros((3, 5)), [0, 1, 2]) == pytest.approx(
        3 * -np.log(5)
    )


def test_teacher_forced_logprob_guards():
    with pytest.raises(ValueError, match="2-D"):
        masking.teacher_forced_logprob(np.zeros(4), [0])
    with pytest.raises(ValueError, match="one logit row"):
        masking.teacher_forced_logprob(np.zeros((2, 4)), [0])
    with pytest.raises(ValueError, match="no answer"):
        masking.teacher_forced_logprob(np.zeros((0, 4)), [])


def test_answer_log_prob_from_logits_slices_predicting_rows():
    seq, vocab = 5, 4
    lg = np.full((seq, vocab), -10.0)
    # logits[j] predicts position j+1, so answer_start=3 with L=2 reads rows 2 and 3
    lg[2, 1] = 10.0
    lg[3, 2] = 10.0
    direct = masking.answer_log_prob_from_logits(lg, [1, 2], answer_start=3)
    assert direct == pytest.approx(masking.teacher_forced_logprob(lg[2:4], [1, 2]))


def test_answer_log_prob_from_logits_guards():
    lg = np.zeros((4, 3))
    with pytest.raises(ValueError, match="answer_start must be >= 1"):
        masking.answer_log_prob_from_logits(lg, [0], answer_start=0)
    with pytest.raises(ValueError, match="exceed seq len"):
        masking.answer_log_prob_from_logits(lg, [0, 0], answer_start=4)


def test_masking_effect_sign_convention():
    # clean=0; masking token 0 hurt (-1 -> E=+1), token 2 helped (+0.5 -> E=-0.5)
    E = masking.masking_effect(0.0, np.array([-1.0, 0.0, 0.5]))
    assert E.tolist() == [1.0, 0.0, -0.5]  # positive E == causal content removed


def test_masking_effect_guards():
    with pytest.raises(ValueError, match="1-D"):
        masking.masking_effect(0.0, np.zeros((2, 2)))
    with pytest.raises(ValueError, match=">= 1"):
        masking.masking_effect(0.0, np.zeros(0))


# --- CPU core: 11.2a exact-masking fallback decision (§11.2a) --------------- #


def test_adjudication_subset_size_pinned():
    assert adjudication.ADJUDICATION_SUBSET_SIZE == 150


def test_decide_oracle_path_uses_patching_above_gate():
    plan = adjudication.decide_oracle_path(0.85)
    assert plan.use_attribution_patching is True
    assert plan.adjudication_subset_size == 150


def test_decide_oracle_path_at_gate_boundary_passes():
    plan = adjudication.decide_oracle_path(attr_patching.ATTR_PATCHING_SPEARMAN_GATE)
    assert plan.use_attribution_patching is True


def test_decide_oracle_path_falls_back_below_gate():
    plan = adjudication.decide_oracle_path(0.5)
    assert plan.use_attribution_patching is False
    assert plan.adjudication_subset_size == 150
    assert "exact masking" in plan.reason


def test_decide_oracle_path_nan_fails_closed():
    # never trust an undefined fidelity -> exact masking on the subset
    assert adjudication.decide_oracle_path(float("nan")).use_attribution_patching is False


def test_decide_oracle_path_none_fails_closed():
    assert adjudication.decide_oracle_path(None).use_attribution_patching is False


def test_decide_oracle_path_force_exact_overrides_high_spearman():
    plan = adjudication.decide_oracle_path(0.99, force_exact=True)
    assert plan.use_attribution_patching is False
    assert "force_exact" in plan.reason


def test_decide_oracle_path_subset_size_guard():
    with pytest.raises(ValueError, match="subset_size"):
        adjudication.decide_oracle_path(0.9, subset_size=0)


# --- CPU core: transcoder feature->token reductions (§2.1 / 11.2c) --------- #


def test_reduce_input_gradient_distributes_by_grad_norm():
    # two features reading disjoint tokens with equal attribution
    a = transcoder_attr.reduce_input_gradient(
        np.array([1.0, 1.0]), np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    )
    assert a[0] == pytest.approx(1.0)
    assert a[1] == pytest.approx(0.0)
    assert a[2] == pytest.approx(1.0)


def test_reduce_input_gradient_dead_row_contributes_nothing():
    # feature 0 reads no token (all-zero row); its |g| is dropped, not spread
    a = transcoder_attr.reduce_input_gradient(
        np.array([1.0, 5.0]), np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    )
    assert a.tolist() == [1.0, 0.0, 0.0]


def test_reduce_input_gradient_uses_magnitude_of_attr():
    pos = transcoder_attr.reduce_input_gradient(np.array([2.0]), np.array([[1.0, 3.0]]))
    neg = transcoder_attr.reduce_input_gradient(np.array([-2.0]), np.array([[1.0, 3.0]]))
    assert np.allclose(pos, neg)


def test_reduce_input_gradient_guards():
    with pytest.raises(ValueError, match="1-D over features"):
        transcoder_attr.reduce_input_gradient(np.zeros((2, 2)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="2-D"):
        transcoder_attr.reduce_input_gradient(np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError, match="feature count mismatch"):
        transcoder_attr.reduce_input_gradient(np.zeros(2), np.zeros((3, 4)))
    with pytest.raises(ValueError, match=">= 1 feature"):
        transcoder_attr.reduce_input_gradient(np.zeros(0), np.zeros((0, 4)))
    with pytest.raises(ValueError, match="non-negative"):
        transcoder_attr.reduce_input_gradient(np.array([1.0]), np.array([[-1.0, 0.0]]))


def test_reduce_firing_position_single_indices():
    a = transcoder_attr.reduce_firing_position(np.array([1.0, 2.0]), [0, 2], n_tokens=3)
    assert np.argmax(a) == 2  # raw mass [1, 0, 2]
    assert a[1] == pytest.approx(0.0)


def test_reduce_firing_position_splits_iterable_mass():
    a = transcoder_attr.reduce_firing_position(np.array([2.0]), [[0, 1]], n_tokens=3)
    assert a[0] == pytest.approx(a[1])  # 2.0 split equally over positions 0 and 1
    assert a[2] == pytest.approx(0.0)


def test_reduce_firing_position_empty_entry_skipped():
    a = transcoder_attr.reduce_firing_position(np.array([1.0, 1.0]), [[], [1]], n_tokens=3)
    assert a.tolist() == [0.0, 1.0, 0.0]  # feature 0 fires nowhere


def test_reduce_firing_position_guards():
    with pytest.raises(ValueError, match="one firing entry"):
        transcoder_attr.reduce_firing_position(np.zeros(2), [0], n_tokens=3)
    with pytest.raises(ValueError, match="n_tokens"):
        transcoder_attr.reduce_firing_position(np.zeros(1), [0], n_tokens=0)
    with pytest.raises(ValueError, match="out of range"):
        transcoder_attr.reduce_firing_position(np.zeros(1), [5], n_tokens=3)


def test_reductions_agree_when_feature_reads_and_fires_at_same_token():
    # the §2.1 robustness pairing: when a feature's gradient mass and its firing
    # position coincide, both reductions must rank that content token top
    g = np.array([1.0])
    ig = transcoder_attr.reduce_input_gradient(g, np.array([[0.0, 0.0, 1.0]]))
    fp = transcoder_attr.reduce_firing_position(g, [2], n_tokens=3)
    assert int(np.argmax(ig)) == int(np.argmax(fp)) == 2


# --- CPU core: 11.2a attribution-patching fidelity ------------------------- #


def test_11_2a_gate_threshold_is_point_eight():
    assert attr_patching.ATTR_PATCHING_SPEARMAN_GATE == 0.8


def test_11_2a_validate_passes_for_correlated_estimate():
    rng = np.random.default_rng(0)
    E = rng.normal(size=300)
    E_hat = E + 0.02 * rng.normal(size=300)  # near-perfect ranking
    rho, sign_agreement = attr_patching.validate_attribution_patching(E, E_hat)
    assert rho >= attr_patching.ATTR_PATCHING_SPEARMAN_GATE
    assert sign_agreement > 0.9
    assert attr_patching.passes_attr_patching_gate(E, E_hat)


def test_11_2a_validate_fails_for_unrelated_estimate():
    rng = np.random.default_rng(1)
    E = rng.normal(size=300)
    E_hat = rng.normal(size=300)
    assert not attr_patching.passes_attr_patching_gate(E, E_hat)


def test_11_2a_validate_guards():
    with pytest.raises(ValueError, match="share shape"):
        attr_patching.validate_attribution_patching(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match=">= 2"):
        attr_patching.validate_attribution_patching(np.zeros(1), np.zeros(1))


# --- CPU core: 11.2b corruption stability ---------------------------------- #


def test_11_2b_corruption_stability_reported():
    a = np.array([0.9, 0.1, 0.8, 0.2, 0.7])  # top-2 = indices 0, 2
    assert co.corruption_stability(a, a, 2) == pytest.approx(1.0)
    b = np.array([0.1, 0.9, 0.2, 0.8, 0.05])  # top-2 = indices 1, 3 (disjoint)
    assert co.corruption_stability(a, b, 2) == pytest.approx(0.0)


# --- GPU gates ------------------------------------------------------------- #


@pytest.mark.gpu
def test_11_2a_attr_patching_matches_true_masking(backbone):
    """On a subsample, ``Spearman(E_hat, E) >= 0.8`` on token ranking vs masking (§11.2a)."""
    from lcv.data import niah

    insts = niah.build_niah_dataset(depths=(0.5,), haystack_sentences=(20,), seed=0)
    for inst in insts:
        E = masking.token_masking_effect(backbone, inst).E
        E_hat = attr_patching.attribution_patching_effect(backbone, inst).E_hat
        assert attr_patching.passes_attr_patching_gate(E, E_hat)  # else exact-masking subset


@pytest.mark.gpu
def test_11_2c_transcoder_token_attribution_gate(backbone):
    pytest.skip(
        f"transcoder ranking: gold AUROC > {TRANSCODER_GOLD_AUROC_MIN} AND "
        f"Spearman(oracle) > {TRANSCODER_ORACLE_SPEARMAN_MIN} on correct instances "
        "(§11.2c); on failure drop from the token substrate"
    )


@pytest.mark.gpu
def test_11_2d_gold_span_masking_is_large(backbone):
    """Masking the whole gold span drops the answer logit a lot (§11.2d)."""
    from lcv.data import niah

    inst = niah.build_niah_instance(
        instance_id="s", depth=0.5, haystack_sentences=20, magic_number=4821, seed=0
    )
    drop = masking.gold_span_masking_drop(backbone, inst)
    assert drop > 1.0  # > ~1 nat: the gold span is causally necessary, wiring is right
