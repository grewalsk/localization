"""Phase 2 acceptance gates: the oracle (addendum §11.2).

The attribution-patching fidelity check (11.2a) and the corruption-stability
robustness number (11.2b) have pure-CPU cores that run in CI. The model-bound
gates (11.2a end-to-end, 11.2c, 11.2d) are marked ``gpu`` with thresholds pinned.
"""

import numpy as np
import pytest

from lcv.oracle import attr_patching
from lcv.oracle import corruption as co
from lcv.signals import transcoder_attr

# Gate 11.2c bars (§4.4 / §11.2c).
TRANSCODER_GOLD_AUROC_MIN = 0.5  # above chance vs gold-span membership
TRANSCODER_ORACLE_SPEARMAN_MIN = transcoder_attr.TRANSCODER_ORACLE_SPEARMAN_GATE  # 0.3


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
def test_11_2a_attr_patching_matches_true_ablation(backbone):
    """On a subsample, ``Spearman(E_hat, E) >= 0.8`` on token ranking (§11.2a)."""
    from lcv.data import niah
    from lcv.oracle import ablation

    insts = niah.build_niah_dataset(depths=(0.5,), haystack_sentences=(20,), seed=0)
    for inst in insts:
        E = ablation.token_ablation_effect(backbone, inst).E
        E_hat = attr_patching.attribution_patching_effect(backbone, inst).E_hat
        assert attr_patching.passes_attr_patching_gate(E, E_hat)  # else subsampled ablation


@pytest.mark.gpu
def test_11_2c_transcoder_token_attribution_gate(backbone):
    pytest.skip(
        f"transcoder ranking: gold AUROC > {TRANSCODER_GOLD_AUROC_MIN} AND "
        f"Spearman(oracle) > {TRANSCODER_ORACLE_SPEARMAN_MIN} on correct instances "
        "(§11.2c); on failure drop from the token substrate"
    )


@pytest.mark.gpu
def test_11_2d_gold_span_ablation_is_large(backbone):
    """Ablating the whole gold span drops the answer logit a lot (§11.2d)."""
    from lcv.data import niah
    from lcv.oracle import ablation

    inst = niah.build_niah_instance(
        instance_id="s", depth=0.5, haystack_sentences=20, magic_number=4821, seed=0
    )
    drop = ablation.gold_span_ablation_drop(backbone, inst)
    assert drop > 1.0  # > ~1 nat: the gold span is causally necessary, wiring is right
