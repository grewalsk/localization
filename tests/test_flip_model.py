"""Tests for the §9.3 leakage-safe flip-prediction model.

Two regimes are pinned: one where D(x) genuinely drives flips (it must beat the
confounds and be significant) and one where flips are confound-driven and D(x)
is noise (it must NOT manufacture signal). Plus the leakage/structural guards.
"""

import numpy as np
import pytest

from lcv.compression import flip_model as fm
from lcv.contracts import CompressionMethod, Confounds, DisagreementScore, FlipRecord, Substrate


def _dataset(*, mode: str, seed: int = 0, n: int = 400) -> fm.FlipDataset:
    """Synthetic flip data. mode='d': D(x) drives flips; mode='conf': a confound
    (length) drives them and D(x) is independent noise."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.0, 2.0, n)
    length = rng.uniform(-1.0, 1.0, n)
    gold_depth = rng.uniform(0.0, 1.0, n)
    answer_entropy = rng.uniform(0.0, 3.0, n)
    confounds = np.column_stack([length, gold_depth, answer_entropy])
    if mode == "d":
        logit = -1.5 + 2.0 * (d - 1.0)
    elif mode == "conf":
        logit = -0.5 + 2.5 * length  # D independent of the label
    else:  # pragma: no cover
        raise ValueError(mode)
    p = 1.0 / (1.0 + np.exp(-logit))
    flip = (rng.uniform(0.0, 1.0, n) < p).astype(int)
    return fm.FlipDataset(
        flip=flip,
        disagreement=d,
        confounds=confounds,
        instance_ids=tuple(f"i{i}" for i in range(n)),
    )


# --- Dataset assembly + leakage guard -------------------------------------- #


def test_build_flip_dataset_filters_to_full_cache_correct():
    flips = [
        FlipRecord("a", CompressionMethod.H2O, 128, correct_full=True, correct_compressed=False),
        FlipRecord("b", CompressionMethod.H2O, 128, correct_full=True, correct_compressed=True),
        # not correct under full cache -> excluded (flips are defined only there)
        FlipRecord("c", CompressionMethod.H2O, 128, correct_full=False, correct_compressed=False),
    ]
    ds = {
        i: DisagreementScore(i, Substrate.TOKEN_ATTRIBUTION, value=0.5, n_methods=3)
        for i in ("a", "b", "c")
    }
    cf = {i: Confounds(length=10.0, gold_depth=0.5, answer_entropy=1.0) for i in ("a", "b", "c")}
    data = fm.build_flip_dataset(flips, ds, cf)
    assert data.instance_ids == ("a", "b")
    assert data.n == 2
    assert data.flip.tolist() == [1, 0]  # a flipped, b did not
    assert data.confounds.shape == (2, 3)


def test_build_flip_dataset_raises_on_missing_features():
    flips = [FlipRecord("a", CompressionMethod.H2O, 128, True, False)]
    ds = {"a": DisagreementScore("a", Substrate.TOKEN_ATTRIBUTION, value=0.5, n_methods=3)}
    with pytest.raises(ValueError, match="missing D"):
        fm.build_flip_dataset(flips, ds, {})  # no confounds


def test_build_flip_dataset_raises_when_no_correct_instances():
    flips = [
        FlipRecord("c", CompressionMethod.H2O, 128, correct_full=False, correct_compressed=False)
    ]
    with pytest.raises(ValueError, match="no full-cache-correct"):
        fm.build_flip_dataset(flips, {}, {})


def test_disagreement_score_cannot_carry_leakage():
    # structural leakage guard: a leaky D(x) cannot even be constructed (§9.3)
    with pytest.raises(ValueError, match="full-cache"):
        DisagreementScore(
            "a", Substrate.TOKEN_ATTRIBUTION, value=0.5, n_methods=3, from_full_cache=False
        )


def test_base_flip_rate():
    ds = _dataset(mode="d")
    assert 0.0 < ds.base_flip_rate < 1.0  # gate 11.3b band


# --- The model: D(x) drives flips ------------------------------------------ #


def test_d_signal_beats_confounds_and_is_significant():
    report = fm.fit_flip_model(_dataset(mode="d"), seed=0)
    assert report.n == 400
    assert 0.0 < report.base_flip_rate < 1.0
    assert report.auroc_d_only > 0.7
    assert report.auroc_full > report.auroc_confound_only
    assert report.incremental_auroc > 0.1
    assert report.lr_df == 1
    assert report.lr_pvalue < 0.01
    assert 0.0 <= report.brier <= 1.0
    assert report.calibration_curve[0].size > 0


# --- The model: flips are confound-driven, D(x) is noise ------------------- #


def test_d_noise_does_not_manufacture_signal():
    report = fm.fit_flip_model(_dataset(mode="conf"), seed=0)
    # confounds carry the signal; D(x) alone is near chance
    assert report.auroc_confound_only > 0.7
    assert 0.4 < report.auroc_d_only < 0.6
    assert report.incremental_auroc < 0.05
    assert report.lr_pvalue > 0.05


# --- Pieces ---------------------------------------------------------------- #


def test_likelihood_ratio_test_detects_real_term():
    rng = np.random.default_rng(1)
    n = 300
    x = rng.normal(size=n)
    confound = rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.5 * confound + 1.8 * x)))).astype(int)
    stat, df, p = fm.likelihood_ratio_test(
        y, confound.reshape(-1, 1), np.column_stack([x, confound])
    )
    assert df == 1
    assert stat > 0
    assert p < 0.01


def test_likelihood_ratio_test_noise_term_not_significant():
    rng = np.random.default_rng(2)
    n = 300
    confound = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(1.5 * confound)))).astype(int)
    _, _, p = fm.likelihood_ratio_test(
        y, confound.reshape(-1, 1), np.column_stack([noise, confound])
    )
    assert p > 0.05


def test_cv_oof_proba_nan_when_single_class():
    y = np.zeros(20, dtype=int)
    x = np.arange(20.0)
    assert np.all(np.isnan(fm.cv_oof_proba(x, y)))


def test_fit_flip_model_degenerate_single_class_is_nan_not_crash():
    n = 50
    ds = fm.FlipDataset(
        flip=np.zeros(n, dtype=int),
        disagreement=np.linspace(0, 2, n),
        confounds=np.random.default_rng(0).normal(size=(n, 3)),
        instance_ids=tuple(f"i{i}" for i in range(n)),
    )
    report = fm.fit_flip_model(ds)
    assert report.base_flip_rate == 0.0
    assert np.isnan(report.auroc_d_only)
    assert np.isnan(report.incremental_auroc)
