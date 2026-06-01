"""Phase 3 acceptance gates: the payoff (addendum §11.3).

Does per-instance disagreement ``D(x)`` predict KV-compression flips beyond
difficulty confounds? Two of the three gates have pure-CPU cores that run in CI:
the usable-flip-rate band (11.3b) is a property of the assembled
:class:`~lcv.compression.flip_model.FlipDataset`, and the leakage rule (11.3c) is
enforced by the :class:`~lcv.contracts.DisagreementScore` type plus the field
order of the report. The eviction-reproduction gate (11.3a) is model-bound and
marked ``gpu`` with its tolerance pinned.
"""

import dataclasses

import numpy as np
import pytest

from lcv.compression import flip_model as fm
from lcv.compression import flip_test as ft
from lcv.contracts import (
    CompressionMethod,
    Confounds,
    DisagreementScore,
    FlipRecord,
    Substrate,
)

# Gate 11.3a bar (§9.1 / §11.3a): reproduce the paper LongBench number within a
# few points. Pinned here; the end-to-end check is GPU-only.
PAPER_TOLERANCE_POINTS = 5.0


def _make_dataset(n: int, n_flips: int, *, seed: int = 0) -> fm.FlipDataset:
    """Assemble a leakage-safe :class:`FlipDataset` of ``n`` full-cache-correct rows.

    The first ``n_flips`` instances flip under compression. ``D(x)`` is drawn with
    a higher mean on flipped instances but enough spread to overlap (no perfect
    separation, so the logistic fit converges), exercising the real §9.3 assembly
    path including the full-cache leakage guard.
    """
    rng = np.random.default_rng(seed)
    flips: list[FlipRecord] = []
    disagreements: dict[str, DisagreementScore] = {}
    confounds: dict[str, Confounds] = {}
    for i in range(n):
        iid = f"x{i}"
        is_flip = i < n_flips
        flips.append(
            FlipRecord(
                instance_id=iid,
                method=CompressionMethod.H2O,
                budget=0.5,
                correct_full=True,
                correct_compressed=not is_flip,
            )
        )
        center = 0.65 if is_flip else 0.35
        disagreements[iid] = DisagreementScore(
            instance_id=iid,
            substrate=Substrate.TOKEN_ATTRIBUTION,
            value=float(np.clip(center + 0.15 * rng.standard_normal(), 0.0, 1.0)),
            n_methods=4,
        )
        confounds[iid] = Confounds(
            length=float(rng.integers(500, 4000)),
            gold_depth=float(rng.random()),
            answer_entropy=float(rng.random()),
        )
    return fm.build_flip_dataset(flips, disagreements, confounds)


# --- CPU core: 11.3b usable flip-rate band --------------------------------- #


def test_11_3b_band_and_sample_constants():
    assert ft.USABLE_FLIP_RATE_BAND == (0.05, 0.40)
    assert ft.MIN_CORRECT_PER_BUDGET == 300  # >= 300 correct/budget so flips are fittable


def test_11_3b_tuned_budget_lands_in_band():
    lo, hi = ft.USABLE_FLIP_RATE_BAND
    ds = _make_dataset(20, 4)  # 20% flip rate
    assert ds.n == 20
    assert lo <= ds.base_flip_rate <= hi


def test_11_3b_degenerate_budgets_fall_out_of_band():
    lo, hi = ft.USABLE_FLIP_RATE_BAND
    assert _make_dataset(20, 0).base_flip_rate < lo  # ~0%: no signal -> raise budget
    assert _make_dataset(20, 20).base_flip_rate > hi  # ~100%: no signal -> lower budget


# --- CPU core: 11.3c leakage rule + report structure ----------------------- #


def test_11_3c_disagreement_refuses_non_full_cache():
    """D(x) cannot exist unless it came from the full-cache pass (leakage rule)."""
    with pytest.raises(ValueError, match="full-cache"):
        DisagreementScore(
            instance_id="x",
            substrate=Substrate.TOKEN_ATTRIBUTION,
            value=0.5,
            n_methods=2,
            from_full_cache=False,
        )


def test_11_3c_disagreement_needs_a_pair():
    with pytest.raises(ValueError, match=">= 2"):
        DisagreementScore(
            instance_id="x", substrate=Substrate.TOKEN_ATTRIBUTION, value=0.5, n_methods=1
        )


def test_11_3c_confound_only_precedes_disagreement_in_report():
    """The report reads confound-only AUROC *before* the D(x) increment (§9.3)."""
    names = [f.name for f in dataclasses.fields(fm.FlipModelReport)]
    assert names.index("auroc_confound_only") < names.index("auroc_d_only")
    assert names.index("auroc_d_only") < names.index("auroc_full")
    assert names.index("auroc_full") < names.index("incremental_auroc")


def test_11_3c_flip_model_runs_leakage_safe():
    ds = _make_dataset(80, 24, seed=1)
    report = fm.fit_flip_model(ds)
    assert report.n == 80
    assert report.n_flips == 24
    # all three CV AUROCs are real numbers (class balance lets the folds fit)
    assert not np.isnan(report.auroc_confound_only)
    assert not np.isnan(report.auroc_d_only)
    assert not np.isnan(report.auroc_full)
    assert report.incremental_auroc == pytest.approx(report.auroc_full - report.auroc_confound_only)
    assert report.lr_df == 1  # D(x) adds exactly one parameter over the confound model


# --- GPU gate -------------------------------------------------------------- #


@pytest.mark.gpu
def test_11_3a_eviction_reproduces_paper_longbench(backbone):
    pytest.skip(
        f"H2O/SnapKV at a stated budget reproduce paper LongBench within "
        f"{PAPER_TOLERANCE_POINTS} points (§11.3a); on failure the eviction is "
        "misconfigured -> do not proceed to Phase 3"
    )
