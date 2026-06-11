"""Leakage-safe flip-prediction model (addendum §9.3).

The payoff test: does per-instance disagreement ``D(x)`` predict whether KV
compression flips a correct answer to wrong, *beyond* trivial difficulty proxies
(length, gold-span depth, answer-token entropy)? If it doesn't beat the
confound-only model, we've only rediscovered that hard questions break under
compression — not the finding. So we always report the confound-only AUROC
first, then the increment from adding ``D(x)``.

Pipeline (§9.3):

1. Keep only instances correct under the **full** cache.
2. Label ``flip = correct -> incorrect`` at a budget.
3. Fit ``flip ~ D(x)``; 4. fit ``flip ~ D(x) + confounds``; plus the
   confound-only reference.
5. Report 5-fold CV AUROC for each, the incremental AUROC of ``D(x)`` over
   confound-only, and a likelihood-ratio chi-square for the ``D(x)`` term.
6. Report calibration of the ``D(x)`` model.

**Leakage rule (named bug, §9.3 / gate 11.3c):** ``D(x)`` must come strictly from
the full-cache pass. The only way to supply it here is a
:class:`~lcv.contracts.DisagreementScore`, which refuses to exist unless
``from_full_cache`` is True — so the compressed run cannot influence a feature
that predicts its own outcome.

Deterministic numpy/scipy/sklearn/statsmodels; no GPU, no torch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from ..contracts import Confounds, DisagreementScore, FlipRecord

# Below this across-instance variance, D(x) carries no rank information for the
# flip model: every instance has ~the same disagreement, so ``flip ~ D(x)`` would
# regress on floating-point noise. Treated as degenerate (M1).
_D_VARIANCE_TOL = 1e-12

# --------------------------------------------------------------------------- #
# Assembled modelling matrix
# --------------------------------------------------------------------------- #


@dataclass(slots=True, frozen=True)
class FlipDataset:
    """Aligned design matrix for the flip model (one row per instance).

    ``confounds`` columns are ``[length, gold_depth, answer_entropy]`` (see
    :class:`~lcv.contracts.Confounds`). ``disagreement`` is ``D(x)`` from the
    full-cache pass only.
    """

    flip: np.ndarray  # [n] int {0, 1}
    disagreement: np.ndarray  # [n] float
    confounds: np.ndarray  # [n, 3]
    instance_ids: tuple[str, ...]

    @property
    def n(self) -> int:
        return int(self.flip.shape[0])

    @property
    def n_flips(self) -> int:
        return int(self.flip.sum())

    @property
    def base_flip_rate(self) -> float:
        return self.n_flips / self.n if self.n else float("nan")


def build_flip_dataset(
    flips: Sequence[FlipRecord],
    disagreements: Mapping[str, DisagreementScore],
    confounds: Mapping[str, Confounds],
    *,
    require_correct_full: bool = True,
) -> FlipDataset:
    """Assemble the design matrix, enforcing the §9.3 leakage rule.

    Only instances correct under the full cache are kept (flips are defined only
    there). ``D(x)`` arrives as a :class:`DisagreementScore`, which cannot exist
    unless it came from the full-cache pass; we re-assert defensively.
    """
    flip_rows: list[int] = []
    d_rows: list[float] = []
    c_rows: list[np.ndarray] = []
    ids: list[str] = []
    for fr in flips:
        if require_correct_full and not fr.correct_full:
            continue
        ds = disagreements.get(fr.instance_id)
        cf = confounds.get(fr.instance_id)
        if ds is None or cf is None:
            raise ValueError(f"missing D(x) or confounds for instance {fr.instance_id!r}")
        if not ds.from_full_cache:  # belt-and-suspenders; the type already guards this
            raise ValueError(f"leakage: D(x) for {fr.instance_id!r} is not full-cache (§9.3)")
        ids.append(fr.instance_id)
        flip_rows.append(int(fr.flipped))
        d_rows.append(ds.value)
        c_rows.append(cf.as_array())
    if not ids:
        raise ValueError("no full-cache-correct instances to model")
    return FlipDataset(
        flip=np.array(flip_rows, dtype=int),
        disagreement=np.array(d_rows, dtype=float),
        confounds=np.vstack(c_rows),
        instance_ids=tuple(ids),
    )


# --------------------------------------------------------------------------- #
# Pieces: CV AUROC, likelihood-ratio test, calibration
# --------------------------------------------------------------------------- #


def _model() -> object:
    # Standardize so the L2-regularized logistic converges cleanly; scaling does
    # not change AUROC ranking.
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def _as2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X.reshape(-1, 1) if X.ndim == 1 else X


def cv_oof_proba(X: np.ndarray, y: np.ndarray, *, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold P(flip) via stratified k-fold. NaN array if a class is too rare."""
    X = _as2d(X)
    y = np.asarray(y, dtype=int)
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    k = min(n_splits, n_pos, n_neg)
    if k < 2:
        return np.full(y.shape[0], np.nan)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    proba = cross_val_predict(_model(), X, y, cv=skf, method="predict_proba")
    return np.asarray(proba)[:, 1]


def _auroc(y: np.ndarray, proba: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    if np.any(np.isnan(proba)) or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, proba))


def _standardize(X: np.ndarray) -> np.ndarray:
    X = _as2d(X)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (X - mu) / sd


def likelihood_ratio_test(
    y: np.ndarray, X_reduced: np.ndarray, X_full: np.ndarray
) -> tuple[float, int, float]:
    """LRT for the extra term(s) in ``X_full`` over nested ``X_reduced``.

    Returns ``(stat, df, p_value)`` with ``stat = 2(llf_full - llf_reduced)`` and
    ``p`` from chi-square with ``df`` = added parameters. NaN on non-convergence
    or perfect separation.
    """
    y = np.asarray(y, dtype=float)
    Xr = sm.add_constant(_standardize(X_reduced), has_constant="add")
    Xf = sm.add_constant(_standardize(X_full), has_constant="add")
    df = Xf.shape[1] - Xr.shape[1]
    if np.unique(y).size < 2:  # one class -> likelihood undefined; skip the fit
        return float("nan"), int(df), float("nan")
    try:
        red = sm.Logit(y, Xr).fit(disp=0, maxiter=200)
        full = sm.Logit(y, Xf).fit(disp=0, maxiter=200)
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError):
        return float("nan"), df, float("nan")
    stat = 2.0 * (full.llf - red.llf)
    return float(stat), int(df), float(chi2.sf(stat, df))


def calibration(
    y: np.ndarray, proba: np.ndarray, *, n_bins: int = 10
) -> tuple[tuple[np.ndarray, np.ndarray], float]:
    """Calibration curve ``(mean_predicted, fraction_positive)`` and Brier score."""
    y = np.asarray(y, dtype=int)
    if np.any(np.isnan(proba)) or np.unique(y).size < 2:
        return (np.array([]), np.array([])), float("nan")
    from sklearn.calibration import calibration_curve

    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=n_bins, strategy="uniform")
    brier = float(brier_score_loss(y, proba))
    return (mean_pred, frac_pos), brier


# --------------------------------------------------------------------------- #
# Full report
# --------------------------------------------------------------------------- #


@dataclass(slots=True, frozen=True)
class FlipModelReport:
    """Everything gate 11.3 needs to judge whether ``D(x)`` adds signal."""

    n: int
    n_flips: int
    base_flip_rate: float
    d_variance: float  # across-instance variance of D(x)
    degenerate_d: bool  # True -> D(x) carried no signal; its metrics are NaN (M1)
    auroc_confound_only: float
    auroc_d_only: float
    auroc_full: float
    incremental_auroc: float  # full - confound_only
    lr_stat: float
    lr_df: int
    lr_pvalue: float
    brier: float  # calibration of the D(x) model
    calibration_curve: tuple[np.ndarray, np.ndarray]


def fit_flip_model(
    dataset: FlipDataset, *, n_splits: int = 5, seed: int = 0, calibration_bins: int = 10
) -> FlipModelReport:
    """Run the full §9.3 analysis and return the report.

    The headline is :attr:`incremental_auroc` (does ``D(x)`` beat the confounds?)
    and :attr:`lr_pvalue` (is the ``D(x)`` term significant?).

    **Degenerate D(x) (M1).** If ``D(x)`` is constant across instances (variance
    below ``_D_VARIANCE_TOL``) or has any undefined (NaN) entry, it carries no usable
    signal -- fitting ``flip ~ D(x)`` would regress on floating-point noise (spurious
    increment) or crash the sklearn pipeline. In that case :attr:`degenerate_d` is set
    and every D-dependent metric is NaN, while :attr:`auroc_confound_only` -- which
    never touches ``D(x)`` -- is still reported so the confound baseline stands.
    """
    y = dataset.flip.astype(int)
    d_flat = dataset.disagreement.astype(float)
    C = dataset.confounds

    finite = np.isfinite(d_flat)
    d_variance = float(np.nanvar(d_flat)) if finite.any() else float("nan")
    degenerate_d = (
        int(finite.sum()) < 2
        or not finite.all()
        or not np.isfinite(d_variance)
        or d_variance <= _D_VARIANCE_TOL
    )

    # The confound-only model never reads D(x); it stays valid even when D(x) is
    # degenerate, and is the reference the headline increment is measured against.
    p_conf = cv_oof_proba(C, y, n_splits=n_splits, seed=seed)
    auroc_conf = _auroc(y, p_conf)

    if degenerate_d:
        return FlipModelReport(
            n=dataset.n,
            n_flips=dataset.n_flips,
            base_flip_rate=dataset.base_flip_rate,
            d_variance=d_variance,
            degenerate_d=True,
            auroc_confound_only=auroc_conf,
            auroc_d_only=float("nan"),
            auroc_full=float("nan"),
            incremental_auroc=float("nan"),
            lr_stat=float("nan"),
            lr_df=0,
            lr_pvalue=float("nan"),
            brier=float("nan"),
            calibration_curve=(np.array([]), np.array([])),
        )

    D = d_flat.reshape(-1, 1)
    X_full = np.hstack([D, C])

    p_d = cv_oof_proba(D, y, n_splits=n_splits, seed=seed)
    p_full = cv_oof_proba(X_full, y, n_splits=n_splits, seed=seed)

    auroc_d = _auroc(y, p_d)
    auroc_full = _auroc(y, p_full)
    incremental = (
        auroc_full - auroc_conf
        if not (np.isnan(auroc_full) or np.isnan(auroc_conf))
        else float("nan")
    )

    lr_stat, lr_df, lr_p = likelihood_ratio_test(y, C, X_full)
    curve, brier = calibration(y, p_d, n_bins=calibration_bins)

    return FlipModelReport(
        n=dataset.n,
        n_flips=dataset.n_flips,
        base_flip_rate=dataset.base_flip_rate,
        d_variance=d_variance,
        degenerate_d=False,
        auroc_confound_only=auroc_conf,
        auroc_d_only=auroc_d,
        auroc_full=auroc_full,
        incremental_auroc=incremental,
        lr_stat=lr_stat,
        lr_df=lr_df,
        lr_pvalue=lr_p,
        brier=brier,
        calibration_curve=curve,
    )
