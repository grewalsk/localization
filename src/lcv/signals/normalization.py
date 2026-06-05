"""Per-instance importance normalization, shared across token-attribution signals.

Every token-attribution member (accumulated attention, Wu/QR token projections,
transcoder attribution) emits a per-content-token vector normalized the *same* way
within a run (§5/§6), so their rankings are comparable. Factor it here so the
attention signal and the transcoder reduction cannot drift apart. Pure numpy.
"""

from __future__ import annotations

import numpy as np


def normalize_importance(values: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Normalize an importance vector (``minmax`` | ``zscore``), uniform per run (§5).

    A degenerate vector (constant values, so no rank information) maps to all-zeros
    rather than producing a divide-by-zero, in both schemes. Empty input is
    returned unchanged.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v
    if method == "minmax":
        lo, hi = float(v.min()), float(v.max())
        return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    if method == "zscore":
        mu, sd = float(v.mean()), float(v.std())
        return (v - mu) / sd if sd > 0 else np.zeros_like(v)
    raise ValueError(f"unknown normalization {method!r}")
