"""Transcoder token-attribution (§4.4; highest-risk component, gated 11.2c).

The single most error-prone signal and the one least served by a turnkey method.
Transcoder attribution is natively defined over *features*; placing it in the
token-attribution substrate requires an explicit, load-bearing feature->token
reduction (paper §2.1). Recipe (§4.4):

* **Step 1 - find answer-relevant features.** On the clean pass, at the answer
  position read transcoder feature activations at the chosen layer(s) and score
  each feature by its effect on the correct-answer logit, via either direct logit
  attribution ``DLA(f) = act_f * (W_dec[f] . W_U[answer])`` (cheap, ignores
  downstream nonlinearity) or gradient-times-activation (one backward pass). Take
  the top ``M`` features by ``|score|`` (M in 16-64); call ``g_f`` the attribution.
* **Step 2 - reduce features to tokens (input-gradient attribution, primary).**
  Distribute each feature's ``|g_f|`` over content tokens in proportion to how much
  each token's embedding ``e_t`` drives the feature's pre-activation ``z_f``::

      w_{f,t} = || d z_f / d e_t ||_2
      a^{tc}_t = sum_f |g_f| * w_{f,t} / sum_{t'} w_{f,t'}

  then min-max normalize. The reduction *mechanics* are the pure-numpy,
  CPU-tested :func:`reduce_input_gradient`; only ``W = [w_{f,t}]`` (autograd) and
  ``g`` (attribution) need the GPU. **Robustness:** :func:`reduce_firing_position`
  attributes each ``|g_f|`` to where the feature fires; gate conclusions are
  reported only where they hold across both reductions (paper §2.1).

**Gate 11.2c (mandatory):** on correctly-answered instances, the ranking must
place gold tokens above random (AUROC > chance) AND correlate with the masking
oracle (Spearman > ~0.3). **Fallback if it fails:** drop transcoder from the
token-attribution substrate, keep it in the component substrate via read-direction
loadings, and report the failure as a finding (a §13 human checkpoint). The
reductions are CPU-testable; the model-bound feature/gradient extraction is GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..contracts import HeadScore, ImportanceVector, Instance
from ..model.sae_loader import DEFAULT_TRANSCODER_LAYERS
from .normalization import normalize_importance

if TYPE_CHECKING:  # hints only
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

# Top features per instance, |score|-ranked (§4.4 Step 1).
TRANSCODER_TOP_M = 32  # range 16-64
# Gate 11.2c: Spearman(transcoder ranking, oracle) bar on correct instances.
TRANSCODER_ORACLE_SPEARMAN_GATE = 0.3
# Feature->token reductions (paper §2.1): input-gradient primary, firing-position robustness.
REDUCTION_INPUT_GRADIENT = "input_gradient"
REDUCTION_FIRING_POSITION = "firing_position"


def reduce_input_gradient(
    feature_attr: np.ndarray,
    input_grad_norms: np.ndarray,
    *,
    normalization: str = "minmax",
) -> np.ndarray:
    """Input-gradient feature->token reduction (paper §2.1; backbone-portable core).

    ``feature_attr`` is ``g_f`` over the ``F`` active features (signed or already
    ``|g_f|``; the magnitude is taken here). ``input_grad_norms`` is the ``[F, T]``
    matrix ``w_{f,t} = ||d z_f / d e_t||_2 >= 0`` over the ``T`` content tokens.
    Each feature's mass is row-normalized across tokens (a feature that reads no
    token, all-zero row, contributes nothing), weighted by ``|g_f|``, summed over
    features, and min-max normalized to ``[0,1]^T``. Pure numpy -- the GPU only has
    to produce ``g`` and ``W``.
    """
    g = np.abs(np.asarray(feature_attr, dtype=float))
    w = np.asarray(input_grad_norms, dtype=float)
    if g.ndim != 1:
        raise ValueError("feature_attr must be 1-D over features")
    if w.ndim != 2:
        raise ValueError("input_grad_norms must be 2-D [n_features, n_tokens]")
    if w.shape[0] != g.shape[0]:
        raise ValueError(f"feature count mismatch: g={g.shape[0]} vs W rows={w.shape[0]}")
    if g.size == 0:
        raise ValueError("need >= 1 feature")
    if np.any(w < 0):
        raise ValueError("input_grad_norms are L2 norms and must be non-negative")
    row_sums = w.sum(axis=1, keepdims=True)  # [F, 1]
    safe = np.where(row_sums > 0, row_sums, 1.0)  # avoid 0/0; dead rows stay 0 below
    shares = np.where(row_sums > 0, w / safe, 0.0)  # row-stochastic where the feature reads
    a = (g[:, None] * shares).sum(axis=0)  # [T]
    return normalize_importance(a, normalization)


def reduce_firing_position(
    feature_attr: np.ndarray,
    firing_positions: Sequence[int] | Sequence[Sequence[int]],
    n_tokens: int,
    *,
    normalization: str = "minmax",
) -> np.ndarray:
    """Firing-position reduction (paper §2.1 robustness; backbone-portable core).

    Attribute each feature's ``|g_f|`` to the content-token position(s) where it
    fires: ``firing_positions[f]`` is either a single token index or an iterable of
    indices (mass split equally across them). Returns the min-max normalized
    ``[0,1]^T`` vector over ``n_tokens`` content tokens. A conclusion that is stable
    between this and :func:`reduce_input_gradient` licenses the cross-family
    comparison; one that flips is a reduction artifact (H1).
    """
    g = np.abs(np.asarray(feature_attr, dtype=float))
    if g.ndim != 1:
        raise ValueError("feature_attr must be 1-D over features")
    if len(firing_positions) != g.shape[0]:
        raise ValueError(
            f"need one firing entry per feature: {len(firing_positions)} vs {g.shape[0]}"
        )
    if n_tokens <= 0:
        raise ValueError("n_tokens must be positive")
    a = np.zeros(n_tokens, dtype=float)
    for f, pos in enumerate(firing_positions):
        idxs = [int(pos)] if isinstance(pos, (int, np.integer)) else [int(p) for p in pos]
        if not idxs:
            continue
        for p in idxs:
            if not (0 <= p < n_tokens):
                raise ValueError(f"firing position {p} out of range [0, {n_tokens})")
            a[p] += g[f] / len(idxs)
    return normalize_importance(a, normalization)


def transcoder_token_attribution(
    model: HookedTransformer,
    transcoders: dict[int, SAE],
    instance: Instance,
    *,
    top_m: int = TRANSCODER_TOP_M,
    layers: Sequence[int] = DEFAULT_TRANSCODER_LAYERS,
    estimator: str = "grad_x_act",
    answer_token_ids: Sequence[int] | None = None,
) -> ImportanceVector:
    """Per-token transcoder attribution (§4.4), as the conditional TRANSCODER_ATTR member.

    Runs Step 1 (top-``M`` answer-relevant features via ``estimator`` in
    ``{"dla", "grad_x_act"}``) then Step 2 (attribute back to context tokens),
    restricted to ``instance.content_token_mask`` and normalized. Membership in
    the token-attribution substrate is conditional on passing gate 11.2c
    (``CONDITIONAL_MEMBERS`` in contracts).
    """
    raise NotImplementedError("requires GPU phase")


def transcoder_loadings(
    model: HookedTransformer,
    transcoders: dict[int, SAE],
    examples: Sequence[Instance],
    *,
    top_k: int = 20,
) -> HeadScore:
    """Attention-to-feature read-direction loadings (§6), as the TRANSCODER_LOADINGS member.

    The component-substrate transcoder member, retained even when 11.2c fails and
    the token member is dropped. Scores heads by how strongly they write into the
    read directions of the answer-relevant features.
    """
    raise NotImplementedError("requires GPU phase")
