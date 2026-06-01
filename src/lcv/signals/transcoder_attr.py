"""Transcoder token-attribution (§4.4; highest-risk component, gated 11.2c).

The single most error-prone signal and the one least served by a turnkey method.
Recipe (§4.4):

* **Step 1 - find answer-relevant features.** On the clean pass, at the answer
  position read transcoder feature activations at the chosen layer(s) and score
  each feature by its effect on the correct-answer logit, via either direct logit
  attribution ``DLA(f) = act_f * (W_dec[f] . W_U[answer])`` (cheap, ignores
  downstream nonlinearity) or gradient-times-activation (one backward pass). Take
  the top ``M`` features by ``|score|`` (M in 16-64).
* **Step 2 - attribute features back to context tokens.** For each top feature,
  take the gradient of ``act_f`` w.r.t. the residual stream at each context-token
  position, contract over the residual dim, times the clean activation (or the
  corruption delta for an attribution-patching estimate). Weight by the feature's
  Step-1 importance and sum over the ``M`` features -> per-token importance.

**Gate 11.2c (mandatory):** on correctly-answered instances, the ranking must
place gold tokens above random (AUROC > chance) AND correlate with the oracle
(Spearman > ~0.3). **Fallback if it fails:** drop transcoder from the
token-attribution substrate, keep it in the component substrate via read-direction
loadings, and report the failure as a finding (a §13 human checkpoint). GPU phase
only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..contracts import HeadScore, ImportanceVector, Instance
from ..model.sae_loader import DEFAULT_TRANSCODER_LAYERS

if TYPE_CHECKING:  # hints only
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

# Top features per instance, |score|-ranked (§4.4 Step 1).
TRANSCODER_TOP_M = 32  # range 16-64
# Gate 11.2c: Spearman(transcoder ranking, oracle) bar on correct instances.
TRANSCODER_ORACLE_SPEARMAN_GATE = 0.3


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
