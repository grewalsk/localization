"""ITI head selection + truth-direction read (§5.4).

Internal-confidence methods do **not** belong in the token-attribution substrate
(they read the answer position, not the context). ITI contributes two things:

* a **head set** to the component substrate (``ITI_HEADS``): train a linear probe
  on each head's per-token output activations (``hook_z``, head-sliced) to
  classify truthful vs untruthful on a labeled true/false set, rank heads by
  held-out probe AUROC, take the top-K. K defaults to ``DEFAULT_TOP_K`` (=20),
  matched to the retrieval-head count so the cross-method Jaccard compares
  equal-size sets; this deliberately departs from the ITI paper's K=48 (Li et al.
  2023), which was tuned for intervention strength, not set-overlap comparison;
* a **truth direction read at the answer position** (``ITI_DIRECTION``) to the
  answer-position substrate (§6), compared geographically against where the
  token-attribution methods place importance, never merged into them.

The probing set must be distinct from the QA evaluation sets of §10 (the ITI
setup uses a separate true/false statement set to avoid training on an eval set).
GPU phase only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..contracts import HeadScore, Instance, RetrievalHeadSet
from .retrieval_wu import DEFAULT_TOP_K

if TYPE_CHECKING:  # hints only
    from transformer_lens import HookedTransformer


def select_iti_heads(
    model: HookedTransformer,
    truth_false_dataset: Sequence[tuple[str, bool]],
    *,
    top_k: int = DEFAULT_TOP_K,
    n_folds: int = 5,
) -> HeadScore:
    """Rank heads by truth-probe held-out AUROC (§5.4), as the ITI_HEADS component score.

    Trains a linear probe per ``(layer, head)`` on head-sliced ``hook_z`` to
    separate truthful from untruthful statements with ``n_folds`` CV, and returns
    a ``[n_layers, n_heads]`` :class:`HeadScore` of held-out AUROC. ``top_k``
    defaults to ``DEFAULT_TOP_K`` (=20) to match the retrieval-head count for a
    fair Jaccard, deliberately departing from the ITI paper's K=48 (Li et al. 2023):
    here the head set is a comparison target, not an intervention knob.
    """
    raise NotImplementedError("requires GPU phase")


def iti_head_set(score: HeadScore, k: int = DEFAULT_TOP_K) -> RetrievalHeadSet:
    """Top-``k`` ITI truth heads as a :class:`RetrievalHeadSet`."""
    raise NotImplementedError("requires GPU phase")


def iti_truth_direction(
    model: HookedTransformer,
    instance: Instance,
    probe: np.ndarray,
    *,
    answer_position: int | None = None,
) -> float:
    """Read the ITI truth direction at the answer position (§5.4), as ITI_DIRECTION.

    Projects the answer-position activation onto the learned truth direction; the
    answer-position-substrate signal compared geographically against the
    token-attribution placement (§6), not merged into it.
    """
    raise NotImplementedError("requires GPU phase")
