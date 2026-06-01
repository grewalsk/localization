"""KV-compression flip test via an external eviction library + gate 11.3a (§9.1).

Use **H2O** and **SnapKV** as the two token-level eviction methods, swept over
budget. Reimplementing eviction by hand is precisely where silent bugs live, so
drive an existing maintained library that exposes these under a unified HF
``generate`` interface (NVIDIA KVPress is the obvious candidate; verify it is
current and supports the target model before committing). The library is **not**
trusted blindly: gate 11.3a requires the chosen method, at a stated budget, to
roughly reproduce its paper-reported LongBench numbers (within a few points). A
failure means misconfigured eviction that would corrupt every Phase-3 result --
do not proceed.

Correctness labels use the pre-registered deterministic metric (§9.2,
:mod:`lcv.compression.correctness`); flips feed the leakage-safe model in
:mod:`lcv.compression.flip_model`. The leakage rule (§9.3): ``D(x)`` is computed
from the **full-cache** pass only and never from anything here. GPU phase only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from ..contracts import CompressionMethod, FlipRecord, Instance

if TYPE_CHECKING:  # hints only
    from transformer_lens import HookedTransformer

# Base flip rate must land in a usable band, tuned by budget (gate 11.3b).
USABLE_FLIP_RATE_BAND = (0.05, 0.40)
# Aim for >= 300 correctly-answered instances per budget so flips are fittable (§9.3).
MIN_CORRECT_PER_BUDGET = 300


def generate_answer(
    model: HookedTransformer, instance: Instance, *, max_new_tokens: int = 64
) -> str:
    """Greedy full-cache generation (the uncompressed reference answer, §9.3 step 1)."""
    raise NotImplementedError("requires GPU phase")


def generate_answer_compressed(
    model: HookedTransformer,
    instance: Instance,
    method: CompressionMethod,
    budget: float,
    *,
    library: str = "kvpress",
    max_new_tokens: int = 64,
) -> str:
    """Greedy generation under a compressed KV cache (§9.1).

    Applies ``method`` (H2O / SnapKV) at ``budget`` via the external eviction
    ``library``. Token-level eviction is unaffected by the GQA query/KV grouping
    (§3.2), so no head-group pooling is needed here.
    """
    raise NotImplementedError("requires GPU phase")


def build_flip_record(
    model: HookedTransformer,
    instance: Instance,
    method: CompressionMethod,
    budget: float,
    judge: Callable[[str], bool],
) -> FlipRecord:
    """One :class:`FlipRecord` for ``(instance, method, budget)`` (§9.3 steps 1-2).

    Scores full-cache and compressed answers with the pre-registered ``judge``
    (e.g. :func:`lcv.compression.correctness.squad_judge`); ``flipped`` is then
    ``correct_full and not correct_compressed``. Flips are only meaningful on
    instances correct under the full cache.
    """
    raise NotImplementedError("requires GPU phase")


def build_flip_records(
    model: HookedTransformer,
    instances: Sequence[Instance],
    method: CompressionMethod,
    budget: float,
    judge: Callable[[str], bool],
) -> list[FlipRecord]:
    """:class:`FlipRecord` list over ``instances`` at one ``(method, budget)`` (§9.3)."""
    raise NotImplementedError("requires GPU phase")


def reproduce_paper_gate(
    model: HookedTransformer,
    longbench_subset: Sequence[Instance],
    method: CompressionMethod,
    budget: float,
    *,
    library: str = "kvpress",
) -> dict[str, float]:
    """Reproduce the method's paper-reported LongBench score within a few points (gate 11.3a).

    Returns the measured per-task metric(s) so the caller can compare against the
    published number. Failing this gate means the eviction is misconfigured; do
    not proceed to Phase 3.
    """
    raise NotImplementedError("requires GPU phase")
