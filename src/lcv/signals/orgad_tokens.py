"""Orgad exact-answer-token signal (§5.5; RQ4, kept separate).

Answers RQ4 (attribution vs internal confidence) and is reported separately from
the token-attribution matrix. Identify the exact-answer tokens, then read the
truth probe at those answer-position tokens and compare *geographically* against
where the token-attribution methods place importance (§6 answer-position
substrate; never merged into the token matrix).

Identification: for extractive cases, locate the answer string within the
generated tokens (deterministic, cheap, implemented first). Orgad's free-form
extension uses heuristics plus an instruction-tuned LLM to mark exact-answer
tokens; that is optional and explicitly out of the deterministic core (§5.5, §9.2
"no LLM judge in the core"). GPU phase only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..contracts import Instance

if TYPE_CHECKING:  # hints only
    from transformer_lens import HookedTransformer


def locate_answer_tokens(
    instance: Instance,
    generated_offsets: Sequence[tuple[int, int]],
    generated_text: str,
    answer_string: str,
) -> tuple[int, ...]:
    """Token indices of the exact answer span in the generated text (extractive, §5.5).

    Reuses the §3.3 char->token machinery
    (:func:`lcv.data.tokenization.locate_text` +
    :func:`~lcv.data.tokenization.char_span_to_token_indices`) over the generated
    sequence. Returns an empty tuple if the answer string is absent (free-form
    case -> optional LLM-assisted extension, not in the deterministic core).
    """
    raise NotImplementedError("requires GPU phase")


def orgad_answer_token_signal(
    model: HookedTransformer,
    instance: Instance,
    truth_probe: np.ndarray,
    answer_token_indices: Sequence[int],
) -> tuple[tuple[int, ...], np.ndarray]:
    """Truth-probe read at the exact-answer tokens (§5.5), as the ORGAD_TOKENS signal.

    Returns ``(answer_token_indices, probe_values)`` for the answer-position
    substrate. Deliberately **not** an :class:`~lcv.contracts.ImportanceVector`
    (that type is locked to the token-attribution substrate, §6); this output is
    compared geographically against token-attribution placement, not merged.
    """
    raise NotImplementedError("requires GPU phase")
