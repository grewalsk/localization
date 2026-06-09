"""Chat-template rendering + content-token mask + gold mapping (§3.3, gate 11.1e).

The deployment rule (§3.3): run the **full** rendered chat template through the
model, but rank token-level importance over **content tokens only** (role tags
and special tokens excluded). Off-by-one over the template is *the* canonical
silent bug; gate 11.1e checks the mask by decoding it.

This module renders an instance through a real fast tokenizer's
``apply_chat_template`` and builds the content mask + gold-span token indices by
delegating to the torch-free machinery in :mod:`lcv.data.tokenization`. It needs
a fast tokenizer but **no torch**, so it stays CPU-importable; the tokenizer is
passed in (the integration test is marked ``needs_model``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..contracts import Dataset, GoldSpan, Instance
from ..data import tokenization as tok

if TYPE_CHECKING:  # hints only; transformers is pulled in by the caller's tokenizer
    from transformers import PreTrainedTokenizerBase


def render_chat(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    context: str,
    *,
    system_prompt: str | None = None,
    add_generation_prompt: bool = True,
) -> str:
    """Render the deployment-realistic prompt via ``apply_chat_template`` (§3.3).

    The user turn is ``f"{context}\\n\\n{question}"`` (context first, mirroring the
    NIAH renderer). Returns the rendered *string* (``tokenize=False``); the content
    mask is then built from a single offset-mapped re-tokenization, with template
    specials dropped by id and content clipped to the verbatim context window
    (:func:`content_token_mask`) so role tags, an injected system block, and the
    question never enter the content set (§3.3).
    """
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": f"{context}\n\n{question}"})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def content_token_mask(
    tokenizer: PreTrainedTokenizerBase,
    rendered_prompt: str,
    *,
    content_char_range: tuple[int, int] | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    """``(input_ids, offset_mapping, content_mask)`` for a rendered prompt (gate 11.1e).

    The rendered prompt already contains the template's special-token *text*, so we
    re-tokenize with ``add_special_tokens=False`` to avoid prepending a second BOS.
    Those in-text specials get *real* offsets and an all-zero ``special_tokens_mask``
    (the §3.3 trap), so they are dropped **structurally by id** inside
    :func:`lcv.data.tokenization.content_mask_with_tokenizer` -- not by trusting
    their offsets. ``content_char_range`` additionally clips the mask to the verbatim
    context window so role tags and the question are excluded (the canonical fix:
    without it the 51.8% first-token attention sink lands on template tokens).
    """
    return tok.content_mask_with_tokenizer(
        tokenizer,
        rendered_prompt,
        add_special_tokens=False,
        content_char_range=content_char_range,
    )


def build_instance(
    tokenizer: PreTrainedTokenizerBase,
    *,
    instance_id: str,
    dataset: Dataset,
    question: str,
    context: str,
    answer_string: str,
    gold_texts: Sequence[str] = (),
    system_prompt: str | None = None,
    metadata: dict[str, Any] | None = None,
    verify_threshold: float = tok.GOLD_MATCH_THRESHOLD,
) -> Instance:
    """Render, mask, and map gold spans into a validated :class:`Instance` (gate 11.1e).

    Content tokens are restricted to the **context** region of the rendered prompt
    (§3.3): the verbatim ``context`` substring is located in the rendered string and
    its char span becomes the ``content_char_range``, so role tags, an injected
    system block, and the appended question are excluded -- matching the CPU builders
    (:func:`lcv.data.assembly.assemble_text_instance`, :mod:`lcv.data.niah`). This is
    the chat-path half of the canonical content-mask fix; without it every
    token-substrate signal, the oracle sweep, and D(x) are computed over template
    tokens.

    For each gold sentence: locate it in the rendered prompt (exact, else
    whitespace-tolerant), map its char span to token indices intersected with the
    content mask, and verify the decoding fuzzy-matches the gold text at
    ``>= verify_threshold`` (gate 11.1e). A gold text that cannot be located or
    that maps below threshold raises ``ValueError`` rather than silently
    producing an off-by-one span.
    """
    rendered = render_chat(tokenizer, question, context, system_prompt=system_prompt)
    ctx_span = tok.locate_text(rendered, context)
    if ctx_span is None:
        raise ValueError(
            "context substring not found in rendered prompt; cannot restrict content "
            "tokens to the context region (§3.3)"
        )
    input_ids, offsets, mask = content_token_mask(tokenizer, rendered, content_char_range=ctx_span)

    gold_spans: list[GoldSpan] = []
    for gold_text in gold_texts:
        span = tok.locate_text(rendered, gold_text)
        if span is None:
            raise ValueError(f"gold text not found in rendered prompt: {gold_text!r}")
        c0, c1 = span
        idx = tok.char_span_to_token_indices(offsets, c0, c1, content_mask=mask)
        if not tok.verify_gold_mapping(
            rendered, offsets, idx, gold_text, threshold=verify_threshold
        ):
            raise ValueError(
                f"gold-span token mapping failed gate 11.1e (< {verify_threshold}): {gold_text!r}"
            )
        gold_spans.append(
            GoldSpan(char_start=c0, char_end=c1, text=rendered[c0:c1], token_indices=idx)
        )

    return Instance(
        instance_id=instance_id,
        dataset=dataset,
        rendered_prompt=rendered,
        context_tokens=np.asarray(input_ids),
        content_token_mask=np.asarray(mask, dtype=bool),
        answer_string=answer_string,
        gold_spans=tuple(gold_spans),
        question=question,
        metadata=metadata or {},
    )
