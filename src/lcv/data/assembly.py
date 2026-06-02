"""Shared CPU assembly for the natural-text datasets (§3.3, gate 11.1e).

HotpotQA, LongBench and TriviaQA all reduce to the same problem: given a context
string, a question, an answer, and some gold answer/support texts, build a
contract-conformant :class:`Instance` whose gold spans are mapped to token indices
through the §3.3 char->token machinery. That mapping is *the* named bug surface
(gate 11.1e), so it lives here once -- tested once -- and is reused by all three
loaders instead of being re-derived (and re-broken) in each.

Like :mod:`lcv.data.niah`, the CPU core tokenizes with the deterministic
word-level stand-in (:func:`lcv.data.niah.word_tokenize`), **not** the model
tokenizer. Char offsets are authoritative and survive re-tokenization through the
real tokenizer + chat template on GPU via the same :mod:`lcv.data.tokenization`
helpers. So an instance built here is fully valid and gold-mapping-checkable on
CPU, and the GPU pipeline only swaps the tokenizer underneath.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..contracts import Dataset, GoldSpan, Instance
from .niah import word_tokenize
from .tokenization import char_span_to_token_indices, content_mask_from_offsets, locate_text


def assemble_text_instance(
    *,
    instance_id: str,
    dataset: Dataset,
    context: str,
    question: str,
    answer_string: str,
    gold_texts: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
    join: str = "\n\n",
) -> Instance:
    """Build an :class:`Instance` from raw text + gold spans located in the context.

    The prompt is rendered as ``context + join + question``; **content tokens are
    the context only** (§3.3), so token-level rankings never range over the
    question/template. Each gold text is located in the context (exact, then
    whitespace-tolerant, via :func:`locate_text`) and mapped to content-token
    indices via the §3.3 machinery.

    Gold texts that cannot be located, or that map to no content token, are
    dropped -- they would fail gate 11.1e anyway -- but the count is recorded in
    ``metadata["gold_unlocated"]`` so a silent annotation/render mismatch surfaces
    as a number instead of vanishing. ``metadata["n_gold"]`` is the located count.
    Duplicate gold texts collapse to the first occurrence in the context.
    """
    if not context:
        raise ValueError("context must be non-empty (no content tokens otherwise)")
    rendered = f"{context}{join}{question}"
    token_ids, spans = word_tokenize(rendered)
    context_len = len(context)
    # Content == context region only; the appended question is template (§3.3).
    mask = content_mask_from_offsets(spans, content_char_range=(0, context_len))

    golds: list[GoldSpan] = []
    seen: set[tuple[int, int]] = set()
    unlocated = 0
    for raw_gold in gold_texts:
        gold = raw_gold.strip()
        if not gold:
            continue
        loc = locate_text(context, gold)
        if loc is None:
            unlocated += 1
            continue
        c0, c1 = loc
        if (c0, c1) in seen:
            continue
        idx = char_span_to_token_indices(spans, c0, c1, content_mask=mask)
        if not idx:
            unlocated += 1
            continue
        seen.add((c0, c1))
        golds.append(GoldSpan(char_start=c0, char_end=c1, text=gold, token_indices=idx))

    meta: dict[str, Any] = dict(metadata or {})
    meta["gold_unlocated"] = unlocated
    meta["n_gold"] = len(golds)
    return Instance(
        instance_id=instance_id,
        dataset=dataset,
        rendered_prompt=rendered,
        context_tokens=token_ids,
        content_token_mask=mask,
        answer_string=answer_string,
        gold_spans=tuple(golds),
        question=question,
        metadata=meta,
    )
