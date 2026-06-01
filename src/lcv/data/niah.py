"""Synthetic needle-in-a-haystack (addendum §10).

NIAH is the cleanest substrate to bring up first because gold spans are **exact
by construction**: we insert the needle, so we know its character span and which
tokens it occupies without any fuzzy gold->token mapping (contrast HotpotQA,
gate 11.1e). It also gives the easy/hard contrast for free — vary needle depth
and haystack length.

This module is the CPU core, so it tokenizes with a deterministic **word-level
stand-in** (:func:`word_tokenize`), not the model tokenizer. The real pipeline
re-renders through the chat template and the model tokenizer on GPU; the char
offsets here are authoritative and survive that re-tokenization via the §3.3
char->token mapping. The exactness guarantee is checked by
:func:`gold_text_from_tokens` (decode the gold token indices -> get the needle
back verbatim).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from ..contracts import Dataset, GoldSpan, Instance

DEFAULT_QUESTION = "What is the magic number?"
MAGIC_NEEDLE_TEMPLATE = "The magic number is {n}."

# Neutral filler; the haystack content is irrelevant to the needle by design.
_FILLER = (
    "The sky was clear and bright that quiet morning.",
    "She walked slowly along the empty country road.",
    "A train passed somewhere far beyond the hills.",
    "The old library smelled of dust and warm paper.",
    "Rain had fallen gently through most of the night.",
    "He counted the coins twice before paying for tea.",
    "The garden was overgrown with ivy and tall grass.",
    "Someone had left the kitchen window slightly open.",
    "The river moved without any sound past the bridge.",
    "Lights flickered on across the valley at dusk.",
)


# --------------------------------------------------------------------------- #
# Deterministic word-level tokenizer (CPU stand-in)
# --------------------------------------------------------------------------- #


def _word_id(word: str, vocab: int = 50_000) -> int:
    """Stable token id for a word (0 is reserved). Deterministic across runs."""
    h = int.from_bytes(hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest(), "big")
    return (h % (vocab - 1)) + 1


def word_tokenize(text: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Split on whitespace into tokens, returning ``(ids, char_spans)``.

    ``char_spans[i] == (start, end)`` with ``text[start:end]`` the i-th token, so
    char offsets and token indices are mutually invertible.
    """
    ids: list[int] = []
    spans: list[tuple[int, int]] = []
    for m in re.finditer(r"\S+", text):
        ids.append(_word_id(m.group()))
        spans.append((m.start(), m.end()))
    return np.array(ids, dtype=int), spans


# --------------------------------------------------------------------------- #
# Needle insertion
# --------------------------------------------------------------------------- #


def make_haystack(n_sentences: int, rng: np.random.Generator) -> list[str]:
    """Deterministic list of ``n_sentences`` filler sentences."""
    if n_sentences < 1:
        raise ValueError("haystack needs at least one sentence")
    return [_FILLER[int(rng.integers(len(_FILLER)))] for _ in range(n_sentences)]


def insert_needle(
    haystack_sentences: Sequence[str], needle: str, depth: float
) -> tuple[str, int, int]:
    """Insert ``needle`` at fractional ``depth`` (0=start, 1=end).

    Returns ``(context_text, char_start, char_end)`` with
    ``context_text[char_start:char_end] == needle`` exactly.
    """
    if not 0.0 <= depth <= 1.0:
        raise ValueError("depth must be in [0, 1]")
    sents = list(haystack_sentences)
    pos = int(round(depth * len(sents)))  # number of sentences before the needle
    before, after = sents[:pos], sents[pos:]
    text = " ".join([*before, needle, *after])
    prefix = " ".join(before)
    char_start = len(prefix) + (1 if before else 0)  # +1 for the join space
    char_end = char_start + len(needle)
    return text, char_start, char_end


# --------------------------------------------------------------------------- #
# Instance construction
# --------------------------------------------------------------------------- #


def build_niah_instance(
    *,
    instance_id: str,
    depth: float,
    haystack_sentences: int,
    magic_number: int | None = None,
    question: str = DEFAULT_QUESTION,
    needle_template: str = MAGIC_NEEDLE_TEMPLATE,
    seed: int = 0,
) -> Instance:
    """Build one NIAH :class:`Instance` with an exact gold span.

    ``depth`` and ``haystack_sentences`` control the easy/hard contrast. Content
    tokens are the context (haystack + needle); the appended question is template
    (non-content), so token-level rankings range over the context only (§3.3).
    """
    if "{n}" not in needle_template:
        raise ValueError("needle_template must contain '{n}' for the magic number")
    rng = np.random.default_rng(seed)
    if magic_number is None:
        magic_number = int(rng.integers(10_000, 100_000))
    needle = needle_template.format(n=magic_number)

    haystack = make_haystack(haystack_sentences, rng)
    context, c0, c1 = insert_needle(haystack, needle, depth)
    rendered = f"{context}\n\n{question}"

    token_ids, spans = word_tokenize(rendered)
    context_len = len(context)
    # context tokens (incl. needle) are content; question tokens are not (§3.3)
    mask = np.array([start < context_len for start, _ in spans], dtype=bool)
    needle_tokens = tuple(i for i, (a, b) in enumerate(spans) if a < c1 and b > c0)

    gold = GoldSpan(char_start=c0, char_end=c1, text=needle, token_indices=needle_tokens)
    return Instance(
        instance_id=instance_id,
        dataset=Dataset.NIAH,
        rendered_prompt=rendered,
        context_tokens=token_ids,
        content_token_mask=mask,
        answer_string=str(magic_number),
        gold_spans=(gold,),
        question=question,
        metadata={
            "depth": depth,
            "haystack_sentences": haystack_sentences,
            "magic_number": magic_number,
        },
    )


def build_niah_dataset(
    *,
    depths: Sequence[float],
    haystack_sentences: Sequence[int],
    question: str = DEFAULT_QUESTION,
    needle_template: str = MAGIC_NEEDLE_TEMPLATE,
    seed: int = 0,
) -> list[Instance]:
    """Grid of NIAH instances over ``depths`` x ``haystack_sentences``."""
    rng = np.random.default_rng(seed)
    out: list[Instance] = []
    for depth in depths:
        for hs in haystack_sentences:
            out.append(
                build_niah_instance(
                    instance_id=f"niah_d{depth:g}_h{hs}_{len(out)}",
                    depth=depth,
                    haystack_sentences=hs,
                    question=question,
                    needle_template=needle_template,
                    seed=int(rng.integers(1 << 31)),
                )
            )
    return out


def gold_text_from_tokens(instance: Instance, gold: GoldSpan) -> str:
    """Decode a gold span's token indices back to text (exactness check, ~11.1e).

    For synthetic NIAH this returns the needle verbatim (similarity 1.0), which is
    why NIAH validates the gold->token machinery before noisier datasets.
    """
    _, spans = word_tokenize(instance.rendered_prompt)
    idx = gold.token_indices
    if not idx:
        return ""
    return instance.rendered_prompt[spans[idx[0]][0] : spans[idx[-1]][1]]
