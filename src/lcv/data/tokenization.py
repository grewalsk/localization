"""Content-token mask + gold-span -> token mapping (addendum §3.3, gate 11.1e).

The canonical silent bug in this project is an off-by-one over the chat template:
activations must come from the *full* rendered prompt, but token-level rankings
must range over **content tokens only** (role tags and special tokens excluded).
The fix is to build one explicit content-token mask from the tokenizer's offset
mapping and reuse it everywhere.

The reusable machinery here is tokenizer-agnostic and torch-free — it consumes an
``offset_mapping`` (the ``(char_start, char_end)`` per token that any HF *fast*
tokenizer returns) plus an optional ``special_tokens_mask``. Special tokens carry
zero-width offsets ``(0, 0)``, so they drop out of the mask either way.

Gate 11.1e is exactly: map a gold span's characters to token indices, intersect
with the content mask, decode, and fuzzy-match the decoding to the gold text at
``>= 0.9`` similarity (:func:`verify_gold_mapping`). The single real-tokenizer
helper (:func:`content_mask_with_tokenizer`) takes a tokenizer object, so the
module stays importable in CI; its test is marked ``needs_model``.
"""

from __future__ import annotations

from collections.abc import Sequence
from difflib import SequenceMatcher

import numpy as np

OffsetMapping = Sequence[tuple[int, int]]

# Gate 11.1e similarity bar for decoded gold spans.
GOLD_MATCH_THRESHOLD = 0.9


# --------------------------------------------------------------------------- #
# Content-token mask
# --------------------------------------------------------------------------- #


def content_mask_from_offsets(
    offset_mapping: OffsetMapping,
    *,
    special_tokens_mask: Sequence[int] | None = None,
    content_char_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """Boolean content mask from an offset mapping (§3.3).

    A token is content unless it is zero-width (``end <= start``, i.e. a special
    token), flagged by ``special_tokens_mask``, or outside ``content_char_range``
    (use this to exclude the question/template region while keeping the context).
    """
    offsets = list(offset_mapping)
    n = len(offsets)
    mask = np.ones(n, dtype=bool)
    for i, (a, b) in enumerate(offsets):
        if b <= a:  # zero-width span -> special/template token
            mask[i] = False
    if special_tokens_mask is not None:
        sp = np.asarray(special_tokens_mask, dtype=bool)
        if sp.shape[0] != n:
            raise ValueError(f"special_tokens_mask length {sp.shape[0]} != n_tokens {n}")
        mask &= ~sp
    if content_char_range is not None:
        lo, hi = content_char_range
        for i, (a, b) in enumerate(offsets):
            if not (a >= lo and b <= hi):
                mask[i] = False
    return mask


# --------------------------------------------------------------------------- #
# Char span -> token indices
# --------------------------------------------------------------------------- #


def char_span_to_token_indices(
    offset_mapping: OffsetMapping,
    char_start: int,
    char_end: int,
    *,
    content_mask: np.ndarray | None = None,
) -> tuple[int, ...]:
    """Token indices overlapping ``[char_start, char_end)`` (§3.3 / 11.1e).

    Zero-width (special) tokens are skipped; if ``content_mask`` is given, the
    result is intersected with it so template tokens never enter a gold mapping.
    """
    if char_end <= char_start:
        raise ValueError("empty char span")
    idx: list[int] = []
    for i, (a, b) in enumerate(offset_mapping):
        if b <= a:  # skip zero-width specials
            continue
        if a < char_end and b > char_start:  # half-open overlap
            if content_mask is None or bool(content_mask[i]):
                idx.append(i)
    return tuple(idx)


def reconstruct_span_text(
    text: str, offset_mapping: OffsetMapping, token_indices: Sequence[int]
) -> str:
    """Substring of ``text`` covered by ``token_indices`` (first start..last end)."""
    if len(token_indices) == 0:
        return ""
    starts = [offset_mapping[i][0] for i in token_indices]
    ends = [offset_mapping[i][1] for i in token_indices]
    return text[min(starts) : max(ends)]


# --------------------------------------------------------------------------- #
# Fuzzy verification (gate 11.1e)
# --------------------------------------------------------------------------- #


def similarity_ratio(a: str, b: str) -> float:
    """difflib similarity in [0, 1] (1.0 = identical). Pure stdlib."""
    return SequenceMatcher(None, a, b).ratio()


def verify_gold_mapping(
    text: str,
    offset_mapping: OffsetMapping,
    token_indices: Sequence[int],
    gold_text: str,
    *,
    threshold: float = GOLD_MATCH_THRESHOLD,
) -> bool:
    """Gate 11.1e: decode the mapped tokens and fuzzy-match to the gold text."""
    decoded = reconstruct_span_text(text, offset_mapping, token_indices)
    return similarity_ratio(decoded.strip(), gold_text.strip()) >= threshold


# --------------------------------------------------------------------------- #
# Locating a gold sentence in the rendered context
# --------------------------------------------------------------------------- #


def _normalize_ws_with_map(s: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to single spaces, keeping a map back to originals."""
    out: list[str] = []
    idx_map: list[int] = []
    prev_space = False
    for i, ch in enumerate(s):
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                idx_map.append(i)
                prev_space = True
        else:
            out.append(ch)
            idx_map.append(i)
            prev_space = False
    return "".join(out), idx_map


def locate_text(haystack: str, needle: str) -> tuple[int, int] | None:
    """Char span of ``needle`` in ``haystack`` (exact, else whitespace-tolerant).

    HotpotQA supporting facts usually appear verbatim; the normalized fallback
    handles rendered-vs-gold whitespace differences and still returns offsets into
    the *original* ``haystack``.
    """
    i = haystack.find(needle)
    if i >= 0:
        return (i, i + len(needle))
    norm_hay, idx_map = _normalize_ws_with_map(haystack)
    norm_needle = " ".join(needle.split())
    if not norm_needle:
        return None
    j = norm_hay.find(norm_needle)
    if j < 0:
        return None
    start = idx_map[j]
    end = idx_map[j + len(norm_needle) - 1] + 1
    return (start, end)


# --------------------------------------------------------------------------- #
# Real-tokenizer bridge (test marked needs_model)
# --------------------------------------------------------------------------- #


def content_mask_with_tokenizer(
    tokenizer, text: str, *, add_special_tokens: bool = True
) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    """Tokenize ``text`` with a real (fast) tokenizer and build the content mask.

    Returns ``(input_ids, offset_mapping, content_mask)``. The tokenizer is passed
    in, so this module never imports transformers and stays CI-importable; the
    integration test that constructs a tokenizer is marked ``needs_model``.
    """
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        add_special_tokens=add_special_tokens,
    )
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    mask = content_mask_from_offsets(offsets, special_tokens_mask=enc["special_tokens_mask"])
    return np.asarray(enc["input_ids"]), offsets, mask
