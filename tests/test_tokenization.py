"""Tests for §3.3 content-token mask + gate-11.1e gold-span mapping.

The synthetic offset mapping below mimics what a real fast tokenizer returns:
special tokens carry zero-width ``(0, 0)`` offsets, content tokens carry real
char spans. That lets the whole 11.1e pipeline (mask -> char->token -> decode ->
fuzzy match) be tested on CPU without a tokenizer; one real-tokenizer check is
marked ``needs_model``.
"""

import pytest

from lcv.data import tokenization as tok

# text = "The magic number is 4821."  with BOS/EOS specials around it
TEXT = "The magic number is 4821."
OFFSETS = [(0, 0), (0, 3), (4, 9), (10, 16), (17, 19), (20, 25), (0, 0)]
SPECIAL = [1, 0, 0, 0, 0, 0, 1]  # BOS ... EOS


# --- content mask ---------------------------------------------------------- #


def test_content_mask_excludes_zero_width_specials():
    mask = tok.content_mask_from_offsets(OFFSETS)
    assert mask.tolist() == [False, True, True, True, True, True, False]


def test_content_mask_honors_special_tokens_mask():
    mask = tok.content_mask_from_offsets(OFFSETS, special_tokens_mask=SPECIAL)
    assert mask.tolist() == [False, True, True, True, True, True, False]


def test_content_mask_restricts_to_char_range():
    # keep only the context [0, 16): "The magic number"
    mask = tok.content_mask_from_offsets(OFFSETS, content_char_range=(0, 16))
    assert mask.tolist() == [False, True, True, True, False, False, False]


def test_content_mask_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        tok.content_mask_from_offsets(OFFSETS, special_tokens_mask=[1, 0, 0])


# --- char span -> token indices -------------------------------------------- #


def test_char_span_to_token_indices_overlap():
    # "4821." occupies chars [20, 25) -> token index 5
    assert tok.char_span_to_token_indices(OFFSETS, 20, 25) == (5,)
    # "magic number" -> chars [4, 16) -> tokens 2, 3
    assert tok.char_span_to_token_indices(OFFSETS, 4, 16) == (2, 3)


def test_char_span_to_token_indices_respects_content_mask():
    mask = tok.content_mask_from_offsets(OFFSETS, content_char_range=(0, 16))
    # span covers tokens 3 and 4, but token 4 ("is") is outside the content range
    assert tok.char_span_to_token_indices(OFFSETS, 10, 19, content_mask=mask) == (3,)


def test_char_span_empty_raises():
    with pytest.raises(ValueError, match="empty char span"):
        tok.char_span_to_token_indices(OFFSETS, 5, 5)


# --- reconstruction + fuzzy verify (11.1e) --------------------------------- #


def test_reconstruct_span_text():
    assert tok.reconstruct_span_text(TEXT, OFFSETS, (5,)) == "4821."
    assert tok.reconstruct_span_text(TEXT, OFFSETS, (2, 3)) == "magic number"
    assert tok.reconstruct_span_text(TEXT, OFFSETS, ()) == ""


def test_similarity_ratio_bounds():
    assert tok.similarity_ratio("abc", "abc") == pytest.approx(1.0)
    assert tok.similarity_ratio("abcdef", "xyz") < 0.5


def test_gold_match_threshold_is_point_nine():
    assert tok.GOLD_MATCH_THRESHOLD == 0.9


def test_verify_gold_mapping_pipeline():
    # exact mapping -> similarity 1.0
    idx = tok.char_span_to_token_indices(OFFSETS, 20, 25)
    assert tok.verify_gold_mapping(TEXT, OFFSETS, idx, "4821.") is True
    # a near-miss gold string still clears 0.9
    assert tok.verify_gold_mapping(TEXT, OFFSETS, (2, 3), "magic numbers") is True
    # an unrelated gold string fails
    assert tok.verify_gold_mapping(TEXT, OFFSETS, idx, "something else entirely") is False


# --- locating gold sentences ----------------------------------------------- #


def test_locate_text_exact():
    hay = "alpha beta gamma delta"
    assert tok.locate_text(hay, "beta gamma") == (6, 16)
    assert hay[6:16] == "beta gamma"


def test_locate_text_whitespace_tolerant():
    hay = "alpha   beta\n\ngamma delta"  # collapsed whitespace differs from gold
    span = tok.locate_text(hay, "beta gamma")
    assert span is not None
    start, end = span
    # offsets index back into the ORIGINAL haystack
    assert hay[start:end] == "beta\n\ngamma"


def test_locate_text_absent_returns_none():
    assert tok.locate_text("alpha beta", "zeta") is None


def test_locate_then_map_then_verify_end_to_end():
    # the full 11.1e flow on a content-only string (no specials)
    text = "the cat sat on the mat"
    ids, offsets = [], []
    pos = 0
    for word in text.split():
        start = text.index(word, pos)
        offsets.append((start, start + len(word)))
        ids.append(len(ids))
        pos = start + len(word)
    span = tok.locate_text(text, "sat on")
    assert span is not None
    idx = tok.char_span_to_token_indices(offsets, *span)
    assert tok.verify_gold_mapping(text, offsets, idx, "sat on") is True


# --- real tokenizer (needs model weights / transformers) ------------------- #


@pytest.mark.needs_model
def test_content_mask_with_real_tokenizer_excludes_specials():
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    except Exception as exc:  # network/gating issues -> skip, not fail
        pytest.skip(f"tokenizer unavailable: {exc}")

    text = "The magic number is 4821."
    input_ids, offsets, mask = tok.content_mask_with_tokenizer(tokenizer, text)
    assert input_ids.shape[0] == len(offsets) == mask.shape[0]
    # every masked-out token is a zero-width special; every content token has a span
    for keep, (a, b) in zip(mask.tolist(), offsets, strict=True):
        assert keep == (b > a)
    # the content tokens reconstruct the original text closely
    content_idx = tuple(i for i, k in enumerate(mask.tolist()) if k)
    assert tok.verify_gold_mapping(text, offsets, content_idx, text) is True
