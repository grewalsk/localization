"""Tests for §10 synthetic NIAH.

The whole point of NIAH is exactness: the gold span must bracket the needle at
the character level AND decode back to the needle from its token indices, with
depth/length controllable. These tests pin that, plus the content-token mask
(context is content, the appended question is not).
"""

import numpy as np
import pytest

from lcv.contracts import Dataset
from lcv.data import niah


def _context_of(inst):
    return inst.rendered_prompt.split("\n\n", 1)[0]


# --- word tokenizer -------------------------------------------------------- #


def test_word_tokenize_offsets_invert_and_ids_stable():
    text = "alpha beta  gamma"
    ids, spans = niah.word_tokenize(text)
    assert [text[a:b] for a, b in spans] == ["alpha", "beta", "gamma"]
    assert ids.shape == (3,)
    # same word -> same id, regardless of position
    ids2, _ = niah.word_tokenize("gamma gamma")
    assert ids2[0] == ids2[1] == ids[2]


# --- insertion exactness --------------------------------------------------- #


def test_insert_needle_char_span_is_exact():
    hay = ["one two.", "three four.", "five six."]
    for depth in (0.0, 0.5, 1.0):
        text, c0, c1 = niah.insert_needle(hay, "NEEDLE_HERE.", depth)
        assert text[c0:c1] == "NEEDLE_HERE."


def test_insert_needle_depth_moves_the_needle():
    hay = ["a b.", "c d.", "e f.", "g h."]
    _, start0, _ = niah.insert_needle(hay, "N.", 0.0)
    _, start1, _ = niah.insert_needle(hay, "N.", 1.0)
    assert start0 == 0
    assert start1 > start0


def test_insert_needle_depth_out_of_range():
    with pytest.raises(ValueError, match="depth"):
        niah.insert_needle(["x."], "N.", 1.5)


# --- instance construction ------------------------------------------------- #


def test_build_instance_is_valid_and_gold_is_char_exact():
    inst = niah.build_niah_instance(
        instance_id="i0", depth=0.5, haystack_sentences=6, magic_number=4821, seed=0
    )
    assert inst.dataset is Dataset.NIAH
    assert inst.answer_string == "4821"
    assert inst.content_token_mask.shape == inst.context_tokens.shape
    assert inst.n_content > 0
    gold = inst.gold_spans[0]
    # char-level exactness
    assert inst.rendered_prompt[gold.char_start : gold.char_end] == "The magic number is 4821."
    assert "4821" in gold.text


def test_gold_decodes_back_to_needle_from_tokens():
    # the exact-by-construction guarantee (analogue of gate 11.1e at similarity 1.0)
    for depth in (0.0, 0.3, 1.0):
        inst = niah.build_niah_instance(
            instance_id=f"i_{depth}", depth=depth, haystack_sentences=5, magic_number=7, seed=3
        )
        gold = inst.gold_spans[0]
        assert niah.gold_text_from_tokens(inst, gold) == "The magic number is 7."


def test_content_mask_marks_context_not_question():
    inst = niah.build_niah_instance(
        instance_id="i0", depth=0.4, haystack_sentences=5, magic_number=99, seed=1
    )
    gold = inst.gold_spans[0]
    mask = inst.content_token_mask
    # every needle token is content
    assert all(mask[i] for i in gold.token_indices)
    # the appended question contributes non-content tokens
    assert not mask.all()
    # content count equals the number of words in the context
    assert int(mask.sum()) == len(_context_of(inst).split())


def test_depth_controls_position_within_context():
    common = dict(haystack_sentences=8, magic_number=55, seed=2)
    early = niah.build_niah_instance(instance_id="e", depth=0.0, **common)
    late = niah.build_niah_instance(instance_id="l", depth=1.0, **common)
    assert early.gold_spans[0].char_start == 0
    # at depth 1.0 the needle sits at the very end of the context
    assert late.gold_spans[0].char_end == len(_context_of(late))
    assert late.gold_spans[0].char_start > early.gold_spans[0].char_start


def test_length_control_changes_size():
    short = niah.build_niah_instance(
        instance_id="s", depth=0.5, haystack_sentences=3, magic_number=1, seed=0
    )
    long = niah.build_niah_instance(
        instance_id="l", depth=0.5, haystack_sentences=30, magic_number=1, seed=0
    )
    assert long.n_tokens > short.n_tokens


def test_build_instance_is_deterministic():
    kw = dict(instance_id="i0", depth=0.5, haystack_sentences=6, magic_number=4821, seed=42)
    a = niah.build_niah_instance(**kw)
    b = niah.build_niah_instance(**kw)
    np.testing.assert_array_equal(a.context_tokens, b.context_tokens)
    np.testing.assert_array_equal(a.content_token_mask, b.content_token_mask)
    assert a.rendered_prompt == b.rendered_prompt


def test_build_instance_rejects_bad_template():
    with pytest.raises(ValueError, match=r"\{n\}"):
        niah.build_niah_instance(
            instance_id="x", depth=0.5, haystack_sentences=3, needle_template="no placeholder"
        )


# --- dataset grid ---------------------------------------------------------- #


def test_build_dataset_grid_size_and_unique_ids():
    ds = niah.build_niah_dataset(depths=(0.0, 0.5, 1.0), haystack_sentences=(4, 8), seed=0)
    assert len(ds) == 6  # 3 depths x 2 lengths
    assert len({i.instance_id for i in ds}) == 6
    for inst in ds:
        gold = inst.gold_spans[0]
        assert niah.gold_text_from_tokens(inst, gold) == gold.text  # exact across the grid
