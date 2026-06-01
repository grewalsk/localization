"""Phase 1 acceptance gates: signals are correct (addendum §11.1).

The gold-span -> token mapping (11.1e) has a CPU core (synthetic offset mapping)
that runs in CI plus a ``needs_model`` integration through the real chat-template
path. The signal-correctness gates (11.1a/b/c/d) need the model and are marked
``gpu``, with thresholds/bars pinned as constants.
"""

import numpy as np
import pytest

from lcv.contracts import Dataset
from lcv.data import niah
from lcv.data import tokenization as tok

# Pre-registered bar for 11.1a: fraction of the detected top-20 Wu heads that must
# overlap the released Llama-3.1-8B set (§11.1a).
WU_PUBLISHED_OVERLAP_MIN = 0.5
# Fill from the released head_score reference to enable the 11.1a gate (§5.2/§11.1a).
PUBLISHED_WU_HEADS: frozenset[tuple[int, int]] = frozenset()

# Synthetic offset mapping mimicking a fast tokenizer: specials are zero-width.
TEXT = "The magic number is 4821."
OFFSETS = [(0, 0), (0, 3), (4, 9), (10, 16), (17, 19), (20, 25), (0, 0)]


# --- CPU core: 11.1e gold-span mapping ------------------------------------- #


def test_11_1e_content_mask_excludes_zero_width_specials():
    mask = tok.content_mask_from_offsets(OFFSETS)
    assert mask.tolist() == [False, True, True, True, True, True, False]


def test_11_1e_gold_span_decodes_and_fuzzy_matches():
    idx = tok.char_span_to_token_indices(OFFSETS, 20, 25)  # "4821."
    assert tok.verify_gold_mapping(TEXT, OFFSETS, idx, "4821.") is True
    # a near-miss gold still clears the bar; the bar is 0.9
    assert tok.verify_gold_mapping(TEXT, OFFSETS, (2, 3), "magic numbers") is True
    assert tok.GOLD_MATCH_THRESHOLD == 0.9


def test_11_1e_unrelated_gold_fails_the_gate():
    idx = tok.char_span_to_token_indices(OFFSETS, 20, 25)
    assert tok.verify_gold_mapping(TEXT, OFFSETS, idx, "something else entirely") is False


# --- needs_model: 11.1e through the real chat template --------------------- #


@pytest.mark.needs_model
def test_11_1e_build_instance_maps_gold_and_excludes_specials(chat_tokenizer):
    from lcv.model import chat_template

    context = "Alpha beta gamma. The magic number is 4821. Delta epsilon zeta."
    inst = chat_template.build_instance(
        chat_tokenizer,
        instance_id="g0",
        dataset=Dataset.NIAH,
        question="What is the magic number?",
        context=context,
        answer_string="4821",
        gold_texts=["The magic number is 4821."],
    )
    # build_instance raises unless the decode fuzzy-matches >= 0.9, so reaching
    # here *is* gate 11.1e passing; pin the structural consequences too.
    gold = inst.gold_spans[0]
    assert "4821" in gold.text
    assert all(inst.content_token_mask[i] for i in gold.token_indices)  # gold is content
    assert int(inst.content_token_mask.sum()) < inst.n_tokens  # template specials excluded


@pytest.mark.needs_model
def test_11_1e_build_instance_rejects_unlocatable_gold(chat_tokenizer):
    from lcv.model import chat_template

    with pytest.raises(ValueError, match="not found"):
        chat_template.build_instance(
            chat_tokenizer,
            instance_id="g1",
            dataset=Dataset.NIAH,
            question="?",
            context="hello world",
            answer_string="x",
            gold_texts=["this sentence is absent from the context"],
        )


# --- GPU gates ------------------------------------------------------------- #


@pytest.mark.gpu
def test_11_1a_wu_reproduces_published_heads(backbone):
    """Detected top-20 Wu heads overlap the released set above the bar (§11.1a)."""
    from lcv.signals import retrieval_wu

    if not PUBLISHED_WU_HEADS:
        pytest.skip("register the released Llama-3.1-8B Wu head set to enable 11.1a")
    insts = niah.build_niah_dataset(depths=(0.1, 0.5, 0.9), haystack_sentences=(20,), seed=0)
    score = retrieval_wu.detect_wu_retrieval_heads(backbone, insts)
    detected = set(retrieval_wu.wu_retrieval_head_set(score, 20).head_ids)
    overlap = len(detected & PUBLISHED_WU_HEADS) / 20
    assert overlap >= WU_PUBLISHED_OVERLAP_MIN  # else the scoring rule is wrong


@pytest.mark.gpu
def test_11_1b_ablating_wu_heads_collapses_niah(backbone):
    pytest.skip(
        "ablate top-N Wu heads -> NIAH accuracy collapses while random-N is ~harmless "
        "(§11.1b); requires the head-ablation + NIAH-eval harness"
    )


@pytest.mark.gpu
def test_11_1c_qrhead_degrades_at_least_as_much_as_wu(backbone):
    pytest.skip(
        "masking top-32 QRHeads degrades NIAH >= masking top-32 Wu heads (§11.1c); "
        "requires the head-masking + NIAH-eval harness"
    )


@pytest.mark.gpu
def test_11_1d_accumulated_attention_ranks_needle(backbone):
    """Accumulated-attention importance ranks needle tokens above chance (§11.1d)."""
    from lcv.agreement import auroc_vs_gold
    from lcv.signals import attention_hh

    inst = niah.build_niah_instance(
        instance_id="d", depth=0.5, haystack_sentences=20, magic_number=4821, seed=0
    )
    imp = attention_hh.accumulated_attention_importance(backbone, inst)
    content_positions = np.flatnonzero(inst.content_token_mask)
    gold_tokens = set(inst.gold_spans[0].token_indices)
    gold_mask = np.array([p in gold_tokens for p in content_positions], dtype=bool)
    assert auroc_vs_gold(imp.values, gold_mask) > 0.5  # above chance vs needle membership
