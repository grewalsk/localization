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
from lcv.signals import retrieval_qr

# Pre-registered bar for 11.1a: fraction of the detected top-20 Wu heads that must
# overlap the released reference set (§11.1a).
WU_PUBLISHED_OVERLAP_MIN = 0.5

# Released Wu retrieval-head reference for gate 11.1a. EMPTY on purpose (verified
# 2026-06-07): Wu et al. (github.com/nightdessert/Retrieval_Head, head_score/)
# released scores for 7 models -- llama-2-7b-80k, llama-2-13b-64k, Mistral-7B-v0.2,
# Mixtral-8x7B-v0.1, Qwen1.5-14B(-Chat), Yi-6B-200K -- but NOT Llama-3.1-8B (our
# PRIMARY_MODEL), so an exact released Llama-3.1-8B Wu set does not exist. The
# QRHead repo (github.com/princeton-pli/QRHead) ships QR-head sets for
# Llama-3.1-8B-Instruct (configs/*_qr_head_LME.yaml, top-16) but recomputes
# Wu/RetHead on the fly and stores no Wu list either. To run 11.1a as a
# reproduce-published-numbers check, detect on a model Wu *did* release: each
# head_score JSON is {"layer-head": [per-example scores]}, a retrieval head is
# mean(score) >= 0.1 (Wu's threshold), and llama-2-7b-80k yields 33 such heads
# (top: 16-19, 11-15, 8-26, ...). Do NOT transcribe a Llama-3.1-8B set from a
# secondary source -- that fabricates the number this gate exists to check.
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
    # The question carries a sentinel word absent from the context, so a content
    # mask that (wrongly) includes the question is detectable by decoding.
    question = "Reply only with the zzqqsentinel marker."
    inst = chat_template.build_instance(
        chat_tokenizer,
        instance_id="g0",
        dataset=Dataset.NIAH,
        question=question,
        context=context,
        answer_string="4821",
        gold_texts=["The magic number is 4821."],
    )
    # build_instance raises unless the decode fuzzy-matches >= 0.9, so reaching
    # here *is* gate 11.1e passing; pin the structural consequences too.
    gold = inst.gold_spans[0]
    assert "4821" in gold.text
    assert all(inst.content_token_mask[i] for i in gold.token_indices)  # gold is content

    # The content mask must cover the context region ONLY (§3.3): the question, the
    # role/header template, and any special tokens are excluded. This holds for any
    # template because content is clipped to the verbatim context char range -- not
    # left to depend on whether the tokenizer emits true specials. Decoding just the
    # content tokens must contain the needle but neither the question sentinel nor
    # the template markers.
    n_content = int(inst.content_token_mask.sum())
    assert 0 < n_content < inst.n_tokens  # question + template were dropped
    decoded = chat_tokenizer.decode(inst.context_tokens[inst.content_token_mask])
    assert "4821" in decoded  # context kept
    assert "zzqqsentinel" not in decoded  # question excluded (M11)
    assert "[INST]" not in decoded  # role/template markers excluded


@pytest.mark.needs_model
def test_11_1e_chat_path_matches_cpu_content_text(chat_tokenizer):
    """M11: the chat path (build_instance) and the CPU path (assemble_text_instance)
    restrict content to the *same* context text -- no template/question leakage."""
    from lcv.data.assembly import assemble_text_instance
    from lcv.model import chat_template

    context = "Alpha beta gamma. The magic number is 4821. Delta epsilon zeta."
    question = "Reply only with the zzqqsentinel marker."

    chat_inst = chat_template.build_instance(
        chat_tokenizer,
        instance_id="x0",
        dataset=Dataset.NIAH,
        question=question,
        context=context,
        answer_string="4821",
    )
    cpu_inst = assemble_text_instance(
        instance_id="x0",
        dataset=Dataset.NIAH,
        context=context,
        question=question,
        answer_string="4821",
    )

    chat_content = chat_tokenizer.decode(chat_inst.context_tokens[chat_inst.content_token_mask])
    _, spans = niah.word_tokenize(cpu_inst.rendered_prompt)
    cpu_idx = tuple(int(i) for i in np.flatnonzero(cpu_inst.content_token_mask))
    cpu_content = tok.reconstruct_span_text(cpu_inst.rendered_prompt, spans, cpu_idx)

    # both keep the needle, neither leaks the question sentinel
    for content in (chat_content, cpu_content):
        assert "4821" in content
        assert "zzqqsentinel" not in content
    # and they agree on the context text (whitespace-normalized)
    sim = tok.similarity_ratio(" ".join(chat_content.split()), " ".join(cpu_content.split()))
    assert sim >= 0.9


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


# --- CPU core: QRscore head detection (§5.3 / arXiv:2506.09944 Eq. 1-3) ----- #


def test_qr_score_aggregates_query_to_gold_attention():
    # head 0 attends to the gold key (3); head 1 attends elsewhere (0).
    A = np.zeros((1, 2, 3, 4))
    A[0, 0, :, 3] = [0.5, 0.5, 0.5]
    A[0, 1, :, 0] = [0.9, 0.9, 0.9]
    s = retrieval_qr.qr_score_from_patterns(A, query_positions=[0, 1, 2], gold_key_positions=[3])
    assert s.shape == (1, 2)
    assert s[0, 0] == pytest.approx(0.5)  # mean query->gold attention
    assert s[0, 1] == pytest.approx(0.0)  # attends off-gold => no QRscore
    assert s[0, 0] > s[0, 1]


def test_qr_score_divides_by_query_length():
    # |q|-normalization (Eq. 1): four query rows each pay 1.0 onto the gold key.
    A = np.zeros((1, 1, 4, 2))
    A[0, 0, :, 1] = 1.0
    s = retrieval_qr.qr_score_from_patterns(A, query_positions=[0, 1, 2, 3], gold_key_positions=[1])
    assert s[0, 0] == pytest.approx(1.0)


def test_qr_score_guards():
    A = np.zeros((1, 1, 2, 2))
    with pytest.raises(ValueError, match="n_layers"):
        retrieval_qr.qr_score_from_patterns(np.zeros((2, 2)), [0], [0])
    with pytest.raises(ValueError, match="query token"):
        retrieval_qr.qr_score_from_patterns(A, [], [0])
    with pytest.raises(ValueError, match="gold"):
        retrieval_qr.qr_score_from_patterns(A, [0], [])
    with pytest.raises(ValueError, match="out of range"):
        retrieval_qr.qr_score_from_patterns(A, [9], [0])


def test_aggregate_qr_scores_means_over_examples():
    # Eq. 3: average the per-example QRscore matrices over the detection set T.
    agg = retrieval_qr.aggregate_qr_scores([np.array([[1.0, 0.0]]), np.array([[3.0, 2.0]])])
    assert agg.tolist() == [[2.0, 1.0]]


def test_aggregate_qr_scores_guard():
    with pytest.raises(ValueError, match=">= 1 detection example"):
        retrieval_qr.aggregate_qr_scores([])


def test_qr_token_importance_projects_query_attention_to_tokens():
    # content keys {1,2,3}; key 2 gets the most query attention, key 1 a little.
    A = np.zeros((1, 1, 2, 4))
    A[0, 0, :, 2] = [0.6, 0.4]
    A[0, 0, :, 1] = [0.1, 0.1]
    imp = retrieval_qr.qr_token_importance_from_patterns(
        A, np.array([False, True, True, True]), query_positions=[0, 1], instance_id="x"
    )
    assert imp.method.value == "qr_attention"
    assert imp.values.shape == (3,)
    assert np.argmax(imp.values) == 1  # content index 1 == key position 2


# --- CPU core: exact head-pair selection (M5; non-rectangular head sets) ---- #


def test_summed_attention_per_key_pairs_exclude_out_of_set_heads():
    """A non-rectangular head set selected by ``head_pairs`` ignores the off-set
    heads a ``layers`` x ``heads`` Cartesian product would wrongly average in (M5)."""
    from lcv.signals.attention_hh import summed_attention_per_key

    # 4 layers x 4 heads, q=2, k=3. In-set heads (0,0)->key0 and (3,3)->key2;
    # off-set heads (0,3) and (3,0) (present only in the Cartesian) -> key1.
    a = np.zeros((4, 4, 2, 3))
    a[0, 0, :, 0] = 1.0
    a[3, 3, :, 2] = 1.0
    a[0, 3, :, 1] = 1.0
    a[3, 0, :, 1] = 1.0

    pairs = summed_attention_per_key(a, query_positions=[0, 1], head_pairs=[(0, 0), (3, 3)])
    assert pairs.tolist() == [1.0, 0.0, 1.0]  # off-set key1 gets nothing

    cart = summed_attention_per_key(a, query_positions=[0, 1], layers=[0, 3], heads=[0, 3])
    assert cart[1] > 0.0  # the Cartesian product leaks off-set heads into key1


def test_summed_attention_per_key_guards():
    from lcv.signals.attention_hh import summed_attention_per_key

    a = np.zeros((2, 2, 1, 2))
    with pytest.raises(ValueError, match="head_pairs OR layers/heads"):
        summed_attention_per_key(a, head_pairs=[(0, 0)], layers=[0])
    with pytest.raises(ValueError, match="out of range"):
        summed_attention_per_key(a, head_pairs=[(0, 9)])


def test_qr_token_importance_routes_exact_head_pairs():
    """qr_token_importance projects an exact QR head set via ``head_pairs`` (M5)."""
    a = np.zeros((2, 2, 1, 3))
    a[0, 0, :, 0] = 1.0  # in-set head -> key0
    a[1, 1, :, 2] = 1.0  # in-set head -> key2
    a[0, 1, :, 1] = 1.0  # off-set head -> key1
    imp = retrieval_qr.qr_token_importance_from_patterns(
        a,
        np.array([True, True, True]),
        query_positions=[0],
        instance_id="x",
        head_pairs=[(0, 0), (1, 1)],
    )
    assert imp.values[1] < imp.values[0]  # off-set head excluded
    assert imp.values[1] < imp.values[2]


def test_accumulated_attention_routes_exact_head_pairs():
    """Wu/accumulated projection of an exact head set excludes off-set heads (M5)."""
    from lcv.signals.attention_hh import accumulated_attention_from_patterns

    a = np.zeros((2, 2, 1, 3))
    a[0, 0, :, 0] = 1.0
    a[1, 1, :, 2] = 1.0
    a[0, 1, :, 1] = 1.0  # off-set head -> key1
    imp = accumulated_attention_from_patterns(
        a, np.array([True, True, True]), instance_id="x", head_pairs=[(0, 0), (1, 1)]
    )
    assert imp.values[1] == 0.0  # off-set head contributes nothing


# --- CPU core: per-layer streaming == whole-tensor (M6; 4K-OOM avoidance) -- #


def test_accumulate_query_key_sums_matches_full_tensor():
    """Streaming the per-layer query-sum then selecting heads reproduces the
    whole-tensor ``summed_attention_per_key`` exactly, for every selection mode (M6)."""
    from lcv.signals.attention_hh import (
        accumulate_query_key_sums,
        select_head_mean,
        summed_attention_per_key,
    )

    rng = np.random.default_rng(0)
    a = rng.random((4, 3, 5, 6))  # [L, H, q, k]
    layer_stream = [a[layer] for layer in range(a.shape[0])]

    for qpos in (None, [0, 2, 4]):
        reduced = accumulate_query_key_sums(layer_stream, query_positions=qpos)
        assert reduced.shape == (4, 3, 6)  # [L, H, k]
        # default all-head mean
        np.testing.assert_allclose(
            select_head_mean(reduced), summed_attention_per_key(a, query_positions=qpos)
        )
        # exact non-rectangular pair set
        pairs = [(0, 0), (3, 2), (1, 1)]
        np.testing.assert_allclose(
            select_head_mean(reduced, head_pairs=pairs),
            summed_attention_per_key(a, query_positions=qpos, head_pairs=pairs),
        )
        # rectangular layers x heads
        np.testing.assert_allclose(
            select_head_mean(reduced, layers=[0, 3], heads=[0, 2]),
            summed_attention_per_key(a, query_positions=qpos, layers=[0, 3], heads=[0, 2]),
        )


def test_qr_score_from_layer_stream_matches_full_tensor():
    """Streaming QRscore per layer reproduces ``qr_score_from_patterns`` exactly (M6)."""
    rng = np.random.default_rng(1)
    a = rng.random((4, 3, 5, 6))  # [L, H, q, k]
    layer_stream = [a[layer] for layer in range(a.shape[0])]
    qpos, gpos = [0, 1, 4], [2, 5]
    streamed = retrieval_qr.qr_score_from_layer_stream(layer_stream, qpos, gpos)
    full = retrieval_qr.qr_score_from_patterns(a, qpos, gpos)
    assert streamed.shape == (4, 3)  # [L, H]
    np.testing.assert_allclose(streamed, full)


def test_accumulate_query_key_sums_guards():
    from lcv.signals.attention_hh import accumulate_query_key_sums

    with pytest.raises(ValueError, match="empty"):
        accumulate_query_key_sums([])
    with pytest.raises(ValueError, match="must be"):
        accumulate_query_key_sums([np.zeros((3, 5))])  # 2-D, not [n_heads, q, k]


def test_qr_score_from_layer_stream_guards():
    a_layer = np.zeros((3, 5, 6))  # [n_heads, q, k]
    with pytest.raises(ValueError, match="query token"):
        retrieval_qr.qr_score_from_layer_stream([a_layer], [], [0])
    with pytest.raises(ValueError, match="gold-document"):
        retrieval_qr.qr_score_from_layer_stream([a_layer], [0], [])
    with pytest.raises(ValueError, match="empty"):
        retrieval_qr.qr_score_from_layer_stream([], [0], [0])
    with pytest.raises(ValueError, match="must be"):
        retrieval_qr.qr_score_from_layer_stream([np.zeros((3, 5))], [0], [0])
    with pytest.raises(ValueError, match="out of range"):
        retrieval_qr.qr_score_from_layer_stream([a_layer], [9], [0])


def test_qr_head_set_selects_top_heads_from_score():
    from lcv.contracts import HeadScore, Method, RetrievalSource

    hs = HeadScore(method=Method.QR_HEADS, scores=np.array([[0.1, 0.9], [0.5, 0.2]]))
    rset = retrieval_qr.qr_head_set(hs, k=2)
    assert rset.source == RetrievalSource.QRHEAD
    assert rset.head_ids[0] == (0, 1)  # highest score
    assert len(rset.head_ids) == 2


def test_wu_retrieval_head_set_selects_top_heads_from_score():
    # Pure-CPU selection over an already-computed detection score (the gate-11.1a
    # input); no model needed, mirrors qr_head_set (M2).
    from lcv.contracts import HeadScore, Method, RetrievalSource
    from lcv.signals import retrieval_wu

    hs = HeadScore(method=Method.WU_HEADS, scores=np.array([[0.1, 0.9], [0.5, 0.2]]))
    rset = retrieval_wu.wu_retrieval_head_set(hs, k=2)
    assert rset.source == RetrievalSource.WU
    assert rset.head_ids[0] == (0, 1)  # highest score
    assert rset.head_ids == ((0, 1), (1, 0))  # descending, stable tie order
    assert rset.scores[(0, 1)] == 0.9


# --- GPU gates ------------------------------------------------------------- #


@pytest.mark.gpu
def test_11_1a_wu_reproduces_published_heads(backbone):
    """Detected top-20 Wu heads overlap the released set above the bar (§11.1a)."""
    from lcv.signals import retrieval_wu

    if not PUBLISHED_WU_HEADS:
        pytest.skip(
            "no Wu-released Llama-3.1-8B head set exists (see PUBLISHED_WU_HEADS note); "
            "enable 11.1a by detecting on a Wu-released model, e.g. llama-2-7b-80k"
        )
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
