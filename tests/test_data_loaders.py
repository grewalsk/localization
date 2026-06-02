"""Tests for the natural-text loaders (§10) and the shared gold-mapping core.

These pin the §3.3 / gate 11.1e contract on real loader output without any model
or network: a synthetic example in the *exact columnar shape* HF returns goes
through the pure ``build_*_instance`` core, and we check that

* each gold text is located and its token indices decode back to it at >= 0.9
  similarity (gate 11.1e), reusing the same word-level tokenizer the core uses;
* content tokens are the **context only** -- the appended question never enters
  the content mask (§3.3);
* gold tokens are all content tokens;
* a schema drift (missing contract field) fails loudly with a ValueError;
* unlocatable gold (yes/no, absent aliases) is counted, not silently dropped.

Real HF downloads are exercised only when ``LCV_RUN_DATA_DOWNLOADS=1`` so CI
stays offline and fast.
"""

from __future__ import annotations

import os

import pytest

from lcv.contracts import Dataset
from lcv.data.assembly import assemble_text_instance
from lcv.data.hotpotqa import (
    build_hotpot_instance,
    parse_hotpot_example,
    resolve_supporting_sentences,
)
from lcv.data.longbench import build_longbench_instance, parse_longbench_example
from lcv.data.niah import word_tokenize
from lcv.data.tokenization import verify_gold_mapping
from lcv.data.triviaqa import build_triviaqa_instance, parse_triviaqa_example

RUN_DOWNLOADS = os.environ.get("LCV_RUN_DATA_DOWNLOADS") == "1"


# --------------------------------------------------------------------------- #
# Shared assertions (mirror gate 11.1e on loader output)
# --------------------------------------------------------------------------- #


def _assert_gold_maps_back(inst) -> None:
    """Every gold span decodes back to its text from its content-token indices."""
    _, spans = word_tokenize(inst.rendered_prompt)
    assert inst.gold_spans, "expected at least one located gold span"
    for g in inst.gold_spans:
        assert g.token_indices, f"gold {g.text!r} mapped to no token"
        assert verify_gold_mapping(inst.rendered_prompt, spans, g.token_indices, g.text)
        for ti in g.token_indices:
            assert inst.content_token_mask[ti], "gold token must be a content token"


def _assert_content_is_context_only(inst) -> None:
    """Content tokens lie in the context region; the question contributes none."""
    _, spans = word_tokenize(inst.rendered_prompt)
    ctx_len = inst.rendered_prompt.rindex("\n\n")  # join before the question
    assert not inst.content_token_mask.all(), "question tokens must be non-content"
    for i, (_a, b) in enumerate(spans):
        if inst.content_token_mask[i]:
            assert b <= ctx_len, "a content token spilled into the question region"


# --------------------------------------------------------------------------- #
# HotpotQA
# --------------------------------------------------------------------------- #


def _hotpot_example() -> dict:
    return {
        "id": "5a8b57f25542995d1e6f1371",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "answer": "yes",
        "type": "comparison",
        "level": "hard",
        "supporting_facts": {
            "title": ["Scott Derrickson", "Ed Wood"],
            "sent_id": [0, 0],
        },
        "context": {
            "title": ["Scott Derrickson", "Ed Wood", "A Distractor Page"],
            "sentences": [
                ["Scott Derrickson is an American filmmaker.", " He was born in Denver."],
                ["Edward Davis Wood Jr. was an American filmmaker and actor."],
                ["This paragraph has nothing to do with the question at all."],
            ],
        },
    }


def test_hotpot_build_maps_supporting_facts_to_gold_tokens():
    inst = build_hotpot_instance(_hotpot_example())
    assert inst.dataset is Dataset.HOTPOTQA
    assert inst.metadata["n_gold"] == 2
    assert inst.metadata["gold_unlocated"] == 0
    assert inst.metadata["n_supporting"] == 2
    golds = {g.text for g in inst.gold_spans}
    assert "Scott Derrickson is an American filmmaker." in golds
    assert "Edward Davis Wood Jr. was an American filmmaker and actor." in golds
    _assert_gold_maps_back(inst)
    _assert_content_is_context_only(inst)


def test_hotpot_resolve_counts_out_of_range_sent_ids():
    titles = ["A", "B"]
    sentences = [["a0.", "a1."], ["b0."]]
    # B has one sentence; sent_id 5 is out of range; unknown title "C" too.
    gold, oor = resolve_supporting_sentences(titles, sentences, ["A", "B", "C"], [1, 5, 0])
    assert gold == ["a1."]
    assert oor == 2


def test_hotpot_out_of_range_surfaces_in_metadata():
    ex = _hotpot_example()
    ex["supporting_facts"] = {"title": ["Ed Wood"], "sent_id": [9]}
    inst = build_hotpot_instance(ex)
    assert inst.metadata["sf_out_of_range"] == 1
    assert inst.metadata["n_gold"] == 0


@pytest.mark.parametrize("missing", ["question", "answer", "context", "supporting_facts"])
def test_hotpot_missing_field_raises(missing):
    ex = _hotpot_example()
    del ex[missing]
    with pytest.raises(ValueError, match="§10|missing"):
        parse_hotpot_example(ex)


def test_hotpot_noncolumnar_context_raises():
    ex = _hotpot_example()
    ex["context"] = [{"title": "X", "sentences": ["y."]}]  # list-of-dicts, not columnar
    with pytest.raises(ValueError, match="columnar"):
        parse_hotpot_example(ex)


# --------------------------------------------------------------------------- #
# LongBench
# --------------------------------------------------------------------------- #


def _longbench_example() -> dict:
    return {
        "input": "What nationality was the filmmaker who directed the cult classic?",
        "context": (
            "Edward Davis Wood Jr. was an American filmmaker and actor. "
            "He is best remembered today for a series of low-budget films."
        ),
        "answers": ["American"],
        "length": 21,
        "dataset": "hotpotqa",
        "language": "en",
        "all_classes": [],
        "_id": "abc123",
    }


def test_longbench_build_locates_answer_span():
    inst = build_longbench_instance(_longbench_example())
    assert inst.dataset is Dataset.LONGBENCH
    assert inst.answer_string == "American"
    assert inst.metadata["n_gold"] == 1
    assert inst.metadata["subset"] == "hotpotqa"
    _assert_gold_maps_back(inst)
    _assert_content_is_context_only(inst)


def test_longbench_unlocatable_answer_is_counted_not_raised():
    ex = _longbench_example()
    ex["answers"] = ["yes"]  # not present in the context
    ex["context"] = "The two filmmakers shared the same country of origin."
    inst = build_longbench_instance(ex)
    assert inst.metadata["n_gold"] == 0
    assert inst.metadata["gold_unlocated"] == 1


@pytest.mark.parametrize("missing", ["input", "context", "answers"])
def test_longbench_missing_field_raises(missing):
    ex = _longbench_example()
    del ex[missing]
    with pytest.raises(ValueError, match="§10|missing"):
        parse_longbench_example(ex)


# --------------------------------------------------------------------------- #
# TriviaQA
# --------------------------------------------------------------------------- #


def _triviaqa_example() -> dict:
    return {
        "question": "Who directed the film Plan 9 from Outer Space?",
        "question_id": "qw_123",
        "question_source": "http://example.org",
        "entity_pages": {
            "doc_source": ["wikipedia"],
            "filename": ["Ed_Wood.txt"],
            "title": ["Ed Wood"],
            "wiki_context": [
                "Edward Davis Wood Jr. was an American filmmaker who directed "
                "Plan 9 from Outer Space, later called the worst film ever made."
            ],
        },
        "search_results": {
            "description": [],
            "filename": [],
            "rank": [],
            "title": [],
            "url": [],
            "search_context": [],
        },
        "answer": {
            "value": "Ed Wood",
            "aliases": ["Edward D. Wood Jr.", "Edward Davis Wood Jr."],
            "normalized_aliases": ["edward davis wood jr", "edward d wood jr"],
            "normalized_value": "ed wood",
            "matched_wiki_entity_name": "",
            "type": "WikipediaEntity",
        },
    }


def test_triviaqa_build_locates_via_alias():
    inst = build_triviaqa_instance(_triviaqa_example())
    assert inst.dataset is Dataset.TRIVIAQA
    assert inst.answer_string == "Ed Wood"
    # Only the alias "Edward Davis Wood Jr." is present verbatim; value + the other
    # alias are absent and counted as unlocated.
    assert inst.metadata["n_gold"] == 1
    assert inst.metadata["gold_unlocated"] == 2
    assert inst.gold_spans[0].text == "Edward Davis Wood Jr."
    _assert_gold_maps_back(inst)
    _assert_content_is_context_only(inst)


def test_triviaqa_falls_back_to_search_context():
    ex = _triviaqa_example()
    ex["entity_pages"]["wiki_context"] = []  # no wiki page
    ex["search_results"] = {
        "description": ["snippet"],
        "filename": ["x"],
        "rank": [0],
        "title": ["t"],
        "url": ["u"],
        "search_context": ["Trivia: Edward Davis Wood Jr. is credited as the director."],
    }
    inst = build_triviaqa_instance(ex)
    assert inst.metadata["n_gold"] == 1
    _assert_gold_maps_back(inst)


def test_triviaqa_empty_evidence_raises():
    ex = _triviaqa_example()
    ex["entity_pages"]["wiki_context"] = []
    with pytest.raises(ValueError, match="empty evidence"):
        build_triviaqa_instance(ex)


def test_triviaqa_missing_answer_raises():
    ex = _triviaqa_example()
    del ex["answer"]
    with pytest.raises(ValueError, match="§10|answer"):
        parse_triviaqa_example(ex)


# --------------------------------------------------------------------------- #
# Shared assembly guard
# --------------------------------------------------------------------------- #


def test_assemble_requires_nonempty_context():
    with pytest.raises(ValueError, match="non-empty"):
        assemble_text_instance(
            instance_id="x",
            dataset=Dataset.HOTPOTQA,
            context="",
            question="q?",
            answer_string="a",
        )


def test_assemble_collapses_duplicate_gold():
    inst = assemble_text_instance(
        instance_id="x",
        dataset=Dataset.HOTPOTQA,
        context="The cat sat on the mat. The cat sat on the mat.",
        question="Where did the cat sit?",
        answer_string="mat",
        gold_texts=["The cat sat on the mat.", "The cat sat on the mat."],
    )
    # Duplicate text collapses to the first occurrence (one span).
    assert inst.metadata["n_gold"] == 1


# --------------------------------------------------------------------------- #
# Real downloads (opt-in: LCV_RUN_DATA_DOWNLOADS=1)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not RUN_DOWNLOADS, reason="set LCV_RUN_DATA_DOWNLOADS=1 to hit HF")
def test_load_hotpotqa_real():
    pytest.importorskip("datasets")
    from lcv.data.hotpotqa import load_hotpotqa

    insts = load_hotpotqa(split="validation", n=3, seed=0)
    assert len(insts) == 3
    assert all(i.dataset is Dataset.HOTPOTQA for i in insts)
    assert any(i.gold_spans for i in insts)


@pytest.mark.skipif(not RUN_DOWNLOADS, reason="set LCV_RUN_DATA_DOWNLOADS=1 to hit HF")
def test_load_longbench_real():
    pytest.importorskip("datasets")
    from lcv.data.longbench import load_longbench

    insts = load_longbench(configs=("hotpotqa",), split="test", n=2, seed=0)
    assert len(insts) == 2
    assert all(i.dataset is Dataset.LONGBENCH for i in insts)


@pytest.mark.skipif(not RUN_DOWNLOADS, reason="set LCV_RUN_DATA_DOWNLOADS=1 to hit HF")
def test_load_triviaqa_real():
    pytest.importorskip("datasets")
    from lcv.data.triviaqa import load_triviaqa

    insts = load_triviaqa(split="validation", n=3, seed=0)
    assert all(i.dataset is Dataset.TRIVIAQA for i in insts)
    assert any(i.gold_spans for i in insts)
