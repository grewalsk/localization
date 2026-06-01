"""Tests for the §9.2 pre-registered correctness metric.

These pin the *definition* of "correct" used to label flips: SQuAD-style F1 with
a 0.5 primary threshold, exact match as robustness, NIAH-by-construction, and a
deterministic core with no LLM judge.
"""

import pytest

from lcv.compression import correctness as cr

# --- Normalization --------------------------------------------------------- #


def test_normalize_strips_articles_punct_case_space():
    assert cr.normalize_answer("  The Quick, Brown FOX! ") == "quick brown fox"
    assert cr.normalize_answer("An Apple.") == "apple"
    assert cr.normalize_answer("THE") == ""  # a bare article normalizes away


# --- Token F1 -------------------------------------------------------------- #


def test_token_f1_identical_and_disjoint():
    assert cr.token_f1("quick brown fox", "quick brown fox") == pytest.approx(1.0)
    assert cr.token_f1("cat", "dog") == pytest.approx(0.0)


def test_token_f1_partial_overlap_value():
    # pred tokens {quick, brown, fox}, gold {quick, brown}: P=2/3, R=1 -> F1=0.8
    assert cr.token_f1("the quick brown fox", "quick brown") == pytest.approx(0.8)


def test_token_f1_empty_convention():
    assert cr.token_f1("", "") == pytest.approx(1.0)  # both empty -> 1.0
    assert cr.token_f1("", "cat") == pytest.approx(0.0)  # one empty -> 0.0
    assert cr.token_f1("the", "cat") == pytest.approx(0.0)  # pred normalizes to empty


def test_exact_match_after_normalization():
    assert cr.exact_match("The answer", "answer") is True
    assert cr.exact_match("Paris.", "paris") is True
    assert cr.exact_match("cat", "dog") is False
    # F1 == 1.0 but order differs -> not an exact match (EM is stricter)
    assert cr.token_f1("brown fox", "fox brown") == pytest.approx(1.0)
    assert cr.exact_match("brown fox", "fox brown") is False


# --- Scoring over gold sets + thresholds ----------------------------------- #


def test_score_answer_takes_best_over_golds():
    s = cr.score_answer("paris", ["london", "paris", "berlin"])
    assert s.f1 == pytest.approx(1.0)
    assert s.exact_match is True


def test_score_answer_requires_a_gold():
    with pytest.raises(ValueError, match="at least one gold"):
        cr.score_answer("x", [])


def test_primary_threshold_is_half():
    assert cr.PRIMARY_F1_THRESHOLD == 0.5


def test_is_correct_f1_thresholds_and_exact():
    s = cr.AnswerScore(f1=0.6, exact_match=False)
    assert cr.is_correct(s) is True  # default 0.5
    assert cr.is_correct(s, 0.3) is True
    assert cr.is_correct(s, 0.7) is False
    assert cr.is_correct(s, cr.EXACT) is False  # no exact match
    assert s.correct() is True  # method delegates to is_correct

    s2 = cr.AnswerScore(f1=0.4, exact_match=True)
    assert cr.is_correct(s2, 0.5) is False
    assert cr.is_correct(s2, 0.3) is True
    assert cr.is_correct(s2, "em") is True  # alias for exact


def test_is_correct_rejects_unknown_string_threshold():
    s = cr.AnswerScore(f1=1.0, exact_match=True)
    with pytest.raises(ValueError, match="unknown threshold"):
        cr.is_correct(s, "fuzzy")


def test_sweep_covers_preregistered_grid():
    s = cr.AnswerScore(f1=0.45, exact_match=False)
    sweep = cr.sweep_correctness(s)
    assert set(sweep) == {0.3, 0.5, cr.EXACT}
    assert sweep[0.3] is True
    assert sweep[0.5] is False
    assert sweep[cr.EXACT] is False


# --- NIAH ------------------------------------------------------------------ #


def test_niah_correct_substring_of_answer():
    assert cr.niah_correct("The magic number is 4821.", "4821") is True
    assert cr.niah_correct("I don't know", "4821") is False


def test_niah_correct_empty_needle_raises():
    with pytest.raises(ValueError, match="empty needle"):
        cr.niah_correct("anything", "")


# --- Judges ---------------------------------------------------------------- #


def test_squad_judge_is_a_thresholded_callable():
    judge = cr.squad_judge(["quick brown fox"], f1_threshold=0.5)
    assert judge("quick brown") is True  # F1 0.8 >= 0.5
    assert judge("totally unrelated") is False
    strict = cr.squad_judge(["quick brown fox"], f1_threshold=0.9)
    assert strict("quick brown") is False  # F1 0.8 < 0.9


def test_niah_judge_callable():
    judge = cr.niah_judge("4821")
    assert judge("answer: 4821") is True
    assert judge("answer: 0000") is False
