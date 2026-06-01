"""Pre-registered correctness metric (addendum §9.2).

The flip test labels an instance "correct" or not, so the *definition* of
correct is load-bearing and must be fixed before looking at results:

* **Synthetic NIAH** — exact by construction; the inserted needle must appear in
  the answer (:func:`niah_correct`).
* **Natural QA** (HotpotQA, LongBench QA subsets) — SQuAD-style normalization
  (lowercase, strip articles + punctuation, collapse whitespace), then
  **token-F1 >= 0.5 is the pre-registered primary** (:data:`PRIMARY_F1_THRESHOLD`),
  with exact match reported as a robustness column.

The core is **deterministic**: no LLM judge (cost, nondeterminism, hidden
dependency). An LLM-judge pass is an optional robustness check that lives
elsewhere, never in this module.

Scores are computed once and stored threshold-free (:class:`AnswerScore`) so the
§9.2 sensitivity sweep over {0.3, 0.5, exact match} re-derives labels without
re-scoring. LongBench subsets that define their own official per-task metric
should plug in a custom judge rather than overriding these defaults.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# The pre-registered primary threshold (§9.2). Frozen before results are seen.
PRIMARY_F1_THRESHOLD = 0.5

# Thresholds reported for the Phase-3 sensitivity analysis (§9.2): F1 cutoffs
# plus exact match. "exact" is the sentinel meaning "require exact match".
EXACT = "exact"
SWEEP_THRESHOLDS: tuple[float | str, ...] = (0.3, 0.5, EXACT)

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = frozenset(string.punctuation)


# --------------------------------------------------------------------------- #
# SQuAD-style normalization + token metrics
# --------------------------------------------------------------------------- #


def normalize_answer(s: str) -> str:
    """SQuAD normalization: lowercase, drop punctuation + articles, fix spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _tokens(s: str) -> list[str]:
    return normalize_answer(s).split()


def token_f1(prediction: str, gold: str) -> float:
    """SQuAD token-level F1 over normalized tokens.

    Follows the official convention for empties: if either side has no tokens,
    F1 is 1.0 only when both are empty, else 0.0.
    """
    pred_toks, gold_toks = _tokens(prediction), _tokens(gold)
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_toks)
    recall = n_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> bool:
    """Normalized exact match (stricter than F1 == 1.0: order matters)."""
    return normalize_answer(prediction) == normalize_answer(gold)


# --------------------------------------------------------------------------- #
# Threshold-free score + correctness decisions
# --------------------------------------------------------------------------- #


@dataclass(slots=True, frozen=True)
class AnswerScore:
    """Deterministic, threshold-free scores of one prediction vs a gold set.

    Stored once; :func:`is_correct` derives the boolean label under any threshold
    so the §9.2 sweep needs no re-scoring. ``f1`` is the max token-F1 over golds;
    ``exact_match`` is True if any gold matches exactly.
    """

    f1: float
    exact_match: bool

    def correct(self, threshold: float | str = PRIMARY_F1_THRESHOLD) -> bool:
        return is_correct(self, threshold)


def is_correct(score: AnswerScore, threshold: float | str = PRIMARY_F1_THRESHOLD) -> bool:
    """Label a score correct. ``threshold`` is an F1 cutoff or ``"exact"``."""
    if isinstance(threshold, str):
        if threshold.lower() in {EXACT, "em", "exact_match"}:
            return score.exact_match
        raise ValueError(f"unknown threshold {threshold!r}; use a float or {EXACT!r}")
    return score.f1 >= threshold


def score_answer(prediction: str, golds: Sequence[str]) -> AnswerScore:
    """Best score of ``prediction`` over one or more acceptable gold answers."""
    golds = list(golds)
    if not golds:
        raise ValueError("need at least one gold answer")
    f1 = max(token_f1(prediction, g) for g in golds)
    em = any(exact_match(prediction, g) for g in golds)
    return AnswerScore(f1=f1, exact_match=em)


def sweep_correctness(
    score: AnswerScore, thresholds: Sequence[float | str] = SWEEP_THRESHOLDS
) -> dict[float | str, bool]:
    """Correctness label under each threshold, for the §9.2 sensitivity report."""
    return {t: is_correct(score, t) for t in thresholds}


# --------------------------------------------------------------------------- #
# Synthetic NIAH
# --------------------------------------------------------------------------- #


def niah_correct(prediction: str, needle: str) -> bool:
    """Synthetic NIAH correctness: the inserted needle must appear in the answer.

    Exact by construction (we insert the needle). We normalize both sides and
    require the needle as a substring, since the model typically wraps the
    magic-number needle in a sentence rather than emitting it bare.
    """
    needle_norm = normalize_answer(needle)
    if not needle_norm:
        raise ValueError("empty needle")
    return needle_norm in normalize_answer(prediction)


# --------------------------------------------------------------------------- #
# Judges (uniform correctness callables for the flip pipeline, §9.3)
# --------------------------------------------------------------------------- #

CorrectnessJudge = Callable[[str], bool]


def squad_judge(
    golds: Sequence[str], *, f1_threshold: float | str = PRIMARY_F1_THRESHOLD
) -> CorrectnessJudge:
    """A ``prediction -> bool`` judge for natural QA at a fixed threshold."""
    golds = list(golds)
    return lambda prediction: is_correct(score_answer(prediction, golds), f1_threshold)


def niah_judge(needle: str) -> CorrectnessJudge:
    """A ``prediction -> bool`` judge for synthetic NIAH."""
    return lambda prediction: niah_correct(prediction, needle)
