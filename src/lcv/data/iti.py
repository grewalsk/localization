"""ITI truth/false probing set (§5.4; the ITI_HEADS / ITI_DIRECTION substrate input).

ITI (Inference-Time Intervention, Li et al. 2023) does **not** localize over the
context the way the retrieval-head and attribution methods do; it reads internal
truthfulness at the answer position. Its detector
(:func:`lcv.signals.iti_heads.select_iti_heads`) trains a linear probe on each
head's ``hook_z`` to separate *truthful* from *untruthful* statements, so it needs
a labeled ``Sequence[tuple[str, bool]]`` -- a statement and whether it is true.

The probing set must be **disjoint from the §10 QA evaluation sets** (NIAH,
HotpotQA, LongBench, TriviaQA) so head selection never trains on an eval
instance. The ITI paper uses **TruthfulQA**, which is exactly such a held-out
truthfulness set, and forms statements by pairing each question with each
candidate answer in the ``multiple_choice`` config.

Schema (verified against the live datasets-server, 2026-06-06):
``truthfulqa/truthful_qa`` (alias ``truthful_qa``), config ``multiple_choice``,
split ``validation`` (817 rows)::

    question:    str
    mc1_targets: {choices: list[str], labels: list[int]}  # exactly one label == 1
    mc2_targets: {choices: list[str], labels: list[int]}  # multi-hot (>=1 correct)

ITI trains on ``mc2_targets`` (multiple correct + multiple incorrect answers per
question), so that is the default here; ``mc1`` is available for the
single-correct variant. The ``generation`` config (``correct_answers`` /
``incorrect_answers`` as ``list[str]``) is the alternative phrasing and is parsed
by :func:`build_iti_statements_from_generation`.

A "statement" is the QA pair ``"Q: {question} A: {choice}"`` -- the unit ITI
probes, whose last token is the end of the candidate answer. The label is
``bool(target_label)``. The build cores are pure ``dict -> list[(str, bool)]``;
:func:`load_iti_truthfulqa` imports ``datasets`` lazily.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

TRUTHFULQA_HF_ID = "truthfulqa/truthful_qa"  # alias: bare `truthful_qa`
TRUTHFULQA_MC_CONFIG = "multiple_choice"
TRUTHFULQA_GEN_CONFIG = "generation"

# A statement is the (question, candidate answer) pair ITI probes; the probed
# activation is the last token of this string.
ITI_QA_FORMAT = "Q: {question} A: {answer}"

_VALID_TARGETS = ("mc1", "mc2")


def parse_truthfulqa_mc_example(raw: Mapping[str, Any], *, targets: str = "mc2") -> dict[str, Any]:
    """Pull ``question`` + the chosen ``{targets}_targets`` out of one MC example.

    ``targets`` is ``"mc1"`` or ``"mc2"``. Raises ``ValueError`` (referencing §5.4)
    if ``question`` or the ``choices``/``labels`` of the target block are missing or
    length-mismatched -- a schema drift must fail loudly, never silently mislabel.
    """
    if targets not in _VALID_TARGETS:
        raise ValueError(f"targets must be one of {_VALID_TARGETS}, got {targets!r}")
    if "question" not in raw:
        raise ValueError("TruthfulQA example missing 'question' (§5.4 schema)")
    key = f"{targets}_targets"
    block = raw.get(key)
    if not isinstance(block, Mapping):
        raise ValueError(f"TruthfulQA example missing dict {key!r} (§5.4 schema)")
    choices = list(block.get("choices", []) or [])
    labels = list(block.get("labels", []) or [])
    if not choices:
        raise ValueError(f"{key} has no choices (§5.4 schema)")
    if len(choices) != len(labels):
        raise ValueError(f"{key} choices/labels length mismatch: {len(choices)} vs {len(labels)}")
    return {
        "question": str(raw["question"]),
        "choices": [str(c) for c in choices],
        "labels": [int(x) for x in labels],
    }


def build_iti_statements(
    raw: Mapping[str, Any],
    *,
    targets: str = "mc2",
    qa_format: str = ITI_QA_FORMAT,
) -> list[tuple[str, bool]]:
    """Labeled ``(statement, is_true)`` pairs from one MC example (pure CPU core).

    Each candidate answer becomes one statement ``qa_format.format(question, answer)``
    with label ``bool(target_label)``. A single TruthfulQA question yields several
    statements (its correct and incorrect answers), which is exactly the per-head
    probe's training signal.
    """
    ex = parse_truthfulqa_mc_example(raw, targets=targets)
    q = ex["question"]
    return [
        (qa_format.format(question=q, answer=choice), bool(label))
        for choice, label in zip(ex["choices"], ex["labels"], strict=True)
    ]


def build_iti_statements_from_generation(
    raw: Mapping[str, Any],
    *,
    qa_format: str = ITI_QA_FORMAT,
) -> list[tuple[str, bool]]:
    """Labeled pairs from the ``generation`` config (``correct``/``incorrect_answers``).

    The alternative phrasing of the same signal: ``correct_answers`` -> True,
    ``incorrect_answers`` -> False. Raises ``ValueError`` if ``question`` is absent
    or both answer lists are empty (nothing to label).
    """
    if "question" not in raw:
        raise ValueError("TruthfulQA example missing 'question' (§5.4 schema)")
    q = str(raw["question"])
    correct = [str(a) for a in (raw.get("correct_answers", []) or [])]
    incorrect = [str(a) for a in (raw.get("incorrect_answers", []) or [])]
    if not correct and not incorrect:
        raise ValueError("generation example has no correct/incorrect answers (§5.4)")
    out: list[tuple[str, bool]] = []
    for a in correct:
        out.append((qa_format.format(question=q, answer=a), True))
    for a in incorrect:
        out.append((qa_format.format(question=q, answer=a), False))
    return out


def load_iti_truthfulqa(
    *,
    split: str = "validation",
    targets: str = "mc2",
    n: int | None = None,
    seed: int = 0,
    qa_format: str = ITI_QA_FORMAT,
    hf_id: str = TRUTHFULQA_HF_ID,
    config: str = TRUTHFULQA_MC_CONFIG,
) -> list[tuple[str, bool]]:
    """Load TruthfulQA and flatten to ``(statement, is_true)`` pairs (downloads).

    ``n`` deterministically subsamples the number of *questions* (each contributes
    several statements). The ``datasets`` import is lazy; this runs only under
    ``LCV_RUN_DATA_DOWNLOADS=1``. Returns the labeled probing set consumed by
    :func:`lcv.signals.iti_heads.select_iti_heads`.
    """
    try:
        import numpy as np
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only with the data extra
        raise ImportError(
            "load_iti_truthfulqa needs the 'data' extra (datasets). Install lcv[data]."
        ) from exc

    ds = load_dataset(hf_id, config, split=split)
    indices: Sequence[int] = range(len(ds))
    if n is not None and n < len(ds):
        rng = np.random.default_rng(seed)
        indices = sorted(int(i) for i in rng.permutation(len(ds))[:n])

    out: list[tuple[str, bool]] = []
    for i in indices:
        out.extend(build_iti_statements(ds[i], targets=targets, qa_format=qa_format))
    return out
