"""HotpotQA distractor loader (§10, gate 11.1e).

HotpotQA is the canonical home of the named gold-mapping bug: gold support is
given as ``supporting_facts = [title, sentence_id]`` pairs, *not* as char spans.
Turning those into token indices means (1) resolving each pair to its sentence
text through the ``context`` structure, (2) locating that sentence in the
rendered context, (3) mapping chars -> content-token indices. Steps 2-3 are the
shared :func:`lcv.data.assembly.assemble_text_instance` machinery; step 1 (the
HotpotQA-specific lookup) lives here.

Schema (verified against the live datasets-server, 2026-06-02):
``hotpotqa/hotpot_qa`` / config ``distractor`` / splits ``{train, validation}``::

    id: str
    question: str
    answer: str
    type: str
    level: str
    supporting_facts: {title: list[str], sent_id: list[int]}   # columnar
    context: {title: list[str], sentences: list[list[str]]}     # columnar

Both ``supporting_facts`` and ``context`` come back from HF in **columnar** form
(a dict of parallel lists) when you index a single example -- not a list of
dicts. The parser assumes that form and raises if a contract field is missing.

The build core (:func:`build_hotpot_instance`) is pure ``dict -> Instance`` with
no ``datasets`` dependency, so it is fully CPU-testable on synthetic fixtures.
:func:`load_hotpotqa` is the thin HF wrapper and imports ``datasets`` lazily.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..contracts import Dataset, Instance
from .assembly import assemble_text_instance

HOTPOTQA_HF_ID = "hotpotqa/hotpot_qa"  # was bare `hotpot_qa` (renamed; §10 errata 2026-06-02)
HOTPOTQA_CONFIG = "distractor"

# Paragraphs are joined by blank lines; sentences within a paragraph by a single
# space. Because each gold sentence is concatenated in verbatim, it is always a
# contiguous substring of the rendered context regardless of the join chars, so
# `locate_text` finds it (exact, then whitespace-tolerant).
_PARA_JOIN = "\n\n"
_SENT_JOIN = " "


def parse_hotpot_example(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the contract fields out of one columnar HotpotQA example.

    Raises ``ValueError`` (referencing §10) if a required field is absent or has
    an unexpected shape -- a schema drift should fail loudly here, not silently
    produce an instance with no gold.
    """
    for key in ("question", "answer", "context", "supporting_facts"):
        if key not in raw:
            raise ValueError(f"HotpotQA example missing {key!r} (§10 schema)")

    context = raw["context"]
    sf = raw["supporting_facts"]
    if not (isinstance(context, Mapping) and "title" in context and "sentences" in context):
        raise ValueError(
            "HotpotQA 'context' must be columnar {title: [...], sentences: [[...]]} (§10)"
        )
    if not (isinstance(sf, Mapping) and "title" in sf and "sent_id" in sf):
        raise ValueError(
            "HotpotQA 'supporting_facts' must be columnar {title: [...], sent_id: [...]} (§10)"
        )

    titles = list(context["title"])
    sentences = [list(s) for s in context["sentences"]]
    if len(titles) != len(sentences):
        raise ValueError("HotpotQA context title/sentences length mismatch (§10)")

    return {
        "id": str(raw.get("id", "")),
        "question": str(raw["question"]),
        "answer": str(raw["answer"]),
        "type": str(raw.get("type", "")),
        "level": str(raw.get("level", "")),
        "titles": titles,
        "sentences": sentences,
        "sf_titles": list(sf["title"]),
        "sf_sent_ids": [int(i) for i in sf["sent_id"]],
    }


def render_hotpot_context(titles: Sequence[str], sentences: Sequence[Sequence[str]]) -> str:
    """Concatenate paragraphs into one context string (gold sentences stay verbatim)."""
    paragraphs = [_SENT_JOIN.join(s for s in para) for para in sentences]
    return _PARA_JOIN.join(p for p in paragraphs if p)


def resolve_supporting_sentences(
    titles: Sequence[str],
    sentences: Sequence[Sequence[str]],
    sf_titles: Sequence[str],
    sf_sent_ids: Sequence[int],
) -> tuple[list[str], int]:
    """Map ``(title, sent_id)`` supporting facts to sentence texts.

    Returns ``(gold_sentences, n_out_of_range)``. A fact whose title is unknown or
    whose ``sent_id`` is out of range (HotpotQA has a few, usually from upstream
    truncation) is dropped and counted -- never silently mapped to the wrong text.
    """
    by_title: dict[str, Sequence[str]] = {}
    for title, para in zip(titles, sentences, strict=False):
        by_title.setdefault(title, para)  # first paragraph wins on duplicate titles

    gold: list[str] = []
    out_of_range = 0
    for title, sid in zip(sf_titles, sf_sent_ids, strict=False):
        para = by_title.get(title)
        if para is None or not (0 <= sid < len(para)):
            out_of_range += 1
            continue
        gold.append(para[sid])
    return gold, out_of_range


def build_hotpot_instance(
    raw: Mapping[str, Any],
    *,
    instance_id: str | None = None,
) -> Instance:
    """Build one HotpotQA :class:`Instance` from a columnar example (pure CPU core)."""
    ex = parse_hotpot_example(raw)
    context = render_hotpot_context(ex["titles"], ex["sentences"])
    gold_sentences, out_of_range = resolve_supporting_sentences(
        ex["titles"], ex["sentences"], ex["sf_titles"], ex["sf_sent_ids"]
    )
    if instance_id is not None:
        iid = instance_id
    elif ex["id"]:
        iid = f"hotpot_{ex['id']}"
    else:
        iid = "hotpot"
    return assemble_text_instance(
        instance_id=iid,
        dataset=Dataset.HOTPOTQA,
        context=context,
        question=ex["question"],
        answer_string=ex["answer"],
        gold_texts=gold_sentences,
        metadata={
            "hotpot_type": ex["type"],
            "level": ex["level"],
            "n_supporting": len(ex["sf_titles"]),
            "sf_out_of_range": out_of_range,
        },
    )


def load_hotpotqa(
    *,
    split: str = "validation",
    n: int | None = None,
    seed: int = 0,
    hf_id: str = HOTPOTQA_HF_ID,
    config: str = HOTPOTQA_CONFIG,
) -> list[Instance]:
    """Load HotpotQA distractor and build instances (downloads via ``datasets``).

    ``n`` takes a deterministic random subsample (seeded). The ``datasets`` import
    is lazy so the module stays importable in the torch-free CPU CI; this function
    is only exercised under ``LCV_RUN_DATA_DOWNLOADS=1``.
    """
    try:
        import numpy as np
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only with the data extra
        raise ImportError(
            "load_hotpotqa needs the 'data' extra (datasets). Install lcv[data]."
        ) from exc

    ds = load_dataset(hf_id, config, split=split)
    indices = range(len(ds))
    if n is not None and n < len(ds):
        rng = np.random.default_rng(seed)
        indices = sorted(int(i) for i in rng.permutation(len(ds))[:n])

    out: list[Instance] = []
    for i in indices:
        raw = ds[i]
        out.append(build_hotpot_instance(raw, instance_id=f"hotpot_{split}_{i}"))
    return out
