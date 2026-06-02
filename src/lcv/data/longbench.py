"""LongBench QA-subset loader (§10).

LongBench supplies the realistic long-context distribution. Unlike HotpotQA, gold
here is the **answer string(s)**, not sentence ids: we locate each answer in the
context and map it to content tokens through the shared §3.3 machinery. Short
answers that are not literally present (``yes``/``no``, normalized dates) simply
fail to locate and are counted in ``metadata["gold_unlocated"]`` -- expected, not
an error.

Schema (verified against ``LongBench.py`` raw source, 2026-06-02):
``zai-org/LongBench`` / configs ``{hotpotqa, 2wikimqa, musique, ...}`` (+ ``_e``
length-bucketed variants) / split ``test`` / **requires ``trust_remote_code=True``**
(the dataset is script-based: ``LongBench.py`` + ``data.zip``)::

    input: str          # the question / instruction
    context: str        # the long retrieval context (content)
    answers: list[str]  # gold answer strings -> gold spans
    length: int
    dataset: str
    language: str
    all_classes: list[str]
    _id: str

The build core (:func:`build_longbench_instance`) is pure ``dict -> Instance``;
:func:`load_longbench` imports ``datasets`` lazily and passes
``trust_remote_code=True``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..contracts import Dataset, Instance
from .assembly import assemble_text_instance

LONGBENCH_HF_ID = "zai-org/LongBench"  # was THUDM/LongBench (renamed; §10 errata 2026-06-02)
# QA subsets whose gold is an answer span locatable in the context (§10).
LONGBENCH_QA_CONFIGS = ("hotpotqa", "2wikimqa", "musique")


def parse_longbench_example(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the contract fields out of one LongBench example.

    Raises ``ValueError`` (referencing §10) on missing fields so a schema drift
    fails loudly rather than yielding a gold-less instance.
    """
    for key in ("input", "context", "answers"):
        if key not in raw:
            raise ValueError(f"LongBench example missing {key!r} (§10 schema)")
    answers = list(raw["answers"])
    return {
        "id": str(raw.get("_id", "")),
        "question": str(raw["input"]),
        "context": str(raw["context"]),
        "answers": [str(a) for a in answers],
        "length": int(raw.get("length", 0) or 0),
        "subset": str(raw.get("dataset", "")),
        "language": str(raw.get("language", "")),
    }


def build_longbench_instance(
    raw: Mapping[str, Any],
    *,
    instance_id: str | None = None,
) -> Instance:
    """Build one LongBench :class:`Instance` from an example (pure CPU core).

    ``answer_string`` is the first gold answer (LongBench QA scoring is qa_f1 over
    the answer set); all answers are passed as gold texts to locate in context.
    """
    ex = parse_longbench_example(raw)
    answer_string = ex["answers"][0] if ex["answers"] else ""
    if instance_id is not None:
        iid = instance_id
    elif ex["id"]:
        iid = f"longbench_{ex['id']}"
    else:
        iid = "longbench"
    return assemble_text_instance(
        instance_id=iid,
        dataset=Dataset.LONGBENCH,
        context=ex["context"],
        question=ex["question"],
        answer_string=answer_string,
        gold_texts=ex["answers"],
        metadata={
            "subset": ex["subset"],
            "length": ex["length"],
            "language": ex["language"],
            "n_answers": len(ex["answers"]),
        },
    )


def load_longbench(
    *,
    configs: Sequence[str] = LONGBENCH_QA_CONFIGS,
    split: str = "test",
    n: int | None = None,
    seed: int = 0,
    hf_id: str = LONGBENCH_HF_ID,
) -> list[Instance]:
    """Load LongBench QA subsets and build instances (downloads via ``datasets``).

    Iterates the requested ``configs`` (default: the three multi-hop QA subsets),
    taking up to ``n`` instances *per config* (deterministic subsample). Requires
    ``trust_remote_code=True`` because the dataset ships a loading script.
    """
    try:
        import numpy as np
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only with the data extra
        raise ImportError(
            "load_longbench needs the 'data' extra (datasets). Install lcv[data]."
        ) from exc

    out: list[Instance] = []
    for config in configs:
        ds = load_dataset(hf_id, config, split=split, trust_remote_code=True)
        indices = range(len(ds))
        if n is not None and n < len(ds):
            rng = np.random.default_rng(seed)
            indices = sorted(int(i) for i in rng.permutation(len(ds))[:n])
        for i in indices:
            raw = ds[i]
            out.append(build_longbench_instance(raw, instance_id=f"longbench_{config}_{i}"))
    return out
