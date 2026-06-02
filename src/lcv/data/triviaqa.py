"""TriviaQA (rc) loader (§10).

TriviaQA is the **easy single-hop contrast** against the multi-hop sets: a short
factoid answer over a provided evidence context. Gold is the answer string plus
its aliases, located in the evidence context and mapped to content tokens via the
shared §3.3 machinery. Aliases matter -- the literal ``answer.value`` is often
absent while an alias appears verbatim, so passing all of them maximizes the
located-gold rate (duplicates collapse in :func:`assemble_text_instance`).

Schema (verified against the live datasets-server, 2026-06-02):
``mandarjoshi/trivia_qa`` / config ``rc`` / splits ``{train, validation, test}``::

    question: str
    question_id: str
    question_source: str
    entity_pages:   {doc_source, filename, title, wiki_context}   # columnar (Sequence[dict])
    search_results: {description, filename, rank, title, url, search_context}  # columnar
    answer: {value, aliases, normalized_aliases, normalized_value,
             matched_wiki_entity_name, type}                       # plain dict

``entity_pages`` / ``search_results`` come back **columnar** (dict of parallel
lists): ``raw["entity_pages"]["wiki_context"]`` is a ``list[str]``. ``answer`` is
a plain dict. The evidence context is the concatenation of the Wikipedia
contexts, falling back to the search-result contexts when no wiki page is present.

The build core (:func:`build_triviaqa_instance`) is pure ``dict -> Instance``;
:func:`load_triviaqa` imports ``datasets`` lazily.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts import Dataset, Instance
from .assembly import assemble_text_instance

TRIVIAQA_HF_ID = "mandarjoshi/trivia_qa"  # was bare `trivia_qa` (renamed; §10 errata 2026-06-02)
TRIVIAQA_CONFIG = "rc"

_CTX_JOIN = "\n\n"


def _columnar_list(field: Any, key: str) -> list[Any]:
    """Read one column out of a columnar (dict-of-lists) HF feature, else ``[]``."""
    if isinstance(field, Mapping):
        return list(field.get(key, []) or [])
    return []


def parse_triviaqa_example(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the contract fields out of one TriviaQA (rc) example.

    Raises ``ValueError`` (referencing §10) if ``question`` or ``answer`` is
    missing. Evidence may legitimately be empty (then the instance has no content
    gold), but the core fields must be present.
    """
    if "question" not in raw:
        raise ValueError("TriviaQA example missing 'question' (§10 schema)")
    if "answer" not in raw or not isinstance(raw["answer"], Mapping):
        raise ValueError("TriviaQA example missing dict 'answer' (§10 schema)")

    answer = raw["answer"]
    wiki_contexts = _columnar_list(raw.get("entity_pages"), "wiki_context")
    search_contexts = _columnar_list(raw.get("search_results"), "search_context")
    evidence = [c for c in wiki_contexts if c] or [c for c in search_contexts if c]

    value = str(answer.get("value", "") or "")
    aliases = [str(a) for a in (answer.get("aliases", []) or [])]
    gold_texts = [t for t in [value, *aliases] if t]

    return {
        "id": str(raw.get("question_id", "")),
        "question": str(raw["question"]),
        "answer_value": value,
        "gold_texts": gold_texts,
        "evidence": evidence,
        "n_wiki": len(wiki_contexts),
        "n_search": len(search_contexts),
    }


def build_triviaqa_instance(
    raw: Mapping[str, Any],
    *,
    instance_id: str | None = None,
) -> Instance:
    """Build one TriviaQA :class:`Instance` from an example (pure CPU core).

    ``context`` is the joined evidence; gold texts are the answer value plus
    aliases (located in that evidence). Raises ``ValueError`` if the evidence is
    empty -- an instance with no context has no content tokens (the same guard as
    :func:`assemble_text_instance`).
    """
    ex = parse_triviaqa_example(raw)
    context = _CTX_JOIN.join(ex["evidence"])
    if not context:
        raise ValueError(
            f"TriviaQA example {ex['id'] or '<?>'} has empty evidence context "
            "(no entity_pages.wiki_context or search_results.search_context)"
        )
    if instance_id is not None:
        iid = instance_id
    elif ex["id"]:
        iid = f"triviaqa_{ex['id']}"
    else:
        iid = "triviaqa"
    return assemble_text_instance(
        instance_id=iid,
        dataset=Dataset.TRIVIAQA,
        context=context,
        question=ex["question"],
        answer_string=ex["answer_value"],
        gold_texts=ex["gold_texts"],
        metadata={
            "n_aliases": max(len(ex["gold_texts"]) - 1, 0),
            "n_wiki": ex["n_wiki"],
            "n_search": ex["n_search"],
        },
    )


def load_triviaqa(
    *,
    split: str = "validation",
    n: int | None = None,
    seed: int = 0,
    skip_empty: bool = True,
    hf_id: str = TRIVIAQA_HF_ID,
    config: str = TRIVIAQA_CONFIG,
) -> list[Instance]:
    """Load TriviaQA (rc) and build instances (downloads via ``datasets``).

    ``n`` takes a deterministic random subsample. With ``skip_empty`` (default),
    examples whose evidence context is empty are skipped rather than raising, so a
    requested ``n`` yields ``<= n`` usable instances. The ``datasets`` import is
    lazy; this runs only under ``LCV_RUN_DATA_DOWNLOADS=1``.
    """
    try:
        import numpy as np
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only with the data extra
        raise ImportError(
            "load_triviaqa needs the 'data' extra (datasets). Install lcv[data]."
        ) from exc

    ds = load_dataset(hf_id, config, split=split)
    indices = range(len(ds))
    if n is not None and n < len(ds):
        rng = np.random.default_rng(seed)
        indices = sorted(int(i) for i in rng.permutation(len(ds))[:n])

    out: list[Instance] = []
    for i in indices:
        raw = ds[i]
        try:
            out.append(build_triviaqa_instance(raw, instance_id=f"triviaqa_{split}_{i}"))
        except ValueError:
            if not skip_empty:
                raise
    return out
