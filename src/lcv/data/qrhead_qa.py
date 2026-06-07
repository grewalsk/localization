"""QRHead long-context QA detection set (§5.3; LongMemEval, the QRHead detector input).

QRHead detects retrieval heads from *real-task* long-context QA (not synthetic NIAH
copy-paste): it aggregates each head's attention paid **by the question tokens onto
the gold-document tokens** (:mod:`lcv.signals.retrieval_qr`). So the detection set
must carry, per example, a question, a multi-document context, and a **gold-document
annotation** ``D*`` marking the answer-bearing span. The QRHead paper
(arXiv:2506.09944) detects on ~70 **single-session-user** LongMemEval examples, so
that is the default subset here.

LongMemEval (Wu et al. 2024) ships as raw JSON (not a datasets-server config: the
``answer`` column is sometimes a scalar, which breaks Arrow inference), so the
loader reads the JSON file directly. The original ``xiaowu0162/longmemeval`` is
**deprecated** in favor of ``xiaowu0162/longmemeval-cleaned`` (noisy non-evidence
history removed), which is the default ``hf_id``.

Schema (verified against the dataset authors' README + raw JSON, 2026-06-06).
Each example::

    question_id:          str   # ``*_abs`` suffix => abstention question
    question_type:        str   # single-session-user | ... | multi-session
    question:             str
    answer:               str   # occasionally a scalar non-string; coerced
    question_date:        str
    haystack_dates:       list[str]
    haystack_session_ids: list[str]
    haystack_sessions:    list[session]   # session = list[turn]
    answer_session_ids:   list[str]       # gold/evidence session ids

where ``turn = {"role": "user"|"assistant", "content": str}`` and **evidence turns
carry an extra ``"has_answer": true``**. The gold-document annotation ``D*`` is the
content of those ``has_answer`` turns: the finest gold span LongMemEval provides
and exactly what QRHead restricts attention keys to.

The build core (:func:`build_longmemeval_instance`) is pure
``dict -> Instance`` (gold turns become :class:`~lcv.contracts.GoldSpan`s via the
shared §3.3 machinery in :func:`lcv.data.assembly.assemble_text_instance`);
:func:`load_longmemeval` imports ``huggingface_hub`` + ``json`` lazily.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..contracts import Dataset, Instance
from .assembly import assemble_text_instance

LONGMEMEVAL_HF_ID = "xiaowu0162/longmemeval-cleaned"  # original repo is deprecated
# Split -> data file in the cleaned repo (config "default").
LONGMEMEVAL_SPLIT_FILES: dict[str, str] = {
    "longmemeval_oracle": "longmemeval_oracle.json",
    "longmemeval_s": "longmemeval_s_cleaned.json",
    "longmemeval_m": "longmemeval_m_cleaned.json",
}
# The QRHead paper detects on the single-session-user subset (§5.3).
QRHEAD_DETECTION_QUESTION_TYPE = "single-session-user"

_TURN_JOIN = "\n"
_SESSION_JOIN = "\n\n"


def render_sessions(
    haystack_sessions: Sequence[Sequence[Mapping[str, Any]]],
    *,
    turn_join: str = _TURN_JOIN,
    session_join: str = _SESSION_JOIN,
) -> tuple[str, list[str]]:
    """Flatten sessions to a context string + the gold (``has_answer``) turn texts.

    Each turn renders as ``"{role}: {content}"``; turns are joined within a session
    and sessions are joined into one context. Returns ``(context, evidence_texts)``
    where ``evidence_texts`` is the verbatim ``content`` of every turn flagged
    ``has_answer`` -- the gold-document annotation ``D*`` QRHead keys on. The
    evidence content is a substring of the rendered context, so it locates exactly.
    """
    rendered_sessions: list[str] = []
    evidence_texts: list[str] = []
    for session in haystack_sessions:
        lines: list[str] = []
        for turn in session:
            role = str(turn.get("role", ""))
            content = str(turn.get("content", ""))
            lines.append(f"{role}: {content}" if role else content)
            if turn.get("has_answer"):
                evidence_texts.append(content)
        rendered_sessions.append(turn_join.join(lines))
    return session_join.join(rendered_sessions), evidence_texts


def parse_longmemeval_example(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the contract fields out of one LongMemEval example.

    Raises ``ValueError`` (referencing §5.3) if ``question``, ``answer`` or
    ``haystack_sessions`` is missing. ``answer`` is coerced to ``str`` (it is
    occasionally a scalar). Returns the rendered context, gold evidence texts, and
    provenance counts.
    """
    for field_name in ("question", "answer", "haystack_sessions"):
        if field_name not in raw:
            raise ValueError(f"LongMemEval example missing {field_name!r} (§5.3 schema)")
    sessions = raw["haystack_sessions"]
    if not isinstance(sessions, Sequence) or isinstance(sessions, (str, bytes)):
        raise ValueError("haystack_sessions must be a list of sessions (§5.3 schema)")
    context, evidence_texts = render_sessions(sessions)
    n_turns = sum(len(s) for s in sessions)
    return {
        "id": str(raw.get("question_id", "")),
        "question_type": str(raw.get("question_type", "")),
        "question": str(raw["question"]),
        "answer": str(raw["answer"]),
        "context": context,
        "evidence_texts": evidence_texts,
        "n_sessions": len(sessions),
        "n_turns": n_turns,
        "n_evidence_turns": len(evidence_texts),
        "answer_session_ids": list(raw.get("answer_session_ids", []) or []),
    }


def build_longmemeval_instance(
    raw: Mapping[str, Any],
    *,
    instance_id: str | None = None,
) -> Instance:
    """Build one LongMemEval :class:`Instance` (pure CPU core).

    The context is the rendered haystack sessions; gold texts are the
    ``has_answer`` turn contents (the QRHead gold-document annotation ``D*``).
    Raises ``ValueError`` on empty context (no content tokens) -- the same guard as
    the other natural-text loaders. ``metadata`` records the question type, session
    and evidence-turn counts, and located/unlocated gold (from
    :func:`assemble_text_instance`), so a gold-mapping miss surfaces as a number.
    """
    ex = parse_longmemeval_example(raw)
    if not ex["context"]:
        raise ValueError(
            f"LongMemEval example {ex['id'] or '<?>'} has empty session context (no haystack turns)"
        )
    if instance_id is not None:
        iid = instance_id
    elif ex["id"]:
        iid = f"longmemeval_{ex['id']}"
    else:
        iid = "longmemeval"
    return assemble_text_instance(
        instance_id=iid,
        dataset=Dataset.LONGMEMEVAL,
        context=ex["context"],
        question=ex["question"],
        answer_string=ex["answer"],
        gold_texts=ex["evidence_texts"],
        metadata={
            "question_type": ex["question_type"],
            "n_sessions": ex["n_sessions"],
            "n_turns": ex["n_turns"],
            "n_evidence_turns": ex["n_evidence_turns"],
            "n_answer_sessions": len(ex["answer_session_ids"]),
        },
    )


def load_longmemeval(
    *,
    split: str = "longmemeval_oracle",
    question_type: str | None = QRHEAD_DETECTION_QUESTION_TYPE,
    n: int | None = None,
    seed: int = 0,
    skip_empty: bool = True,
    hf_id: str = LONGMEMEVAL_HF_ID,
    filename: str | None = None,
) -> list[Instance]:
    """Load LongMemEval and build QRHead detection instances (downloads the JSON).

    ``split`` selects the data file (``oracle``/``s``/``m`` via
    :data:`LONGMEMEVAL_SPLIT_FILES`, or pass ``filename`` directly).
    ``question_type`` filters to a subset (default ``single-session-user``, the
    QRHead detection subset; pass ``None`` for all types). ``n`` deterministically
    subsamples *after* filtering. With ``skip_empty`` (default), examples whose
    context is empty are skipped rather than raising. ``huggingface_hub`` + ``json``
    are imported lazily; this runs only under ``LCV_RUN_DATA_DOWNLOADS=1``.
    """
    try:
        import json

        import numpy as np
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - exercised only with the data extra
        raise ImportError(
            "load_longmemeval needs the 'data' extra (huggingface_hub). Install lcv[data]."
        ) from exc

    fname = filename or LONGMEMEVAL_SPLIT_FILES.get(split)
    if fname is None:
        raise ValueError(
            f"unknown split {split!r}; pass one of {sorted(LONGMEMEVAL_SPLIT_FILES)} "
            "or an explicit filename"
        )
    path = hf_hub_download(repo_id=hf_id, filename=fname, repo_type="dataset")
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)

    if question_type is not None:
        rows = [r for r in rows if r.get("question_type") == question_type]

    indices: Sequence[int] = range(len(rows))
    if n is not None and n < len(rows):
        rng = np.random.default_rng(seed)
        indices = sorted(int(i) for i in rng.permutation(len(rows))[:n])

    out: list[Instance] = []
    for i in indices:
        raw = rows[i]
        try:
            out.append(build_longmemeval_instance(raw, instance_id=f"longmemeval_{split}_{i}"))
        except ValueError:
            if not skip_empty:
                raise
    return out
