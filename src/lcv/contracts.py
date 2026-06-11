"""Core data contracts.

The objects every downstream module exchanges, with two addendum invariants
baked into the types so they cannot be violated silently:

* **Substrates are never merged** (§6). ``substrate_of`` / ``assert_same_substrate``
  make a cross-substrate comparison raise rather than produce a meaningless
  number.
* **The leakage rule** (§9.3, gate 11.3c). ``DisagreementScore`` refuses to exist
  unless it was computed from the full-cache pass.

All types are torch-free (numpy only) so they import and validate in CI.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class Substrate(str, Enum):
    """Commensurable comparison spaces. Agreement is defined only within one."""

    TOKEN_ATTRIBUTION = "token-attribution"  # importance over content tokens
    COMPONENT = "component"  # score over query heads
    ANSWER_POSITION = "answer-position"  # RQ4, reported separately


class Method(str, Enum):
    """Localization signals, namespaced by the substrate they live in.

    Wu and QRHead appear in two substrates (attention summed onto tokens vs the
    native head set); they are distinct members so each maps to one substrate.
    """

    # token-attribution substrate (§5.1, §5.2, §5.3, §4.4)
    ACCUMULATED_ATTENTION = "accumulated_attention"
    WU_ATTENTION = "wu_attention"
    QR_ATTENTION = "qr_attention"
    TRANSCODER_ATTR = "transcoder_attr"  # conditional on gate 11.2c (§4.4)
    # component substrate (§5.2, §5.3, §5.4)
    WU_HEADS = "wu_heads"
    QR_HEADS = "qr_heads"
    ITI_HEADS = "iti_heads"
    TRANSCODER_LOADINGS = "transcoder_loadings"
    # answer-position substrate (§5.4, §5.5)
    ITI_DIRECTION = "iti_direction"
    ORGAD_TOKENS = "orgad_tokens"


class CompressionMethod(str, Enum):
    """Token-level KV eviction methods swept in the flip test (§9.1)."""

    H2O = "h2o"
    SNAPKV = "snapkv"


class CorruptionType(str, Enum):
    """Robustness corruption for the oracle-ranking stability check (§8.2). Never Gaussian.

    The primary oracle is token masking (Definition 2, §8.1) and the ``E_hat``
    estimator linearizes that masking directly -- no corrupted reference (§8.3, M9).
    These corruptions are secondary: gate 11.2b re-derives the oracle ranking under
    a corruption and reports its overlap with the masking ranking.
    """

    TOKEN_SWAP = "token_swap"  # primary corruption (robustness gate 11.2b)
    DOCUMENT_SWAP = "document_swap"  # coarser, for multi-hop


class Granularity(str, Enum):
    """Oracle effect granularity (§8.4)."""

    TOKEN = "token"
    SENTENCE = "sentence"  # span/sentence-level where gold is sentence-level


class Dataset(str, Enum):
    NIAH = "niah"
    HOTPOTQA = "hotpotqa"
    LONGBENCH = "longbench"
    TRIVIAQA = "triviaqa"
    LONGMEMEVAL = "longmemeval"  # QRHead detection set (§5.3); not a §10 eval set


class RetrievalSource(str, Enum):
    """The two contested retrieval-head detectors (§5.3); their disagreement is
    a headline number."""

    WU = "wu"
    QRHEAD = "qrhead"


# Membership map (§6). transcoder_attr is conditional (kept only if gate 11.2c
# passes); it is listed so projections know the full candidate set.
SUBSTRATE_MEMBERS: dict[Substrate, tuple[Method, ...]] = {
    Substrate.TOKEN_ATTRIBUTION: (
        Method.ACCUMULATED_ATTENTION,
        Method.WU_ATTENTION,
        Method.QR_ATTENTION,
        Method.TRANSCODER_ATTR,
    ),
    Substrate.COMPONENT: (
        Method.WU_HEADS,
        Method.QR_HEADS,
        Method.ITI_HEADS,
        Method.TRANSCODER_LOADINGS,
    ),
    Substrate.ANSWER_POSITION: (
        Method.ITI_DIRECTION,
        Method.ORGAD_TOKENS,
    ),
}

CONDITIONAL_MEMBERS: frozenset[Method] = frozenset({Method.TRANSCODER_ATTR})

_METHOD_TO_SUBSTRATE: dict[Method, Substrate] = {
    m: s for s, members in SUBSTRATE_MEMBERS.items() for m in members
}


def substrate_of(method: Method) -> Substrate:
    """Return the substrate a method belongs to (§6)."""
    try:
        return _METHOD_TO_SUBSTRATE[method]
    except KeyError:
        raise ValueError(f"{method!r} is not a registered substrate member") from None


def assert_same_substrate(methods: Iterable[Method]) -> Substrate:
    """Return the shared substrate, or raise if methods span more than one.

    This is the structural guard behind "substrates are never merged" (§6).
    """
    subs = {substrate_of(m) for m in methods}
    if len(subs) == 0:
        raise ValueError("no methods given")
    if len(subs) != 1:
        raise ValueError(
            f"methods span multiple substrates {sorted(s.value for s in subs)}; "
            "substrates are never merged (§6)"
        )
    return subs.pop()


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class GoldSpan:
    """A gold answer/support span located in the rendered context.

    Char offsets are authoritative; ``token_indices`` (positions in the full
    rendered sequence) are filled by the gold-span→token mapping and verified by
    gate 11.1e.
    """

    char_start: int
    char_end: int  # exclusive
    text: str = ""
    token_indices: tuple[int, ...] = ()


@dataclass(slots=True)
class Instance:
    """One evaluation instance. Activations come from the full rendered prompt;
    token-level rankings are computed over content tokens only (§3.3)."""

    instance_id: str
    dataset: Dataset
    rendered_prompt: str
    context_tokens: np.ndarray  # int token ids of the full rendered sequence
    content_token_mask: np.ndarray  # bool; True = content (non-template) token
    answer_string: str
    gold_spans: tuple[GoldSpan, ...] = ()
    question: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.context_tokens = np.asarray(self.context_tokens)
        self.content_token_mask = np.asarray(self.content_token_mask, dtype=bool)
        if self.content_token_mask.shape != self.context_tokens.shape:
            raise ValueError(
                "content_token_mask shape "
                f"{self.content_token_mask.shape} != context_tokens shape "
                f"{self.context_tokens.shape}"
            )

    @property
    def n_content(self) -> int:
        return int(self.content_token_mask.sum())

    @property
    def n_tokens(self) -> int:
        return int(self.context_tokens.shape[0])


@dataclass(slots=True)
class ImportanceVector:
    """A token-substrate signal output: normalized importance over the content
    tokens of one instance (§5, §6)."""

    instance_id: str
    method: Method
    values: np.ndarray  # 1-D, length == instance.n_content
    normalization: str = "minmax"  # "minmax" | "zscore"; uniform per run

    def __post_init__(self) -> None:
        if substrate_of(self.method) is not Substrate.TOKEN_ATTRIBUTION:
            raise ValueError(
                f"{self.method!r} is not a token-attribution method; "
                "ImportanceVector lives in the token-attribution substrate (§6)"
            )
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim != 1:
            raise ValueError("importance values must be 1-D over content tokens")

    @property
    def substrate(self) -> Substrate:
        return Substrate.TOKEN_ATTRIBUTION


@dataclass(slots=True)
class HeadScore:
    """A component-substrate signal output: a score over query heads.

    ``scores`` is 2-D ``[n_layers, n_heads]`` (Llama-3.1-8B: 32x32 query heads,
    §3.2). ``instance_id is None`` denotes a global head set.
    """

    method: Method
    scores: np.ndarray  # [n_layers, n_heads]
    instance_id: str | None = None

    def __post_init__(self) -> None:
        if substrate_of(self.method) is not Substrate.COMPONENT:
            raise ValueError(
                f"{self.method!r} is not a component method; HeadScore lives in "
                "the component substrate (§6)"
            )
        self.scores = np.asarray(self.scores, dtype=float)
        if self.scores.ndim != 2:
            raise ValueError("head scores must be 2-D [n_layers, n_heads]")

    @property
    def substrate(self) -> Substrate:
        return Substrate.COMPONENT

    def top_k_heads(self, k: int) -> list[tuple[int, int]]:
        """Return the top-k ``(layer, head)`` pairs by score, descending.

        Ties break by ascending flat index (lower layer, then lower head) via a stable
        sort of the negated scores, so the head set fed into the Wu-vs-QR Jaccard is
        deterministic; ``argsort(scores)[::-1]`` would reverse ties unstably (M1).
        """
        n_heads = self.scores.shape[1]
        flat_order = np.argsort(-self.scores, axis=None, kind="stable")[:k]
        return [(int(i // n_heads), int(i % n_heads)) for i in flat_order]


@dataclass(slots=True)
class RetrievalHeadSet:
    """A selected set of retrieval heads from one detector (§5.3). The Wu-vs-QR
    Jaccard over ``head_ids`` is a headline disagreement number."""

    source: RetrievalSource
    head_ids: tuple[tuple[int, int], ...]  # (layer, head) pairs
    scores: dict[tuple[int, int], float] = field(default_factory=dict)

    def as_set(self) -> frozenset[tuple[int, int]]:
        return frozenset(self.head_ids)


@dataclass(slots=True)
class OracleEffect:
    """Per-token (or per-span) causal effect for one instance (§8).

    ``E`` is the true **masking** effect (Definition 2, §8.1: token ``t``'s K/V
    removed from attention at every layer); ``E_hat`` is the attribution-patching
    estimate validated against it by gate 11.2a.
    """

    instance_id: str
    E: np.ndarray  # 1-D over content tokens (or spans)
    granularity: Granularity = Granularity.TOKEN
    E_hat: np.ndarray | None = None
    corruption: CorruptionType | None = None

    def __post_init__(self) -> None:
        self.E = np.asarray(self.E, dtype=float)
        if self.E.ndim != 1:
            raise ValueError("E must be 1-D over content tokens/spans")
        if self.E_hat is not None:
            self.E_hat = np.asarray(self.E_hat, dtype=float)
            if self.E_hat.shape != self.E.shape:
                raise ValueError("E_hat must match E shape")


@dataclass(slots=True)
class DisagreementScore:
    """``D(x) = 1 - mean pairwise Spearman`` within a substrate (§7).

    Leakage rule (§9.3, gate 11.3c): D(x) is the feature carried into the flip
    test and **must** be computed from the full-cache pass only. The type
    refuses to exist otherwise, so a leak cannot be introduced silently.
    """

    instance_id: str
    substrate: Substrate
    value: float
    n_methods: int
    from_full_cache: bool = True

    def __post_init__(self) -> None:
        if not self.from_full_cache:
            raise ValueError(
                "D(x) must be computed from the full-cache pass only "
                "(leakage rule, §9.3 / gate 11.3c)"
            )
        if self.n_methods < 2:
            raise ValueError("D(x) needs >= 2 methods to form a pair")


@dataclass(slots=True)
class Confounds:
    """The §9.3 difficulty proxies D(x) must beat: length, gold-span depth, and
    answer-token entropy (a confidence proxy)."""

    length: float
    gold_depth: float
    answer_entropy: float

    def as_array(self) -> np.ndarray:
        return np.array([self.length, self.gold_depth, self.answer_entropy], dtype=float)


@dataclass(slots=True)
class FlipRecord:
    """A Phase-3 label: did compression flip a correct answer to wrong (§9.3)?

    Flips are defined only on instances correct under the full cache (step 1).
    """

    instance_id: str
    method: CompressionMethod
    budget: float  # KV budget (token count or fraction)
    correct_full: bool
    correct_compressed: bool

    @property
    def flipped(self) -> bool:
        return self.correct_full and not self.correct_compressed
