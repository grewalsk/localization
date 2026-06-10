"""Robustness corruption for the oracle-ranking stability check (addendum §8.2).

**The primary oracle is token masking** (Definition 2, §8.1, :mod:`lcv.oracle.
masking`), and the attribution-patching estimator linearizes that masking
directly (§8.3, :mod:`lcv.oracle.attr_patching`) -- it needs **no** corrupted
reference (M9). Token-swap's one *secondary* role is the oracle-ranking
robustness check (gate 11.2b: re-derive the ranking under a swap and report its
overlap with the masking ranking). The choice of corruption is load-bearing:
Zhang & Nanda show Gaussian-noise corruption is fragile and
hyperparameter-sensitive, so it is **forbidden** here.

* **Token-swap (primary corruption).** Replace the gold span / needle with a
  different plausible span so the answer is no longer supported. Length-preserving,
  so the corrupted run stays token-aligned with the clean run (so its content-token
  ranking lines up with the clean ranking for the gate 11.2b overlap). Note the
  swap perturbs neighboring positions' residual streams through their attention to
  the swapped token, which masking does not -- one reason masking, not swap, is the
  primary oracle (§8.2).
* **Document-swap (coarser).** Replace the supporting document with a distractor;
  used for the multi-hop sets.

Everything is deterministic given a seed, and we can emit *two* distinct variants
so gate 11.2b can show the oracle's top-token set is stable across corruptions
(i.e., the oracle is not as arbitrary as the methods it judges).

Pure numpy — operates in token-id space. The char->token mapping that locates the
gold span lives in the tokenization module; the true masking E(t) and E_hat live
in the GPU oracle modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..agreement import top_k_jaccard
from ..contracts import CorruptionType

# The pre-registered default corruption (§8.2); secondary to the masking oracle.
PRIMARY = CorruptionType.TOKEN_SWAP


def assert_not_gaussian(corruption: CorruptionType | str) -> CorruptionType:
    """Return the corruption as an enum, refusing Gaussian/noise corruption (§8.2)."""
    if isinstance(corruption, CorruptionType):
        return corruption
    name = str(corruption).strip().lower()
    if "gauss" in name or "noise" in name:
        raise ValueError(
            "Gaussian-noise corruption is forbidden (§8.2); use token_swap or document_swap"
        )
    try:
        return CorruptionType(name)
    except ValueError:
        raise ValueError(
            f"unknown corruption {corruption!r}; use token_swap or document_swap"
        ) from None


@dataclass(slots=True, frozen=True)
class Corruption:
    """A corrupted token sequence plus provenance for the robustness check (gate 11.2b)."""

    tokens: np.ndarray  # corrupted token ids (1-D)
    corruption_type: CorruptionType
    changed_span: tuple[int, int] | None  # token [start, end) that was replaced
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", np.asarray(self.tokens))


def fit_to_length(candidate: np.ndarray, length: int) -> np.ndarray:
    """Tile/truncate a candidate token span to exactly ``length`` tokens.

    Lets an arbitrary distractor span fill a target gold span while keeping the
    sequence length fixed (token-swap must be length-preserving).
    """
    c = np.asarray(candidate)
    if c.ndim != 1:
        raise ValueError("candidate must be 1-D")
    if c.size == 0:
        raise ValueError("empty candidate span")
    if length <= 0:
        raise ValueError("length must be positive")
    if c.size == length:
        return c.copy()
    if c.size > length:
        return c[:length].copy()
    reps = int(np.ceil(length / c.size))
    return np.tile(c, reps)[:length].copy()


def token_swap(
    tokens: np.ndarray,
    span: tuple[int, int],
    replacement: np.ndarray,
    *,
    seed: int = 0,
) -> Corruption:
    """Length-preserving replacement of ``tokens[start:end]`` (§8.2 primary).

    ``replacement`` is fit to the span length, so the result has the same total
    length as ``tokens`` and stays position-aligned with the clean run.
    """
    tokens = np.asarray(tokens)
    if tokens.ndim != 1:
        raise ValueError("tokens must be 1-D")
    start, end = span
    n = tokens.shape[0]
    if not (0 <= start < end <= n):
        raise ValueError(f"bad span {span} for sequence length {n}")
    rep = fit_to_length(replacement, end - start)
    out = tokens.copy()
    out[start:end] = rep
    return Corruption(
        tokens=out,
        corruption_type=CorruptionType.TOKEN_SWAP,
        changed_span=(start, end),
        seed=seed,
    )


@dataclass(slots=True)
class TokenSwapCorruptor:
    """Seeded token-swap from a pool of plausible replacement spans (§8.2).

    The pool is typically gold/needle spans from *other* instances, so the swap
    is semantically plausible but no longer supports this instance's answer.
    """

    candidate_spans: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        spans = tuple(np.asarray(c) for c in self.candidate_spans)
        if not spans:
            raise ValueError("need at least one candidate span")
        for c in spans:
            if c.ndim != 1 or c.size == 0:
                raise ValueError("candidate spans must be non-empty 1-D")
        self.candidate_spans = spans

    def _pick(self, *, seed: int, length: int, avoid: np.ndarray | None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.candidate_spans))
        for idx in order:
            rep = fit_to_length(self.candidate_spans[idx], length)
            if avoid is None or not np.array_equal(rep, avoid):
                return rep
        # every candidate collapses to `avoid` at this length; return one anyway
        return fit_to_length(self.candidate_spans[int(order[0])], length)

    def corrupt(self, tokens: np.ndarray, span: tuple[int, int], *, seed: int = 0) -> Corruption:
        tokens = np.asarray(tokens)
        start, end = span
        original = tokens[start:end]
        rep = self._pick(seed=seed, length=end - start, avoid=original)
        return token_swap(tokens, span, rep, seed=seed)

    def two_variants(
        self, tokens: np.ndarray, span: tuple[int, int], *, seeds: tuple[int, int] = (0, 1)
    ) -> tuple[Corruption, Corruption]:
        """Two distinct seeded corruptions for the gate-11.2b stability check.

        Variant B is chosen to differ from variant A's replacement where the pool
        allows; with a single candidate the two may coincide (documented).
        """
        tokens = np.asarray(tokens)
        start, end = span
        original = tokens[start:end]
        a = self.corrupt(tokens, span, seed=seeds[0])
        rep_a = a.tokens[start:end]
        rng = np.random.default_rng(seeds[1])
        rep_b = None
        for idx in rng.permutation(len(self.candidate_spans)):
            cand = fit_to_length(self.candidate_spans[int(idx)], end - start)
            if not np.array_equal(cand, rep_a) and not np.array_equal(cand, original):
                rep_b = cand
                break
        if rep_b is None:
            rep_b = self._pick(seed=seeds[1], length=end - start, avoid=original)
        b = token_swap(tokens, span, rep_b, seed=seeds[1])
        return a, b


def document_swap(
    documents: list[np.ndarray],
    support_idx: int,
    distractor: np.ndarray,
    *,
    fit_length: bool = False,
    seed: int = 0,
) -> Corruption:
    """Replace the supporting document with a distractor (§8.2 coarser variant).

    ``documents`` are the token arrays concatenated to form the context.
    ``fit_length=True`` keeps the total length fixed (for token-aligned patching);
    otherwise the distractor's own length is used.
    """
    docs = [np.asarray(d) for d in documents]
    if not docs:
        raise ValueError("no documents")
    if not (0 <= support_idx < len(docs)):
        raise ValueError(f"support_idx {support_idx} out of range for {len(docs)} documents")
    dist = np.asarray(distractor)
    if dist.ndim != 1 or dist.size == 0:
        raise ValueError("distractor must be non-empty 1-D")
    if fit_length:
        dist = fit_to_length(dist, docs[support_idx].size)
    start = int(sum(d.size for d in docs[:support_idx]))
    new_docs = list(docs)
    new_docs[support_idx] = dist
    out = np.concatenate(new_docs)
    return Corruption(
        tokens=out,
        corruption_type=CorruptionType.DOCUMENT_SWAP,
        changed_span=(start, start + int(dist.size)),
        seed=seed,
    )


def corruption_stability(effect_a: np.ndarray, effect_b: np.ndarray, k: int) -> float:
    """Top-k token-set overlap of two oracle estimates from two corruptions (11.2b).

    1.0 means the oracle's top tokens are identical across the two variants;
    low values mean the oracle is corruption-sensitive.
    """
    return top_k_jaccard(effect_a, effect_b, k)
