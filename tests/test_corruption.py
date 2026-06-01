"""Tests for §8.2 corruption construction.

Pins the invariants that make token-swap usable as a patching reference: it is
length-preserving, only the gold span changes, it is deterministic + seedable,
Gaussian corruption is refused, and two distinct variants can be emitted for the
gate-11.2b stability check.
"""

import numpy as np
import pytest

from lcv.contracts import CorruptionType
from lcv.oracle import corruption as co

# --- Gaussian is forbidden ------------------------------------------------- #


def test_primary_is_token_swap():
    assert co.PRIMARY is CorruptionType.TOKEN_SWAP


def test_assert_not_gaussian_refuses_noise_accepts_valid():
    with pytest.raises(ValueError, match="forbidden"):
        co.assert_not_gaussian("gaussian")
    with pytest.raises(ValueError, match="forbidden"):
        co.assert_not_gaussian("noise")
    assert co.assert_not_gaussian("token_swap") is CorruptionType.TOKEN_SWAP
    assert co.assert_not_gaussian(CorruptionType.DOCUMENT_SWAP) is CorruptionType.DOCUMENT_SWAP
    with pytest.raises(ValueError, match="unknown corruption"):
        co.assert_not_gaussian("blur")


# --- fit_to_length --------------------------------------------------------- #


def test_fit_to_length_equal_truncate_tile():
    np.testing.assert_array_equal(co.fit_to_length([1, 2, 3], 3), [1, 2, 3])
    np.testing.assert_array_equal(co.fit_to_length([1, 2, 3, 4], 2), [1, 2])
    np.testing.assert_array_equal(co.fit_to_length([7, 8], 5), [7, 8, 7, 8, 7])


def test_fit_to_length_guards():
    with pytest.raises(ValueError, match="empty"):
        co.fit_to_length([], 3)
    with pytest.raises(ValueError, match="positive"):
        co.fit_to_length([1, 2], 0)


# --- token_swap ------------------------------------------------------------ #


def test_token_swap_is_length_preserving_and_local():
    tokens = np.array([10, 11, 12, 13, 14])
    c = co.token_swap(tokens, (1, 3), np.array([99, 98]))
    assert c.tokens.shape == tokens.shape  # length preserved
    np.testing.assert_array_equal(c.tokens, [10, 99, 98, 13, 14])  # only span changed
    np.testing.assert_array_equal(tokens, [10, 11, 12, 13, 14])  # input untouched
    assert c.changed_span == (1, 3)
    assert c.corruption_type is CorruptionType.TOKEN_SWAP


def test_token_swap_fits_replacement_to_span():
    tokens = np.array([1, 2, 3, 4, 5])
    c = co.token_swap(tokens, (0, 4), np.array([9]))  # replacement tiled to width 4
    np.testing.assert_array_equal(c.tokens, [9, 9, 9, 9, 5])


def test_token_swap_rejects_bad_span():
    tokens = np.arange(5)
    for span in [(-1, 2), (2, 2), (3, 10)]:
        with pytest.raises(ValueError, match="bad span"):
            co.token_swap(tokens, span, np.array([0]))


# --- TokenSwapCorruptor ---------------------------------------------------- #


def test_corruptor_changes_span_and_is_deterministic():
    pool = (np.array([100, 101]), np.array([200, 201, 202]))
    corruptor = co.TokenSwapCorruptor(pool)
    tokens = np.array([1, 2, 3, 4, 5])
    a = corruptor.corrupt(tokens, (1, 3), seed=7)
    b = corruptor.corrupt(tokens, (1, 3), seed=7)
    np.testing.assert_array_equal(a.tokens, b.tokens)  # same seed -> identical
    assert not np.array_equal(a.tokens[1:3], tokens[1:3])  # span actually changed
    np.testing.assert_array_equal(a.tokens[[0, 3, 4]], tokens[[0, 3, 4]])  # rest intact


def test_corruptor_empty_pool_and_bad_candidates_raise():
    with pytest.raises(ValueError, match="at least one candidate"):
        co.TokenSwapCorruptor(())
    with pytest.raises(ValueError, match="non-empty 1-D"):
        co.TokenSwapCorruptor((np.zeros((2, 2)),))


def test_two_variants_differ_and_preserve_length():
    pool = (np.array([100, 101]), np.array([200, 201]), np.array([300, 301]))
    corruptor = co.TokenSwapCorruptor(pool)
    tokens = np.array([1, 2, 3, 4, 5])
    a, b = corruptor.two_variants(tokens, (1, 3), seeds=(0, 1))
    assert a.tokens.shape == tokens.shape == b.tokens.shape
    assert not np.array_equal(a.tokens[1:3], b.tokens[1:3])  # genuinely two variants
    # everything outside the span is identical across both
    np.testing.assert_array_equal(a.tokens[[0, 3, 4]], tokens[[0, 3, 4]])
    np.testing.assert_array_equal(b.tokens[[0, 3, 4]], tokens[[0, 3, 4]])


# --- document_swap --------------------------------------------------------- #


def test_document_swap_replaces_right_doc_and_offsets():
    docs = [np.array([1, 2]), np.array([3, 4, 5]), np.array([6])]
    c = co.document_swap(docs, 1, np.array([97, 98, 99, 100]))
    np.testing.assert_array_equal(c.tokens, [1, 2, 97, 98, 99, 100, 6])
    assert c.changed_span == (2, 6)  # doc 1 started at offset 2
    assert c.corruption_type is CorruptionType.DOCUMENT_SWAP


def test_document_swap_fit_length_keeps_total():
    docs = [np.array([1, 2]), np.array([3, 4, 5]), np.array([6])]
    total = sum(d.size for d in docs)
    c = co.document_swap(docs, 1, np.array([99]), fit_length=True)
    assert c.tokens.size == total  # distractor tiled to the replaced doc's width
    np.testing.assert_array_equal(c.tokens, [1, 2, 99, 99, 99, 6])


def test_document_swap_guards():
    docs = [np.array([1, 2]), np.array([3, 4])]
    with pytest.raises(ValueError, match="out of range"):
        co.document_swap(docs, 5, np.array([9]))
    with pytest.raises(ValueError, match="distractor"):
        co.document_swap(docs, 0, np.array([]))
    with pytest.raises(ValueError, match="no documents"):
        co.document_swap([], 0, np.array([9]))


# --- stability metric (11.2b) ---------------------------------------------- #


def test_corruption_stability_identical_and_disjoint():
    a = np.array([0.9, 0.1, 0.8, 0.2])
    assert co.corruption_stability(a, a, 2) == pytest.approx(1.0)
    assert co.corruption_stability([1.0, 0, 0, 0], [0, 0, 0, 1.0], 1) == pytest.approx(0.0)
