"""Pytest configuration.

The §11 GPU/model acceptance gates are collected but auto-skipped wherever the
instrumentation stack is unavailable (local CPU dev, CI). Run them on the rented
H100, where torch + CUDA + model weights are present. CI additionally deselects
them with ``-m "not gpu"``; this hook covers ad-hoc local runs.
"""

from __future__ import annotations

import importlib.util

import pytest


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _cuda_available() -> bool:
    if not _installed("torch"):
        return False
    import torch  # noqa: PLC0415

    return bool(torch.cuda.is_available())


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    cuda = _cuda_available()
    have_model = _installed("transformers")
    skip_gpu = pytest.mark.skip(reason="no GPU (torch+CUDA); run on the rented H100")
    skip_model = pytest.mark.skip(reason="transformers/model weights unavailable")
    for item in items:
        if "gpu" in item.keywords and not cuda:
            item.add_marker(skip_gpu)
        if "needs_model" in item.keywords and not have_model:
            item.add_marker(skip_model)


# --------------------------------------------------------------------------- #
# Shared fixtures for the §11 phase gates
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def backbone():
    """Primary TransformerLens backbone for the GPU gates (§3.1, §11).

    Only requested by ``gpu``-marked tests, which the collection hook skips when
    CUDA is absent, so the heavy load never runs in CI / on CPU dev.
    """
    pytest.importorskip("transformer_lens")
    from lcv.model.backbone import load_backbone

    return load_backbone()


@pytest.fixture(scope="session")
def gemma_backbone():
    """Replication backbone (gemma-2-9b-it) for the Phase-4 gates (§3.1, §11.4)."""
    pytest.importorskip("transformer_lens")
    from lcv.model.backbone import REPLICATION_MODEL, load_backbone

    return load_backbone(REPLICATION_MODEL)


@pytest.fixture(scope="session")
def chat_tokenizer():
    """A real fast tokenizer for the 11.1e gold-span gate (marked ``needs_model``)."""
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    except Exception as exc:  # network / gating issues -> skip, not fail
        pytest.skip(f"tokenizer unavailable: {exc}")
