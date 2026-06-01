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
