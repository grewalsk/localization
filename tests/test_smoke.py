"""Smoke test: the package imports cleanly without a GPU stack."""

import sys

import lcv


def test_version() -> None:
    assert lcv.__version__ == "0.0.0"


def test_import_is_torch_free() -> None:
    # Importing lcv must not transitively import torch. This keeps CI and local
    # CPU dev light; the GPU stack is reached only in the instrumentation phase.
    assert "torch" not in sys.modules
