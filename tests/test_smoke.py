"""Smoke test: the package imports cleanly without a GPU stack."""

import subprocess
import sys

import lcv


def test_version() -> None:
    assert lcv.__version__ == "0.0.0"


def test_import_is_torch_free() -> None:
    # Importing lcv must not transitively import torch. This keeps CI and local
    # CPU dev light; the GPU stack is reached only in the instrumentation phase.
    # Checked in a *fresh* interpreter so the result is independent of other tests
    # in this process (e.g. the needs_model CPU smoke) that legitimately import
    # torch -- sys.modules is process-global, so an in-process check is polluted
    # the moment any sibling test touches torch. CI runs torch-free regardless;
    # this keeps the contract honest even in a venv with the smoke/gpu extras.
    code = (
        "import sys, importlib; importlib.import_module('lcv'); "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"import lcv transitively pulled torch:\n{result.stderr}"
