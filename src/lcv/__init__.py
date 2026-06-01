"""localization-convergent-validity (``lcv``).

Research harness testing whether the localization signals deployed in LLM
inference agree per-instance on *where information lives*, whether each tracks a
per-instance causal oracle, and whether their disagreement ``D(x)`` predicts
KV-compression failure.

The module layout mirrors the engineering addendum §12. This top-level package
is intentionally import-light: importing ``lcv`` must not pull in torch or any
GPU instrumentation, so the CPU analysis/stats/data plumbing stays importable
and testable in CI without a GPU.
"""

__version__ = "0.0.0"
