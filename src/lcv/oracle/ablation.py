"""Legacy alias for the masking oracle (§8.1).

Earlier drafts named the oracle "token ablation"; the paper (Definition 2) calls
the identical operation **masking**, and the canonical implementation now lives in
:mod:`lcv.oracle.masking`. This module re-exports it under the old names so
existing references keep working. Prefer importing from :mod:`lcv.oracle.masking`.
"""

from __future__ import annotations

from .masking import (
    answer_log_prob,
)
from .masking import (
    gold_span_masking_drop as gold_span_ablation_drop,
)
from .masking import (
    token_masking_effect as token_ablation_effect,
)

__all__ = ["answer_log_prob", "gold_span_ablation_drop", "token_ablation_effect"]
