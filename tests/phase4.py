"""Phase 4 acceptance gate: replication (addendum §11.4).

Re-run gates 11.0 through 11.3 on ``gemma-2-9b-it`` with the *original* Gemma Scope
(SAE-only). Gemma 2 has no transcoder suite -- transcoders only exist for Gemma 3
(Gemma Scope 2) -- so the transcoder-based signals are dropped from the replication
and the omission is reported as a finding (§13.6, gate 11.2c); RQ1-RQ3 replicate
over the attention-based signals. The disagreement structure and the
``D(x)``-predicts-flips result either replicate or they do not; either outcome is
reportable. The whole gate is model-bound and marked ``gpu``; the only thing
checkable in CI is that the replication backbone is pinned to the right model so
the H100 run targets Gemma, not the primary.
"""

import pytest

# Expected replication target (§3.1 / §11.4). Pinned so the GPU rerun cannot
# silently point at the primary model.
EXPECTED_REPLICATION_MODEL = "google/gemma-2-9b-it"
EXPECTED_PRIMARY_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


# --- CPU core: replication target is pinned -------------------------------- #


def test_11_4_replication_backbone_is_gemma():
    from lcv.model.backbone import PRIMARY_MODEL, REPLICATION_MODEL

    assert REPLICATION_MODEL == EXPECTED_REPLICATION_MODEL
    assert PRIMARY_MODEL == EXPECTED_PRIMARY_MODEL  # the two backbones stay distinct
    assert REPLICATION_MODEL != PRIMARY_MODEL


# --- GPU gate -------------------------------------------------------------- #


@pytest.mark.gpu
def test_11_4_phases_replicate_on_gemma(gemma_backbone):
    pytest.skip(
        "re-run gates 11.0-11.3 on gemma-2-9b-it with the original Gemma Scope "
        "(SAE-only; transcoder signals dropped -- no Gemma-2 transcoders, §13.6); "
        "the disagreement structure and D(x)-predicts-flips result either replicate "
        "or they do not -- both are reportable"
    )
