"""Phase 0 acceptance gates: environment and parity (addendum §11.0).

A green phase means "the instrumentation is provably doing what we think," not
"it ran." The CPU-runnable cores here — the next-token KL statistic, the
SAE-reconstruction-error metric, and SAE-name parsing with the §4.2 ``LXA``
refusal — run in CI. The model/parity gates (11.0a/b/d) are marked ``gpu`` and
run on the rented H100, with their thresholds pinned as constants below.
"""

import numpy as np
import pytest
from scipy.special import log_softmax, softmax

from lcv.model import sae_loader

# Gate 11.0b: next-token KL(TL || HF) bound on a fixed 10-prompt set (§11.0b).
PARITY_KL_THRESHOLD = 1e-3
PARITY_PROMPTS = (
    "The capital of France is",
    "Water is made of hydrogen and",
    "2 + 2 =",
    "The opposite of hot is",
    "Once upon a time,",
    "The first president of the United States was",
    "The chemical symbol for gold is",
    "A triangle has",
    "The sun rises in the",
    "To be or not to be,",
)


def next_token_kl(logits_p: np.ndarray, logits_q: np.ndarray) -> float:
    """Worst-case ``KL(softmax(p) || softmax(q))`` over the last axis (gate 11.0b).

    Pure-CPU statistic for the TL-vs-HF parity check; rows are independent
    distributions and we report the max KL across them.
    """
    lp = log_softmax(logits_p, axis=-1)
    lq = log_softmax(logits_q, axis=-1)
    p = softmax(logits_p, axis=-1)
    return float(np.sum(p * (lp - lq), axis=-1).max())


# --- CPU cores ------------------------------------------------------------- #


def test_11_0b_kl_is_zero_for_identical_logits():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(10, 256))
    assert next_token_kl(logits, logits) == pytest.approx(0.0, abs=1e-9)


def test_11_0b_kl_small_for_tiny_perturbation_clears_bar():
    rng = np.random.default_rng(1)
    p = rng.normal(size=256)
    q = p + 1e-4 * rng.normal(size=256)
    kl = next_token_kl(p, q)
    assert kl > 0.0
    assert kl < PARITY_KL_THRESHOLD


def test_11_0c_reconstruction_error_metric():
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]])
    assert sae_loader.reconstruction_error(a, a) == pytest.approx(0.0)
    err = sae_loader.reconstruction_error(a, a + 0.1)  # residual leaves variance
    assert 0.0 < err < 1.0
    z = np.ones((4, 3))  # zero-variance activations -> undefined
    assert np.isnan(sae_loader.reconstruction_error(z, z))
    with pytest.raises(ValueError, match="shape mismatch"):
        sae_loader.reconstruction_error(a, a[:2])


def test_11_0c_sae_name_parsing():
    assert sae_loader.parse_sae_name("L15TC-8x").is_transcoder
    n = sae_loader.parse_sae_name("L0R-8x")
    assert (n.layer, n.position, n.expansion, n.n_features) == (0, "R", 8, 32_768)
    assert sae_loader.parse_sae_name("L31M-32x").n_features == 131_072


def test_11_0c_attention_sae_lxa_refused_per_4_2():
    with pytest.raises(ValueError, match="attention-output"):
        sae_loader.parse_sae_name("L15A-8x")


def test_11_0c_bad_sae_names_raise():
    with pytest.raises(ValueError, match="unparseable"):
        sae_loader.parse_sae_name("nonsense")
    with pytest.raises(ValueError, match="unparseable"):
        sae_loader.parse_sae_name("L7R-16x")  # 16x is not a valid width
    with pytest.raises(ValueError, match="out of range"):
        sae_loader.parse_sae_name("L99R-8x")  # layer > 31


# --- CPU core: verified Llama-Scope provenance (gate 11.0c resolution) ------ #
# Pins the checkpoint resolution verified against the HF API + the SAELens
# pretrained_saes.yaml registry (2026-06-07), so the GPU phase cannot silently
# load the wrong checkpoint or invent a nonexistent transcoder release.


def test_11_0c_sae_lens_release_for_residual_and_mlp():
    # R/M are registered with conversion_func "llama_scope"; id == l<layer><pos>_<exp>x.
    assert sae_loader.llama_scope_sae_lens_release("L0R-8x") == ("llama_scope_lxr_8x", "l0r_8x")
    assert sae_loader.llama_scope_sae_lens_release("L16M-32x") == (
        "llama_scope_lxm_32x",
        "l16m_32x",
    )


def test_11_0c_transcoder_absent_from_sae_lens_registry():
    # Verified: SAELens registers R/M/A only -- no LXTC release. Transcoders must
    # load directly, so resolving one as a SAELens release must fail loudly.
    with pytest.raises(ValueError, match="no SAELens release|load directly"):
        sae_loader.llama_scope_sae_lens_release("L8TC-8x")


def test_11_0c_checkpoint_ref_templates_resolve():
    # The transcoder direct-download path (per-layer final.safetensors lives here).
    assert sae_loader.llama_scope_checkpoint_ref("L8TC-8x") == (
        "fnlp/Llama3_1-8B-Base-LXTC-8x",
        "Llama3_1-8B-Base-L8TC-8x",
    )
    assert sae_loader.llama_scope_checkpoint_ref("L0R-32x") == (
        "fnlp/Llama3_1-8B-Base-LXR-32x",
        "Llama3_1-8B-Base-L0R-32x",
    )


def test_11_0c_checkpoint_ref_rejects_lxa():
    with pytest.raises(ValueError, match="attention-output"):
        sae_loader.llama_scope_checkpoint_ref("L8A-8x")


# --- GPU gates ------------------------------------------------------------- #


@pytest.mark.gpu
def test_11_0a_greedy_generation_coherent():
    pytest.skip("manual human read of 5 sanity prompts for coherence (§11.0a)")


@pytest.mark.gpu
def test_11_0b_tl_hf_logit_parity(backbone):
    """TL logits match HF-eager within next-token KL < 1e-3 (§11.0b)."""
    from lcv.model.backbone import PRIMARY_MODEL, forward_logits, hf_reference_logits

    for prompt in PARITY_PROMPTS:
        tokens = backbone.to_tokens(prompt)
        tl = forward_logits(backbone, tokens)[0, -1].float().cpu().numpy()
        hf = hf_reference_logits(PRIMARY_MODEL, tokens)[0, -1].float().cpu().numpy()
        assert next_token_kl(tl, hf) < PARITY_KL_THRESHOLD  # else -> nnsight fallback


@pytest.mark.gpu
def test_11_0d_attention_shape_and_row_sums(backbone):
    """Eager attention is ``[batch, n_layers, 32, q, k]`` with rows summing to 1 (§11.0d)."""
    from lcv.model.backbone import N_QUERY_HEADS, eager_attention_patterns

    tokens = backbone.to_tokens(PARITY_PROMPTS[0])
    patterns = eager_attention_patterns(backbone, tokens)
    assert patterns.shape[2] == N_QUERY_HEADS  # 32 query heads per layer
    row_sums = patterns.sum(dim=-1).float().cpu().numpy()
    assert np.allclose(row_sums, 1.0, atol=1e-3)  # softmax axis correct, no FlashAttention
