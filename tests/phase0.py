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

# The short prompts above only stress the *last* token of tiny inputs. TL-vs-HF
# divergence (rotary phase, eager vs fused attention, bf16 accumulation) grows with
# sequence length, so the gate must also hold deep into the ~4k context the
# experiments actually use -- before any of the six signals run against TL hooks
# (the §11 TL-memory watch-out). Parity is checked at strided positions, not just
# the final token, across a >=4k-token prompt.
LONG_PARITY_MIN_TOKENS = 4096
LONG_PARITY_STRIDE = 512


def next_token_kl(logits_p: np.ndarray, logits_q: np.ndarray) -> float:
    """Worst-case ``KL(softmax(p) || softmax(q))`` over the last axis (gate 11.0b).

    Pure-CPU statistic for the TL-vs-HF parity check; rows are independent
    distributions and we report the max KL across them.
    """
    lp = log_softmax(logits_p, axis=-1)
    lq = log_softmax(logits_q, axis=-1)
    p = softmax(logits_p, axis=-1)
    return float(np.sum(p * (lp - lq), axis=-1).max())


def build_long_parity_text(min_chars: int = 24_000) -> str:
    """A long, deterministic, *varied* prompt for the >=4k-token parity gate (§11.0b).

    Concatenates the short parity prompts with numbered narrative filler until it
    passes ``min_chars`` -- varied tokens (not one token repeated) so the forward pass
    exercises position-dependent rotary embeddings and the attention path across the
    whole context, where TL-vs-HF divergence accumulates. The GPU gate tokenizes this
    and slices to ``LONG_PARITY_MIN_TOKENS``; ``min_chars`` is set well above
    ``4096 * 4`` so the slice never runs short regardless of the tokenizer.
    """
    chunks: list[str] = []
    total = 0
    i = 0
    while total < min_chars:
        base = PARITY_PROMPTS[i % len(PARITY_PROMPTS)]
        chunk = f"Section {i}. {base} and the account continued onward in detail. "
        chunks.append(chunk)
        total += len(chunk)
        i += 1
    return "".join(chunks)


def parity_check_positions(seq_len: int, stride: int = LONG_PARITY_STRIDE) -> list[int]:
    """0-based positions at which to compare TL vs HF logits over a long context (§11.0b).

    Strided across the sequence with the final token always included, so divergence
    that only shows up deep into a 4k context is caught (a last-token-only check would
    miss it). Empty for a non-positive length.
    """
    if seq_len <= 0:
        return []
    positions = list(range(0, seq_len, stride))
    if positions[-1] != seq_len - 1:
        positions.append(seq_len - 1)
    return positions


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


def test_11_0b_long_parity_text_long_enough_and_varied():
    text = build_long_parity_text()
    # >= 4096 tokens even at a pessimistic ~4 chars/token, and deterministic.
    assert len(text) >= LONG_PARITY_MIN_TOKENS * 4
    assert text == build_long_parity_text()
    # varied, not one token repeated: several distinct base prompts appear.
    assert sum(prompt in text for prompt in PARITY_PROMPTS) >= 5


def test_11_0b_parity_positions_strided_with_final_token():
    pos = parity_check_positions(LONG_PARITY_MIN_TOKENS)
    assert pos[0] == 0
    assert pos[-1] == LONG_PARITY_MIN_TOKENS - 1  # final token always checked
    assert LONG_PARITY_STRIDE in pos and 2 * LONG_PARITY_STRIDE in pos
    # final token appended even when it does not land on the stride
    assert parity_check_positions(10, stride=4) == [0, 4, 8, 9]
    assert parity_check_positions(0) == []


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


# --- CPU core: Llama-Scope transcoder conversion (gate 11.0c, the §4.3 trap) - #
# Pins the TopK->JumpReLU + two-factor dataset-norm conversion against the actual
# hyperparams.json fields fetched from HF (2026-06-07), so the GPU loader cannot
# silently reuse the SAELens single-factor SAE fold and inflate every transcoder
# output by ~19x.

# Verbatim from the published checkpoints (fnlp/Llama3_1-8B-Base-LX{TC,R}-8x, L8).
_L8TC_HP = {
    "d_model": 4096,
    "d_sae": 32768,
    "hook_point_in": "blocks.8.ln2.hook_normalized",
    "hook_point_out": "blocks.8.hook_mlp_out",
    "jump_relu_threshold": 0.10400390625,
    "dataset_average_activation_norm": {"in": 64.0, "out": 3.390625},
    "top_k": 50,
    "act_fn": "jumprelu",
}
_L8R_HP = {
    "d_model": 4096,
    "d_sae": 32768,
    "hook_point_in": "blocks.8.hook_resid_post",
    "hook_point_out": "blocks.8.hook_resid_post",
    "jump_relu_threshold": 0.2021484375,
    "dataset_average_activation_norm": {"in": 6.3125, "out": 6.3125},
    "top_k": 50,
    "act_fn": "jumprelu",
}


def test_11_0c_hyperparams_parse_and_transcoder_flag():
    tc = sae_loader.parse_llama_scope_hyperparams(_L8TC_HP)
    assert tc.is_transcoder is True  # in (ln2.normalized) != out (mlp_out)
    assert (tc.d_model, tc.d_sae, tc.top_k) == (4096, 32768, 50)
    assert tc.avg_norm_in == 64.0 and tc.avg_norm_out == 3.390625
    sae = sae_loader.parse_llama_scope_hyperparams(_L8R_HP)
    assert sae.is_transcoder is False  # residual SAE reconstructs the same hook
    assert sae.avg_norm_in == sae.avg_norm_out


def test_11_0c_hyperparams_reject_malformed():
    with pytest.raises(ValueError, match="missing/invalid"):
        sae_loader.parse_llama_scope_hyperparams({"d_model": 4096})  # no norms etc.
    with pytest.raises(ValueError, match="act_fn"):
        sae_loader.parse_llama_scope_hyperparams({**_L8TC_HP, "act_fn": "topk"})
    with pytest.raises(ValueError, match="dataset activation norm"):
        bad = {**_L8TC_HP, "dataset_average_activation_norm": {"in": 0.0, "out": 3.0}}
        sae_loader.parse_llama_scope_hyperparams(bad)


def test_11_0c_transcoder_conversion_two_factor():
    # The crux: input is a LayerNorm output (norm ~= sqrt(4096) = 64 -> nsf_in = 1),
    # output is mlp_out (norm 3.39 -> nsf_out ~= 18.9). The decoder must use nsf_out.
    conv = sae_loader.llama_scope_conversion(sae_loader.parse_llama_scope_hyperparams(_L8TC_HP))
    assert conv.nsf_in == pytest.approx(1.0)  # sqrt(4096) / 64
    assert conv.nsf_out == pytest.approx(64.0 / 3.390625)
    assert conv.nsf_out == pytest.approx(18.8754, abs=1e-3)
    assert conv.threshold_scaled == pytest.approx(0.10400390625)  # * nsf_in (=1)
    assert conv.decoder_inflation == pytest.approx(conv.nsf_out / conv.nsf_in)
    assert conv.decoder_inflation == pytest.approx(18.8754, abs=1e-3)  # SAELens would over-scale


def test_11_0c_sae_conversion_collapses_to_single_factor():
    # For an R/M SAE (in == out) the two factors coincide: no transcoder correction,
    # reproduces the SAELens single-factor fold exactly (decoder_inflation == 1).
    conv = sae_loader.llama_scope_conversion(sae_loader.parse_llama_scope_hyperparams(_L8R_HP))
    assert conv.nsf_in == pytest.approx(64.0 / 6.3125)
    assert conv.nsf_out == pytest.approx(conv.nsf_in)
    assert conv.decoder_inflation == pytest.approx(1.0)
    assert conv.threshold_scaled == pytest.approx(0.2021484375 * conv.nsf_in)


def _tiny_checkpoint(seed: int = 0, d_in: int = 4, d_sae: int = 6, d_out: int = 4):
    """A miniature Llama-Scope-shaped safetensors dict (raw, un-folded)."""
    rng = np.random.default_rng(seed)
    return {
        "encoder.weight": rng.normal(size=(d_sae, d_in)),  # [d_sae, d_in]
        "encoder.bias": rng.normal(size=d_sae),
        "decoder.weight": rng.normal(size=(d_out, d_sae)),  # [d_out, d_sae]
        "decoder.bias": rng.normal(size=d_out),
    }


def test_11_0c_fold_maps_keys_and_orientation():
    raw = _tiny_checkpoint()
    conv = sae_loader.LlamaScopeConversion(nsf_in=2.0, nsf_out=8.0, threshold_scaled=0.5)
    folded = sae_loader.fold_llama_scope_checkpoint(raw, conv)
    assert folded["W_enc"].shape == (4, 6)  # encoder.weight.T
    assert folded["W_dec"].shape == (6, 4)  # decoder.weight.T
    np.testing.assert_allclose(folded["W_enc"], raw["encoder.weight"].T * 2.0)
    np.testing.assert_allclose(folded["W_dec"], raw["decoder.weight"].T / 8.0)
    np.testing.assert_allclose(folded["b_dec"], raw["decoder.bias"] / 8.0)
    np.testing.assert_allclose(folded["b_enc"], raw["encoder.bias"])  # b_enc not folded
    np.testing.assert_allclose(folded["threshold"], np.full(6, 0.5))


def test_11_0c_fold_missing_tensor_raises():
    raw = _tiny_checkpoint()
    del raw["decoder.bias"]
    conv = sae_loader.LlamaScopeConversion(nsf_in=1.0, nsf_out=1.0, threshold_scaled=0.0)
    with pytest.raises(ValueError, match="missing tensors"):
        sae_loader.fold_llama_scope_checkpoint(raw, conv)


def test_11_0c_transcoder_forward_matches_explicit_jumprelu():
    raw = _tiny_checkpoint(seed=1)
    conv = sae_loader.LlamaScopeConversion(nsf_in=1.3, nsf_out=7.0, threshold_scaled=0.2)
    folded = sae_loader.fold_llama_scope_checkpoint(raw, conv)
    x = np.random.default_rng(2).normal(size=(5, 4))
    recon, feats = sae_loader.transcoder_reconstruct(x, folded)
    # explicit reference: relu(pre) * (pre > threshold), then decode
    pre = x @ folded["W_enc"] + folded["b_enc"]
    feats_ref = np.maximum(pre, 0.0) * (pre > folded["threshold"])
    recon_ref = feats_ref @ folded["W_dec"] + folded["b_dec"]
    np.testing.assert_allclose(feats, feats_ref)
    np.testing.assert_allclose(recon, recon_ref)
    assert recon.shape == (5, 4)


def test_11_0c_single_factor_fold_is_the_clean_but_wrong_trap():
    # Demonstrate the bug numerically: a single-factor (nsf_in) decoder fold keeps the
    # *same* sparse code but inflates the reconstruction by exactly nsf_out / nsf_in.
    raw = _tiny_checkpoint(seed=3)
    hp = sae_loader.parse_llama_scope_hyperparams(_L8TC_HP)
    conv = sae_loader.llama_scope_conversion(hp)
    wrong = sae_loader.LlamaScopeConversion(
        nsf_in=conv.nsf_in, nsf_out=conv.nsf_in, threshold_scaled=conv.threshold_scaled
    )
    x = np.random.default_rng(4).normal(size=(7, 4))
    recon_ok, feats_ok = sae_loader.transcoder_reconstruct(
        x, sae_loader.fold_llama_scope_checkpoint(raw, conv)
    )
    recon_bad, feats_bad = sae_loader.transcoder_reconstruct(
        x, sae_loader.fold_llama_scope_checkpoint(raw, wrong)
    )
    np.testing.assert_allclose(feats_ok, feats_bad)  # encoder/threshold untouched
    np.testing.assert_allclose(recon_bad, recon_ok * conv.decoder_inflation)
    assert conv.decoder_inflation > 18.0  # ~18.9x at layer 8


def test_11_0c_l0_band_and_expected():
    assert sae_loader.EXPECTED_L0 == 50
    acts = np.array([[1.0, 0.0, 2.0, 0.0], [0.0, 0.0, 3.0, 4.0]])
    assert sae_loader.mean_l0(acts) == pytest.approx(2.0)
    assert sae_loader.mean_l0(np.zeros(4)) == pytest.approx(0.0)  # 1-D handled
    assert sae_loader.l0_within_expected(50.0)
    assert sae_loader.l0_within_expected(48.7)
    assert not sae_loader.l0_within_expected(5.0)  # threshold too high -> dead
    assert not sae_loader.l0_within_expected(200.0)  # threshold too low -> dense


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
def test_11_0b_tl_hf_parity_holds_at_4k_context(backbone):
    """Parity survives a >=4k-token context, checked at strided positions (gate 11.0b).

    The short-prompt gate above can pass while TL and HF drift apart deep into a long
    context: rotary embeddings, eager softmax, and bf16 rounding all accumulate with
    sequence length. A last-token-only check would miss mid-sequence divergence, so we
    compare the full next-token distribution at strided positions across the whole 4k
    window and take the worst case. If this fails, the signals must run against the
    nnsight/HF backbone instead of TransformerLens.
    """
    import torch

    from lcv.model.backbone import PRIMARY_MODEL, forward_logits, hf_reference_logits

    tokens = backbone.to_tokens(build_long_parity_text())
    assert tokens.shape[1] >= LONG_PARITY_MIN_TOKENS  # filler long enough to fill 4k
    tokens = tokens[:, :LONG_PARITY_MIN_TOKENS]
    idx = torch.tensor(parity_check_positions(tokens.shape[1]), device=tokens.device)

    # Index the strided positions on-device, so neither full [seq, vocab] logit tensor
    # is materialised on the CPU as numpy (~2 GB each at the Llama-3 vocab).
    tl = forward_logits(backbone, tokens)[0].index_select(0, idx).float().cpu().numpy()
    hf = hf_reference_logits(PRIMARY_MODEL, tokens)[0].index_select(0, idx).float().cpu().numpy()
    assert next_token_kl(tl, hf) < PARITY_KL_THRESHOLD  # worst-case over checked positions


@pytest.mark.gpu
def test_11_0d_attention_shape_and_row_sums(backbone):
    """Eager attention is ``[batch, n_layers, 32, q, k]`` with rows summing to 1 (§11.0d)."""
    from lcv.model.backbone import N_QUERY_HEADS, eager_attention_patterns

    tokens = backbone.to_tokens(PARITY_PROMPTS[0])
    patterns = eager_attention_patterns(backbone, tokens)
    assert patterns.shape[2] == N_QUERY_HEADS  # 32 query heads per layer
    row_sums = patterns.sum(dim=-1).float().cpu().numpy()
    assert np.allclose(row_sums, 1.0, atol=1e-3)  # softmax axis correct, no FlashAttention
