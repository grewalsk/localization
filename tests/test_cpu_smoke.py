"""CPU end-to-end smoke test: the pipeline spine runs on a real transformer.

This exercises the *plumbing*, not the science: a toy model (gpt2) on CPU with
eager attention, fed through the same signal core the H100 path uses. It proves
the wiring (tokenize -> content mask -> eager attention -> accumulated-attention
importance -> agreement) runs end-to-end on a real HF model. It proves no
published number — gpt2 has no localization ground truth. The validated-result
path stays on TransformerLens + Llama-3.1-8B (§11), untouched by this module.

Marked ``needs_model`` (downloads gpt2, ~0.5 GB): auto-skipped in CI and on any
box without ``transformers``. ``torch`` is additionally importorskip'd so a
transformers-without-torch env skips rather than errors.
"""

from __future__ import annotations

import numpy as np
import pytest

from lcv.agreement import auroc_vs_gold, disagreement, spearman
from lcv.data import niah
from lcv.data.tokenization import char_span_to_token_indices, verify_gold_mapping
from lcv.signals.attention_hh import accumulated_attention_from_patterns

pytestmark = pytest.mark.needs_model

# gpt2: 12 layers x 12 heads, ungated, fast BPE tokenizer with char offsets.
_N_LAYERS = 12
_N_HEADS = 12


@pytest.fixture(scope="module")
def smoke():
    """One gpt2 eager forward pass over a synthetic NIAH instance (loaded once)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from lcv.model.hf_backbone import hf_eager_forward, load_hf_backbone

    inst = niah.build_niah_instance(
        instance_id="cpu-smoke", depth=0.5, haystack_sentences=6, magic_number=4821, seed=0
    )
    rendered = inst.rendered_prompt
    context_len = len(rendered.split("\n\n", 1)[0])
    try:
        model, tokenizer = load_hf_backbone("gpt2")
    except Exception as exc:  # network / hub issues -> skip, not fail
        pytest.skip(f"gpt2 unavailable: {exc}")
    fwd = hf_eager_forward(model, tokenizer, rendered, content_char_range=(0, context_len))
    return inst, rendered, context_len, fwd


def test_attention_shape_is_gpt2(smoke):
    _, _, _, fwd = smoke
    n_layers, n_heads, q, k = fwd.attentions.shape
    assert (n_layers, n_heads) == (_N_LAYERS, _N_HEADS)
    assert q == k  # square causal attention over the sequence
    assert k == fwd.input_ids.shape[0] == len(fwd.offsets)


def test_attention_rows_sum_to_one(smoke):
    """Gate 11.0d analog on a toy model: each attention row is a distribution."""
    _, _, _, fwd = smoke
    row_sums = fwd.attentions.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-3)


def test_content_mask_splits_context_from_question(smoke):
    _, _, _, fwd = smoke
    mask = fwd.content_mask
    assert mask.dtype == bool
    assert mask.any() and not mask.all()  # some content, some non-content (question/specials)
    assert mask.shape[0] == fwd.attentions.shape[-1]


def test_gold_needle_maps_back_through_real_tokenizer(smoke):
    """Gate 11.1e on the real gpt2 tokenizer: needle chars -> tokens -> decodes back."""
    inst, rendered, _, fwd = smoke
    gold = inst.gold_spans[0]
    gold_tok = char_span_to_token_indices(
        fwd.offsets, gold.char_start, gold.char_end, content_mask=fwd.content_mask
    )
    assert len(gold_tok) >= 1
    assert verify_gold_mapping(rendered, fwd.offsets, gold_tok, gold.text) is True
    # the located needle tokens are content tokens (not the appended question)
    assert all(fwd.content_mask[i] for i in gold_tok)


def test_importance_vector_is_well_formed(smoke):
    inst, _, _, fwd = smoke
    n_content = int(fwd.content_mask.sum())
    imp = accumulated_attention_from_patterns(
        fwd.attentions, fwd.content_mask, instance_id=inst.instance_id
    )
    assert imp.values.shape == (n_content,)
    assert np.all(np.isfinite(imp.values))
    assert imp.values.min() >= 0.0 and imp.values.max() <= 1.0  # minmax-normalized


def test_two_query_regions_agree_finitely(smoke):
    """Two query regions of the same signal -> a finite internal D(x) in [0, 2]."""
    inst, _, context_len, fwd = smoke
    question_positions = [i for i, (a, b) in enumerate(fwd.offsets) if a >= context_len and b > a]
    imp_all = accumulated_attention_from_patterns(
        fwd.attentions, fwd.content_mask, instance_id=inst.instance_id
    )
    imp_q = accumulated_attention_from_patterns(
        fwd.attentions,
        fwd.content_mask,
        instance_id=inst.instance_id,
        query_positions=question_positions,
    )
    assert imp_all.values.shape == imp_q.values.shape
    rho = spearman(imp_all.values, imp_q.values)
    assert np.isfinite(rho) and -1.0 <= rho <= 1.0
    dx = disagreement([imp_all.values, imp_q.values])
    assert np.isfinite(dx) and 0.0 <= dx <= 2.0


def test_needle_auroc_is_defined(smoke):
    """AUROC of importance vs needle membership is a finite probability.

    Plumbing only: gpt2 has no localization ground truth, so we assert the metric
    is *well-defined* (finite, in [0, 1]) — NOT that it clears chance. The
    above-chance bar is gate 11.1d on Llama-3.1-8B (tests/phase1.py), not here.
    """
    inst, _, _, fwd = smoke
    imp = accumulated_attention_from_patterns(
        fwd.attentions, fwd.content_mask, instance_id=inst.instance_id
    )
    gold = inst.gold_spans[0]
    gold_tok = set(
        char_span_to_token_indices(
            fwd.offsets, gold.char_start, gold.char_end, content_mask=fwd.content_mask
        )
    )
    content_positions = np.where(fwd.content_mask)[0]
    gold_content_mask = np.array([p in gold_tok for p in content_positions], dtype=bool)
    auroc = auroc_vs_gold(imp.values, gold_content_mask)
    assert np.isfinite(auroc) and 0.0 <= auroc <= 1.0
