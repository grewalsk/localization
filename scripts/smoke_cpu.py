"""CPU smoke demo: run the pipeline spine on a toy model (default gpt2).

    uv run --extra smoke python scripts/smoke_cpu.py

Builds one synthetic NIAH instance, runs a real HF eager forward pass on CPU,
maps the gold needle to real sub-word tokens (gate 11.1e on a real tokenizer),
computes accumulated-attention importance over the content tokens, and reports
the gold-vs-importance AUROC plus the internal-consistency D(x) between two query
regions. This proves the wiring runs end-to-end on a real transformer. It proves
no published number: gpt2 is a toy and has no localization ground truth. The
science path stays on TransformerLens + Llama-3.1-8B on the H100 (§11).
"""

from __future__ import annotations

import argparse

import numpy as np

from lcv.agreement import auroc_vs_gold, disagreement, spearman
from lcv.data import niah
from lcv.data.tokenization import char_span_to_token_indices, verify_gold_mapping
from lcv.model.hf_backbone import DEFAULT_CPU_MODEL, hf_eager_forward, load_hf_backbone
from lcv.signals.attention_hh import accumulated_attention_from_patterns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_CPU_MODEL, help="HF model id (CPU)")
    parser.add_argument("--haystack", type=int, default=6, help="filler sentences")
    parser.add_argument("--depth", type=float, default=0.5, help="needle depth in [0,1]")
    args = parser.parse_args()

    inst = niah.build_niah_instance(
        instance_id="smoke",
        depth=args.depth,
        haystack_sentences=args.haystack,
        magic_number=4821,
        seed=0,
    )
    rendered = inst.rendered_prompt
    context = rendered.split("\n\n", 1)[0]
    context_len = len(context)
    gold = inst.gold_spans[0]

    print(f"model            : {args.model} (CPU, eager attention)")
    print(f"prompt chars     : {len(rendered)}  (context {context_len})")
    print(f"needle           : {gold.text!r}")

    model, tokenizer = load_hf_backbone(args.model)
    fwd = hf_eager_forward(model, tokenizer, rendered, content_char_range=(0, context_len))

    n_layers, n_heads, q, k = fwd.attentions.shape
    n_content = int(fwd.content_mask.sum())
    row_sums = fwd.attentions.sum(axis=-1)
    print(f"tokens           : {k}  (content {n_content})")
    print(f"attention shape  : [{n_layers} layers, {n_heads} heads, {q} q, {k} k]")
    print(f"rows sum to 1    : {np.allclose(row_sums, 1.0, atol=1e-3)}  (gate 11.0d analog)")

    # Gate 11.1e on the real tokenizer: needle chars -> real tokens -> decode back.
    gold_tok = char_span_to_token_indices(
        fwd.offsets, gold.char_start, gold.char_end, content_mask=fwd.content_mask
    )
    maps_back = verify_gold_mapping(rendered, fwd.offsets, gold_tok, gold.text)
    print(f"gold -> {len(gold_tok)} real tokens, decodes back >= 0.9: {maps_back}")

    # Two query regions of the same signal (a robustness contrast, §5.1).
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

    # Gold membership over the content tokens, for AUROC.
    content_positions = np.where(fwd.content_mask)[0]
    gold_set = set(gold_tok)
    gold_content_mask = np.array([p in gold_set for p in content_positions], dtype=bool)

    auroc = auroc_vs_gold(imp_all.values, gold_content_mask)
    rho = spearman(imp_all.values, imp_q.values)
    dx = disagreement([imp_all.values, imp_q.values])
    print(f"needle AUROC     : {auroc:.3f}  (importance ranks needle vs filler; chance 0.5)")
    print(f"region Spearman  : {rho:.3f}  (all-positions vs question-region)")
    print(f"internal D(x)    : {dx:.3f}  (1 - mean pairwise Spearman; 0 = agree)")

    order = np.argsort(imp_all.values)[::-1][:5]
    print("top-5 attended content tokens:")
    for rank, ci in enumerate(order, 1):
        p = int(content_positions[ci])
        a, b = fwd.offsets[p]
        print(f"  {rank}. {rendered[a:b]!r:20s} importance={imp_all.values[ci]:.3f}")

    print("\nOK: spine ran end-to-end on a real transformer (plumbing only, not a result).")


if __name__ == "__main__":
    main()
