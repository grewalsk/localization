"""CPU plumbing backbone: plain Hugging Face eager attention on a tiny model.

This is **not** the science instrumentation. The published-number path is
TransformerLens on CUDA (:mod:`lcv.model.backbone`), gated by the §11 acceptance
tests and tied to specific models (Llama-3.1-8B + Llama Scope). This module is a
*separate, smaller* path whose only job is to let the pipeline spine
(tokenize -> content mask -> eager attention -> importance -> agreement) run
end-to-end on a laptop with a toy model like ``gpt2``. It validates plumbing,
never a published result: a tiny base LM has no localization numbers to check
against, so passing here means "the wiring runs", not "the method is right".

Design intent (STATE.md watch-out: keep signals backbone-portable). Attention is
read here and handed back as plain numpy in the shape the signal cores expect
(``[n_layers, n_heads, q, k]``). The signal cores never touch a model object, so
swapping this CPU accessor for the TransformerLens ``hook_pattern`` extractor on
the H100 means re-implementing only this file, not the signals. torch and
transformers are imported lazily so the package stays CPU-importable without them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..data.tokenization import content_mask_from_offsets

if TYPE_CHECKING:  # hints only; never imported on the torch-free CPU import path
    pass

# Toy default. Small, ungated, fast BPE tokenizer with char offsets, eager
# attention supported. 12 layers x 12 heads, 1024-token context.
DEFAULT_CPU_MODEL = "gpt2"


@dataclass(slots=True)
class HFForward:
    """One eager forward pass, reduced to plain numpy for the signal cores."""

    input_ids: np.ndarray  # [seq]
    offsets: list[tuple[int, int]]  # (char_start, char_end) per token
    content_mask: np.ndarray  # [seq] bool; True = content (non-template) token
    attentions: np.ndarray  # [n_layers, n_heads, q, k], rows over k sum to 1


def load_hf_backbone(
    model_name: str = DEFAULT_CPU_MODEL,
    *,
    device: str = "cpu",
    dtype: str = "float32",
) -> tuple[Any, Any]:
    """Load a tiny HF causal LM with **eager** attention + its fast tokenizer.

    Eager attention is mandatory for the same reason as the GPU path: only eager
    exposes the attention pattern (SDPA/FlashAttention do not). Returns
    ``(model, tokenizer)``; the model is in eval mode on ``device``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype=getattr(torch, dtype),
    )
    model.to(device).eval()
    return model, tokenizer


def hf_eager_forward(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    content_char_range: tuple[int, int] | None = None,
    device: str = "cpu",
) -> HFForward:
    """Run one eager forward pass and return attention + the §3.3 content mask.

    ``content_char_range`` restricts content tokens to a char window (e.g.
    ``(0, len(context))`` to exclude the appended question, matching §3.3). The
    returned ``attentions`` is stacked ``[n_layers, n_heads, q, k]`` with the batch
    dim squeezed; each attention row over keys sums to 1 (the gate 11.0d shape
    contract, here on a toy model).
    """
    import torch

    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"][0].tolist()]
    special = enc["special_tokens_mask"][0].tolist()
    mask = content_mask_from_offsets(
        offsets, special_tokens_mask=special, content_char_range=content_char_range
    )

    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            output_attentions=True,
        )

    # out.attentions: tuple of n_layers tensors, each [batch, n_heads, q, k].
    attn = torch.stack(out.attentions, dim=0)[:, 0]  # [n_layers, n_heads, q, k]
    return HFForward(
        input_ids=enc["input_ids"][0].cpu().numpy(),
        offsets=offsets,
        content_mask=mask,
        attentions=attn.to(torch.float32).cpu().numpy(),
    )
