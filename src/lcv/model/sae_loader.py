"""SAE / transcoder loading by Llama-Scope / Gemma-Scope name (§4).

The naming convention ``L[Layer][Position]-[Expansion]x`` (§4.1) is parsed here,
and the load-bearing §4.2 rule is enforced structurally: **attention-output SAEs
(`LXA`) are never used** (the Llama-Scope authors warn they are mostly dead
features), so :func:`parse_sae_name` refuses them rather than letting an agent
guess from the pattern and quietly poison the component substrate.

Name parsing and the reconstruction-error metric are pure-CPU and tested; the
actual checkpoint load needs ``sae_lens`` + a GPU and is stubbed.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # hints only; sae_lens/torch live in the GPU phase
    from sae_lens import SAE

# L<layer><position>-<expansion>x  (§4.1). Position A is matched so it can be
# rejected with a clear §4.2 message rather than an "unparseable" error.
_NAME_RE = re.compile(r"^L(\d{1,2})(R|A|M|TC)-(8|32)x$")

# Expansion factor -> feature count: 8x = 32K, 32x = 128K (§4.1).
_FEATURES: dict[str, int] = {"8": 32_768, "32": 131_072}

# Token-attribution substrate uses transcoders (LXTC), 8x width default (§4.3).
DEFAULT_POSITION = "TC"
DEFAULT_EXPANSION = 8
# Three depths ~25/50/65% of 32 layers (§4.3); a §13.3 human checkpoint.
DEFAULT_TRANSCODER_LAYERS: tuple[int, ...] = (8, 16, 21)

# Llama-Scope SAEs/transcoders are trained with TopK (k=50, the `top_k` field of
# every hyperparams.json) and post-processed to a scalar-threshold JumpReLU, so the
# per-token active-feature count (L0) sits at ~50. Gate 11.0c checks the *measured*
# mean L0 against this; a wildly off L0 means the threshold conversion (below) or the
# hook point is wrong. The band is generous (a TopK=50 model relaxed to JumpReLU
# drifts as latents fall under/over the single threshold); tighten it against the
# first real checkpoint measurement.
EXPECTED_L0 = 50
L0_ACCEPT_RANGE: tuple[float, float] = (25.0, 100.0)

# --------------------------------------------------------------------------- #
# Verified checkpoint provenance (HF API + SAELens registry, 2026-06-07)
# --------------------------------------------------------------------------- #
# Llama-Scope ships one HF repo per (position, expansion); the umbrella names
# `fnlp/Llama-Scope` / `OpenMOSS-Team/Llama-Scope` are *not* the checkpoint repos.
# The real per-position repos resolve (`fnlp/*` 307-redirects to the canonical
# `OpenMOSS-Team/*`, identical content), each holding one subdirectory per layer:
#   Llama3_1-8B-Base-L<layer><POS>-<exp>x/checkpoints/final.safetensors
#                                        /hyperparams.json
#                                        /lm_config.json
LLAMA_SCOPE_REPOS = ("fnlp/Llama-Scope", "OpenMOSS-Team/Llama-Scope")  # umbrella, not loadable
LLAMA_SCOPE_REPO_TEMPLATE = "fnlp/Llama3_1-8B-Base-LX{position}-{expansion}x"
LLAMA_SCOPE_LAYER_SUBDIR_TEMPLATE = "Llama3_1-8B-Base-L{layer}{position}-{expansion}x"

# SAELens registry status (`sae_lens/pretrained_saes.yaml`, conversion_func
# "llama_scope"): only R and M positions are registered and load via
# `SAE.from_pretrained(release, sae_id)`. The transcoder (TC) safetensors exist
# on HF but are **absent from the SAELens registry** (verified: zero `*TC*`
# entries), so TC must be loaded directly from the per-layer `final.safetensors`
# -- `SAE.from_pretrained("llama_scope_lxtc_*", ...)` would miss the registry and
# is a clean-but-wrong trap. (LXA is registered upstream but banned here, §4.2.)
LLAMA_SCOPE_SAELENS_RELEASES: dict[tuple[str, int], str] = {
    ("R", 8): "llama_scope_lxr_8x",
    ("R", 32): "llama_scope_lxr_32x",
    ("M", 8): "llama_scope_lxm_8x",
    ("M", 32): "llama_scope_lxm_32x",
}


@dataclass(frozen=True, slots=True)
class SAEName:
    """A parsed Llama-Scope / Gemma-Scope checkpoint name (§4.1)."""

    layer: int
    position: str  # "R" | "M" | "TC"  (never "A", §4.2)
    expansion: int  # 8 | 32
    n_features: int

    @property
    def is_transcoder(self) -> bool:
        return self.position == "TC"


def parse_sae_name(name: str) -> SAEName:
    """Parse ``L<layer><R|M|TC>-<8|32>x`` and enforce the §4.2 ``LXA`` ban.

    Raises ``ValueError`` for attention-output SAEs (``LXA``), out-of-range
    layers, or names that do not match the convention. This is the structural
    guard that keeps dead-feature attention SAEs out of every substrate.
    """
    m = _NAME_RE.match(name)
    if m is None:
        raise ValueError(f"unparseable SAE name {name!r} (expected L<layer><R|M|TC>-<8|32>x, §4.1)")
    layer, position, exp = int(m.group(1)), m.group(2), m.group(3)
    if position == "A":
        raise ValueError(
            f"attention-output SAEs are excluded on the authors' dead-feature "
            f"warning (§4.2): {name!r}"
        )
    if not 0 <= layer <= 31:
        raise ValueError(f"layer {layer} out of range 0..31 ({name!r})")
    return SAEName(layer=layer, position=position, expansion=int(exp), n_features=_FEATURES[exp])


def llama_scope_checkpoint_ref(name: str) -> tuple[str, str]:
    """Return ``(repo_id, layer_subdir)`` for a Llama-Scope name (verified provenance).

    The checkpoint is ``<repo_id>/<layer_subdir>/checkpoints/final.safetensors``.
    This is the direct-download path the **transcoder** substrate must use (TC is
    not in the SAELens registry; see :func:`llama_scope_sae_lens_release`); R/M may
    use it too. Rejects ``LXA`` via :func:`parse_sae_name` (§4.2).
    """
    p = parse_sae_name(name)
    repo = LLAMA_SCOPE_REPO_TEMPLATE.format(position=p.position, expansion=p.expansion)
    subdir = LLAMA_SCOPE_LAYER_SUBDIR_TEMPLATE.format(
        layer=p.layer, position=p.position, expansion=p.expansion
    )
    return repo, subdir


def llama_scope_sae_lens_release(name: str) -> tuple[str, str]:
    """Return the SAELens ``(release, sae_id)`` for an R/M Llama-Scope name.

    Resolves the ``conversion_func: llama_scope`` registry entry so a GPU-phase
    ``SAE.from_pretrained(release, sae_id)`` loads the right checkpoint. Raises
    ``ValueError`` for transcoders (``TC``): they are **absent from the SAELens
    registry** (verified 2026-06-07), so they cannot be resolved this way and must
    be loaded from the per-layer ``final.safetensors`` via
    :func:`llama_scope_checkpoint_ref`. Rejects ``LXA`` (§4.2).
    """
    p = parse_sae_name(name)
    release = LLAMA_SCOPE_SAELENS_RELEASES.get((p.position, p.expansion))
    if release is None:
        raise ValueError(
            f"{name!r}: position {p.position!r} has no SAELens release (registry has "
            "R/M only; transcoders load directly via llama_scope_checkpoint_ref, §4.3)"
        )
    sae_id = f"l{p.layer}{p.position.lower()}_{p.expansion}x"
    return release, sae_id


# --------------------------------------------------------------------------- #
# Llama-Scope checkpoint -> inference SAE/transcoder conversion (the §4.3 trap)
# --------------------------------------------------------------------------- #
# Llama-Scope ships raw `final.safetensors` + `hyperparams.json`. Turning those
# into a runnable JumpReLU module is the canonical clean-but-wrong trap. SAELens
# has a loader for the R/M *SAEs* (`get_llama_scope_config_from_hf`), which folds a
# single `norm_scaling_factor = sqrt(d_model) / dataset_average_activation_norm["in"]`
# into the encoder and scales the scalar JumpReLU threshold to match. It has **no**
# Llama-Scope *transcoder* loader (verified against `pretrained_sae_loaders.py`,
# 2026-06-07): the registry's `llama_scope` entry builds an SAE that reads only the
# `["in"]` norm and reconstructs the same hook. A transcoder reconstructs a
# *different* activation (MLP output, `dataset_average_activation_norm["out"]`),
# whose dataset norm is ~19x smaller than the input's, so the decoder must be
# un-normalised by `nsf_out`, not `nsf_in`. Reusing the SAE loader inflates every
# transcoder output by `nsf_in / nsf_out` (~18.9x at layer 8). These pure-CPU
# functions pin the corrected two-factor conversion so the GPU loader can't drift.


@dataclass(frozen=True, slots=True)
class LlamaScopeHyperparams:
    """The load-bearing fields of a Llama-Scope ``hyperparams.json`` (§4.3).

    Captured verbatim from the checkpoint so the TopK->JumpReLU + dataset-norm
    conversion is driven by the *file*, never a guessed constant.
    ``avg_norm_in``/``avg_norm_out`` differ only for transcoders (input = pre-MLP
    LayerNorm output, norm ~= sqrt(d_model); output = MLP output, much smaller); for
    R/M SAEs they are equal and the conversion collapses to the SAELens single
    factor.
    """

    d_model: int
    d_sae: int
    hook_point_in: str
    hook_point_out: str
    jump_relu_threshold: float  # scalar, all features (unlike Gemma Scope's vector)
    avg_norm_in: float
    avg_norm_out: float
    top_k: int
    act_fn: str

    @property
    def is_transcoder(self) -> bool:
        """A transcoder reconstructs a different hook than it reads (in != out)."""
        return self.hook_point_in != self.hook_point_out


def parse_llama_scope_hyperparams(raw: Mapping[str, Any]) -> LlamaScopeHyperparams:
    """Parse a Llama-Scope ``hyperparams.json`` dict (pure-CPU, no HF / torch).

    Pulls only the fields the conversion needs and validates them, so a malformed or
    unexpected checkpoint fails here rather than silently mis-converting. The
    ``dataset_average_activation_norm`` must carry **both** ``in`` and ``out`` -- the
    two scales a transcoder spans, and the exact key the upstream SAE loader ignores.
    """
    try:
        norms = raw["dataset_average_activation_norm"]
        hp = LlamaScopeHyperparams(
            d_model=int(raw["d_model"]),
            d_sae=int(raw["d_sae"]),
            hook_point_in=str(raw["hook_point_in"]),
            hook_point_out=str(raw["hook_point_out"]),
            jump_relu_threshold=float(raw["jump_relu_threshold"]),
            avg_norm_in=float(norms["in"]),
            avg_norm_out=float(norms["out"]),
            top_k=int(raw["top_k"]),
            act_fn=str(raw.get("act_fn", "jumprelu")),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed Llama-Scope hyperparams: missing/invalid {exc}") from exc
    if hp.act_fn != "jumprelu":
        raise ValueError(f"expected a JumpReLU checkpoint, got act_fn={hp.act_fn!r} (§4.3)")
    if hp.d_model <= 0 or hp.d_sae <= 0:
        raise ValueError(f"non-positive dims d_model={hp.d_model} d_sae={hp.d_sae}")
    if hp.avg_norm_in <= 0.0 or hp.avg_norm_out <= 0.0:
        raise ValueError(
            f"non-positive dataset activation norm in={hp.avg_norm_in} out={hp.avg_norm_out}"
        )
    return hp


@dataclass(frozen=True, slots=True)
class LlamaScopeConversion:
    """Scalars that turn a raw Llama-Scope checkpoint into an inference SAE / TC (§4.3).

    ``nsf_in`` folds into the encoder (``W_enc *= nsf_in``) and the threshold, exactly
    as SAELens does; ``nsf_out`` un-normalises the decoder. For an R/M SAE the two are
    equal, so this reproduces SAELens bit-for-bit; for a transcoder they diverge and
    :attr:`decoder_inflation` is the factor by which the SAELens single-factor loader
    would inflate the output -- the clean-but-wrong error this conversion removes.
    """

    nsf_in: float
    nsf_out: float
    threshold_scaled: float

    @property
    def decoder_inflation(self) -> float:
        """``nsf_out / nsf_in``: how much a single-factor (SAELens) fold over-scales TC output.

        The correct decoder divides by ``nsf_out``; the SAELens single-factor loader
        divides by ``nsf_in``, so its reconstruction is ``nsf_out / nsf_in`` times too
        large (~18.9x at layer 8, 1.0 -- i.e. no error -- for an R/M SAE).
        """
        return self.nsf_out / self.nsf_in


def llama_scope_conversion(hp: LlamaScopeHyperparams) -> LlamaScopeConversion:
    """Compute the (input, output) norm-scaling factors and the scaled JumpReLU threshold.

    ``nsf = sqrt(d_model) / dataset_average_activation_norm`` -- the SAELens input
    formula (``get_llama_scope_config_from_hf``), applied **separately** to the output
    scale so a transcoder un-normalises with its own ``["out"]`` norm. The threshold is
    multiplied by ``nsf_in`` to track the encoder fold (the SAELens loader does the same
    so ``sae.fold_activation_norm_scaling_factor`` stays consistent). ``d_out`` equals
    ``d_model`` for the Llama MLP transcoders (decoder reconstructs a d_model vector).
    """
    nsf_in = math.sqrt(hp.d_model) / hp.avg_norm_in
    nsf_out = math.sqrt(hp.d_model) / hp.avg_norm_out
    return LlamaScopeConversion(
        nsf_in=nsf_in,
        nsf_out=nsf_out,
        threshold_scaled=hp.jump_relu_threshold * nsf_in,
    )


# Safetensors tensor names in a Llama-Scope checkpoint, mapped to SAELens weight
# orientation: encoder.weight [d_sae, d_in] -> W_enc [d_in, d_sae];
# decoder.weight [d_out, d_sae] -> W_dec [d_sae, d_out]; biases are 1-D, copied as-is.
LLAMA_SCOPE_SAFETENSOR_KEYS = ("encoder.weight", "encoder.bias", "decoder.weight", "decoder.bias")


def fold_llama_scope_checkpoint(
    raw: Mapping[str, np.ndarray], conv: LlamaScopeConversion
) -> dict[str, np.ndarray]:
    """Map the four safetensors tensors to inference weights and fold the norm scaling (§4.3).

    Pure-CPU mirror of what the GPU loader must do to the checkpoint tensors, so the
    conversion is unit-tested off the GPU. Encoder side matches SAELens
    (``W_enc *= nsf_in``); the decoder uses ``nsf_out`` (the transcoder correction).
    Returns ``W_enc``/``b_enc``/``W_dec``/``b_dec``/``threshold`` ready for
    :func:`transcoder_reconstruct`.
    """
    missing = [k for k in LLAMA_SCOPE_SAFETENSOR_KEYS if k not in raw]
    if missing:
        raise ValueError(f"checkpoint missing tensors {missing} (have {sorted(raw)})")
    w_enc = np.asarray(raw["encoder.weight"], dtype=float).T * conv.nsf_in
    b_enc = np.asarray(raw["encoder.bias"], dtype=float)
    w_dec = np.asarray(raw["decoder.weight"], dtype=float).T / conv.nsf_out
    b_dec = np.asarray(raw["decoder.bias"], dtype=float) / conv.nsf_out
    threshold = np.full(b_enc.shape[0], conv.threshold_scaled, dtype=float)
    return {"W_enc": w_enc, "b_enc": b_enc, "W_dec": w_dec, "b_dec": b_dec, "threshold": threshold}


def transcoder_reconstruct(
    activations: np.ndarray, folded: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """JumpReLU encode + decode on raw activations (pure-CPU reference, §4.3).

    ``activations`` are the raw values at ``hook_point_in`` (``[..., d_in]``). Returns
    ``(reconstruction, feature_acts)``: the reconstruction is in the raw
    ``hook_point_out`` space (a transcoder's MLP output), ``feature_acts`` the sparse
    code. The activation is ``relu(pre) * (pre > threshold)`` with the scalar folded
    threshold -- the exact JumpReLU SAELens applies (strict ``>``).
    """
    x = np.asarray(activations, dtype=float)
    pre = x @ folded["W_enc"] + folded["b_enc"]
    feature_acts = np.where(pre > folded["threshold"], np.maximum(pre, 0.0), 0.0)
    recon = feature_acts @ folded["W_dec"] + folded["b_dec"]
    return recon, feature_acts


def mean_l0(feature_acts: np.ndarray) -> float:
    """Mean number of active (non-zero) features per token -- the gate 11.0c L0."""
    f = np.asarray(feature_acts, dtype=float)
    if f.ndim == 1:
        f = f[None, :]
    return float(np.count_nonzero(f, axis=-1).mean())


def l0_within_expected(measured_l0: float) -> bool:
    """Whether a measured mean L0 sits in the post-TopK=50 JumpReLU band (gate 11.0c)."""
    lo, hi = L0_ACCEPT_RANGE
    return bool(lo <= measured_l0 <= hi)


def reconstruction_error(activations: np.ndarray, reconstruction: np.ndarray) -> float:
    """``1 - explained_variance`` of an SAE reconstruction (gate 11.0c).

    Pure-CPU array math (no torch): given clean activations and their SAE
    reconstruction, returns the fraction of variance left unexplained. Gate 11.0c
    requires this to fall within the layer's published Llama-Scope range; a
    base-trained SAE on Instruct activations (§3.3) will sit at the higher end.
    NaN if the activations have zero variance.
    """
    a = np.asarray(activations, dtype=float)
    r = np.asarray(reconstruction, dtype=float)
    if a.shape != r.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {r.shape}")
    total = float(np.sum((a - a.mean(axis=0)) ** 2))
    if total == 0.0:
        return float("nan")
    return float(np.sum((a - r) ** 2) / total)


def load_sae(name: str, *, device: str = "cuda", release: str | None = None) -> SAE:
    """Load an SAE/transcoder checkpoint by name at its hook point (§4.1, gate 11.0c).

    Parses ``name`` (rejecting ``LXA`` via :func:`parse_sae_name`), then resolves
    by position (verified 2026-06-07):

    * **R/M** go through the SAELens registry via :func:`llama_scope_sae_lens_release`
      (``SAE.from_pretrained(release, sae_id)``); SAELens applies its own single-factor
      norm fold, correct because input and output are the same hook.
    * **Transcoders (TC)** are absent from the SAELens registry, so they load directly
      from the per-layer ``final.safetensors`` (:func:`llama_scope_checkpoint_ref`) and
      its ``hyperparams.json`` (:func:`parse_llama_scope_hyperparams`). The GPU loader
      must then apply :func:`llama_scope_conversion` + :func:`fold_llama_scope_checkpoint`
      -- the **two-factor** fold (``nsf_in`` into the encoder/threshold, ``nsf_out`` into
      the decoder). Reusing the SAELens single-factor path here is the clean-but-wrong
      trap: it leaves TC output inflated by ``nsf_out / nsf_in`` (~18.9x at layer 8).

    Gate 11.0c then verifies the checkpoint loads at its named hook point with
    reconstruction error in range (:func:`reconstruction_error`) and the expected
    JumpReLU/TopK L0 (:func:`l0_within_expected`); a failure means the wrong hook
    point, checkpoint, or conversion.
    """
    parse_sae_name(name)  # refuse LXA / malformed before touching the GPU
    raise NotImplementedError("requires GPU phase")
