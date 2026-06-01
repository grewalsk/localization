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

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

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

# Checkpoint homes (§4.1); confirm which SAELens resolves before relying on either.
LLAMA_SCOPE_REPOS = ("fnlp/Llama-Scope", "OpenMOSS-Team/Llama-Scope")


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

    Parses ``name`` (rejecting ``LXA`` via :func:`parse_sae_name`), resolves the
    SAELens release/hook point (``fnlp/Llama-Scope`` vs ``OpenMOSS-Team`` per
    §4.1), and loads onto ``device``. Gate 11.0c then verifies the checkpoint
    loads at its named hook point with reconstruction error in range and the
    expected JumpReLU/TopK L0; a failure means the wrong hook point or checkpoint.
    """
    parse_sae_name(name)  # refuse LXA / malformed before touching the GPU
    raise NotImplementedError("requires GPU phase")
