"""Model backbone, chat-template content-mask, and SAE/transcoder loading.

Addendum §2 (instrumentation), §3.3 (chat template), §4 (SAEs), §12. The
torch/TransformerLens-dependent code here is reached only in the GPU phase;
import torch lazily so the package stays CPU-clean.
"""
