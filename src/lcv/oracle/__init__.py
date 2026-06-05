"""Causal oracle.

The primary oracle is **token masking** ``E(t)`` (paper Definition 2, §8.1):
:mod:`lcv.oracle.masking` (with :mod:`lcv.oracle.ablation` as a legacy alias). The
attribution-patching approximation ``E_hat`` and its fidelity gate live in
:mod:`lcv.oracle.attr_patching` (§8.3); the pinned 150-instance exact-masking
fallback in :mod:`lcv.oracle.adjudication`. The corruption module (§8.2) supplies
the token-swap robustness corruption (and the patching reference), demoted from
the primary oracle. Masking is read as an *upper-bound proxy* for the per-token
information loss KV eviction induces, not the identical operation; the empirical
bridge to eviction is the flip test (§9).
"""
