"""Localization signals.

Accumulated attention (§5.1), Wu retrieval heads (§5.2), QRHead (§5.3),
transcoder attribution (§4.4), ITI heads (§5.4), Orgad answer-token probe
(§5.5). Each token-substrate signal emits a normalized per-token importance
vector over the content-token mask; each component signal emits a per-head
score (§6, §12).
"""
