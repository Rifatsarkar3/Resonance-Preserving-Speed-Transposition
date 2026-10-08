"""Label-free fidelity perception: which features does the synthesiser reproduce faithfully?

Compares each feature's marginal distribution in the synthesised target data with real, UNLABELLED windows
from the new environment. No target labels are used, so the screen is available to the agent online.
"""
import numpy as np


def marginal_smd(X_synth, X_real):
    """Per-feature standardised mean difference of the two marginal distributions."""
    pooled = np.sqrt(0.5 * (X_synth.var(axis=0) + X_real.var(axis=0))) + 1e-8
    return np.abs(X_synth.mean(axis=0) - X_real.mean(axis=0)) / pooled


def fidelity_screen(X_synth, X_unlabelled, tau=0.5, min_keep=20):
    """Allowed-feature mask: SMD <= tau; if fewer than min_keep pass, the min_keep most faithful features."""
    smd = marginal_smd(X_synth, X_unlabelled)
    allowed = smd <= tau
    if allowed.sum() < min_keep:
        allowed = np.zeros_like(allowed)
        allowed[np.argsort(smd)[:min_keep]] = True
    return allowed, smd
