"""Spectrum Consistency Regularizer (SCR) Loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpectrumConsistencyRegularizer(nn.Module):
    """
    Spectrum Consistency Regularizer constrains frequency-bin attention weights
    to peak at theoretical fault frequencies.

    Uses KL divergence between predicted attention distribution and physics-based target.
    Dynamically adjusts Gaussian variance based on frequency resolution to account for
    different sampling rates across datasets (CWRU 12kHz, JNU 50kHz, PU 64kHz).
    """

    def __init__(self, sigma: float = 2.0, target_hz_width: float = 5.0):
        super().__init__()
        self.sigma = sigma
        self.target_hz_width = target_hz_width

    def forward(
        self,
        attn_weights: torch.Tensor,
        fault_freq_bins: torch.Tensor,
        window_len: int,
        fs_sampling: float,
    ) -> torch.Tensor:
        """
        Compute SCR loss with dynamic variance normalization.

        Args:
            attn_weights: (B, n_heads, N_tokens+1, N_tokens+1) from transformer layer
            fault_freq_bins: (B, 4) indices of [BPFO, BPFI, BSF, FTF] in frequency domain
            window_len: Window length in samples
            fs_sampling: Sampling frequency in Hz

        Returns:
            scr_loss: Scalar KL divergence loss
        """
        B = attn_weights.shape[0]

        # Compute dynamic sigma based on frequency resolution.
        if isinstance(fs_sampling, torch.Tensor):
            fs_sampling = fs_sampling[0].item()
        freq_resolution = fs_sampling / window_len
        sigma_dynamic = max(1.0, self.target_hz_width / freq_resolution)

        # Handle placeholder attention weights (model returns dummy attention for now)
        if attn_weights.dim() == 2 and attn_weights.shape[1] == 1:
            # Placeholder attention: create uniform distribution over frequency bins
            attn_freq = torch.ones(B, window_len, device=attn_weights.device, dtype=torch.float32) / window_len
        else:
            # Extract attention over frequency tokens (skip [CLS] token column)
            attn_freq = attn_weights[:, :, 0, 1:]  # (B, n_heads, N_tokens)
            attn_freq = attn_freq.mean(dim=1)  # (B, N_tokens) - average over heads
            # Ensure float32 for stability
            attn_freq = attn_freq.float()

        # Normalize to probability distribution (single normalization for stability)
        attn_sum = attn_freq.sum(dim=1, keepdim=True).clamp(min=1e-8)
        attn_freq = attn_freq / attn_sum
        # Clamp to valid probability range to prevent log(0) and numerical issues
        attn_freq = torch.clamp(attn_freq, min=1e-10, max=1.0 - 1e-10)

        # Build target distribution from fault frequency bins
        n_tokens = attn_freq.shape[1]
        target = torch.zeros_like(attn_freq)

        # Safeguard: ensure fault_freq_bins is valid
        fault_freq_bins = fault_freq_bins.to(torch.long)

        # Precompute Gaussian kernels with dynamic sigma for efficiency and stability
        for b in range(B):
            for fault_bin_idx in range(fault_freq_bins.shape[1]):
                fault_bin_val = fault_freq_bins[b, fault_bin_idx]
                fault_bin_int = int(fault_bin_val.item())
                if 0 <= fault_bin_int < n_tokens:
                    bin_distances = torch.arange(n_tokens, device=attn_freq.device, dtype=attn_freq.dtype)
                    distance_sq = (bin_distances - fault_bin_int) ** 2
                    gaussian = torch.exp(-distance_sq / (2 * sigma_dynamic ** 2))
                    target[b] += gaussian

        # Normalize target to probability distribution
        target_sum = target.sum(dim=1, keepdim=True).clamp(min=1e-8)
        target = target / target_sum
        target = torch.clamp(target, min=1e-8, max=1.0)

        # Safeguard: if target is degenerate (all zeros or NaN), use uniform distribution
        if torch.isnan(target).any() or (target.sum(dim=1) < 1e-6).any():
            # Use uniform distribution when target is degenerate
            target = torch.ones_like(target) / n_tokens

        # Final clamp to ensure valid probabilities for both distributions
        attn_freq = torch.clamp(attn_freq, min=1e-10, max=1.0 - 1e-10)
        target = torch.clamp(target, min=1e-10, max=1.0 - 1e-10)

        # Renormalize after final clamp to ensure they're valid probability distributions
        attn_freq = attn_freq / attn_freq.sum(dim=1, keepdim=True).clamp(min=1e-8)
        target = target / target.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # Use numerically stable KL divergence
        # KL(target || attn_freq) = sum(target * log(target / attn_freq))
        # = sum(target * (log(target) - log(attn_freq)))
        kl_loss = (target * (torch.log(target) - torch.log(attn_freq))).sum(dim=1).mean()

        return kl_loss
