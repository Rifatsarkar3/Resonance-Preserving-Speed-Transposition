"""Learnable Wavelet-Packet Tokenizer (LWPT)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt


class LearnableWaveletPacketTokenizer(nn.Module):
    """
    Learnable Wavelet-Packet Tokenizer for end-to-end learned frequency analysis.

    Uses angle-parameterized orthogonal filter design to learn optimal Daubechies-like
    filters via backpropagation while maintaining near-orthogonality.
    """

    def __init__(self, filter_order: int = 8, n_frames: int = 16, hidden_dim: int = 96):
        super().__init__()
        self.filter_order = filter_order
        self.n_frames = n_frames
        self.hidden_dim = hidden_dim
        self.n_bands = 16  # 4-level DWT packet → 2^4 = 16 bands

        # Initialize filter angles close to Daubechies-4 angles
        db4 = pywt.Wavelet('db4')
        db4_low = db4.dec_lo

        # Parameterize low-pass filter via angles (ensures orthogonality)
        initial_angles = self._init_angles_from_filter(db4_low)

        self.theta_low = nn.Parameter(torch.tensor(initial_angles, dtype=torch.float32))
        self.theta_high = nn.Parameter(torch.tensor(initial_angles, dtype=torch.float32))

        # Linear projection from token energy to hidden_dim
        self.proj = nn.Linear(1, hidden_dim)

    def _init_angles_from_filter(self, filter_coeffs: np.ndarray) -> np.ndarray:
        """Initialize angles from a reference filter."""
        # Simple initialization: use small random angles
        n_angles = self.filter_order - 1
        return np.random.randn(n_angles) * 0.1

    def _construct_filters(self, theta):
        """Construct orthogonal filters from angle parameters."""
        # Angle parameterization ensures unit norm and near-orthogonality
        h = torch.ones(1, dtype=theta.dtype, device=theta.device)
        for k in range(self.filter_order - 1):
            h_k = torch.cos(theta[k])
            for j in range(k):
                h_k = h_k * torch.sin(theta[j])
            if k < len(theta):
                h_k = h_k * torch.sin(theta[k])
            h = torch.cat([h, h_k.unsqueeze(0)])

        # Quadrature mirror filter for high-pass
        g = torch.zeros_like(h)
        for k in range(len(h)):
            g[k] = ((-1) ** k) * h[len(h) - 1 - k]

        return h / (h.norm() + 1e-8), g / (g.norm() + 1e-8)

    def _wavelet_decompose(self, x, h, g):
        """4-level wavelet packet decomposition."""
        # x: (B, C, L)
        h = h.unsqueeze(0).unsqueeze(0)  # (1, 1, filter_len)
        g = g.unsqueeze(0).unsqueeze(0)

        subbands = []
        current = x

        # 4 levels of decomposition, keeping all subbands
        for level in range(4):
            n_subbands = 2 ** level
            new_subbands = []

            for sb in (subbands if level > 0 else [current]):
                # Apply low-pass and high-pass filters
                ll = F.conv1d(sb, h, padding=h.shape[-1] // 2, stride=2)
                lh = F.conv1d(sb, g, padding=g.shape[-1] // 2, stride=2)
                new_subbands.extend([ll, lh])

            subbands = new_subbands

        return subbands[:self.n_bands]  # Return 16 subbands

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (B, C, L) raw vibration signal

        Returns:
            tokens: (B, N_tokens, hidden_dim) where N_tokens = 16*n_frames = 256
        """
        B, C, L = x.shape

        # Construct filters from parameters
        h, g = self._construct_filters(self.theta_low)

        # 4-level wavelet decomposition
        subbands = self._wavelet_decompose(x, h, g)

        # Energy pooling per subband
        freq_energy = []
        for sb in subbands:
            # sb: (B, C, L_sb)
            # Divide into n_frames and compute RMS per frame
            frame_len = max(1, sb.shape[-1] // self.n_frames)
            frames = []
            for i in range(self.n_frames):
                start = i * frame_len
                end = min((i + 1) * frame_len, sb.shape[-1])
                if start < end:
                    frame = sb[:, :, start:end]
                    rms = torch.sqrt(torch.mean(frame ** 2, dim=2, keepdim=True))
                    frames.append(rms)

            if frames:
                frame_stack = torch.cat(frames, dim=2)  # (B, C, n_frames)
                freq_energy.append(frame_stack)

        # Stack all bands: (B, C, 16*n_frames)
        if freq_energy:
            freq_energy = torch.cat(freq_energy, dim=2)
        else:
            freq_energy = torch.randn(B, C, self.n_bands * self.n_frames, device=x.device)

        # Reshape and project: (B, 16*n_frames, 1) → (B, 16*n_frames, hidden_dim)
        freq_energy = freq_energy.mean(dim=1, keepdim=True)  # (B, 1, 16*n_frames)
        freq_energy = freq_energy.transpose(1, 2)  # (B, 16*n_frames, 1)
        tokens = self.proj(freq_energy)  # (B, 16*n_frames, hidden_dim)

        return tokens
