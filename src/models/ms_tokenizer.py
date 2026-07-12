"""Multi-Scale Inception Tokenizer: amplitude-invariant CNN frontend for WaPIGT."""
import torch
import torch.nn as nn


class MultiScaleTokenizer(nn.Module):
    """
    Two-stage inception tokenizer for condition-invariant signal tokenization.

    Stage 1 (high-res, ~750 steps): captures fine-grained fault impulses.
    Stage 2 (mid-res, ~375 steps): captures slower periodicity and envelope patterns
    that are more robust to shaft-speed changes.

    Input instance normalization removes amplitude differences between operating
    conditions (speed/load), the root cause of LWPT's failure on cross-condition tasks.
    """

    def __init__(self, hidden_dim: int = 96, n_tokens: int = 256):
        super().__init__()
        self.n_tokens = n_tokens

        # Wide stem: k=64 stride=16 matches TICNN/MSCNN proven frontend
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )

        # Stage-1 inception (32→128 ch, operates on ~750 steps)
        self.branch_k3  = self._branch(32, 32, 3)
        self.branch_k7  = self._branch(32, 32, 7)
        self.branch_k15 = self._branch(32, 32, 15)
        self.branch_k31 = self._branch(32, 32, 31)

        # Spatial reduction between inception stages
        self.stage2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Stage-2 inception (128→128 ch, operates on ~375 steps)
        self.inc2_k3  = self._branch(128, 32, 3)
        self.inc2_k7  = self._branch(128, 32, 7)
        self.inc2_k15 = self._branch(128, 32, 15)
        self.inc2_k31 = self._branch(128, 32, 31)
        self.merge2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )

        # Pool to exactly n_tokens then project to hidden_dim
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(n_tokens),
            nn.Conv1d(128, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

    @staticmethod
    def _branch(in_ch: int, out_ch: int, k: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Instance normalization per sample — amplitude invariant across conditions
        mu    = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True).clamp(min=1e-8)
        x = (x - mu) / sigma

        # Ensure (B, 1, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)

        x = self.stem(x)                                        # (B, 32,  ~750)

        # Stage-1 inception
        x = torch.cat([
            self.branch_k3(x), self.branch_k7(x),
            self.branch_k15(x), self.branch_k31(x),
        ], dim=1)                                               # (B, 128, ~750)

        x = self.stage2(x)                                      # (B, 128, ~375)

        # Stage-2 inception — deeper processing at reduced resolution
        x = self.merge2(torch.cat([
            self.inc2_k3(x), self.inc2_k7(x),
            self.inc2_k15(x), self.inc2_k31(x),
        ], dim=1))                                              # (B, 128, ~375)

        x = self.proj(x)                                        # (B, hidden_dim, 256)

        return x.transpose(1, 2)                                # (B, n_tokens, hidden_dim)
