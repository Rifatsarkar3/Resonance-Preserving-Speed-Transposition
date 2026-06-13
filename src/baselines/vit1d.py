"""ViT-1D: Vision Transformer adapted for 1D signals."""
import torch
import torch.nn as nn
import math


class ViT1D(nn.Module):
    """Vision Transformer adapted for 1D bearing signals."""

    def __init__(self,
                 signal_length: int = 12000,
                 n_classes: int = 4,
                 dropout: float = 0.3,
                 patch_size: int = 64,
                 d_model: int = 128,
                 nhead: int = 4,
                 nlayers: int = 2):
        super().__init__()
        self.signal_length = signal_length
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.d_model = d_model

        n_patches = signal_length // patch_size
        self.n_patches = n_patches

        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size),
        )

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, 256)
        self.relu_fc = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor, bearing_params=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Patch embedding
        x = self.patch_embed(x)  # (B, d_model, n_patches)
        x = x.transpose(1, 2)    # (B, n_patches, d_model)

        # Add positional encoding
        x = x + self.pos_embed

        # Add class token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches+1, d_model)

        # Transformer
        x = self.transformer(x)

        # Use class token for classification
        x = x[:, 0]  # (B, d_model)

        # Classification
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits, None
