"""PhysFormer: Physics-Informed Transformer baseline."""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)


class PhysFormer(nn.Module):
    """Physics-Informed Transformer baseline for bearing fault diagnosis."""

    def __init__(self,
                 signal_length: int = 12000,
                 n_classes: int = 4,
                 dropout: float = 0.3,
                 d_model: int = 64,
                 nhead: int = 4,
                 nlayers: int = 2):
        super().__init__()
        self.signal_length = signal_length
        self.n_classes = n_classes
        self.d_model = d_model

        # Input embedding
        self.conv_embed = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=64),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.linear_embed = nn.Linear(32, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=200)

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
        self.fc1 = nn.Linear(d_model, 128)
        self.relu_fc = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor, bearing_params=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Embed
        x = self.conv_embed(x)  # (B, 32, L)
        x = x.transpose(1, 2)   # (B, L, 32)
        x = self.linear_embed(x)  # (B, L, d_model)

        # Positional encoding
        x = self.pos_encoder(x)

        # Transformer
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)  # (B, d_model)

        # Classification
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits, None
