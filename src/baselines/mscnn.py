"""MSCNN: Multi-Scale CNN baseline."""
import torch
import torch.nn as nn


class MultiScaleBlock(nn.Module):
    """Multi-scale feature extraction block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.scale2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.scale3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        return torch.cat([s1, s2, s3], dim=1)


class MSCNN(nn.Module):
    """Multi-Scale CNN for bearing fault diagnosis."""

    def __init__(self,
                 signal_length: int = 12000,
                 n_classes: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.signal_length = signal_length
        self.n_classes = n_classes

        # Initial conv
        self.conv_init = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        # Multi-scale blocks
        self.ms_block1 = MultiScaleBlock(32, 32)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.ms_block2 = MultiScaleBlock(96, 64)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(192, 256)
        self.relu_fc = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor, bearing_params=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv_init(x)
        x = self.ms_block1(x)
        x = self.pool1(x)

        x = self.ms_block2(x)
        x = self.pool2(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits, None
