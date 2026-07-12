"""TICNN: Transfer Inception CNN baseline."""
import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    """Inception module adapted for 1D signals."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class TICNN(nn.Module):
    """Transfer Inception CNN for bearing fault diagnosis."""

    def __init__(self,
                 signal_length: int = 12000,
                 n_classes: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.signal_length = signal_length
        self.n_classes = n_classes

        # Initial convolution
        self.conv_init = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        # Inception blocks
        self.inception1 = InceptionBlock(32, 128)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.inception2 = InceptionBlock(128, 256)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 256)
        self.relu_fc = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor, bearing_params=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv_init(x)
        x = self.inception1(x)
        x = self.pool1(x)

        x = self.inception2(x)
        x = self.pool2(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits, None
