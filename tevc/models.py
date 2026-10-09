"""1D-CNN classifiers trained inside each recipe evaluation: "TICNN" (D1 default) and "WDCNN" (Gate-0).

The names are registry keys kept for cache compatibility (the key is part of every cached evaluation), not claims
of correspondence to published models: "TICNN" here is a 1-D Inception CNN, NOT the TICNN of Zhang et al. (2018,
training interference); "WDCNN" is WDCNN-style (wide first kernel) with 4 convolution blocks and global pooling,
whereas the original has 5 blocks. Papers must name and specify them by their architecture as written below.
Layers and construction order match the earlier RPST baseline implementations, so seeded runs reproduce the
cached evaluations exactly. Input (N, L) or (N, 1, L); output logits (N, n_classes).
"""
import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    """1D Inception module: 1x1, 1x1-3, 1x1-5 and pool-1x1 branches with out_channels // 4 channels each."""

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
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class TICNN(nn.Module):
    """Transfer Inception CNN: strided wide first convolution, two Inception stages, two-layer head."""

    def __init__(self, signal_length=12000, n_classes=4, dropout=0.3):
        super().__init__()
        self.signal_length = signal_length
        self.n_classes = n_classes
        self.conv_init = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.inception1 = InceptionBlock(32, 128)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.inception2 = InceptionBlock(128, 256)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 256)
        self.relu_fc = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool1(self.inception1(self.conv_init(x)))
        x = self.pool2(self.inception2(x))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu_fc(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class WDCNN(nn.Module):
    """Wide first-layer 1D-CNN: one strided wide convolution, three 3-tap convolutions, two-layer head."""

    def __init__(self, signal_length=12000, n_classes=4, dropout=0.3):
        super().__init__()
        self.signal_length = signal_length
        self.n_classes = n_classes
        self.conv1 = nn.Conv1d(1, 16, kernel_size=64, stride=16, padding=24)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 256)
        self.relu_fc = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu_fc(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


MODELS = {"WDCNN": WDCNN, "TICNN": TICNN}
