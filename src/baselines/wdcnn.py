"""WDCNN: 1D-CNN baseline for bearing fault diagnosis."""
import torch
import torch.nn as nn


class WDCNN(nn.Module):
    """
    Wavelet 1D-CNN (WDCNN) baseline.

    Standard 1D-CNN with learnable 1D convolutions for bearing fault detection.
    Serves as the gold-standard baseline for comparison.
    """

    def __init__(self,
                 signal_length: int = 12000,
                 n_classes: int = 4,
                 dropout: float = 0.3):
        """Initialize WDCNN.

        Args:
            signal_length: Length of input signal
            n_classes: Number of fault classes
            dropout: Dropout rate
        """
        super().__init__()
        self.signal_length = signal_length
        self.n_classes = n_classes

        # Feature extraction: learnable 1D convolutions
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

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 256)
        self.relu_fc = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor, bearing_params=None):
        """Forward pass.

        Args:
            x: Input signal (batch_size, 1, signal_length)
            bearing_params: Bearing parameters (unused, for API compatibility)

        Returns:
            logits: Classification logits (batch_size, n_classes)
            None: No attention maps
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # Classification
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits, None
