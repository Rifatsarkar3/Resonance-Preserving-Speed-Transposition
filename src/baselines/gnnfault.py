"""GNNFault: Graph Neural Network baseline for bearing fault diagnosis."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """Simple graph convolution layer for signal correlation graphs."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Apply graph convolution.

        Args:
            x: Node features (B, N, in_channels)
            adj: Adjacency matrix (B, N, N) or (N, N)

        Returns:
            x: Updated features (B, N, out_channels)
        """
        # Graph convolution: AXW
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)

        x = torch.matmul(adj, x)  # (B, N, in_channels)
        x = self.linear(x)  # (B, N, out_channels)
        x = self.bn(x)

        return F.relu(x)


class GNNFault(nn.Module):
    """Graph Neural Network for bearing fault diagnosis via signal correlation."""

    def __init__(self,
                 signal_length: int = 12000,
                 n_classes: int = 4,
                 dropout: float = 0.3,
                 n_nodes: int = 128):
        super().__init__()
        self.signal_length = signal_length
        self.n_classes = n_classes
        self.n_nodes = n_nodes
        self.dropout = nn.Dropout(dropout)

        # Signal to node features
        self.signal_embed = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=64, stride=64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(n_nodes),
        )

        # Graph convolution layers
        self.gconv1 = GraphConvLayer(64, 128)
        self.gconv2 = GraphConvLayer(128, 256)

        # Classification head
        self.fc1 = nn.Linear(256 * n_nodes, 512)
        self.relu_fc = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, n_classes)

    def _build_adjacency(self, x: torch.Tensor) -> torch.Tensor:
        """Build adjacency matrix from node features (correlation-based).

        Args:
            x: Node features (B, N, C)

        Returns:
            adj: Adjacency matrix (B, N, N)
        """
        batch_size, n_nodes, _ = x.shape

        # Compute pairwise cosine similarity
        x_norm = F.normalize(x, dim=-1)  # (B, N, C)
        adj = torch.bmm(x_norm, x_norm.transpose(1, 2))  # (B, N, N)

        # Threshold to keep top-k connections
        adj = (adj > 0.5).float()

        # Add self-loops
        eye = torch.eye(n_nodes, device=x.device).unsqueeze(0)
        adj = adj + eye

        # Normalize
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)

        return adj

    def forward(self, x: torch.Tensor, bearing_params=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Embed signal to node features
        x = self.signal_embed(x)  # (B, 64, n_nodes)
        x = x.transpose(1, 2)     # (B, n_nodes, 64)

        # Build adjacency matrix
        adj = self._build_adjacency(x)

        # Graph convolution
        x = self.gconv1(x, adj)
        x = self.dropout(x)

        x = self.gconv2(x, adj)
        x = self.dropout(x)

        # Global average pooling
        x = x.mean(dim=1)  # (B, 256)

        # Classification
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits, None
