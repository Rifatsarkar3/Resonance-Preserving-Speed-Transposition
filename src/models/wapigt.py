"""WaPIGT: Multi-Scale Physics-Informed Graph Transformer."""
import torch
import torch.nn as nn
import math
from src.models.ms_tokenizer import MultiScaleTokenizer
from src.models.piffg import PhysicsInformedFaultFrequencyGraph


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, max_len: int = 1000, hidden_dim: int = 96):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        if hidden_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """x: (B, seq_len, hidden_dim)"""
        return self.pe[:, : x.shape[1], :]


class TransformerEncoder(nn.Module):
    """Transformer encoder with graph bias injection and attention capture."""

    def __init__(
        self,
        hidden_dim: int = 96,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_dim: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads

        # Create individual encoder layers to allow graph bias injection and attention capture
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                activation='relu',
                batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, graph_proj_list=None, graph_embed=None, src_key_padding_mask=None):
        """
        Forward pass with optional graph bias injection and attention capture.

        Args:
            x: (B, seq_len, hidden_dim)
            graph_proj_list: List[nn.Module] of graph projection layers
            graph_embed: (B, hidden_dim) graph embedding from PIFFG
            src_key_padding_mask: (B, seq_len) optional padding mask

        Returns:
            output: (B, seq_len, hidden_dim) transformed sequence
            attn_layer2: (B, n_heads, seq_len, seq_len) attention from layer 2
        """
        output = x
        attn_layer2 = None

        for layer_idx, layer in enumerate(self.layers):
            # Graph bias injection at each layer
            if graph_proj_list is not None and graph_embed is not None:
                graph_bias = graph_proj_list[layer_idx](graph_embed)  # (B, hidden_dim)
                graph_bias = graph_bias.unsqueeze(1)  # (B, 1, hidden_dim)
                output = output + graph_bias

            # Forward through transformer layer
            # We need to manually extract attention for layer 2
            if layer_idx == 1:  # Layer 2 (0-indexed)
                # Create a modified forward to capture attention
                # PyTorch TransformerEncoderLayer doesn't expose attention easily,
                # so we access the attention module directly
                self_attn = layer.self_attn

                # Manual forward through the layer to capture attention
                # This mimics what TransformerEncoderLayer does internally
                norm_output = layer.norm1(output)
                attn_output, attn_weights = self_attn(
                    norm_output, norm_output, norm_output,
                    need_weights=True,
                    average_attn_weights=False
                )
                attn_layer2 = attn_weights  # (B, n_heads, seq_len, seq_len)
                output = output + layer.dropout1(attn_output)

                # FFN part
                norm_output = layer.norm2(output)
                ffn_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(norm_output))))
                output = output + layer.dropout2(ffn_output)
            else:
                # Standard layer forward
                output = layer(output, src_key_padding_mask=src_key_padding_mask)

        output = self.norm(output)

        # If attention wasn't captured (no layer 2), create a dummy one
        if attn_layer2 is None:
            B, seq_len, _ = output.shape
            attn_layer2 = torch.ones(B, self.n_heads, seq_len, seq_len, device=output.device)

        return output, attn_layer2


class WaPIGT(nn.Module):
    """
    Multi-Scale Physics-Informed Graph Transformer (WaPIGT-MS).

    Components:
    1. MultiScaleTokenizer: amplitude-invariant inception CNN tokenizer (256 tokens)
    2. PIFFG: Physics-informed fault-frequency graph
    3. Transformer: With graph bias injection and attention capture

    filter_order and n_frames are kept for API backward compatibility but unused.
    """

    def __init__(
        self,
        n_classes: int = 4,
        hidden_dim: int = 96,
        n_encoder_layers: int = 4,
        n_heads: int = 8,
        mlp_dim: int = 384,
        dropout: float = 0.1,
        n_gat_heads: int = 4,
        gat_dropout: float = 0.2,
        filter_order: int = 8,   # unused — kept for backward compat
        n_frames: int = 16,      # unused — kept for backward compat
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # Component 1: Multi-Scale Tokenizer (replaces LWPT)
        self.tokenizer = MultiScaleTokenizer(hidden_dim=hidden_dim, n_tokens=256)

        # Component 2: PIFFG
        self.piffg = PhysicsInformedFaultFrequencyGraph(
            hidden_dim=hidden_dim,
            n_gat_heads=n_gat_heads,
            dropout=gat_dropout,
        )

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(max_len=257, hidden_dim=hidden_dim)

        # Graph projection layers (one per encoder layer)
        self.graph_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_encoder_layers)
        ])

        # Transformer encoder
        self.encoder = TransformerEncoder(
            hidden_dim=hidden_dim,
            n_layers=n_encoder_layers,
            n_heads=n_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        # Classification head
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, bearing_params_list, fs_sampling: float = 64000.0):
        """
        Forward pass with graph bias injection and attention capture.

        Args:
            x: (B, C, L) raw vibration signal
            bearing_params_list: List[Dict] bearing parameters
            fs_sampling: Sampling frequency

        Returns:
            Tuple[logits, attn_weights, cls_repr]:
                logits:      (B, n_classes) model predictions
                attn_weights:(B, n_heads, 257, 257) attention from transformer layer 2
                cls_repr:    (B, hidden_dim) CLS embedding before classifier head
        """
        B = x.shape[0]

        # Step 1: Tokenize via Multi-Scale Tokenizer
        freq_tokens = self.tokenizer(x)  # (B, 256, hidden_dim)

        # Step 2: Physics graph embedding via PIFFG
        graph_embed = self.piffg(bearing_params_list, fs_sampling)  # (B, hidden_dim)

        # Step 3: Prepend [CLS] token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, hidden_dim)
        sequence = torch.cat([cls_token, freq_tokens], dim=1)  # (B, 257, hidden_dim)

        # Step 4: Add positional encoding
        pos_enc = self.pos_encoder(sequence)  # (B, 257, hidden_dim)
        sequence = sequence + pos_enc

        # Step 5 & 6: Inject graph embedding at each encoder layer and capture attention
        encoder_output, attn_weights = self.encoder(
            sequence,
            graph_proj_list=self.graph_proj,
            graph_embed=graph_embed,
        )  # (B, 257, hidden_dim), (B, n_heads, 257, 257)

        # Step 7: Extract [CLS] representation
        cls_repr = encoder_output[:, 0, :]  # (B, hidden_dim)

        # Step 8: Classification head
        logits = self.classifier(cls_repr)  # (B, n_classes)

        return logits, attn_weights, cls_repr

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
