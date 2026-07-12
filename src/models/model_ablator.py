"""Model ablation utilities for component-level analysis."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
from src.models.wapigt import WaPIGT


class FixedWaveletTokenizer(nn.Module):
    """
    Fixed wavelet tokenizer using non-learnable Daubechies-4 filters.

    Same structure as LWPT but filters are frozen to baseline orthogonal wavelets.
    Used for -LWPT ablation to isolate the learnable wavelet component's contribution.
    """

    def __init__(self, filter_order: int = 8, n_frames: int = 16, hidden_dim: int = 96):
        super().__init__()
        self.filter_order = filter_order
        self.n_frames = n_frames
        self.hidden_dim = hidden_dim
        self.n_bands = 16

        # Use fixed Daubechies-4 filters (non-learnable)
        db4 = pywt.Wavelet('db4')
        h_np = np.array(db4.dec_lo, dtype=np.float32) / np.sqrt(2)  # Normalize
        g_np = np.array(db4.dec_hi, dtype=np.float32) / np.sqrt(2)

        # Register as buffers (not parameters - won't update)
        self.register_buffer('h_filter', torch.tensor(h_np, dtype=torch.float32))
        self.register_buffer('g_filter', torch.tensor(g_np, dtype=torch.float32))

        # Linear projection from token energy to hidden_dim
        self.proj = nn.Linear(1, hidden_dim)

    def _wavelet_decompose(self, x):
        """4-level wavelet packet decomposition using fixed filters."""
        h = self.h_filter.unsqueeze(0).unsqueeze(0)  # (1, 1, filter_len)
        g = self.g_filter.unsqueeze(0).unsqueeze(0)

        subbands = []
        current = x

        for level in range(4):
            new_subbands = []
            for sb in (subbands if level > 0 else [current]):
                ll = F.conv1d(sb, h, padding=h.shape[-1] // 2, stride=2)
                lh = F.conv1d(sb, g, padding=g.shape[-1] // 2, stride=2)
                new_subbands.extend([ll, lh])
            subbands = new_subbands

        return subbands[:self.n_bands]

    def forward(self, x):
        """Forward pass with fixed filters."""
        B, C, L = x.shape

        subbands = self._wavelet_decompose(x)

        # Energy pooling per subband
        freq_energy = []
        for sb in subbands:
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
                frame_stack = torch.cat(frames, dim=2)
                freq_energy.append(frame_stack)

        if freq_energy:
            freq_energy = torch.cat(freq_energy, dim=2)
        else:
            freq_energy = torch.randn(B, C, self.n_bands * self.n_frames, device=x.device)

        freq_energy = freq_energy.mean(dim=1, keepdim=True)
        freq_energy = freq_energy.transpose(1, 2)
        tokens = self.proj(freq_energy)

        return tokens


class WaPIGTAblated(WaPIGT):
    """
    WaPIGT variant with configurable component ablations.

    Allows removal of:
    - LWPT: Use fixed wavelet tokenizer instead
    - PIFFG: Zero out graph embedding (no physics injection)
    - SCR: Handled at loss level, not in model
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
        filter_order: int = 8,
        n_frames: int = 16,
        ablate_lwpt: bool = False,
        ablate_piffg: bool = False,
    ):
        super().__init__(
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            n_encoder_layers=n_encoder_layers,
            n_heads=n_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            n_gat_heads=n_gat_heads,
            gat_dropout=gat_dropout,
            filter_order=filter_order,
            n_frames=n_frames,
        )

        self.ablate_lwpt = ablate_lwpt
        self.ablate_piffg = ablate_piffg

        # Replace LWPT with fixed version if ablating
        if ablate_lwpt:
            self.lwpt = FixedWaveletTokenizer(
                filter_order=filter_order,
                n_frames=n_frames,
                hidden_dim=hidden_dim,
            )

    def forward(self, x, bearing_params_list, fs_sampling: float = 64000.0):
        """
        Forward pass with optional ablations.

        Args:
            x: (B, C, L) raw vibration signal
            bearing_params_list: List[Dict] bearing parameters
            fs_sampling: Sampling frequency

        Returns:
            Tuple[logits, attn_weights]
        """
        B = x.shape[0]

        # Step 1: Tokenize via LWPT or fixed variant
        freq_tokens = self.lwpt(x)  # (B, 256, hidden_dim)

        # Step 2: Physics graph embedding via PIFFG (or zero if ablated)
        if self.ablate_piffg:
            graph_embed = torch.zeros(B, self.hidden_dim, device=x.device)
        else:
            graph_embed = self.piffg(bearing_params_list, fs_sampling)

        # Step 3: Prepend [CLS] token
        cls_token = self.cls_token.expand(B, -1, -1)
        sequence = torch.cat([cls_token, freq_tokens], dim=1)

        # Step 4: Add positional encoding
        pos_enc = self.pos_encoder(sequence)
        sequence = sequence + pos_enc

        # Step 5 & 6: Graph injection and attention capture
        encoder_output, attn_weights = self.encoder(
            sequence,
            graph_proj_list=self.graph_proj,
            graph_embed=graph_embed,
        )

        # Step 7: Extract [CLS] representation
        cls_repr = encoder_output[:, 0, :]

        # Step 8: Classification head
        logits = self.classifier(cls_repr)

        return logits, attn_weights


class ModelAblator:
    """Utility for loading checkpoints and creating ablated model variants."""

    @staticmethod
    def _infer_dims_from_state_dict(state_dict):
        """Infer model dimensions from state_dict keys."""
        hidden_dim = 96  # default
        n_heads = 8  # default
        n_encoder_layers = 4  # default
        n_classes = 4  # default

        # Infer hidden_dim from lwpt.proj.bias shape [hidden_dim]
        if 'lwpt.proj.bias' in state_dict:
            hidden_dim = state_dict['lwpt.proj.bias'].shape[0]

        # Infer n_heads from encoder layers
        for key in state_dict.keys():
            if 'encoder.layers.' in key and '.self_attn.' in key:
                # Found attention layer
                n_encoder_layers = max(n_encoder_layers, int(key.split('.')[2]) + 1)

        # Infer n_classes from classifier.bias shape [n_classes]
        if 'classifier.bias' in state_dict:
            n_classes = state_dict['classifier.bias'].shape[0]

        # Infer n_heads from piffg attention weights if present
        if 'piffg.gat1.att_src' in state_dict:
            # Shape is [1, n_heads, hidden_dim]
            n_heads = state_dict['piffg.gat1.att_src'].shape[1]

        mlp_dim = hidden_dim * 4  # Standard ratio

        return {
            'hidden_dim': hidden_dim,
            'n_encoder_layers': n_encoder_layers,
            'n_heads': n_heads,
            'mlp_dim': mlp_dim,
            'n_classes': n_classes,
        }

    @staticmethod
    def create_ablated_model(
        checkpoint_path: str,
        ablation_type: str,
        device: str = "cuda",
    ) -> WaPIGTAblated:
        """
        Load a checkpoint and create an ablated variant.

        Args:
            checkpoint_path: Path to checkpoint file
            ablation_type: One of ["full", "-LWPT", "-PIFFG", "-SCR"]
            device: Device to load model on

        Returns:
            WaPIGTAblated model with weights loaded
        """
        # Load with weights_only=False to handle numpy scalar serialization
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Load weights from checkpoint
        state_dict = checkpoint.get("model_state_dict", {})
        if not state_dict:
            state_dict = checkpoint.get("model_state", checkpoint.get("state_dict", {}))

        # Infer model dimensions from state_dict
        dims = ModelAblator._infer_dims_from_state_dict(state_dict)

        # Create ablated model with inferred config
        ablate_lwpt = ablation_type == "-LWPT"
        ablate_piffg = ablation_type == "-PIFFG"

        model = WaPIGTAblated(
            n_classes=dims['n_classes'],
            hidden_dim=dims['hidden_dim'],
            n_encoder_layers=dims['n_encoder_layers'],
            n_heads=dims['n_heads'],
            mlp_dim=dims['mlp_dim'],
            dropout=0.3,
            n_gat_heads=4,
            gat_dropout=0.3,
            filter_order=8,
            n_frames=16,
            ablate_lwpt=ablate_lwpt,
            ablate_piffg=ablate_piffg,
        ).to(device)

        # Load state dict, allowing for shape mismatches if ablation changed architecture
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        return model
