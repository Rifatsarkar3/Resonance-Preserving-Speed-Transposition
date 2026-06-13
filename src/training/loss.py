"""WaPIGT Loss: Cross-Entropy + Spectrum Consistency Regularizer + Triplet Loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WaPIGTLoss(nn.Module):
    """
    Combined loss for WaPIGT-MS:
      CE + λ_scr × SCR(warmup) + λ_trip × TripletLoss(warmup)

    Triplet loss uses online batch-hard mining (hardest positive + hardest negative
    per anchor) which forces condition-invariant inter-class separation — the key
    mechanism borrowed from TICNN that enables cross-condition generalization.
    """

    def __init__(
        self,
        n_classes: int,
        scr_module: nn.Module,
        scr_lambda: float = 0.1,
        scr_warmup_epochs: int = 10,
        n_epochs: int = 200,
        triplet_lambda: float = 0.1,
        triplet_margin: float = 0.5,
        triplet_warmup_epochs: int = 20,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.ce_loss = nn.CrossEntropyLoss()
        self.scr_module = scr_module
        self.scr_lambda = scr_lambda
        self.scr_warmup_epochs = scr_warmup_epochs
        self.n_epochs = n_epochs
        self.triplet_lambda = triplet_lambda
        self.triplet_margin = triplet_margin
        self.triplet_warmup_epochs = triplet_warmup_epochs
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _batch_hard_triplet_loss(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Online batch-hard triplet mining.

        For each anchor: hardest positive = same-class sample with max distance;
        hardest negative = different-class sample with min distance.
        Margin loss: max(0, d(a,p) - d(a,n) + margin).
        """
        B = embeddings.shape[0]
        if B < 2:
            return embeddings.sum() * 0.0  # gradient-safe zero

        # L2-normalize embeddings before distance computation
        emb = F.normalize(embeddings, p=2, dim=1)
        dist = torch.cdist(emb, emb, p=2)  # (B, B)

        same_class = labels.unsqueeze(0) == labels.unsqueeze(1)   # (B, B)
        eye        = torch.eye(B, dtype=torch.bool, device=embeddings.device)
        pos_mask   = same_class & ~eye
        neg_mask   = ~same_class

        if not pos_mask.any() or not neg_mask.any():
            return embeddings.sum() * 0.0

        # Hardest positive per anchor
        pos_dist     = dist.masked_fill(~pos_mask, -1e9)
        hardest_pos  = pos_dist.max(dim=1)[0]

        # Hardest negative per anchor
        neg_dist     = dist.masked_fill(~neg_mask, 1e9)
        hardest_neg  = neg_dist.min(dim=1)[0]

        # Only include anchors that have at least one positive AND one negative
        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        if not valid.any():
            return embeddings.sum() * 0.0

        loss = F.relu(hardest_pos[valid] - hardest_neg[valid] + self.triplet_margin)
        return loss.mean()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attn_weights: torch.Tensor | None = None,
        fault_freq_bins: torch.Tensor | None = None,
        window_len: int = 1024,
        fs_sampling: float = 12000.0,
        embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            logits:          (B, n_classes)
            labels:          (B,)
            attn_weights:    (B, n_heads, N+1, N+1) — optional
            fault_freq_bins: (B, 4) — optional
            window_len:      signal length in samples
            fs_sampling:     sampling frequency
            embeddings:      (B, hidden_dim) CLS repr for triplet loss — optional
        """
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            import logging
            logging.getLogger(__name__).error("Invalid logits detected — clamping.")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

        if (labels < 0).any() or (labels >= logits.shape[1]).any():
            labels = torch.clamp(labels, 0, logits.shape[1] - 1)

        ce = self.ce_loss(logits, labels)
        loss = ce

        # SCR with linear warmup
        if attn_weights is not None and fault_freq_bins is not None:
            scr_w = self.scr_lambda * min(
                self.epoch / max(self.scr_warmup_epochs, 1), 1.0
            )
            if scr_w > 0:
                scr = self.scr_module(
                    attn_weights, fault_freq_bins, window_len, fs_sampling
                )
                loss = loss + scr_w * scr

        # Triplet loss with linear warmup
        if embeddings is not None and self.triplet_lambda > 0:
            trip_w = self.triplet_lambda * min(
                self.epoch / max(self.triplet_warmup_epochs, 1), 1.0
            )
            if trip_w > 0:
                trip = self._batch_hard_triplet_loss(embeddings, labels)
                loss = loss + trip_w * trip

        return loss
