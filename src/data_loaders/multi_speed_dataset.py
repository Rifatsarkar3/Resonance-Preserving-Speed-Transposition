"""Multi-speed mixed dataset for domain robustness training on JNU."""
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional


class MultiSpeedMixedDataset(Dataset):
    """
    Wraps a primary dataset and auxiliary dataset to mix samples during training.

    For JNU training with domain adaptation:
    - Primary dataset: training speed only
    - Auxiliary dataset: other speeds

    During training: 90% primary samples, 10% auxiliary samples (for robustness)
    During val/test: Always use primary dataset (100% pure, no leakage)
    """

    def __init__(
        self,
        primary_dataset: Dataset,
        auxiliary_dataset: Optional[Dataset] = None,
        auxiliary_ratio: float = 0.1,
        is_training: bool = True,
    ):
        """
        Args:
            primary_dataset: Main training dataset (e.g., 600rpm)
            auxiliary_dataset: Auxiliary dataset for mixing (e.g., 800rpm + 1000rpm combined)
            auxiliary_ratio: Fraction of batches to sample from auxiliary (default 0.1 = 10%)
            is_training: If False, always return primary samples (no mixing)
        """
        self.primary_dataset = primary_dataset
        self.auxiliary_dataset = auxiliary_dataset
        self.auxiliary_ratio = auxiliary_ratio
        self.is_training = is_training

        # Total samples are from primary dataset only (no leakage)
        self.n_samples = len(primary_dataset)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # During validation/test, always use primary dataset (100% pure)
        if not self.is_training or self.auxiliary_dataset is None:
            return self.primary_dataset[idx]

        # During training, probabilistically use auxiliary samples
        if np.random.random() < self.auxiliary_ratio and len(self.auxiliary_dataset) > 0:
            # Sample from auxiliary dataset
            aux_idx = np.random.randint(0, len(self.auxiliary_dataset))
            return self.auxiliary_dataset[aux_idx]
        else:
            # Sample from primary dataset
            return self.primary_dataset[idx]
