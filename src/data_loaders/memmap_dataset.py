"""PyTorch Dataset backed by memory-mapped numpy arrays."""
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapDataset(Dataset):
    """
    PyTorch Dataset using numpy memmap for RAM-efficient data loading.

    Never loads the full array into memory - uses memory-mapped views only.
    """

    def __init__(
        self,
        memmap_dir: str,
        split: str = "train",
        transform=None,
        bearing_geometry: Optional[Dict] = None,
    ):
        """
        Args:
            memmap_dir: Directory containing segments.npy, labels.npy, metadata.json
            split: "train", "val", or "test"
            transform: Optional data transformation
            bearing_geometry: Bearing parameters for PIFFG (optional)
        """
        self.memmap_dir = Path(memmap_dir)
        self.split = split
        self.transform = transform
        self.bearing_geometry = bearing_geometry or {}

        # Load metadata
        metadata_path = self.memmap_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Open memmaps in read-only mode
        self.segments = np.memmap(
            self.memmap_dir / "segments.npy",
            dtype=np.float32,
            mode='r',
            shape=tuple(self.metadata['shape']),
        )
        self.labels = np.memmap(
            self.memmap_dir / "labels.npy",
            dtype=np.int64,
            mode='r',
            shape=(self.metadata['n_samples'],),
        )
        self.bearing_ids = np.memmap(
            self.memmap_dir / "bearing_ids.npy",
            dtype=np.int32,
            mode='r',
            shape=(self.metadata['n_samples'],),
        )

        self.n_samples = self.metadata['n_samples']
        self.n_classes = self.metadata['n_classes']
        self.window_len = self.metadata['window_len']
        self.fs_sampling = self.metadata['fs']
        self.scaling_min = self.metadata.get('scaling_min', 0.0)
        self.scaling_max = self.metadata.get('scaling_max', 1.0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.

        Returns:
            {
                "signal": Tensor (1, window_len) or (C, window_len)
                "label": int
                "bearing_id": int
                "bearing_params": dict
            }
        """
        # Get signal from memmap (returns view, not copy)
        signal = self.segments[idx]  # (C, L) or (L,)

        # Make a copy only when converting to tensor
        signal_tensor = torch.from_numpy(signal.copy()).float()

        # Normalize using training statistics
        if self.scaling_max > self.scaling_min:
            signal_tensor = (signal_tensor - self.scaling_min) / (self.scaling_max - self.scaling_min + 1e-8)

        # Get label
        label = int(self.labels[idx])

        # Get bearing ID
        bearing_id = int(self.bearing_ids[idx])

        # Get bearing parameters for PIFFG
        bearing_params = self.bearing_geometry.copy()
        bearing_params['bearing_id'] = bearing_id

        if self.transform:
            signal_tensor = self.transform(signal_tensor)

        return {
            "signal": signal_tensor,
            "label": label,
            "bearing_id": bearing_id,
            "bearing_params": bearing_params,
        }
