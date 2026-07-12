"""CWRU (Case Western Reserve University) bearing dataset loader."""
import numpy as np
from pathlib import Path
import json
from scipy.io import loadmat


def process_cwru_dataset(config) -> None:
    """
    Process CWRU dataset: Load .mat files → window → normalize → memmap.

    CWRU files are MATLAB v5 format, scipy.io.loadmat compatible.
    """
    cwru_raw_root = Path(config.data.cwru_raw_root)
    processed_root = Path(config.data.processed_root) / "CWRU"
    processed_root.mkdir(parents=True, exist_ok=True)

    # CWRU bearing metadata (SKF 6205-2RS JEM)
    bearing_metadata = {
        "N_balls": 9,
        "d_mm": 7.94,
        "D_mm": 39.04,
        "alpha_deg": 0.0,
    }

    # Load conditions (HP levels)
    loads = ["0HP", "1HP", "2HP", "3HP"]

    # Fault classes
    fault_classes = {
        "Normal": 0,
        "B007": 1, "B014": 2, "B021": 3,  # Ball faults
        "IR007": 4, "IR014": 5, "IR021": 6,  # Inner race
        "OR007@6": 7, "OR014@6": 8, "OR021@6": 9,  # Outer race @ 6 o'clock position
    }

    # Shaft frequencies by load
    shaft_frequencies = {
        "0HP": 29.95,
        "1HP": 29.83,
        "2HP": 29.67,
        "3HP": 29.53,
    }

    print(f"CWRU Dataset Loader")
    print(f"  Raw root: {cwru_raw_root}")
    print(f"  Output: {processed_root}")
    print(f"  Loads: {loads}")
    print(f"  Fault classes: {len(fault_classes)}")
    print(f"  Sampling rate: 12,000 Hz")
    print(f"  Note: Implement full loader with scipy.io.loadmat()")


def cwru_loader(config):
    """Placeholder for full CWRU loading logic."""
    print("CWRU loader - implement with scipy.io.loadmat()")
