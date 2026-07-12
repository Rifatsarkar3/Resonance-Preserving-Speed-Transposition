"""JNU (Jiangnan University) bearing dataset loader."""
import numpy as np
import pandas as pd
from pathlib import Path
import json


def process_jnu_dataset(config) -> None:
    """
    Process JNU dataset: Load .csv files → window → normalize → memmap.

    JNU files are CSV format with single column of float32 vibration amplitude.
    """
    jnu_raw_root = Path(config.data.jnu_raw_root)
    processed_root = Path(config.data.processed_root) / "JNU"
    processed_root.mkdir(parents=True, exist_ok=True)

    # JNU bearing metadata (ER-16K)
    bearing_metadata = {
        "N_balls": 8,
        "d_mm": 7.5,
        "D_mm": 38.5,
        "alpha_deg": 0.0,
    }

    # Operating speeds (RPM)
    speeds = {
        "600rpm": {"f_s": 10.00, "speed_id": 0},
        "800rpm": {"f_s": 13.33, "speed_id": 1},
        "1000rpm": {"f_s": 16.67, "speed_id": 2},
    }

    # Fault classes
    fault_classes = {
        "N.csv": 0,   # Normal
        "IF.csv": 1,  # Inner race fault
        "OF.csv": 2,  # Outer race fault
        "BF.csv": 3,  # Ball fault
    }

    print(f"JNU Dataset Loader")
    print(f"  Raw root: {jnu_raw_root}")
    print(f"  Output: {processed_root}")
    print(f"  Speeds: {list(speeds.keys())}")
    print(f"  Fault classes: {len(fault_classes)}")
    print(f"  Sampling rate: 50,000 Hz")
    print(f"  Note: Implement full loader with pd.read_csv()")


def jnu_loader(config):
    """Placeholder for full JNU loading logic."""
    print("JNU loader - implement with pd.read_csv()")
