#!/usr/bin/env python
"""Generate dataset splits JSON files for training."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loaders.split_strategy import (
    generate_pu_splits,
    generate_pu_lobo_splits,
    generate_cwru_splits,
    generate_jnu_splits,
)


def generate_all_splits():
    """Generate and save all dataset splits."""
    splits_root = Path("data/splits")
    splits_root.mkdir(parents=True, exist_ok=True)

    # Generate PU splits (standard + LOBO)
    pu_splits = {**generate_pu_splits(None), **generate_pu_lobo_splits(None)}
    with open(splits_root / "pu_splits.json", "w") as f:
        json.dump(pu_splits, f, indent=2)
    print(f"[OK] Generated {len(pu_splits)} PU splits (including LOBO)")

    # Generate CWRU splits
    cwru_splits = generate_cwru_splits(None)
    with open(splits_root / "cwru_splits.json", "w") as f:
        json.dump(cwru_splits, f, indent=2)
    print(f"[OK] Generated {len(cwru_splits)} CWRU splits")

    # Generate JNU splits
    jnu_splits = generate_jnu_splits(None)
    with open(splits_root / "jnu_splits.json", "w") as f:
        json.dump(jnu_splits, f, indent=2)
    print(f"[OK] Generated {len(jnu_splits)} JNU splits")

    print(f"\nSplits saved to {splits_root}/")


if __name__ == "__main__":
    generate_all_splits()
