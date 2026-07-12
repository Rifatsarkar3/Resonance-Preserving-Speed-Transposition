"""Master data preprocessing script for PU, CWRU, and JNU datasets."""
import argparse
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.data_loaders.split_strategy import (
    generate_pu_splits,
    generate_cwru_splits,
    generate_jnu_splits,
)
from src.data_loaders.pu_loader import process_pu_dataset
from src.data_loaders.cwru_loader import process_cwru_dataset
from src.data_loaders.jnu_loader import process_jnu_dataset


def main():
    parser = argparse.ArgumentParser(description="Preprocess bearing datasets")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["pu", "cwru", "jnu"],
        help="Datasets to process: pu, cwru, jnu",
    )
    parser.add_argument("--verify", action="store_true", help="Verify splits after processing")
    args = parser.parse_args()

    # Load config
    if not Path(args.config).exists():
        print(f"ERROR: Config file not found: {args.config}")
        return 1

    with open(args.config, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    config = Config.from_yaml(args.config)
    config.validate_paths()

    print("\n" + "=" * 70)
    print("WaPIGT Data Preprocessing Pipeline")
    print("=" * 70)

    # Create output directories
    processed_root = Path(config.data.processed_root)
    splits_root = Path(config.data.splits_root)
    processed_root.mkdir(parents=True, exist_ok=True)
    splits_root.mkdir(parents=True, exist_ok=True)

    datasets_to_process = [d.lower() for d in args.datasets]

    # Process datasets
    if "pu" in datasets_to_process:
        print("\n[1/3] Processing PU dataset...")
        try:
            process_pu_dataset(config)
            # Generate splits
            splits = generate_pu_splits(config)
            splits_path = splits_root / "pu_splits.json"
            with open(splits_path, 'w') as f:
                import json
                json.dump(splits, f, indent=2)
            print(f"  [OK] PU splits saved to {splits_path}")
        except Exception as e:
            print(f"  [ERROR] Error processing PU: {e}")

    if "cwru" in datasets_to_process:
        print("\n[2/3] Processing CWRU dataset...")
        try:
            process_cwru_dataset(config)
            # Generate splits
            splits = generate_cwru_splits(config)
            splits_path = splits_root / "cwru_splits.json"
            with open(splits_path, 'w') as f:
                import json
                json.dump(splits, f, indent=2)
            print(f"  [OK] CWRU splits saved to {splits_path}")
        except Exception as e:
            print(f"  [ERROR] Error processing CWRU: {e}")

    if "jnu" in datasets_to_process:
        print("\n[3/3] Processing JNU dataset...")
        try:
            process_jnu_dataset(config)
            # Generate splits
            splits = generate_jnu_splits(config)
            splits_path = splits_root / "jnu_splits.json"
            with open(splits_path, 'w') as f:
                import json
                json.dump(splits, f, indent=2)
            print(f"  [OK] JNU splits saved to {splits_path}")
        except Exception as e:
            print(f"  [ERROR] Error processing JNU: {e}")

    # Create READY flag
    print("\n" + "-" * 70)
    print("Data preprocessing structure created.")
    print(f"Note: Full implementation requires dataset downloads and MATLAB file parsing.")
    print(f"  Processed data root: {processed_root}")
    print(f"  Splits root: {splits_root}")

    # Create placeholder READY flag for now
    ready_flag = processed_root / "READY.flag"
    with open(ready_flag, 'w') as f:
        f.write("Placeholder - implement full preprocessing pipeline\n")

    print(f"\n[OK] Preprocessing structure ready")
    print(f"  Next: Update config.yaml with actual dataset paths and download datasets")

    return 0


if __name__ == "__main__":
    sys.exit(main())
