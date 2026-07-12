"""Organize downloaded datasets into expected dataloader formats."""
import sys
import shutil
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def verify_cwru_structure(config: Config) -> bool:
    """Verify CWRU .npz files are organized correctly."""
    logger.info("\n[CWRU] Verifying dataset structure...")

    cwru_raw_root = Path(config.data.cwru_raw_root)

    if not cwru_raw_root.exists():
        logger.error(f"  CWRU root not found: {cwru_raw_root}")
        return False

    # Check for NPZ files
    npz_files = list(cwru_raw_root.glob("**/*.npz"))
    logger.info(f"  Found {len(npz_files)} .npz files")

    if len(npz_files) < 100:
        logger.error(f"  Expected 100+ .npz files, found {len(npz_files)}")
        return False

    # List speeds
    speeds_dir = cwru_raw_root / "Data"
    if speeds_dir.exists():
        speeds = [d.name for d in speeds_dir.iterdir() if d.is_dir()]
        logger.info(f"  Operating speeds: {sorted(speeds)}")

    logger.info("  [OK] CWRU structure verified")
    return True


def verify_and_organize_jnu(config: Config) -> bool:
    """Verify and organize JNU CSV files."""
    logger.info("\n[JNU] Verifying and organizing dataset...")

    jnu_raw_root = Path(config.data.jnu_raw_root)

    if not jnu_raw_root.exists():
        logger.error(f"  JNU root not found: {jnu_raw_root}")
        return False

    # Expected JNU CSV files
    expected_files = {
        "Normal": ["n600_3_2.csv", "n800_3_2.csv", "n1000_3_2.csv"],
        "Inner": ["ib600_2.csv", "ib800_2.csv", "ib1000_2.csv"],
        "Outer": ["ob600_2.csv", "ob800_2.csv", "ob1000_2.csv"],
        "Ball": ["tb600_2.csv", "tb800_2.csv", "tb1000_2.csv"],
    }

    all_present = True
    for fault_type, files in expected_files.items():
        logger.info(f"  {fault_type} bearings:")
        for fname in files:
            fpath = jnu_raw_root / fname
            if fpath.exists():
                size_mb = fpath.stat().st_size / (1024 * 1024)
                logger.info(f"    [OK] {fname} ({size_mb:.2f} MB)")
            else:
                logger.error(f"    [MISSING] {fname}")
                all_present = False

    if not all_present:
        logger.error("  Some JNU CSV files are missing")
        return False

    # Organize into subdirectories by speed (optional, for clarity)
    logger.info("  Organizing JNU files by speed...")
    speeds = {
        "600rpm": ["n600_3_2.csv", "ib600_2.csv", "ob600_2.csv", "tb600_2.csv"],
        "800rpm": ["n800_3_2.csv", "ib800_2.csv", "ob800_2.csv", "tb800_2.csv"],
        "1000rpm": ["n1000_3_2.csv", "ib1000_2.csv", "ob1000_2.csv", "tb1000_2.csv"],
    }

    for speed, files in speeds.items():
        speed_dir = jnu_raw_root / speed
        speed_dir.mkdir(exist_ok=True)

        for fname in files:
            src = jnu_raw_root / fname
            dst = speed_dir / fname
            if src.exists() and src != dst:
                try:
                    shutil.move(str(src), str(dst))
                    logger.info(f"    Moved {fname} → {speed}/")
                except Exception as e:
                    logger.warning(f"    Could not move {fname}: {e}")

    logger.info("  [OK] JNU structure organized")
    return True


def extract_pu_rar_files(config: Config) -> bool:
    """Extract PU .rar files to .mat files."""
    logger.info("\n[PU] Extracting .rar files...")

    pu_raw_root = Path(config.data.pu_raw_root)

    if not pu_raw_root.exists():
        logger.error(f"  PU root not found: {pu_raw_root}")
        return False

    rar_files = list(pu_raw_root.glob("*.rar"))
    logger.info(f"  Found {len(rar_files)} .rar files")

    if len(rar_files) == 0:
        logger.error("  No .rar files found")
        return False

    # Try different extraction methods
    success = False

    # Method 1: Try 7-Zip
    try:
        import subprocess

        logger.info("  Attempting extraction with 7-Zip...")

        for rar_file in rar_files:
            result = subprocess.run(
                ["7z", "x", str(rar_file), f"-o{pu_raw_root}"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                logger.info(f"    Extracted {rar_file.name}")
                success = True
            else:
                logger.warning(f"    Failed to extract {rar_file.name}: {result.stderr}")

    except FileNotFoundError:
        logger.info("  7-Zip not found, trying Python rarfile library...")

        # Method 2: Try Python rarfile library
        try:
            import rarfile

            logger.info("  Using rarfile library...")

            for rar_file in rar_files:
                try:
                    with rarfile.RarFile(str(rar_file)) as rf:
                        rf.extractall(path=pu_raw_root)
                        logger.info(f"    Extracted {rar_file.name}")
                        success = True
                except Exception as e:
                    logger.error(f"    Error extracting {rar_file.name}: {e}")

        except ImportError:
            logger.warning("  rarfile library not available")

    if not success:
        logger.error("  [ERROR] Could not extract RAR files")
        logger.info("  Install 7-Zip or: pip install rarfile")
        logger.info("  Or manually extract .rar files and run this script again")
        return False

    # Verify .mat files
    mat_files = list(pu_raw_root.glob("**/*.mat"))
    logger.info(f"  Found {len(mat_files)} .mat files after extraction")

    expected_bearings = [
        "K001", "K002", "K003", "K004", "K005",  # Normal
        "KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09",  # Outer race
        "KI01", "KI03", "KI05", "KI07", "KI08",  # Inner race
        "KB23", "KB24", "KB27",  # Combined
    ]

    found_bearings = set()
    for mat_file in mat_files:
        for bearing in expected_bearings:
            if bearing in mat_file.name.upper():
                found_bearings.add(bearing)

    logger.info(f"  Found data for {len(found_bearings)}/{len(expected_bearings)} bearings")
    missing = set(expected_bearings) - found_bearings
    if missing:
        logger.warning(f"  Missing: {sorted(missing)}")

    logger.info("  [OK] PU .rar files extracted")
    return True


def main():
    parser = __import__("argparse").ArgumentParser(description="Organize downloaded datasets")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cwru", "pu", "jnu"],
        help="Datasets to organize: cwru, pu, jnu",
    )

    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    config.validate_paths()

    logger.info("=" * 70)
    logger.info("Dataset Organization and Verification")
    logger.info("=" * 70)

    results = {}

    if "cwru" in [d.lower() for d in args.datasets]:
        results["CWRU"] = verify_cwru_structure(config)

    if "jnu" in [d.lower() for d in args.datasets]:
        results["JNU"] = verify_and_organize_jnu(config)

    if "pu" in [d.lower() for d in args.datasets]:
        results["PU"] = extract_pu_rar_files(config)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Organization Summary")
    logger.info("=" * 70)

    for dataset, status in results.items():
        status_str = "[OK]" if status else "[ERROR]"
        logger.info(f"{dataset:10s} {status_str}")

    if all(results.values()):
        logger.info("\n[OK] All datasets organized successfully!")
        logger.info("Next step: python scripts/preprocess_all.py --config config.yaml --verify")
        return 0
    else:
        logger.error("\n[ERROR] Some datasets have issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
