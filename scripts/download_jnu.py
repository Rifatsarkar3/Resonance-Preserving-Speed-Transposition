"""Download and organize JNU (Jiangnan University) bearing dataset.

Source: https://github.com/ClarkGableWang/JNU-Bearing-Dataset
"""
import argparse
import sys
import shutil
import zipfile
from pathlib import Path
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Try to import requests for better download handling
try:
    import requests
except ImportError:
    requests = None


def download_with_git(repo_url: str, output_dir: Path) -> bool:
    """Download dataset using git clone."""
    logger.info("Attempting download with git...")

    temp_dir = Path("temp_jnu_download")

    # Clean up if exists
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up existing temp directory: {temp_dir}")

    try:
        # Clone with depth=1 for faster download
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(temp_dir)],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            logger.warning(f"Git clone failed: {result.stderr}")
            return False

        logger.info("[OK] Repository cloned successfully")

        # Copy CSV files
        logger.info("Organizing CSV files...")
        csv_files = list(temp_dir.glob("**/*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files")

        if csv_files:
            for csv_file in csv_files:
                relative_path = csv_file.relative_to(temp_dir)
                dest_path = output_dir / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(csv_file, dest_path)
                logger.info(f"  Copied: {relative_path}")

            logger.info("[OK] All CSV files organized")
        else:
            logger.warning("No CSV files found")

        # Copy README
        readme_file = temp_dir / "README.md"
        if readme_file.exists():
            shutil.copy2(readme_file, output_dir / "README.md")
            logger.info("[OK] Copied README.md")

        # Cleanup
        logger.info("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
        logger.info("[OK] Temporary files removed")

        return True

    except subprocess.TimeoutExpired:
        logger.error("Git clone timed out")
        return False
    except Exception as e:
        logger.error(f"Git download failed: {e}")
        return False


def download_with_requests(repo_url: str, output_dir: Path) -> bool:
    """Download dataset using requests + zipfile."""
    if requests is None:
        logger.warning("requests library not available")
        return False

    logger.info("Attempting download with requests...")

    # Convert GitHub repo URL to ZIP download URL
    zip_url = repo_url.replace("https://github.com/", "https://github.com/").rstrip(
        "/"
    ) + "/archive/refs/heads/main.zip"

    zip_path = Path("jnu_dataset.zip")
    temp_dir = Path("temp_jnu_download")

    try:
        logger.info(f"Downloading from: {zip_url}")

        # Download ZIP file
        response = requests.get(zip_url, timeout=300, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        pct = (downloaded / total_size) * 100
                        logger.info(f"  Downloaded: {pct:.1f}%")

        logger.info("[OK] ZIP file downloaded")

        # Extract ZIP
        logger.info("Extracting ZIP file...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        logger.info("[OK] ZIP file extracted")

        # Find extracted directory
        extracted_dirs = [d for d in temp_dir.iterdir() if d.is_dir()]
        if not extracted_dirs:
            logger.error("No extracted directories found")
            return False

        source_dir = extracted_dirs[0]

        # Copy CSV files
        logger.info("Organizing CSV files...")
        csv_files = list(source_dir.glob("**/*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files")

        if csv_files:
            for csv_file in csv_files:
                relative_path = csv_file.relative_to(source_dir)
                dest_path = output_dir / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(csv_file, dest_path)
                logger.info(f"  Copied: {relative_path}")

            logger.info("[OK] All CSV files organized")
        else:
            logger.warning("No CSV files found")

        # Copy README
        readme_file = source_dir / "README.md"
        if readme_file.exists():
            shutil.copy2(readme_file, output_dir / "README.md")
            logger.info("[OK] Copied README.md")

        # Cleanup
        logger.info("Cleaning up temporary files...")
        zip_path.unlink()
        shutil.rmtree(temp_dir)
        logger.info("[OK] Temporary files removed")

        return True

    except requests.RequestException as e:
        logger.error(f"Download failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download JNU bearing dataset from GitHub"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw/JNU",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "git", "requests"],
        help="Download method (auto=try git first, then requests)",
    )

    args = parser.parse_args()

    # Resolve output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("JNU Dataset Downloader")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir.absolute()}")

    repo_url = "https://github.com/ClarkGableWang/JNU-Bearing-Dataset"

    # Try download methods
    success = False

    if args.method in ["auto", "git"]:
        success = download_with_git(repo_url, output_dir)

    if not success and args.method in ["auto", "requests"]:
        logger.info("\nGit method failed or not selected, trying requests...")
        success = download_with_requests(repo_url, output_dir)

    if not success:
        logger.error("\n[ERROR] All download methods failed")
        logger.info("\nManual download options:")
        logger.info(f"1. Git: git clone {repo_url} {output_dir}")
        logger.info(
            f"2. Browser: Download ZIP from https://github.com/ClarkGableWang/JNU-Bearing-Dataset"
        )
        logger.info(f"3. Extract CSV files to: {output_dir}")
        return 1

    # Verify dataset
    logger.info("\n[Step] Verifying dataset structure...")

    csv_files = list(output_dir.glob("**/*.csv"))
    logger.info(f"  CSV files: {len(csv_files)}")

    # List directories
    subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
    if subdirs:
        logger.info("  Directories:")
        for subdir in subdirs:
            csv_count = len(list(subdir.glob("**/*.csv")))
            logger.info(f"    - {subdir.name} ({csv_count} CSV files)")

    logger.info("\n" + "=" * 70)
    logger.info("Download Complete!")
    logger.info(f"JNU dataset saved to: {output_dir.absolute()}")
    logger.info("\nNext step:")
    logger.info(
        "  python scripts/preprocess_all.py --config config.yaml --datasets jnu --verify"
    )
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
