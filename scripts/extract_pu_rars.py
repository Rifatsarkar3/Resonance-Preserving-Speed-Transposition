#!/usr/bin/env python3
"""Extract PU RAR files to MAT files.

Supports multiple extraction methods:
1. subprocess with system 7z
2. patool library (cross-platform)
3. rarfile library (if available)
4. Manual guidance if none available
"""
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional


def extract_with_7z(rar_path: Path, output_dir: Path) -> bool:
    """Extract using system 7z command."""
    try:
        result = subprocess.run(
            ["7z", "x", str(rar_path), f"-o{output_dir}", "-y"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def extract_with_patool(rar_path: Path, output_dir: Path) -> bool:
    """Extract using patool library."""
    try:
        import patool
        patool.create_archive(str(output_dir / rar_path.stem), [str(rar_path)])
        return True
    except (ImportError, Exception):
        return False


def extract_with_rarfile(rar_path: Path, output_dir: Path) -> bool:
    """Extract using rarfile library."""
    try:
        import rarfile
        with rarfile.RarFile(str(rar_path)) as rf:
            rf.extractall(path=output_dir)
        return True
    except (ImportError, Exception):
        return False


def find_7z() -> Optional[str]:
    """Find 7z executable in common locations."""
    common_paths = [
        Path("C:\\Program Files\\7-Zip\\7z.exe"),
        Path("C:\\Program Files (x86)\\7-Zip\\7z.exe"),
        Path("C:\\ProgramData\\chocolatey\\bin\\7z.exe"),
        Path("/usr/bin/7z"),
        Path("/usr/local/bin/7z"),
    ]

    for path in common_paths:
        if path.exists():
            return str(path)

    return None


def main():
    print("=" * 70)
    print("PU Dataset RAR Extraction")
    print("=" * 70)

    pu_raw_dir = Path("e:/Paper A/data/raw/PU")
    if not pu_raw_dir.exists():
        print(f"[ERROR] PU directory not found: {pu_raw_dir}")
        return 1

    rar_files = list(pu_raw_dir.glob("*.rar"))
    print(f"\nFound {len(rar_files)} RAR files")

    if len(rar_files) == 0:
        print("[INFO] No RAR files to extract")
        return 0

    # Try extraction methods in order
    success_count = 0
    failed_files = []

    print("\nAttempting extraction...")

    # Try 7z first
    sevenzip_path = find_7z()
    if sevenzip_path:
        print(f"\n[Method] Using 7-Zip from: {sevenzip_path}\n")

        for rar_file in rar_files:
            print(f"  Extracting {rar_file.name}...", end=" ")
            try:
                result = subprocess.run(
                    [sevenzip_path, "x", str(rar_file), f"-o{pu_raw_dir}", "-y"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    print("[OK]")
                    success_count += 1
                else:
                    print("[FAILED]")
                    failed_files.append(rar_file.name)
            except Exception as e:
                print(f"[ERROR: {e}]")
                failed_files.append(rar_file.name)

    # If 7z not found, try rarfile library
    else:
        print("[Method] Using rarfile library\n")
        try:
            import rarfile

            for rar_file in rar_files:
                print(f"  Extracting {rar_file.name}...", end=" ")
                try:
                    with rarfile.RarFile(str(rar_file)) as rf:
                        rf.extractall(path=pu_raw_dir)
                    print("[OK]")
                    success_count += 1
                except Exception as e:
                    print(f"[FAILED: {e}]")
                    failed_files.append(rar_file.name)

        except ImportError:
            print("[ERROR] No extraction method available!")
            print("\nPlease install one of:")
            print("  1. 7-Zip: https://www.7-zip.org/download.html")
            print("  2. Python rarfile: pip install rarfile")
            print("  3. Manual: Extract RAR files using Windows Explorer")
            return 1

    # Verify extraction
    print("\n" + "-" * 70)
    print("Verification:")

    mat_files = list(pu_raw_dir.glob("**/*.mat"))
    print(f"  Found {len(mat_files)} .mat files after extraction")

    expected_bearings = {
        "Normal": ["K001", "K002", "K003", "K004", "K005"],
        "Outer Race": ["KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09"],
        "Inner Race": ["KI01", "KI03", "KI05", "KI07", "KI08"],
        "Combined": ["KB23", "KB24", "KB27"],
    }

    for fault_type, bearings in expected_bearings.items():
        found = 0
        for bearing in bearings:
            if any(bearing in f.name.upper() for f in mat_files):
                found += 1
        print(f"    {fault_type}: {found}/{len(bearings)} bearings")

    print("\n" + "=" * 70)
    if len(mat_files) > 0:
        print("[OK] Extraction successful!")
        print("Next step: python scripts/preprocess_all.py --config config.yaml --verify")
        return 0
    else:
        print("[ERROR] No MAT files found after extraction")
        if failed_files:
            print(f"Failed files: {failed_files}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
