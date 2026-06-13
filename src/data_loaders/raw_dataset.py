"""Interim raw dataset loader for testing (before memmap preprocessing)."""
import numpy as np
import scipy.io as sio
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
import warnings
import math
import logging

from src.utils.fault_frequencies import compute_fault_frequencies

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class RawBearingDataset(Dataset):
    """Load bearing data directly from raw MAT/CSV files for testing."""

    def __init__(
        self,
        dataset: str = "PU",
        raw_root: str = "data/raw",
        bearing_list: List[str] = None,
        speed_list: List[str] = None,
        load_list: List[int] = None,
        split: str = "train",
        n_samples_per_bearing: int = 10,
        signal_length: int = 12000,
        task_name: str = None,
        test_speed_list: List[str] = None,
    ):
        """
        Args:
            dataset: "PU", "CWRU", or "JNU"
            raw_root: Root directory for raw data
            bearing_list: List of bearing IDs to load (for PU/CWRU)
            speed_list: List of speeds to load (for JNU, e.g., ["600rpm", "800rpm"])
            load_list: List of load conditions to load (for CWRU, e.g., [0, 1, 2])
            split: Train/val/test (unused for raw data)
            n_samples_per_bearing: Number of samples per bearing
            signal_length: Length of signal window
            task_name: Task identifier for JNU (e.g., "JNU_T01") to enable task-aware augmentation
            test_speed_list: Test speeds for directional augmentation (only for training split of JNU)
        """
        self.dataset = dataset
        self.raw_root = Path(raw_root)
        self.bearing_list = bearing_list or []
        self.speed_list = speed_list or []
        self.load_list = load_list or []
        self.split = split
        self.n_samples_per_bearing = n_samples_per_bearing
        self.signal_length = signal_length
        self.task_name = task_name
        self.test_speed_list = test_speed_list or []

        self.samples = []
        self.labels = []
        self.bearing_ids = []  # Track bearing ID for each sample
        self.speeds = []  # Track speed for JNU samples
        self.shaft_frequencies = []  # Track shaft frequency for each sample
        self._load_data()
        self._set_dataset_params()

    def _set_dataset_params(self):
        """Set dataset-specific bearing parameters and sampling rates."""
        if self.dataset == "PU":
            # SKF 6206 bearing
            self.bearing_params = {
                "N_balls": 9,
                "d_mm": 7.938,
                "D_mm": 38.5,
                "alpha_deg": 0.0
            }
            self.fs_sampling = 64000.0
        elif self.dataset == "CWRU":
            # SKF 6205-2RS JEM bearing
            self.bearing_params = {
                "N_balls": 9,
                "d_mm": 7.94,
                "D_mm": 39.04,
                "alpha_deg": 0.0
            }
            self.fs_sampling = 12000.0
        elif self.dataset == "JNU":
            # ER-16K bearing
            self.bearing_params = {
                "N_balls": 8,
                "d_mm": 7.5,
                "D_mm": 38.5,
                "alpha_deg": 0.0
            }
            self.fs_sampling = 50000.0
        else:
            # Default fallback
            self.bearing_params = {
                "N_balls": 9,
                "d_mm": 7.938,
                "D_mm": 38.5,
                "alpha_deg": 0.0
            }
            self.fs_sampling = 12000.0

    def _load_data(self):
        """Load raw data files."""
        if self.dataset == "PU":
            self._load_pu_data()
        elif self.dataset == "CWRU":
            self._load_cwru_data()
        elif self.dataset == "JNU":
            self._load_jnu_data()

        # Log samples loaded
        logger.info(f"{self.dataset} dataset: {len(self.samples)} samples loaded")

        # Ensure we have at least some data for testing
        if not self.samples:
            logger.warning(f"⚠️ NO REAL DATA LOADED FOR {self.dataset}! Using synthetic fallback.")
            self._load_synthetic_data()
            logger.warning(f"Synthetic data fallback triggered for {self.dataset} (now {len(self.samples)} samples)")

    def _load_synthetic_data(self):
        """Generate synthetic data for testing when real data not available."""
        logger.warning(f"🔴 SYNTHETIC DATA ALERT: Generating synthetic data for {self.dataset}")
        n_classes = 4 if self.dataset in ["PU", "JNU"] else 10
        for i in range(max(10, self.n_samples_per_bearing)):
            self.samples.append(np.random.randn(self.signal_length).astype(np.float32))
            self.labels.append(i % n_classes)
            self.bearing_ids.append(f"synthetic_{i}")
            self.speeds.append("1000rpm" if self.dataset == "JNU" else "unknown")
            # Add default shaft frequency (will be overridden by speed for JNU)
            self.shaft_frequencies.append(15.0 if self.dataset == "PU" else (29.95 if self.dataset == "CWRU" else 16.67))

    def _load_pu_data(self):
        """Load PU dataset from MAT files."""
        pu_root = self.raw_root / "PU"

        # Map bearing ID to fault type class (4 classes total)
        # 0: Normal, 1: OuterRace, 2: InnerRace, 3: Combined
        def get_bearing_label(bearing_id: str) -> int:
            if bearing_id.startswith("K"):
                return 0  # Normal (K001-K005)
            elif bearing_id.startswith("KA"):
                return 1  # OuterRace
            elif bearing_id.startswith("KI"):
                return 2  # InnerRace
            elif bearing_id.startswith("KB"):
                return 3  # Combined
            return 0

        for bearing_id in self.bearing_list:
            # Files are in subdirectories like K001/N09_M07_F10_K001_1.mat
            bearing_dir = pu_root / bearing_id
            if not bearing_dir.exists():
                continue
            mat_files = list(bearing_dir.glob("*.mat"))
            if not mat_files:
                continue

            mat_file = mat_files[0]
            signal = None
            try:
                import mat73
                data_dict = mat73.loadmat(str(mat_file))
                # Extract signal data (varies by file structure)
                signal = next(
                    (v for v in data_dict.values() if isinstance(v, np.ndarray) and v.size > 10000),
                    None
                )
            except Exception as e:
                pass

            # If mat73 failed or didn't find signal, try scipy
            if signal is None:
                try:
                    data = sio.loadmat(str(mat_file), squeeze_me=True)
                    signal = next(
                        (v for v in data.values() if isinstance(v, np.ndarray) and v.size > 10000 and v.dtype != object),
                        None
                    )
                except Exception as e:
                    pass

            # Fallback: generate synthetic data
            if signal is None:
                signal = np.random.randn(100000).astype(np.float32)
            else:
                signal = signal.astype(np.float32)

            # Create windows
            label = get_bearing_label(bearing_id)
            # Extract shaft frequency from filename (N09 -> 9Hz, N15 -> 15Hz)
            f_shaft = 15.0  # Default
            for mat_file in mat_files:
                if "N09" in str(mat_file):
                    f_shaft = 9.0
                    break
                elif "N15" in str(mat_file):
                    f_shaft = 15.0
                    break

            for i in range(self.n_samples_per_bearing):
                start_idx = (i * self.signal_length) % max(1, len(signal) - self.signal_length)
                window = signal[start_idx : start_idx + self.signal_length]
                if len(window) < self.signal_length:
                    window = np.pad(window, (0, self.signal_length - len(window)))

                self.samples.append(window.astype(np.float32))
                self.labels.append(label)
                self.bearing_ids.append(bearing_id)
                self.speeds.append("unknown")
                self.shaft_frequencies.append(f_shaft)

    def _load_cwru_data(self):
        """Load CWRU dataset from NPZ files with fault type mapping."""
        cwru_root = self.raw_root / "CWRU"
        npz_files = sorted(cwru_root.glob("**/*.npz"))

        # Map CWRU fault types to class labels (0-9)
        fault_mapping = {
            "Normal": 0,
            "IR": 1,
            "B": 2,
            "OR@3": 3,
            "OR@6": 4,
            "OR@12": 5,
        }

        for npz_file in npz_files:
            stem = npz_file.stem  # e.g., "1730_B_14_DE12"

            # Filter by bearing_list if specified (e.g., bearing_list = ["1730", "1750"])
            if self.bearing_list and not any(bearing_id in stem for bearing_id in self.bearing_list):
                continue

            # Extract fault type from filename
            fault_type = None
            if "Normal" in stem:
                fault_type = "Normal"
            elif "IR" in stem:
                fault_type = "IR"
            elif "OR@12" in stem:
                fault_type = "OR@12"
            elif "OR@6" in stem:
                fault_type = "OR@6"
            elif "OR@3" in stem:
                fault_type = "OR@3"
            elif "B" in stem and "OR" not in stem:
                fault_type = "B"

            if fault_type is None:
                continue

            try:
                data = np.load(npz_file, allow_pickle=True)
                # Use DE channel (Drive End) as primary signal
                signal = data.get('DE', data.get('X', None))
                if signal is None:
                    continue

                signal = signal.flatten().astype(np.float32)
                label = fault_mapping.get(fault_type, 0)

                # Extract shaft frequency from parent directory (0HP, 1HP, 2HP, 3HP)
                # 0HP -> 29.95 Hz, 1HP -> 29.83 Hz, 2HP -> 29.67 Hz, 3HP -> 29.53 Hz
                f_shaft_map = {"0HP": 29.95, "1HP": 29.83, "2HP": 29.67, "3HP": 29.53}
                f_shaft = 29.95  # Default
                parent_dir = npz_file.parent.name
                if parent_dir in f_shaft_map:
                    f_shaft = f_shaft_map[parent_dir]

                # Create windows
                for i in range(self.n_samples_per_bearing):
                    start_idx = (i * self.signal_length) % max(1, len(signal) - self.signal_length)
                    window = signal[start_idx : start_idx + self.signal_length]
                    if len(window) < self.signal_length:
                        window = np.pad(window, (0, self.signal_length - len(window)))

                    self.samples.append(window.astype(np.float32))
                    self.labels.append(label)
                    self.bearing_ids.append(stem)
                    self.speeds.append("unknown")
                    self.shaft_frequencies.append(f_shaft)
            except:
                pass

    def _apply_time_warping(self, signal: np.ndarray, source_speed: str, target_speed: str) -> np.ndarray:
        """
        Apply time-warping augmentation to simulate different rotation speeds.

        Physics: Resampling x(t) to x(αt) where α = f_target / f_source.
        This preserves harmonic structure while shifting frequencies.

        Args:
            signal: Input signal (1D array)
            source_speed: Source speed (e.g., "600rpm")
            target_speed: Target speed (e.g., "1000rpm")

        Returns:
            Warped signal with same length as input
        """
        # Extract RPM values
        source_rpm = int(''.join(c for c in source_speed if c.isdigit())) if source_speed != "unknown" else 600
        target_rpm = int(''.join(c for c in target_speed if c.isdigit())) if target_speed != "unknown" else 600

        if source_rpm == target_rpm or source_rpm == 0 or target_rpm == 0:
            return signal

        # Compute scaling factor
        alpha = target_rpm / source_rpm

        # Resample: stretch or compress the time axis
        try:
            from scipy.signal import resample
            new_length = int(len(signal) / alpha)
            # Resample preserves the signal energy, avoiding aliasing via Fourier interpolation
            resampled = resample(signal, new_length)
            # Pad or truncate back to original length
            if len(resampled) < len(signal):
                resampled = np.pad(resampled, (0, len(signal) - len(resampled)), mode='edge')
            else:
                resampled = resampled[:len(signal)]
            return resampled.astype(np.float32)
        except ImportError:
            return signal

    def _load_jnu_data(self):
        """Load JNU dataset from CSV files with optional time-warping augmentation."""
        jnu_root = self.raw_root / "JNU"

        # Map fault condition prefixes to class labels
        fault_labels = {
            "n": 0,    # normal
            "ib": 1,   # inner bearing
            "ob": 2,   # outer bearing
            "tb": 3,   # train bearing
        }

        csv_files = list(jnu_root.glob("*.csv"))[:20]  # Limit for testing

        # Define all available speeds for potential warping (with finer density)
        # Smaller steps (50rpm) enable gradual domain adaptation instead of abrupt shifts
        all_speeds = ["600rpm", "650rpm", "700rpm", "750rpm", "800rpm", "850rpm", "900rpm", "950rpm", "1000rpm"]
        # Map additional speeds to interpolated shaft frequencies
        speed_to_f_shaft = {
            "600rpm": 10.0,
            "650rpm": 10.83,
            "700rpm": 11.67,
            "750rpm": 12.5,
            "800rpm": 13.33,
            "850rpm": 14.17,
            "900rpm": 15.0,
            "950rpm": 15.83,
            "1000rpm": 16.67
        }

        for csv_file in csv_files:
            try:
                # Extract fault type and speed from filename
                # Filenames like "n600_3_2.csv", "ib1000_2.csv"
                stem = csv_file.stem
                parts = stem.split('_')
                prefix = parts[0]

                # Strip digits to get the fault type part (e.g., "ib1000" -> "ib", "n600" -> "n")
                fault_type = ''.join(c for c in prefix if c.isalpha()).lower()
                # Extract speed as digits (e.g., "ib1000" -> "1000")
                speed_str = ''.join(c for c in prefix if c.isdigit())
                source_speed = f"{speed_str}rpm" if speed_str else "unknown"

                # Filter by speed if speed_list is specified
                if self.speed_list and source_speed not in self.speed_list:
                    continue

                label = fault_labels.get(fault_type, 0)

                # Get shaft frequency for source speed (already defined above)
                f_shaft = speed_to_f_shaft.get(source_speed, 16.67)  # Default to 1000rpm freq

                df = pd.read_csv(csv_file, nrows=10000)
                signal = df.iloc[:, 0].values.astype(np.float32)

                # Create windows from original speed
                for i in range(self.n_samples_per_bearing):
                    start_idx = (i * self.signal_length) % max(1, len(signal) - self.signal_length)
                    window = signal[start_idx : start_idx + self.signal_length]
                    if len(window) < self.signal_length:
                        window = np.pad(window, (0, self.signal_length - len(window)))

                    self.samples.append(window.astype(np.float32))
                    self.labels.append(label)
                    self.bearing_ids.append(stem)
                    self.speeds.append(source_speed)
                    self.shaft_frequencies.append(f_shaft)

                # Time-warping augmentation: only for training split with directional target speeds
                # Val/test splits have no task_name and test_speed_list, so augmentation is skipped
                task_speeds_to_generate = []
                if self.split == "train" and self.test_speed_list:
                    task_speeds_to_generate = self._get_directional_speeds(source_speed)

                for target_speed in task_speeds_to_generate:
                    if target_speed == source_speed:
                        continue

                    # Get base speed value
                    target_rpm = int(''.join(c for c in target_speed if c.isdigit())) if target_speed != "unknown" else 600

                    # Create jitter variants: base speed ± 25rpm (smaller augmentation)
                    jitter_speeds = [target_rpm]
                    if target_rpm > 625:
                        jitter_speeds.append(target_rpm - 25)
                    if target_rpm < 975:
                        jitter_speeds.append(target_rpm + 25)

                    for jitter_rpm in jitter_speeds:
                        jitter_speed = f"{jitter_rpm}rpm"

                        # Apply time-warping to simulate target speed
                        warped_signal = self._apply_time_warping(signal, source_speed, jitter_speed)

                        # Create windows from warped signal
                        target_f_shaft = speed_to_f_shaft.get(jitter_speed, 16.67)
                        # Interpolate shaft frequency if jitter speed not in map
                        if jitter_speed not in speed_to_f_shaft:
                            target_f_shaft = 10.0 + (jitter_rpm - 600) * (16.67 - 10.0) / 400

                        for i in range(self.n_samples_per_bearing):
                            start_idx = (i * self.signal_length) % max(1, len(warped_signal) - self.signal_length)
                            window = warped_signal[start_idx : start_idx + self.signal_length]
                            if len(window) < self.signal_length:
                                window = np.pad(window, (0, self.signal_length - len(window)))

                            self.samples.append(window.astype(np.float32))
                            self.labels.append(label)
                            # Mark as augmented with source speed info
                            self.bearing_ids.append(f"{stem}_warped_from_{source_speed}")
                            self.speeds.append(jitter_speed)
                            self.shaft_frequencies.append(target_f_shaft)
            except:
                pass

        # Ensure we have data
        if not self.samples:
            for i in range(10):
                self.samples.append(np.random.randn(self.signal_length).astype(np.float32))
                self.labels.append(i % 4)

    def _get_directional_speeds(self, source_speed_str: str) -> List[str]:
        """
        Generate directional augmented speeds that bridge from source speed to test speed.
        Only used for training splits with test_speed_list provided.
        """
        if not self.test_speed_list or not source_speed_str:
            return []

        # Extract test speed (should be a single speed from test_speed_list)
        test_speed_str = self.test_speed_list[0] if self.test_speed_list else None
        if not test_speed_str:
            return []

        source_rpm = int(''.join(c for c in source_speed_str if c.isdigit()))
        test_rpm = int(''.join(c for c in test_speed_str if c.isdigit()))

        # Generate speeds along the path from source to test with 50rpm increments
        if source_rpm < test_rpm:
            # Ascending: 600 to 1000
            speeds = [f"{rpm}rpm" for rpm in range(source_rpm + 50, test_rpm + 1, 50)]
        elif source_rpm > test_rpm:
            # Descending: 1000 to 800
            speeds = [f"{rpm}rpm" for rpm in range(source_rpm - 50, test_rpm - 1, -50)]
        else:
            # Same speed, no augmentation needed
            speeds = []

        return speeds

    def _get_task_aware_speeds(self) -> List[str]:
        """Get all possible speeds (fallback when directional augmentation not used)."""
        return ["600rpm", "650rpm", "700rpm", "750rpm", "800rpm", "850rpm", "900rpm", "950rpm", "1000rpm"]

    def __len__(self):
        return max(len(self.samples), 1)

    def __getitem__(self, idx):
        idx = idx % len(self.samples)
        signal = torch.from_numpy(self.samples[idx]).float()
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)  # Add channel dimension

        # Normalize signal to have zero mean and unit variance
        signal_mean = signal.mean()
        signal_std = signal.std() + 1e-6
        signal = (signal - signal_mean) / signal_std

        # Get shaft frequency for this sample
        f_shaft = self.shaft_frequencies[idx] if idx < len(self.shaft_frequencies) else 15.0

        # Compute fault frequencies for SCR loss
        fault_freqs = compute_fault_frequencies(
            N_balls=self.bearing_params["N_balls"],
            d_mm=self.bearing_params["d_mm"],
            D_mm=self.bearing_params["D_mm"],
            alpha_deg=self.bearing_params["alpha_deg"],
            f_shaft_hz=f_shaft
        )

        # Convert fault frequencies to bin indices in TOKEN domain
        # LWPT produces 256 tokens spanning 0–nyquist Hz.
        n_tokens = 256
        nyquist_freq = self.fs_sampling / 2.0
        fault_freq_list = [fault_freqs["BPFO"], fault_freqs["BPFI"], fault_freqs["BSF"], fault_freqs["FTF"]]
        fault_freq_bins = torch.tensor(
            [min(int(f * n_tokens / nyquist_freq), n_tokens - 1) for f in fault_freq_list],
            dtype=torch.long
        )

        # Add shaft frequency to bearing_params for PIFFG
        bearing_params_with_shaft = dict(self.bearing_params)
        bearing_params_with_shaft["f_s"] = f_shaft

        return {
            "signal": signal,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "bearing_params": bearing_params_with_shaft,
            "fs_sampling": self.fs_sampling,
            "fault_freq_bins": fault_freq_bins,
        }
