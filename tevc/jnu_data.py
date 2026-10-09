"""JNU recordings and time-blocked windowing.

One recording per (class, speed). Fault recordings are 500 499 samples (10 s at 50 kHz),
normal recordings 1 501 499. Windows are taken fully inside a [lo, hi) fraction of a recording,
so the two halves of a recording never share samples.
"""
from pathlib import Path

import numpy as np
import pandas as pd

FS = 50000.0
ROOT = Path("data/raw/JNU")
FAULT_LABELS = {"n": 0, "ib": 1, "ob": 2, "tb": 3}
# Li et al., Sensors 13:8013 (2013), Table 1: N205 (normal, outer-race and roller faults; 10 rollers) and NU205
# (inner-race fault; 11 rollers), roller diameter 7 mm, contact angle 0. Pitch diameter not reported: bore/OD mean
# (25 + 52) / 2 = 38.5 mm.
GEOMETRY = {"N_balls": 10, "N_balls_inner": 11, "d_mm": 7.0, "D_mm": 38.5, "alpha_deg": 0.0}


def load_recordings(jnu_dir):
    """Return {(label, rpm): signal} for every JNU csv (same parsing as the earlier RPST experiments)."""
    recs = {}
    for csv_file in sorted(Path(jnu_dir).glob("*.csv")):
        prefix = csv_file.stem.split("_")[0]
        fault = "".join(c for c in prefix if c.isalpha()).lower()
        rpm = int("".join(c for c in prefix if c.isdigit()))
        recs[(FAULT_LABELS[fault], rpm)] = pd.read_csv(csv_file).iloc[:, 0].values.astype(np.float32)
    return recs


def window_starts(n_samples, length, hop, lo=0.0, hi=1.0, cap=None):
    """Window start indices inside [lo, hi) of the recording, evenly thinned to at most `cap`."""
    a, b = int(lo * n_samples), int(hi * n_samples)
    starts = np.arange(a, b - length + 1, hop)
    if cap is not None and len(starts) > cap:
        starts = starts[np.linspace(0, len(starts) - 1, cap).round().astype(int)]
    return starts


def make_windows(signals, length, hop, lo=0.0, hi=1.0, cap=None):
    """signals: {label: 1-D array}, one recording per label.

    Returns windows (N, length), labels (N,), recording ids (N,), start indices (N,).
    """
    W, y, rec, st = [], [], [], []
    for label, sig in sorted(signals.items()):
        for s in window_starts(len(sig), length, hop, lo, hi, cap):
            W.append(sig[s:s + length])
            y.append(label)
            rec.append(label)
            st.append(s)
    return np.stack(W), np.array(y), np.array(rec), np.array(st)
