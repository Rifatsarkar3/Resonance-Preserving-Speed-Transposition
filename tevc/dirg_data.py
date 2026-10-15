"""Politecnico di Torino rolling-bearing rig, variable speed and load (Daga, Fasana, Marchesiello & Garibaldi,
MSSP 120:252, 2019; Zenodo 10.5281/zenodo.3559553, CC BY 4.0, VariableSpeedAndLoad.zip). RESS spec §5.

Bearing B1: 10 rollers, d = 9.0 mm, pitch diameter 40.5 mm, contact angle 0 (BPFO 3.89, BPFI 6.11, BSF 2.14 f_s).
Classes: 0A healthy; 1A, 2A, 3A inner-ring indentation 450/250/150 µm; 4A, 5A, 6A roller indentation
450/250/150 µm.
Files CnA_fff_vvv_m.mat: n = class, fff = nominal speed (Hz), vvv = load-cell output (mV, 0.499 mV/N), m = 2 for a
repeated measurement. Each file holds one 512000 x 6 matrix in m/s^2 (10 s at 51.2 kHz) named like the file;
columns 1-3 are accelerometer A1 on B1 (axial x, radial y, radial z).
Nominal 1000 N (load cell 496-506 mV) is the only load run at all five speeds. It has one record per class and
speed; 1A exists only as repeat m = 2. The rig has no tachometer and the actual speed is below nominal,
increasingly with load, so speeds are estimated label-free (estimate_speed_hz).
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io as sio

FS = 51200.0
ROOT = Path("data/raw/DIRG")
SPEEDS_HZ = (100, 200, 300, 400, 500)
LOAD_MV = (450, 550)            # nominal 1000 N
RADIAL = {"y": 1, "z": 2}       # matrix columns of channels 2 and 3 (accelerometer A1, radial)
CHANNEL = "z"                   # frozen 2026-10-15 by the channel rule on the 300 Hz records only (z 100.00 % vs
                                # y 98.93 %, 5-fold blocked CV; outputs/tevc/dirg_channel.json)
GEOMETRY = {"N_balls": 10, "d_mm": 9.0, "D_mm": 40.5, "alpha_deg": 0.0}
SPEED_BAND = (0.85, 1.00)       # search band for the shaft speed, x nominal
ZERO_PAD = 8


def record_path(cls, speed_hz, root=None):
    """The single nominal-1000 N record of class cls (0-6) at nominal speed speed_hz."""
    root = Path(root or ROOT)
    hits = [p for p in sorted(root.glob(f"C{cls}A_{speed_hz}_*_*.mat"))
            if LOAD_MV[0] <= int(p.stem.split("_")[2]) <= LOAD_MV[1]]
    if len(hits) != 1:
        raise FileNotFoundError(f"C{cls}A at {speed_hz} Hz: {len(hits)} records at 1000 N in {root}, expected 1")
    return hits[0]


@lru_cache(maxsize=None)
def record(cls, speed_hz, column):
    """One channel (matrix column) of a record, float32."""
    path = record_path(cls, speed_hz)
    return np.asarray(sio.loadmat(path)[path.stem][:, column], dtype=np.float32)


def estimate_speed_hz(W, fs, nominal_hz, band=SPEED_BAND, pad=ZERO_PAD, chunk=64):
    """Label-free shaft speed (Hz): peak of the mean Hann amplitude spectrum of the windows inside band x nominal,
    zero-padded pad-fold and refined by parabolic interpolation of the log amplitude."""
    W = np.asarray(W, dtype=np.float64)
    n, hann = W.shape[1] * pad, np.hanning(W.shape[1])
    A = sum(np.abs(np.fft.rfft((c - c.mean(axis=1, keepdims=True)) * hann, n=n, axis=1)).sum(axis=0)
            for c in np.array_split(W, max(1, len(W) // chunk))) / len(W)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    idx = np.flatnonzero((f >= band[0] * nominal_hz) & (f <= band[1] * nominal_hz))
    k = int(idx[np.argmax(A[idx])])
    a, b, c = np.log(A[k - 1:k + 2] + 1e-30)
    offset = 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c < 0 else 0.0
    return float(f[k] + offset * (f[1] - f[0]))


class DIRG:
    """Adapter with the window/cap attributes of tevc.datasets.JNU/PU (RESS spec §5.4): 0.1 s windows, hop 0.05 s;
    the library uses whole records, a target record gives U (first 5 s) and T (last 5 s)."""
    name, fs, win, hop = "dirg", FS, 5120, 2560
    cap_fit, cap_eval = 80, 57
    n_classes = 7
    geometry = GEOMETRY
    _span = {"library": (0.0, 1.0), "U": (0.0, 0.5), "T": (0.5, 1.0)}

    @classmethod
    def entries(cls, role, speed_hz, channel=None):
        channel = channel or CHANNEL
        if channel is None:
            raise RuntimeError("DIRG channel not frozen: run tevc.probes.dirg_channel first (spec §5.2)")
        lo, hi = cls._span[role]
        return [(c, record_path(c, speed_hz).stem, record(c, speed_hz, RADIAL[channel]), lo, hi)
                for c in range(cls.n_classes)]
