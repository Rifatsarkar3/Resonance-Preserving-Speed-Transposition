"""University of Ottawa bearing data under time-varying speed (Huang & Baddour; Mendeley v43hmbwxpm v2).

60 recordings = 5 health classes x 4 speed profiles (A up, B down, C up-down, D down-up) x 3 trials, 10 s at
200 kHz. Channel_1 = accelerometer, Channel_2 = encoder square wave with 1024 cycles per revolution (dataset
description). Entries are chunks whose speed stays inside a bin; a chunk id is "<code>-<profile>-<trial>#<a>:<b>"
with [a, b) the chunk's sample range in its recording.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io as sio

FS = 200000.0
ENCODER_CPR = 1024
ROOT = Path("data/raw/OTTAWA")
CLASSES = {"H": 0, "I": 1, "O": 2, "B": 3, "C": 4}
PROFILES = ("A", "B", "C", "D")
WIN, HOP = 20000, 10000        # 0.1 s windows: speed drifts < 1 % inside a window
GRID = 10000                   # 0.05 s cells for the in-bin test
CHUNK_MAX = 200000             # 1 s: speed drifts at most ~1-2 Hz inside a chunk


def pulse_edges(encoder, threshold=2.0):
    """Sample indices of rising edges of the encoder square wave."""
    e = np.asarray(encoder)
    return np.flatnonzero((e[1:] > threshold) & (e[:-1] <= threshold)) + 1


def speed_hz(edges, lo, hi):
    """Mean shaft speed (Hz) over samples [lo, hi): encoder cycles counted / (duration x CPR)."""
    n = np.searchsorted(edges, hi) - np.searchsorted(edges, lo)
    return n / ((hi - lo) / FS * ENCODER_CPR)


@lru_cache(maxsize=None)
def recording(code, profile, trial):
    """(vibration float32, encoder rising-edge indices) of one recording."""
    path = next(ROOT.glob(f"*/{code}-{profile}-{trial}.mat"))
    d = sio.loadmat(path)
    return np.ravel(d["Channel_1"]).astype(np.float32), pulse_edges(np.ravel(d["Channel_2"]))


def in_bin_chunks(edges, n_samples, bin_hz, grid=GRID, chunk_max=CHUNK_MAX, min_len=WIN):
    """[start, end) ranges where every `grid` cell's speed lies inside bin_hz = (lo, hi), cut to <= chunk_max."""
    lo_hz, hi_hz = bin_hz
    starts = np.arange(0, n_samples - grid + 1, grid)
    inside = [lo_hz <= speed_hz(edges, s, s + grid) < hi_hz for s in starts]
    chunks, run_start = [], None
    for s, ok in zip(list(starts) + [n_samples], inside + [False]):
        if ok and run_start is None:
            run_start = int(s)
        elif not ok and run_start is not None:
            for a in range(run_start, int(s), chunk_max):
                b = min(a + chunk_max, int(s))
                if b - a >= min_len:
                    chunks.append((a, b))
            run_start = None
    return chunks


class OTTAWA:
    """Adapter with the window/cap attributes of tevc.datasets.JNU/PU; entries are speed-bin chunks."""
    name, fs, win, hop = "ottawa", FS, WIN, HOP
    cap_fit, cap_eval = 80, 80
    n_classes = 5
    # ER16K bearing (Huang & Baddour 2018): BPFO = 3.57 f_r, BPFI = 5.43 f_r
    geometry = {"N_balls": 9, "d_mm": 7.94, "D_mm": 38.52, "alpha_deg": 0.0}

    @staticmethod
    def entries(trial, bin_hz):
        out = []
        for code, label in CLASSES.items():
            for profile in PROFILES:
                vib, edges = recording(code, profile, trial)
                for a, b in in_bin_chunks(edges, len(vib), bin_hz):
                    out.append((label, f"{code}-{profile}-{trial}#{a}:{b}", vib[a:b], 0.0, 1.0))
        return out

    @staticmethod
    def _parse(rec):
        name, span = str(rec).split("#")
        code, profile, trial = name.split("-")
        a, b = (int(v) for v in span.split(":"))
        return recording(code, profile, int(trial))[1], a, b

    @classmethod
    def chunk_rpm(cls, rec):
        """Measured mean speed (rpm) of a chunk."""
        edges, a, b = cls._parse(rec)
        return 60.0 * speed_hz(edges, a, b)

    @classmethod
    def window_rpm(cls, rec, span_lo, span_hi):
        """Measured mean speed (rpm) of a window, spans relative to its chunk."""
        edges, a, _ = cls._parse(rec)
        return 60.0 * speed_hz(edges, a + int(span_lo), a + int(span_hi))
