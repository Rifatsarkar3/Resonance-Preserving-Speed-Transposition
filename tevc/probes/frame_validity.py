"""EXPLORATORY mechanism probe (MSSP spec §5 M7, after the Lee verdict 3bdb9b9): a phase vocoder (β = 0) re-spaces
impacts only while its frame is shorter than the impact period; with longer frames it treats the impact train as a
pitch and keeps its rate, so the transposed signal keeps the source speed's fault frequency. Candidate mechanism for
DIRG's synthesis ceiling (shaft 100-500 Hz: inner-race impacts ~30 samples apart at 51.2 kHz).

  synthetic  ringing impacts (resonance fs/8, decay time constant period/8, 2 % noise floor: without one the vocoder
             cancels alternate impacts) every P samples, transposed by speed ratios 0.6, 1.5, 2.0 with β = 0 at
             n_fft 64-512 for P = 50-1600 samples (3 seeds): impacts found and their median spacing against the
             expected ones; boundary = the largest frame / P up to which every cell is within 10 %. Then the table:
             per dataset, fault periods from the bearing geometry at the library speed, and frame / shortest period
             for the operator frames the methods use (recipe n_fft set and the TV frame, scaled by frame_scale).
  dirg       DIRG real-data frame sweep: the library transposed with β = 0 at n_fft 16-256 against the real U windows
             (tevc.probes.operator_fidelity metrics), plus resampling and untransposed.

    python -m tevc.probes.frame_validity synthetic   -> outputs/tevc/probe_frame_validity.json
    python -m tevc.probes.frame_validity dirg        -> outputs/tevc/dirg/exploratory_frame_sweep.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, hilbert

from tevc import synthesis
from tevc.datasets import JNU, PU
from tevc.dirg_data import DIRG
from tevc.features import compute_fault_frequencies
from tevc.lee_data import LEE
from tevc.ottawa_data import OTTAWA
from tevc.probes.operator_fidelity import fidelity_by_env
from tevc.recipe import N_FFTS, frame
from tevc.rpstda import TV_N_FFT
from tevc.stream import dirg_stream

OUT = Path("outputs/tevc/probe_frame_validity.json")
SWEEP_OUT = Path("outputs/tevc/dirg/exploratory_frame_sweep.json")
N_SAMPLES = 200_000
SPEED_RATIOS = (0.6, 1.5, 2.0)
PERIODS = (50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600)
TOLERANCE = 0.10
# Library speeds (rpm): PU and JNU nominal; Ottawa = median encoder speed of the 13-16 Hz library chunks; DIRG and Lee
# = the label-free estimates of their streams (dirg/exploratory_fidelity.json, lee_speed_band.json).
LIBRARIES = {"pu": (PU, 900.0), "jnu": (JNU, 600.0), "ottawa": (OTTAWA, 904.1), "dirg": (DIRG, 16690.6),
             "lee": (LEE, 801.7)}
DIRG_SWEEP = (16, 32, 64, 128, 256)


def impact_train(period, seed, n=N_SAMPLES, noise=0.02):
    """Ringing impacts every `period` samples: a resonance at fs/8 decaying with time constant period/8, random
    phase per impact, over a noise floor."""
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    since = np.mod(k, period)
    phase = rng.uniform(0, 2 * np.pi, n // period + 2)[k // period]
    x = np.exp(-since / (period / 8.0)) * np.sin(2 * np.pi * since / 8.0 + phase)
    return (x + noise * rng.standard_normal(n)).astype(np.float32)


def recovery(y, period, edge=2000):
    """(impacts found / expected, median spacing / period) from the envelope peaks of y, for an expected period."""
    e = np.abs(hilbert(y))[edge:-edge]
    peaks, _ = find_peaks(e, height=0.3 * np.percentile(e, 99.5), distance=max(1, int(0.6 * period)))
    spacing = float(np.median(np.diff(peaks)) / period) if len(peaks) > 2 else float("nan")
    return len(peaks) / (len(e) / period), spacing


def boundary(cells):
    """Largest frame / period ratio up to which every cell (all speed ratios) recovers; 0 if the smallest fails."""
    upto = 0.0
    for ratio in sorted({c["frame_ratio"] for c in cells}):
        if not all(c["ok"] for c in cells if c["frame_ratio"] == ratio):
            break
        upto = ratio
    return upto


def synthetic(seeds=3):
    cells = []
    for r in SPEED_RATIOS:
        for period in PERIODS:
            for n_fft in N_FFTS:
                found, spacing = np.mean([recovery(synthesis.transpose(impact_train(period, s), 600.0, 600.0 * r,
                                                                       0.0, n_fft), period / r)
                                          for s in range(seeds)], axis=0)
                cells.append({"speed_ratio": r, "period": period, "n_fft": n_fft, "frame_ratio": n_fft / period,
                              "found": float(found), "spacing": float(spacing),
                              "ok": bool(abs(found - 1) <= TOLERANCE and abs(spacing - 1) <= TOLERANCE)})
                print(f"r {r:3.1f} period {period:5d} n_fft {n_fft:3d} ratio {n_fft / period:6.3f}  found "
                      f"{found:5.2f}  spacing {spacing:5.2f}", flush=True)
    return {"cells": cells, "boundary": boundary(cells)}


def impact_periods(ds, rpm):
    """Impact period in samples per defect at shaft speed rpm: BPFO, BPFI and 2 x BSF (a rolling-element defect
    strikes both races once per spin)."""
    f = compute_fault_frequencies(f_shaft_hz=rpm / 60.0, **ds.geometry)
    return {"BPFO": ds.fs / f["BPFO"], "BPFI": ds.fs / f["BPFI"], "2BSF": ds.fs / (2.0 * f["BSF"])}


def table(limit):
    rows = {}
    for name, (ds, rpm) in LIBRARIES.items():
        periods = impact_periods(ds, rpm)
        shortest = min(periods.values())
        frames = {f"n_fft {n}": frame(n, ds) for n in N_FFTS} | {f"TV n_fft {TV_N_FFT}": frame(TV_N_FFT, ds)}
        rows[name] = {"fs": ds.fs, "library_rpm": rpm, "impact_period_samples": periods,
                      "frames": {k: {"samples": v, "ratio": v / shortest, "in_regime": bool(v / shortest <= limit)}
                                 for k, v in frames.items()}}
    return rows


def dirg_sweep():
    stream = dirg_stream()
    bpfi = impact_periods(DIRG, stream.library_rpm)["BPFI"]
    ops = {"untransposed": None, **{f"beta=0 n_fft={n} (RPST)": (0.0, n) for n in DIRG_SWEEP},
           "beta=1 (resampling)": (1.0, 256)}
    out = {"note": "exploratory, after the DIRG verdict and the Lee verdict", "library_rpm": stream.library_rpm,
           "bpfi_period_samples": bpfi, "frame_ratio": {str(n): n / bpfi for n in DIRG_SWEEP},
           "environments": fidelity_by_env(stream, ops, log=lambda s: print(s, flush=True))}
    SWEEP_OUT.write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("synthetic", "dirg"))
    if ap.parse_args().step == "dirg":
        dirg_sweep()
        return
    syn = synthetic()
    OUT.write_text(json.dumps({"note": "exploratory, after the Lee verdict", "synthetic": syn,
                               "table": table(syn["boundary"])}, indent=1))
    print("boundary", syn["boundary"], "->", OUT, flush=True)


if __name__ == "__main__":
    main()
