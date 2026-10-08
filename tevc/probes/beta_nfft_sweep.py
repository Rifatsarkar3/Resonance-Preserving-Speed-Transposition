"""Exploratory probe (after Gate-0 v2; PU treated as development data): do RPST's physics knobs matter?

beta = resonance-shift exponent (0 = RPST, 1 = resampling); n_fft = phase-vocoder frame (transient smearing).
For each synthesised library: median marginal fidelity SMD vs real unlabelled target windows (reps 3-4), number of
features passing the 0.5 fidelity screen, and k-NN accuracy on real test windows (reps 5-6) with all / screened features.
Run from repo root:  python -m tevc.probes.beta_nfft_sweep   ->  outputs/tevc/probe_beta_nfft.json
"""
import json
from pathlib import Path

import numpy as np

from tevc import features, synthesis
from tevc.datasets import PU, make_windows
from tevc.fidelity import fidelity_screen
from tevc.knn_fitness import KNNEvaluator

OUT = Path("outputs/tevc/probe_beta_nfft.json")
CONFIGS = [("identity (stale)", None)] + [(f"beta={b:.2f} nfft=256", (b, 256)) for b in (0, 0.25, 0.5, 0.75, 1.0)] \
          + [(f"beta=0 nfft={n}", (0.0, n)) for n in (64, 128, 512, 1024)]


def feature_windows(entries, rpm, transform=None, rate=1.0):
    w = make_windows(entries, PU.win, PU.hop, PU.cap_fit, transform, rate)
    return features.extract(w["W"], PU.fs, rpm / 60.0, PU.geometry, znorm=True), w


def main():
    results = {}
    for src, tgt in PU.tasks:
        task = f"{src}->{tgt}"
        A, _ = feature_windows(PU.entries("A", tgt), tgt)          # unlabelled: labels never used
        B, wb = feature_windows(PU.entries("B", tgt), tgt)
        results[task] = {}
        for name, cfg in CONFIGS:
            if cfg is None:
                X, w = feature_windows(PU.entries("library", src), src)
            else:
                beta, n_fft = cfg
                X, w = feature_windows(PU.entries("library", src), tgt,
                                       lambda y, beta=beta, n_fft=n_fft: synthesis.transpose(y, src, tgt, beta, n_fft),
                                       tgt / src)
            allowed, smd = fidelity_screen(X, A)
            ev = KNNEvaluator(X, w["y"], w["rec"], w["span_lo"], w["span_hi"])
            row = {"median_smd": float(np.median(smd)), "n_allowed": int(allowed.sum()),
                   "knn_acc_all": float(1 - ev.test_error(np.ones((1, X.shape[1]), bool), B, wb["y"])[0]),
                   "knn_acc_screened": float(1 - ev.test_error(allowed[None], B, wb["y"])[0])}
            results[task][name] = row
            print(f"PU {task} {name:20s} SMD={row['median_smd']:.3f} n={row['n_allowed']:3d} "
                  f"all={row['knn_acc_all'] * 100:5.1f}% screened={row['knn_acc_screened'] * 100:5.1f}%", flush=True)
    OUT.write_text(json.dumps(results, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
