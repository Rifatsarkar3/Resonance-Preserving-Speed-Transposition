"""Exploratory probe (after Gate-0 v1): how faithfully does each synthesiser reproduce each feature family?

Class-conditional standardised mean difference (SMD) between a candidate dataset and real JNU 1000 rpm data (half B),
median per feature family. Uses the Gate-0 v1 feature cache (run `python -m tevc.run_gate0` first).
Run from repo root:  python -m tevc.probes.fidelity_by_family   ->  outputs/tevc/probe_fidelity.json
"""
import json
from pathlib import Path

import numpy as np

from tevc.features import feature_names

CACHE = Path("outputs/tevc/cache")
OUT = Path("outputs/tevc/probe_fidelity.json")
FAMILIES = ("time", "fft_band", "wpt_loge", "wpt_kurt", "wpt_crest", "env_order")
CANDIDATES = {"real 600 rpm (no transposition)": "real600_full", "RPST 600->1000": "rpst600to1000",
              "resampling 600->1000": "rs600to1000", "real 1000 rpm half A (noise floor)": "real1000_A"}


def family(name):
    if name.startswith("t_"):
        return "time"
    if name.startswith("fft"):
        return "fft_band"
    if name.startswith("wpt"):
        return "wpt_" + name.split("_")[-1]
    return "env_order"


def load(tag):
    z = np.load(CACHE / f"{tag}.npz")
    return z["X"], z["y"]


def class_conditional_smd(Xa, ya, Xb, yb):
    per_class = []
    for c in np.unique(yb):
        a, b = Xa[ya == c], Xb[yb == c]
        pooled = np.sqrt(0.5 * (a.var(axis=0) + b.var(axis=0))) + 1e-8
        per_class.append(np.abs(a.mean(axis=0) - b.mean(axis=0)) / pooled)
    return np.mean(per_class, axis=0)


def main():
    names = np.array(feature_names())
    fams = np.array([family(n) for n in names])
    Xt, yt = load("real1000_B")
    smd = {label: class_conditional_smd(*load(tag), Xt, yt) for label, tag in CANDIDATES.items()}
    table = {f: {label: float(np.median(v[fams == f])) for label, v in smd.items()} for f in FAMILIES}
    rpst, none = smd["RPST 600->1000"], smd["real 600 rpm (no transposition)"]
    result = {"reference": "real 1000 rpm, half B", "median_smd_by_family": table,
              "n_features_rpst_worse_than_no_transposition": int((rpst > none).sum()),
              "worst_rpst_features": names[np.argsort(-rpst)[:8]].tolist()}
    OUT.write_text(json.dumps(result, indent=1))
    for f, row in table.items():
        print(f"{f:10s} " + "  ".join(f"{k[:24]}={v:.2f}" for k, v in row.items()))
    print("worst RPST features:", result["worst_rpst_features"], "->", OUT)


if __name__ == "__main__":
    main()
