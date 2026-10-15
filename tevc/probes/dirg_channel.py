"""DIRG channel rule (RESS spec §5.2), applied before any target-speed record is read. For each radial channel of
accelerometer A1 (y = channel 2, z = channel 3), 5-fold time-blocked cross-validated accuracy of the diagnoser
(TICNN, 40 epochs, seed 42) on the library-speed (300 Hz) windows only. The higher mean is frozen by hand as
tevc.dirg_data.CHANNEL; an exact tie goes to y. Folds are contiguous fifths of every record, and training windows
that overlap a test window are purged.

    python -m tevc.probes.dirg_channel      -> outputs/tevc/dirg_channel.json
"""
import json
import subprocess
from datetime import datetime

import numpy as np

from tevc import dirg_data
from tevc.datasets import make_windows
from tevc.deep_train import predict, train_cnn
from tevc.dirg_data import DIRG, RADIAL
from tevc.run_stream import OUT_DIR

LIBRARY_HZ, K, MODEL, EPOCHS, SEED = 300, 5, "TICNN", 40, 42


def blocked_folds(w, k=K):
    """[(train indices, test indices)]: each record's windows, in time order, split into k contiguous blocks;
    training windows overlapping any test window of the same record are dropped."""
    fold = np.empty(len(w["rec"]), dtype=int)
    for rec in np.unique(w["rec"]):
        idx = np.flatnonzero(w["rec"] == rec)
        for f, part in enumerate(np.array_split(idx[np.argsort(w["span_lo"][idx], kind="stable")], k)):
            fold[part] = f
    out = []
    for f in range(k):
        test = np.flatnonzero(fold == f)
        overlap = np.zeros(len(fold), dtype=bool)
        for i in test:
            overlap |= (w["rec"] == w["rec"][i]) & (w["span_lo"] < w["span_hi"][i]) & (w["span_hi"] > w["span_lo"][i])
        out.append((np.flatnonzero(~overlap), test))
    return out


def main():
    if dirg_data.CHANNEL is not None:
        raise SystemExit(f"channel already frozen ({dirg_data.CHANNEL}); the rule runs once")
    out = {"spec": "docs/superpowers/specs/2026-10-13-ress-rpst-pcs-design.md §5.2", "library_hz": LIBRARY_HZ,
           "model": MODEL, "epochs": EPOCHS, "seed": SEED, "folds": K,
           "head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "time": datetime.now().isoformat(timespec="seconds"), "channels": {}}
    for ch in RADIAL:
        entries = DIRG.entries("library", LIBRARY_HZ, channel=ch)
        w = make_windows(entries, DIRG.win, DIRG.hop, DIRG.cap_fit)
        accs = []
        for train, test in blocked_folds(w):
            model = train_cnn(w["W"][train], w["y"][train], MODEL, EPOCHS, SEED)
            accs.append(float((predict(model, w["W"][test]).argmax(axis=1) == w["y"][test]).mean()))
        out["channels"][ch] = {"records": sorted({str(r) for r in w["rec"]}), "fold_acc": accs,
                               "mean_acc": float(np.mean(accs))}
        print(f"channel {ch}: fold accuracies {[round(a * 100, 1) for a in accs]} mean {np.mean(accs) * 100:.2f}")
    m = {ch: v["mean_acc"] for ch, v in out["channels"].items()}
    out["chosen"] = "y" if m["y"] >= m["z"] else "z"
    (OUT_DIR / "dirg_channel.json").write_text(json.dumps(out, indent=1))
    print("chosen:", out["chosen"])


if __name__ == "__main__":
    main()
