"""Gate-0 v2, pre-registered in docs/superpowers/specs/2026-10-08-tevc-rpst-agent-design.md.

Fixes from the v1 diagnostic: per-window z-norm, label-free fidelity screen, mixed library + synthesised
fitness data; deployed accuracy is the primary metric (criteria V2-a..c + dynamics check).
Run from repo root:
    python -m tevc.run_gate0_v2 --dataset jnu     # development
    python -m tevc.run_gate0_v2 --dataset pu      # confirmatory, design frozen
    (--smoke: first task, one seed, 10 generations)
Output: outputs/tevc/gate0v2_<dataset>.json
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from tevc import features, synthesis
from tevc.datasets import DATASETS, make_windows
from tevc.fidelity import fidelity_screen
from tevc.fs_nsga import run_nsga2
from tevc.knn_fitness import KNNEvaluator
from tevc.run_gate0 import add_igd, jaccard, score

SEEDS = [42, 1337, 2025, 999, 7]
ARMS = ["STALE", "RPST", "RPST+FID", "MIX", "MIX+FID", "MIX-RS+FID", "ORACLE"]
OUT_DIR = Path("outputs/tevc")
CACHE = OUT_DIR / "cache"
KEYS = ("X", "y", "rec", "span_lo", "span_hi")


def feature_set(ds, tag, entries, rpm, cap, transform=None, rate=1.0):
    """Windows + z-normalised features at shaft speed rpm; cached as outputs/tevc/cache/<ds>_<tag>_zn.npz."""
    path = CACHE / f"{ds.name}_{tag}_zn.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in KEYS}
    w = make_windows(entries(), ds.win, ds.hop, cap, transform, rate)
    s = {"X": features.extract(w["W"], ds.fs, rpm / 60.0, ds.geometry, znorm=True),
         **{k: w[k] for k in KEYS[1:]}}
    np.savez(path, **s)
    return s


def concat(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in KEYS}


def build_sets(ds, tasks):
    sets = {}
    for src, tgt in tasks:
        def lib_entries(src=src):
            return ds.entries("library", src)

        sets[f"lib{src}"] = feature_set(ds, f"lib{src}", lib_entries, src, ds.cap_fit)
        sets[f"A{tgt}"] = feature_set(ds, f"A{tgt}", lambda: ds.entries("A", tgt), tgt, ds.cap_eval)
        sets[f"B{tgt}"] = feature_set(ds, f"B{tgt}", lambda: ds.entries("B", tgt), tgt, ds.cap_eval)
        for op_name, op in (("rpst", synthesis.rpst), ("rs", synthesis.resample)):
            synth = feature_set(ds, f"{op_name}{src}to{tgt}", lib_entries, tgt, ds.cap_fit,
                                transform=lambda y, op=op: op(y, src, tgt), rate=tgt / src)
            sets[f"{op_name}{src}to{tgt}"] = synth
            sets[f"mix-{op_name}{src}to{tgt}"] = concat(sets[f"lib{src}"], synth)
    return sets


def paired_p(runs, a, b, metric="deployed_acc"):
    vals = {(r["task"], r["seed"], r["arm"]): r[metric] for r in runs}
    keys = sorted({(r["task"], r["seed"]) for r in runs})
    x, y = np.array([vals[k + (a,)] for k in keys]), np.array([vals[k + (b,)] for k in keys])
    return 1.0 if np.allclose(x, y) else float(wilcoxon(x, y, alternative="greater").pvalue)


def summarize(runs):
    summary = {}
    for task in dict.fromkeys(r["task"] for r in runs):
        summary[task] = {arm: {m: [float(np.mean(v)), float(np.std(v))]
                               for m in ("deployed_acc", "hv", "igd", "deployed_n_features")
                               for v in [[r[m] for r in runs if r["task"] == task and r["arm"] == arm]]}
                         for arm in ARMS}
        by_seed = {}
        for r in runs:
            if r["task"] == task:
                by_seed.setdefault(r["seed"], {})[r["arm"]] = r["deployed_features"]
        summary[task]["jaccard_vs_oracle"] = {arm: float(np.mean([jaccard(d[arm], d["ORACLE"]) for d in by_seed.values()]))
                                              for arm in ARMS if arm != "ORACLE"}
    return summary


def gate_stats(runs, summary):
    mean_acc = {arm: float(np.mean([s[arm]["deployed_acc"][0] for s in summary.values()])) for arm in ARMS}
    gap = mean_acc["ORACLE"] - mean_acc["STALE"]
    return {"mean_deployed_acc": mean_acc,
            "mean_hv": {arm: float(np.mean([s[arm]["hv"][0] for s in summary.values()])) for arm in ARMS},
            "dynamics_gap_pp": 100 * gap,
            "V2a_p_mixfid_gt_stale": paired_p(runs, "MIX+FID", "STALE"),
            "V2b_p_mixfid_gt_mixrsfid": paired_p(runs, "MIX+FID", "MIX-RS+FID"),
            "V2c_gap_closure": (mean_acc["MIX+FID"] - mean_acc["STALE"]) / gap if gap > 0 else None,
            "hv_p_mixfid_gt_stale": paired_p(runs, "MIX+FID", "STALE", "hv"),
            "hv_p_mixfid_gt_mixrsfid": paired_p(runs, "MIX+FID", "MIX-RS+FID", "hv")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    ds = DATASETS[args.dataset]
    tasks, seeds, n_gen = (ds.tasks[:1], SEEDS[:1], 10) if args.smoke else (ds.tasks, SEEDS, 100)
    out_path = OUT_DIR / f"gate0v2_{ds.name}{'_smoke' if args.smoke else ''}.json"
    CACHE.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    sets = build_sets(ds, tasks)
    D = len(features.feature_names())
    ev = {name: KNNEvaluator(s["X"], s["y"], s["rec"], s["span_lo"], s["span_hi"])
          for name, s in sets.items() if not name.startswith("B")}
    screens, fidelity_info = {}, {}
    for src, tgt in tasks:
        for op_name in ("rpst", "rs"):
            key = f"{op_name}{src}to{tgt}"
            allowed, smd = fidelity_screen(sets[key]["X"], sets[f"A{tgt}"]["X"])   # A used WITHOUT labels
            screens[key] = allowed
            fidelity_info[key] = {"n_allowed": int(allowed.sum()), "median_smd": float(np.median(smd))}
    print(f"[{ds.name}] sets ready in {time.time() - t0:.0f}s, D={D}; windows: "
          + ", ".join(f"{k}={len(v['y'])}" for k, v in sets.items()), flush=True)
    print("fidelity screen:", json.dumps(fidelity_info), flush=True)

    all_features_acc = {}
    ones = np.ones((1, D), dtype=bool)
    for src, tgt in tasks:
        B = sets[f"B{tgt}"]
        all_features_acc[f"{src}->{tgt}"] = {
            key: float(1.0 - ev[key].test_error(ones, B["X"], B["y"])[0])
            for key in (f"lib{src}", f"rpst{src}to{tgt}", f"mix-rpst{src}to{tgt}", f"mix-rs{src}to{tgt}", f"A{tgt}")}

    results = {"spec": "docs/superpowers/specs/2026-10-08-tevc-rpst-agent-design.md (Gate-0 v2)",
               "dataset": ds.name, "start_time": datetime.now().isoformat(),
               "config": {"win": ds.win, "hop": ds.hop, "cap_fit": ds.cap_fit, "cap_eval": ds.cap_eval, "k": 5,
                          "pop": 100, "n_gen": n_gen, "seeds": seeds, "D": D, "znorm": True,
                          "fidelity_tau": 0.5, "fidelity_min_keep": 20},
               "fidelity": fidelity_info, "all_features_acc": all_features_acc, "runs": []}
    for seed in seeds:
        selections = {}
        for src, tgt in tasks:
            r_key, rs_key = f"rpst{src}to{tgt}", f"rs{src}to{tgt}"
            plan = {"STALE": (f"lib{src}", None),
                    "RPST": (r_key, None),
                    "RPST+FID": (r_key, r_key),
                    "MIX": (f"mix-{r_key}", None),
                    "MIX+FID": (f"mix-{r_key}", r_key),
                    "MIX-RS+FID": (f"mix-{rs_key}", rs_key),
                    "ORACLE": (f"A{tgt}", None)}
            for arm, (data_key, screen_key) in plan.items():
                sel_id = (data_key, screen_key)
                if sel_id not in selections:
                    selections[sel_id] = run_nsga2(ev[data_key], D, seed, n_gen=n_gen,
                                                   allowed=screens[screen_key] if screen_key else None)
                masks, F = selections[sel_id]
                r = score(masks, F, ev[data_key], sets[f"B{tgt}"])
                results["runs"].append({"task": f"{src}->{tgt}", "seed": seed, "arm": arm, **r})
                print(f"seed {seed:4d} {src}->{tgt:<5d} {arm:11s} acc={r['deployed_acc'] * 100:5.1f}% "
                      f"HV={r['hv']:.4f} nfeat={r['deployed_n_features']}", flush=True)
        out_path.write_text(json.dumps(results))

    add_igd(results["runs"])
    results["summary"] = summarize(results["runs"])
    results["gate_stats"] = None if args.smoke else gate_stats(results["runs"], results["summary"])
    results["end_time"] = datetime.now().isoformat()
    out_path.write_text(json.dumps(results, indent=1))

    print(f"\n{'task':10s} {'arm':11s} {'deployed acc %':>15s} {'HV':>15s} {'nfeat':>6s} {'J~oracle':>8s}")
    for task, s in results["summary"].items():
        for arm in ARMS:
            m = s[arm]
            j = s["jaccard_vs_oracle"].get(arm, 1.0)
            print(f"{task:10s} {arm:11s} {m['deployed_acc'][0] * 100:6.1f}+/-{m['deployed_acc'][1] * 100:4.1f} "
                  f"{m['hv'][0]:.4f}+/-{m['hv'][1]:.4f} {m['deployed_n_features'][0]:6.1f} {j:8.2f}")
    print("\nall-features k-NN acc:", json.dumps(all_features_acc))
    if results["gate_stats"]:
        print("gate stats:", json.dumps(results["gate_stats"], indent=1))
    print(f"done in {time.time() - t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
