"""Gate-0 pilot: does feature re-selection on RPST-synthesised data track the real new-speed optimum?

Spec and pre-registered criteria (G0-a..G0-d): docs/superpowers/specs/2026-10-08-tevc-rpst-agent-design.md
Run from repo root:
    python -m tevc.run_gate0            # 4 tasks x 5 seeds, 100 generations
    python -m tevc.run_gate0 --smoke    # 1 task, 1 seed, 10 generations
Output: outputs/tevc/gate0_results.json (features cached in outputs/tevc/cache/)
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from scipy.stats import wilcoxon

from src.utils.config import Config
from tevc import features, jnu_data, synthesis
from tevc.fs_nsga import nd_indices, run_nsga2
from tevc.knn_fitness import KNNEvaluator

WIN, HOP = 12000, 6000
CAP_FIT, CAP_HALF = 80, 40          # windows per class: fitness sets / each real target half
TASKS = [(600, 800), (600, 1000), (1000, 800), (1000, 600)]
SEEDS = [42, 1337, 2025, 999, 7]
ARMS = ["STALE", "STALE+RETRAIN", "RESEL-RPST", "RESEL-RS", "ORACLE"]
OUT_DIR = Path("outputs/tevc")
CACHE = OUT_DIR / "cache"
HV_REF = np.array([1.0, 1.0])


def feature_set(tag, make_signals, rpm, lo=0.0, hi=1.0, cap=CAP_FIT):
    """Windows + features of one dataset; cached as outputs/tevc/cache/<tag>.npz."""
    path = CACHE / f"{tag}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in ("X", "y", "rec", "start")}
    W, y, rec, start = jnu_data.make_windows(make_signals(), WIN, HOP, lo, hi, cap)
    X = features.extract(W, jnu_data.FS, rpm / 60.0, jnu_data.GEOMETRY)
    np.savez(path, X=X, y=y, rec=rec, start=start)
    return {"X": X, "y": y, "rec": rec, "start": start}


def build_sets(recs, tasks):
    def real(rpm):
        return lambda: {lab: recs[(lab, rpm)] for lab in range(4)}

    def synth(op, src, tgt):
        return lambda: {lab: op(recs[(lab, src)], src, tgt) for lab in range(4)}

    sets = {}
    for src, tgt in tasks:
        sets[f"lib{src}"] = feature_set(f"real{src}_full", real(src), src)
        sets[f"A{tgt}"] = feature_set(f"real{tgt}_A", real(tgt), tgt, 0.0, 0.5, CAP_HALF)
        sets[f"B{tgt}"] = feature_set(f"real{tgt}_B", real(tgt), tgt, 0.5, 1.0, CAP_HALF)
        sets[f"rpst{src}to{tgt}"] = feature_set(f"rpst{src}to{tgt}", synth(synthesis.rpst, src, tgt), tgt)
        sets[f"rs{src}to{tgt}"] = feature_set(f"rs{src}to{tgt}", synth(synthesis.resample, src, tgt), tgt)
    return sets


def score(masks, fit_F, scorer, test):
    """Re-score an arm's non-dominated masks on real test data; deploy = min fitness error, ties -> fewer features."""
    test_err = scorer.test_error(masks, test["X"], test["y"])
    ratio = masks.sum(axis=1) / masks.shape[1]
    true_F = np.column_stack([test_err, ratio])
    deployed = np.lexsort((ratio, fit_F[:, 0]))[0]
    return {"hv": float(HV(ref_point=HV_REF)(true_F[nd_indices(true_F)])),
            "deployed_acc": float(1.0 - test_err[deployed]),
            "deployed_n_features": int(masks[deployed].sum()),
            "deployed_features": np.flatnonzero(masks[deployed]).tolist(),
            "fit_err": fit_F[:, 0].tolist(), "test_err": test_err.tolist(), "ratio": ratio.tolist()}


def add_igd(runs):
    """IGD of each arm's true front against the best-known true front of its task (union of all runs)."""
    for task in sorted({r["task"] for r in runs}):
        task_runs = [r for r in runs if r["task"] == task]
        pts = np.vstack([np.column_stack([r["test_err"], r["ratio"]]) for r in task_runs])
        igd = IGD(pts[nd_indices(pts)])
        for r in task_runs:
            F = np.column_stack([r["test_err"], r["ratio"]])
            r["igd"] = float(igd(F[nd_indices(F)]))


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


def summarize(runs):
    summary = {}
    for task in sorted({r["task"] for r in runs}, key=lambda t: [r["task"] for r in runs].index(t)):
        summary[task] = {}
        for arm in ARMS:
            rs = [r for r in runs if r["task"] == task and r["arm"] == arm]
            summary[task][arm] = {m: [float(np.mean([r[m] for r in rs])), float(np.std([r[m] for r in rs]))]
                                  for m in ("hv", "deployed_acc", "igd", "deployed_n_features")}
        by_seed = {}
        for r in runs:
            if r["task"] == task:
                by_seed.setdefault(r["seed"], {})[r["arm"]] = r["deployed_features"]
        summary[task]["jaccard_stale_vs_oracle"] = float(np.mean([jaccard(d["STALE"], d["ORACLE"]) for d in by_seed.values()]))
        summary[task]["jaccard_rpst_vs_oracle"] = float(np.mean([jaccard(d["RESEL-RPST"], d["ORACLE"]) for d in by_seed.values()]))
    return summary


def gate_stats(runs, summary):
    def paired_p(a, b):
        hv = {(r["task"], r["seed"], r["arm"]): r["hv"] for r in runs}
        keys = sorted({(r["task"], r["seed"]) for r in runs})
        x = np.array([hv[k + (a,)] for k in keys])
        y = np.array([hv[k + (b,)] for k in keys])
        return 1.0 if np.allclose(x, y) else float(wilcoxon(x, y, alternative="greater").pvalue)

    gaps = {t: s["ORACLE"]["hv"][0] - s["STALE"]["hv"][0] for t, s in summary.items()}
    mean_hv = {arm: float(np.mean([s[arm]["hv"][0] for s in summary.values()])) for arm in ARMS}
    denom = mean_hv["ORACLE"] - mean_hv["STALE"]
    return {"mean_hv": mean_hv,
            "G0a_hv_gap_per_task": gaps,
            "G0a_n_tasks_gap_ge_0.03": int(sum(g >= 0.03 for g in gaps.values())),
            "G0b_p_rpst_gt_stale_retrain": paired_p("RESEL-RPST", "STALE+RETRAIN"),
            "G0c_p_rpst_gt_rs": paired_p("RESEL-RPST", "RESEL-RS"),
            "G0d_gap_closure": float((mean_hv["RESEL-RPST"] - mean_hv["STALE"]) / denom) if denom > 0 else None,
            "extra_p_rpst_gt_stale": paired_p("RESEL-RPST", "STALE"),
            "extra_p_retrain_gt_stale": paired_p("STALE+RETRAIN", "STALE")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tasks, seeds, n_gen = (TASKS[1:2], SEEDS[:1], 10) if args.smoke else (TASKS, SEEDS, 100)
    out_path = OUT_DIR / ("gate0_smoke.json" if args.smoke else "gate0_results.json")
    CACHE.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cfg = Config.from_yaml("config.yaml")
    recs = jnu_data.load_recordings(Path(cfg.data.jnu_raw_root) / "JNU")
    sets = build_sets(recs, tasks)
    D = len(features.feature_names())
    ev = {name: KNNEvaluator(s["X"], s["y"], s["rec"], s["start"], s["start"] + WIN) for name, s in sets.items()
          if not name.startswith("B")}
    print(f"features ready in {time.time() - t0:.0f}s, D={D}; windows: "
          + ", ".join(f"{k}={len(v['y'])}" for k, v in sets.items()), flush=True)

    all_features_acc = {}
    for src, tgt in tasks:
        B, ones = sets[f"B{tgt}"], np.ones((1, D), dtype=bool)
        all_features_acc[f"{src}->{tgt}"] = {
            name: float(1.0 - ev[key].test_error(ones, B["X"], B["y"])[0])
            for name, key in [("library", f"lib{src}"), ("rpst", f"rpst{src}to{tgt}"),
                              ("resample", f"rs{src}to{tgt}"), ("oracle_A", f"A{tgt}")]}

    results = {"spec": "docs/superpowers/specs/2026-10-08-tevc-rpst-agent-design.md",
               "start_time": datetime.now().isoformat(),
               "config": {"win": WIN, "hop": HOP, "cap_fit": CAP_FIT, "cap_half": CAP_HALF, "k": 5,
                          "pop": 100, "n_gen": n_gen, "seeds": seeds, "D": D},
               "all_features_acc": all_features_acc, "runs": []}
    for seed in seeds:
        selections = {}
        for src, tgt in tasks:
            plan = {"STALE": (f"lib{src}", f"lib{src}"),
                    "STALE+RETRAIN": (f"lib{src}", f"rpst{src}to{tgt}"),
                    "RESEL-RPST": (f"rpst{src}to{tgt}", f"rpst{src}to{tgt}"),
                    "RESEL-RS": (f"rs{src}to{tgt}", f"rs{src}to{tgt}"),
                    "ORACLE": (f"A{tgt}", f"A{tgt}")}
            for arm, (select_key, train_key) in plan.items():
                if select_key not in selections:
                    selections[select_key] = run_nsga2(ev[select_key], D, seed, n_gen=n_gen)
                masks, F = selections[select_key]
                r = score(masks, F, ev[train_key], sets[f"B{tgt}"])
                results["runs"].append({"task": f"{src}->{tgt}", "seed": seed, "arm": arm, **r})
                print(f"seed {seed:4d} {src}->{tgt:<5d} {arm:14s} HV={r['hv']:.4f} "
                      f"acc={r['deployed_acc'] * 100:5.1f}% nfeat={r['deployed_n_features']}", flush=True)
        out_path.write_text(json.dumps(results))

    add_igd(results["runs"])
    results["summary"] = summarize(results["runs"])
    results["gate_stats"] = gate_stats(results["runs"], results["summary"]) if not args.smoke else None
    results["end_time"] = datetime.now().isoformat()
    out_path.write_text(json.dumps(results, indent=1))

    print(f"\n{'task':10s} {'arm':14s} {'HV':>15s} {'deployed acc %':>16s} {'IGD':>15s}")
    for task, s in results["summary"].items():
        for arm in ARMS:
            m = s[arm]
            print(f"{task:10s} {arm:14s} {m['hv'][0]:.4f}+/-{m['hv'][1]:.4f} "
                  f"{m['deployed_acc'][0] * 100:7.1f}+/-{m['deployed_acc'][1] * 100:4.1f} "
                  f"{m['igd'][0]:.4f}+/-{m['igd'][1]:.4f}")
        print(f"{task:10s} Jaccard(deployed) STALE~ORACLE={s['jaccard_stale_vs_oracle']:.2f} "
              f"RPST~ORACLE={s['jaccard_rpst_vs_oracle']:.2f}")
    print("\nall-features k-NN acc:", json.dumps(all_features_acc))
    if results["gate_stats"]:
        print("gate stats:", json.dumps(results["gate_stats"], indent=1))
    print(f"done in {time.time() - t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
