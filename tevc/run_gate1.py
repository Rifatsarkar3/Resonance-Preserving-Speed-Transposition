"""Gate-1 pilot for D1, pre-registered in docs/superpowers/specs/2026-10-08-tevc-rpst-agent-design.md.

Is the RPST-recipe landscape non-flat, and does a label-free proxy (computed on real unlabelled target windows)
rank recipes by true target accuracy?
Run from repo root:  python -m tevc.run_gate1   (--smoke: JNU only, 2 random recipes, 2 epochs)
Output: outputs/tevc/gate1_results.json
"""
import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from tevc import synthesis
from tevc.datasets import DATASETS, make_windows
from tevc.deep_train import predict, train_cnn
from tevc.proxies import information_maximization, neg_entropy, neg_mmd2, transposition_consistency
from tevc.recipe import ANCHORS, sample, training_set

TASKS = [("jnu", 600, 1000), ("pu", 900, 1500), ("pu", 1500, 900)]
N_RANDOM, MODEL, EPOCHS, SEED = 32, "TICNN", 40, 42
ANCHOR_SEEDS = [42, 1337, 2025, 999, 7]
VIEW_RATIOS = (0.95, 1.05)
PROXIES = ("TC", "IM", "ENT", "DFID")
OUT = Path("outputs/tevc/gate1_results.json")


def view_windows(entries, windows, ratio, win, rpm):
    """Each real window's stretch of signal, transposed to speed ratio `ratio` (RPST, n_fft 256)."""
    transposed = {rec: synthesis.transpose(sig, rpm, rpm * ratio) for _, rec, sig, _, _ in entries}
    out = []
    for rec, st in zip(windows["rec"], windows["span_lo"]):
        t = transposed[rec]
        a = min(int(round(st / ratio)), len(t) - win)
        out.append(t[a:a + win])
    return np.stack(out)


def task_context(ds_name, src, tgt):
    ds = DATASETS[ds_name]
    lib_entries, A_entries = ds.entries("library", src), ds.entries("A", tgt)
    A = make_windows(A_entries, ds.win, ds.hop, ds.cap_eval)
    return {"ds": ds, "src": src, "tgt": tgt, "lib_entries": lib_entries,
            "lib": make_windows(lib_entries, ds.win, ds.hop, ds.cap_fit), "A": A,
            "B": make_windows(ds.entries("B", tgt), ds.win, ds.hop, ds.cap_eval),
            "views": [view_windows(A_entries, A, r, ds.win, tgt) for r in VIEW_RATIOS]}


def evaluate(recipe, seed, ctx, epochs):
    t0 = time.time()
    W, y = training_set(recipe, ctx["ds"], ctx["lib_entries"], ctx["lib"], ctx["src"], ctx["tgt"])
    model = train_cnn(W, y, MODEL, epochs, seed)
    pA, eA = predict(model, ctx["A"]["W"], return_embedding=True)        # A labels never used
    view_probs = [predict(model, V) for V in ctx["views"]]
    pB = predict(model, ctx["B"]["W"])
    _, eT = predict(model, W[np.random.default_rng(0).permutation(len(W))[:400]], return_embedding=True)
    return {"true_acc": float((pB.argmax(axis=1) == ctx["B"]["y"]).mean()),
            "TC": transposition_consistency(pA, view_probs), "IM": information_maximization(pA),
            "ENT": neg_entropy(pA), "DFID": neg_mmd2(eT, eA), "n_train": int(len(y)),
            "seconds": round(time.time() - t0, 1)}


def task_stats(rows):
    main = [r for r in rows if r["seed"] == SEED]
    acc = np.array([r["true_acc"] for r in main])
    cand = (rankdata([r["TC"] for r in main]) + rankdata([r["IM"] for r in main])) / 2
    h2 = np.array([r["true_acc"] for r in rows if r["name"] == "H2_LIKE"])
    best = main[int(np.argmax(cand))]
    betas = np.array([r["recipe"]["beta"] for r in main])
    with_synth = np.array([r["recipe"]["n_targets"] > 0 for r in main])
    stats = {"n_recipes": len(main), "acc_range_pp": 100 * float(acc.max() - acc.min()),
             "acc_std_pp": 100 * float(acc.std()), "h2_seed_std_pp": 100 * float(h2.std()),
             "h2_mean_acc": float(h2.mean()), "rho": {p: float(spearmanr([r[p] for r in main], acc)[0]) for p in PROXIES},
             "rho_CAND": float(spearmanr(cand, acc)[0]), "argmax_CAND": {"name": best["name"], "true_acc": best["true_acc"]},
             "oracle_best": {"name": main[int(np.argmax(acc))]["name"], "true_acc": float(acc.max())},
             "rho_beta_acc": float(spearmanr(betas[with_synth], acc[with_synth])[0]),
             "acc_beta_le_0.25": float(acc[with_synth & (betas <= 0.25)].mean()),
             "acc_beta_ge_0.75": float(acc[with_synth & (betas >= 0.75)].mean())}
    stats["G1a"] = stats["acc_range_pp"] >= 10 and stats["acc_std_pp"] >= 1.5 * stats["h2_seed_std_pp"]
    stats["G1b"] = stats["rho_CAND"] >= 0.5
    stats["G1c"] = best["true_acc"] >= stats["h2_mean_acc"]
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tasks, n_random, epochs = (TASKS[:1], 2, 2) if args.smoke else (TASKS, N_RANDOM, EPOCHS)
    out = OUT.with_name("gate1_smoke.json") if args.smoke else OUT
    rng = np.random.default_rng(0)
    recipes = dict(ANCHORS if not args.smoke else {k: ANCHORS[k] for k in ("NONE", "H2_LIKE")})
    recipes.update({f"R{i:02d}": sample(rng) for i in range(n_random)})

    results = {"spec": "docs/superpowers/specs/2026-10-08-tevc-rpst-agent-design.md (Gate-1)",
               "start_time": datetime.now().isoformat(),
               "config": {"model": MODEL, "epochs": epochs, "seed": SEED, "anchor_seeds": ANCHOR_SEEDS,
                          "view_ratios": VIEW_RATIOS, "n_random": n_random},
               "recipes": {k: asdict(v) for k, v in recipes.items()}, "rows": [], "stats": {}}
    for ds_name, src, tgt in tasks:
        task = f"{ds_name}:{src}->{tgt}"
        t0 = time.time()
        ctx = task_context(ds_name, src, tgt)
        print(f"[{task}] context ready in {time.time() - t0:.0f}s", flush=True)
        rows = []
        for name, recipe in recipes.items():
            for seed in (ANCHOR_SEEDS if name == "H2_LIKE" and not args.smoke else [SEED]):
                m = evaluate(recipe, seed, ctx, epochs)
                rows.append({"task": task, "name": name, "seed": seed, "recipe": asdict(recipe), **m})
                print(f"[{task}] {name:11s} s{seed:<4d} acc={m['true_acc'] * 100:5.1f}% TC={m['TC']:.3f} "
                      f"IM={m['IM']:.3f} ENT={m['ENT']:.3f} DFID={m['DFID']:.4f} n={m['n_train']} {m['seconds']}s", flush=True)
                results["rows"] = [r for r in results["rows"] if r["task"] != task] + rows
                out.write_text(json.dumps(results))
        results["stats"][task] = task_stats(rows)
        print(f"[{task}] stats: {json.dumps(results['stats'][task])}", flush=True)
    if not args.smoke:
        results["gates"] = {g: int(sum(s[g] for s in results["stats"].values())) for g in ("G1a", "G1b", "G1c")}
        print("tasks passing:", results["gates"], "(pass = >= 2 of 3)")
    results["end_time"] = datetime.now().isoformat()
    out.write_text(json.dumps(results, indent=1))
    print("done ->", out)


if __name__ == "__main__":
    main()
