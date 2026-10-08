"""IM class-balance stress test (spec §4/§7, week-1 gate) on PU development data.

IM assumes a class-balanced unlabelled buffer; real streams are mostly healthy. For the 16 Gate-1 recipes
(4 anchors + R00..R11, same generator) and both PU directions, IM is recomputed on U_t subsampled to
50/70/90 % healthy (labels used ONLY to construct the skew) and rank-correlated with true accuracy on the
balanced T_t. Pass: Spearman rho(IM at 70 % healthy, accuracy) >= 0.5 in both directions.
Run from repo root:  python -m tevc.probes.im_imbalance   ->  outputs/tevc/probe_im_imbalance.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from tevc.evaluator import RecipeEvaluator
from tevc.proxies import information_maximization
from tevc.recipe import ANCHORS, sample
from tevc.stream import pu_stream

OUT = Path("outputs/tevc/probe_im_imbalance.json")
HEALTHY_FRACTIONS = (0.5, 0.7, 0.9)
SEED = 42


def gate1_recipes(n_random=12):
    rng = np.random.default_rng(0)                  # same generator and order as tevc.run_gate1
    recipes = dict(ANCHORS)
    recipes.update({f"R{i:02d}": sample(rng) for i in range(n_random)})
    return recipes


def skewed_indices(y, healthy_fraction, seed=0):
    """Indices of a subsample with the given share of class 0 (healthy) and equal shares of the other classes."""
    rng = np.random.default_rng(seed)
    healthy = np.flatnonzero(y == 0)
    others = [np.flatnonzero(y == c) for c in np.unique(y) if c != 0]
    k = max(1, int(len(healthy) * (1 - healthy_fraction) / (healthy_fraction * len(others))))
    picks = [healthy] + [rng.choice(idx, size=min(k, len(idx)), replace=False) for idx in others]
    return np.sort(np.concatenate(picks))


def main():
    results = {}
    for library_rpm, target_rpm in ((900, 1500), (1500, 900)):
        stream = pu_stream(library_rpm=library_rpm, sequence=(target_rpm,))
        env = stream.environments[0]
        evaluator = RecipeEvaluator(stream)
        rows = []
        for name, recipe in gate1_recipes().items():
            pU, pT, _, _ = evaluator.probabilities(env, recipe, SEED)
            row = {"recipe": name, "true_acc": float((pT.argmax(axis=1) == env.T["y"]).mean()),
                   "IM_balanced": information_maximization(pU)}
            for f in HEALTHY_FRACTIONS:
                row[f"IM_{int(f * 100)}"] = information_maximization(pU[skewed_indices(env.U["y"], f)])
            rows.append(row)
            print(f"PU {library_rpm}->{target_rpm} {name:11s} acc={row['true_acc'] * 100:5.1f}% "
                  + " ".join(f"{k}={v:.3f}" for k, v in row.items() if k.startswith("IM")), flush=True)
        acc = [r["true_acc"] for r in rows]
        rho = {k: float(spearmanr([r[k] for r in rows], acc)[0]) for k in rows[0] if k.startswith("IM")}
        results[f"{library_rpm}->{target_rpm}"] = {"rows": rows, "rho": rho}
        print(f"PU {library_rpm}->{target_rpm} rho: {json.dumps(rho)}", flush=True)
    passed = all(v["rho"]["IM_70"] >= 0.5 for v in results.values())
    results["verdict"] = {"criterion": "rho(IM_70, acc) >= 0.5 in both directions", "passed": passed}
    OUT.write_text(json.dumps(results, indent=1))
    print("verdict:", results["verdict"], "->", OUT)


if __name__ == "__main__":
    main()
