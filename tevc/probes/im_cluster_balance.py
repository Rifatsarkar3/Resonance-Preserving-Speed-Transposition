"""Does a label-free, cluster-balanced IM survive healthy-dominated unlabelled buffers?

The agent clusters the unlabelled buffer once in the fixed physics-feature space (k = number of library classes,
known to the agent) and weights each window by 1 / (its cluster size) before computing IM
(`tevc.proxies.cluster_balance_weights` / `weighted_information_maximization`). The weights do not depend on the
candidate model, so every recipe is scored on the same re-balanced buffer. Labels are used only to build the
skewed buffers and to report cluster purity. Probabilities come from the evaluator cache (trained if missing).
PU = exploratory (run after the failed stress test); JNU = pre-registered check (spec, 2026-10-09):
pass if rho(cluster-balanced IM, accuracy) >= 0.5 at 70 % healthy in both directions.
Run from repo root:  python -m tevc.probes.im_cluster_balance [--dataset pu|jnu]
  -> outputs/tevc/probe_im_cluster_balance.json (pu) / probe_im_cluster_balance_jnu.json (jnu)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from tevc import features
from tevc.datasets import DATASETS
from tevc.evaluator import RecipeEvaluator
from tevc.probes.im_imbalance import HEALTHY_FRACTIONS, SEED, gate1_recipes, skewed_indices
from tevc.proxies import cluster_balance_weights, information_maximization, weighted_information_maximization
from tevc.stream import jnu_stream, pu_stream

OUT = Path("outputs/tevc/probe_im_cluster_balance.json")
DIRECTIONS = {"pu": ((900, 1500), (1500, 900)), "jnu": ((600, 1000), (1000, 600))}
STREAMS = {"pu": pu_stream, "jnu": jnu_stream}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DIRECTIONS), default="pu")
    args = ap.parse_args()
    ds = DATASETS[args.dataset]
    out = OUT if args.dataset == "pu" else OUT.with_name(f"probe_im_cluster_balance_{args.dataset}.json")
    results = {}
    for library_rpm, target_rpm in DIRECTIONS[args.dataset]:
        stream = STREAMS[args.dataset](library_rpm=library_rpm, sequence=(target_rpm,))
        env = stream.environments[0]
        evaluator = RecipeEvaluator(stream)
        probs = {name: evaluator.probabilities(env, r, SEED) for name, r in gate1_recipes().items()}
        acc = [float((pT.argmax(axis=1) == env.T["y"]).mean()) for _, pT, _, _ in probs.values()]
        X = features.extract(env.U["W"], ds.fs, target_rpm / 60.0, ds.geometry, znorm=True)
        task = {"rho_plain_IM_balanced": float(spearmanr([information_maximization(pU) for pU, _, _, _ in probs.values()],
                                                         acc)[0]),
                "acc_range": [float(min(acc)), float(max(acc))]}
        for f in HEALTHY_FRACTIONS:
            idx = skewed_indices(env.U["y"], f)
            w, clusters = cluster_balance_weights(X[idx], ds.n_classes)
            plain = [information_maximization(pU[idx]) for pU, _, _, _ in probs.values()]
            balanced = [weighted_information_maximization(pU[idx], w) for pU, _, _, _ in probs.values()]
            task[f"{int(f * 100)}% healthy"] = {"rho_plain_IM": float(spearmanr(plain, acc)[0]),
                                                "rho_cluster_balanced_IM": float(spearmanr(balanced, acc)[0]),
                                                "cluster_purity_ARI": float(adjusted_rand_score(env.U["y"][idx], clusters))}
            print(f"{ds.name} {library_rpm}->{target_rpm} {int(f * 100)}% healthy: "
                  f"{json.dumps(task[f'{int(f * 100)}% healthy'])}", flush=True)
        results[f"{library_rpm}->{target_rpm}"] = task
    results["verdict"] = {"criterion": "rho(cluster-balanced IM, acc) >= 0.5 at 70% healthy in both directions",
                          "passed": all(t["70% healthy"]["rho_cluster_balanced_IM"] >= 0.5 for t in results.values())}
    out.write_text(json.dumps(results, indent=1))
    print("verdict:", results["verdict"], "->", out)


if __name__ == "__main__":
    main()
