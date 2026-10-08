"""Exploratory probe (after the failed IM imbalance stress test; PU = development data): does a label-free,
cluster-balanced IM survive healthy-dominated buffers?

The agent clusters the unlabelled buffer once in the fixed physics-feature space (k = number of library classes,
known to the agent) and weights each window by 1 / (its cluster size) before computing IM. The weights do not depend
on the candidate model, so every recipe is scored on the same re-balanced buffer. Labels are used only to build the
skewed buffers and to report cluster purity. Uses the evaluator cache from tevc.probes.im_imbalance (no retraining).
Run from repo root:  python -m tevc.probes.im_cluster_balance   ->  outputs/tevc/probe_im_cluster_balance.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from tevc import features
from tevc.datasets import PU
from tevc.evaluator import RecipeEvaluator
from tevc.probes.im_imbalance import HEALTHY_FRACTIONS, SEED, gate1_recipes, skewed_indices
from tevc.proxies import information_maximization
from tevc.stream import pu_stream

OUT = Path("outputs/tevc/probe_im_cluster_balance.json")
EPS = 1e-12


def weighted_im(p, w):
    """IM with per-window weights: H(weighted mean p) - weighted mean H(p)."""
    w = w / w.sum()
    pbar = (w[:, None] * p).sum(axis=0)
    return float(-(pbar * np.log(pbar + EPS)).sum() + (w * (p * np.log(p + EPS)).sum(axis=1)).sum())


def cluster_weights(X, k, seed=0):
    """1 / cluster size for each window, clusters from k-means on standardised features."""
    Z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
    return 1.0 / np.bincount(labels)[labels], labels


def main():
    results = {}
    for library_rpm, target_rpm in ((900, 1500), (1500, 900)):
        stream = pu_stream(library_rpm=library_rpm, sequence=(target_rpm,))
        env = stream.environments[0]
        evaluator = RecipeEvaluator(stream)
        probs = {name: evaluator.probabilities(env, r, SEED) for name, r in gate1_recipes().items()}
        acc = [float((pT.argmax(axis=1) == env.T["y"]).mean()) for _, pT, _, _ in probs.values()]
        X = features.extract(env.U["W"], PU.fs, target_rpm / 60.0, PU.geometry, znorm=True)
        task = {}
        for f in HEALTHY_FRACTIONS:
            idx = skewed_indices(env.U["y"], f)
            w, clusters = cluster_weights(X[idx], PU.n_classes)
            plain = [information_maximization(pU[idx]) for pU, _, _, _ in probs.values()]
            balanced = [weighted_im(pU[idx], w) for pU, _, _, _ in probs.values()]
            task[f"{int(f * 100)}% healthy"] = {"rho_plain_IM": float(spearmanr(plain, acc)[0]),
                                                "rho_cluster_balanced_IM": float(spearmanr(balanced, acc)[0]),
                                                "cluster_purity_ARI": float(adjusted_rand_score(env.U["y"][idx], clusters))}
            print(f"PU {library_rpm}->{target_rpm} {int(f * 100)}% healthy: {json.dumps(task[f'{int(f * 100)}% healthy'])}",
                  flush=True)
        results[f"{library_rpm}->{target_rpm}"] = task
    OUT.write_text(json.dumps(results, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
