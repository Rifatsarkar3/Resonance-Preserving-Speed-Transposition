"""Augmented significance analysis for the RPST main benchmark (JNU, 5 seeds).
For each model, pairs (task x seed) = 15 paired no-aug vs RPST accuracies and reports:
  - one-sided paired Wilcoxon signed-rank p (RPST > no-aug)   [matches the manuscript]
  - matched-pairs rank-biserial effect size r in [-1, 1]
  - mean paired gain with a 95% bootstrap CI
Multiplicity across the six models is controlled with Holm-Bonferroni.
-> outputs/rpst_significance.json  (+ a LaTeX-ready table printed to stdout)
"""
import json
import numpy as np
from scipy.stats import wilcoxon

RESULTS = "outputs/g2_definitive_results.json"
MODELS = ["WaPIGT", "WDCNN", "TICNN", "MSCNN", "PhysFormer", "ViT1D"]
LABEL = {"ViT1D": "ViT-1D"}
TASKS = ["JNU_T01", "JNU_T02", "JNU_T03"]
SEEDS = [42, 1337, 2025, 999, 7]
rng = np.random.default_rng(0)

runs = [r for r in json.load(open(RESULTS))["runs"] if r.get("status") == "success"]
acc = {(r["model"], r["task"], r["regime"], r["seed"]): r["acc"] for r in runs}


def rank_biserial(diff):
    """Matched-pairs rank-biserial r = (W+ - W-) / (W+ + W-) on nonzero diffs."""
    d = diff[diff != 0]
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    wp = ranks[d > 0].sum()
    wn = ranks[d < 0].sum()
    return (wp - wn) / (wp + wn)


def boot_ci(diff, n=10000):
    means = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


rows = []
for m in MODELS:
    na = np.array([acc[(m, t, "noaug", s)] for t in TASKS for s in SEEDS]) * 100
    rp = np.array([acc[(m, t, "RPST", s)] for t in TASKS for s in SEEDS]) * 100
    diff = rp - na
    p = wilcoxon(rp, na, alternative="greater", zero_method="wilcox").pvalue
    lo, hi = boot_ci(diff)
    rows.append({"model": m, "n_pairs": len(diff), "noaug": na.mean(), "rpst": rp.mean(),
                 "delta": diff.mean(), "ci_lo": lo, "ci_hi": hi,
                 "r": rank_biserial(diff), "p_raw": float(p)})

# Holm-Bonferroni across the six models
order = sorted(range(len(rows)), key=lambda i: rows[i]["p_raw"])
k = len(rows)
prev = 0.0
for rank, i in enumerate(order):
    adj = min(1.0, (k - rank) * rows[i]["p_raw"])
    adj = max(adj, prev)          # enforce monotonicity
    prev = adj
    rows[i]["p_holm"] = adj

json.dump(rows, open("outputs/rpst_significance.json", "w"), indent=2)

print(f"{'Model':<11}{'no-aug':>8}{'+RPST':>8}{'  delta [95% CI]':>22}{'  r':>7}{'p_raw':>10}{'p_Holm':>10}")
for r in rows:
    print(f"{LABEL.get(r['model'],r['model']):<11}{r['noaug']:>8.1f}{r['rpst']:>8.1f}"
          f"{'  +'+format(r['delta'],'.1f')+' ['+format(r['ci_lo'],'.1f')+', '+format(r['ci_hi'],'.1f')+']':>22}"
          f"{r['r']:>7.2f}{r['p_raw']:>10.4f}{r['p_holm']:>10.4f}")
print("\nLaTeX rows:")
for r in rows:
    print(f"{LABEL.get(r['model'],r['model']):<11}& {r['noaug']:.1f} & {r['rpst']:.1f} & "
          f"$+{r['delta']:.1f}$ [{r['ci_lo']:.1f}, {r['ci_hi']:.1f}] & {r['r']:.2f} & "
          f"{r['p_raw']:.4f} & {r['p_holm']:.4f} \\\\")
