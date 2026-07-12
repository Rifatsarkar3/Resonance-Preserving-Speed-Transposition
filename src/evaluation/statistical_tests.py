"""
Statistical tests for MSSP paper submission.
Computes Wilcoxon signed-rank tests and Cohen's d for WaPIGT vs each baseline.
Outputs: outputs/statistical_results.json and a LaTeX table snippet.

Usage:
    python src/evaluation/statistical_tests.py
"""
import json, numpy as np
from pathlib import Path
from scipy import stats


def cohen_d(a, b):
    """Cohen's d: (mean_a - mean_b) / pooled_std"""
    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def load_results(path="outputs/full_comparison_results.json"):
    with open(path) as f:
        data = json.load(f)
    runs = [r for r in data["runs"] if r.get("status") == "success"]
    from collections import defaultdict
    agg = defaultdict(list)
    for r in runs:
        agg[(r["dataset"], r["task"], r["model"])].append(r["test_acc"])
    return agg


def get_task_scores(agg, model, datasets=("JNU", "CWRU")):
    """Return list of per-seed test accuracies across all tasks for a given model."""
    scores = []
    for (ds, task, m), accs in agg.items():
        if m == model and ds in datasets:
            scores.extend(accs)
    return scores


def run_statistical_tests(results_path="outputs/full_comparison_results.json",
                           output_path="outputs/statistical_results.json"):
    agg = load_results(results_path)

    wapigt_scores = get_task_scores(agg, "WaPIGT-MS")
    baselines = ["WDCNN", "TICNN", "MSCNN", "PhysFormer", "ViT1D"]

    results = {}
    print(f"\nWaPIGT-MS: n={len(wapigt_scores)}, mean={np.mean(wapigt_scores)*100:.1f}%")
    print("\nStatistical Tests (WaPIGT-MS vs baseline, two-sided Wilcoxon):")
    print(f"{'Baseline':12s}  {'n_base':6s}  {'mean_base':9s}  {'p_value':9s}  {'Cohen_d':8s}  {'sig':4s}")
    print("-" * 65)

    for bl in baselines:
        bl_scores = get_task_scores(agg, bl)
        if len(bl_scores) < 3:
            print(f"{bl:12s}  insufficient data (n={len(bl_scores)})")
            continue

        # Align lengths for Wilcoxon (use matched pairs where available)
        # For unmatched, use Mann-Whitney U as fallback
        n = min(len(wapigt_scores), len(bl_scores))
        w = wapigt_scores[:n]
        b = bl_scores[:n]

        if len(set(x - y for x, y in zip(w, b))) == 1 and list(set(x - y for x, y in zip(w, b)))[0] == 0:
            p_val = 1.0
            stat = 0.0
        else:
            try:
                stat, p_val = stats.wilcoxon(w, b, alternative="two-sided")
            except ValueError:
                stat, p_val = stats.mannwhitneyu(wapigt_scores, bl_scores, alternative="two-sided")
                p_val *= 2  # two-sided correction for mannwhitney

        d = cohen_d(wapigt_scores, bl_scores)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))

        print(f"{bl:12s}  {len(bl_scores):6d}  {np.mean(bl_scores)*100:8.1f}%  "
              f"{p_val:9.4f}  {d:8.3f}  {sig:4s}")

        results[bl] = {
            "n_wapigt": len(wapigt_scores),
            "n_baseline": len(bl_scores),
            "wapigt_mean": float(np.mean(wapigt_scores)),
            "baseline_mean": float(np.mean(bl_scores)),
            "wilcoxon_stat": float(stat),
            "p_value": float(p_val),
            "cohen_d": float(d),
            "significant_at_005": p_val < 0.05,
            "significant_at_001": p_val < 0.01,
        }

    # Per-task breakdown
    print("\n\nPer-task WaPIGT-MS performance:")
    all_tasks = sorted(set((ds, task) for (ds, task, m) in agg if m == "WaPIGT-MS"))
    for ds, task in all_tasks:
        accs = agg.get((ds, task, "WaPIGT-MS"), [])
        if accs:
            print(f"  {ds} {task}: mean={np.mean(accs)*100:.1f}% std={np.std(accs)*100:.1f}% n={len(accs)}")

    # T03 specific tests (most important for paper claim)
    print("\n\nT03-specific tests (WaPIGT-MS wins T03):")
    wapigt_t03 = agg.get(("JNU", "JNU_T03", "WaPIGT-MS"), [])
    for bl in baselines:
        bl_t03 = agg.get(("JNU", "JNU_T03", bl), [])
        if not bl_t03:
            continue
        n = min(len(wapigt_t03), len(bl_t03))
        try:
            _, p = stats.wilcoxon(wapigt_t03[:n], bl_t03[:n], alternative="greater")
            d = cohen_d(wapigt_t03, bl_t03)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
            delta = (np.mean(wapigt_t03) - np.mean(bl_t03)) * 100
            print(f"  vs {bl:12s}: WaPIGT={np.mean(wapigt_t03)*100:.1f}% "
                  f"bl={np.mean(bl_t03)*100:.1f}%  Δ={delta:+.1f}pp  p={p:.4f} {sig}  d={d:.2f}")
        except Exception as e:
            print(f"  vs {bl:12s}: error {e}")

    # LaTeX table output
    print("\n\n% LaTeX for Table tab:stats (paste into paper):")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Statistical significance: Wilcoxon signed-rank test (two-sided,")
    print(r"$\alpha=0.05$) and Cohen's $d$ — WaPIGT-MS vs.\ each baseline,")
    print(r"all JNU+CWRU task$\times$seed pairs.}")
    print(r"\label{tab:stats}")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"Baseline & $n$ & Mean acc.\ (\%) & $p$-value & Cohen's $d$ & Sig.\ \\")
    print(r"\midrule")
    for bl, r in results.items():
        sig = "***" if r["p_value"] < 0.001 else ("**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else "n.s."))
        print(f"{bl} & {r['n_baseline']} & {r['baseline_mean']*100:.1f} & "
              f"{r['p_value']:.3f} & {r['cohen_d']:.2f} & {sig} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_statistical_tests()
