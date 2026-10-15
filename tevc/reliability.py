"""Reliability metrics of deployed diagnosers (RESS spec §3.5) and the development comparison table.

Per environment, from the cached test-set probabilities of the deployed model (never retrained): missed-fault rate
P(predicted healthy | true fault), false-alarm rate P(predicted fault | true healthy), macro-F1, and expected
calibration error over 15 equal-width bins of the maximum probability. Class 0 is the healthy class in every
dataset (JNU n, PU K00x, Ottawa H, DIRG 0A).

    python -m tevc.reliability --dataset pu --files outputs/tevc/stream_pu_r10.json outputs/tevc/stream_pu_v2.json
writes outputs/tevc/devtable_<dataset>.json: per method, run-level means of accuracy, AUC and the four metrics, and
one-sided run-level tests of AGENT against every other method. For the rates and ECE (lower is better) the tested
difference is other - AGENT, so a positive mean difference always favours AGENT. A method in a later file replaces
the same method from an earlier file. Development sets only: Ottawa joins after the DIRG freeze (spec §6).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from tevc.baselines import config_from_dict
from tevc.evaluator import RecipeEvaluator
from tevc.run_stream import OUT_DIR, STREAMS, run_level_tests

HEALTHY = 0
N_BINS = 15
LOWER_IS_BETTER = ("missed_fault_rate", "false_alarm_rate", "ece")
KEYS = ("deployed_acc", "auc", "macro_f1") + LOWER_IS_BETTER


def missed_fault_rate(y, pred, healthy=HEALTHY):
    fault = y != healthy
    return float(np.mean(pred[fault] == healthy)) if fault.any() else 0.0


def false_alarm_rate(y, pred, healthy=HEALTHY):
    ok = y == healthy
    return float(np.mean(pred[ok] != healthy)) if ok.any() else 0.0


def macro_f1(y, pred, n_classes):
    return float(f1_score(y, pred, labels=np.arange(n_classes), average="macro", zero_division=0))


def ece(probs, y, n_bins=N_BINS):
    conf, correct = probs.max(axis=1), probs.argmax(axis=1) == y
    b = np.clip(np.digitize(conf, np.linspace(0.0, 1.0, n_bins + 1)[1:-1], right=True), 0, n_bins - 1)
    return float(sum((b == k).mean() * abs(correct[b == k].mean() - conf[b == k].mean())
                     for k in range(n_bins) if (b == k).any()))


def metrics(probs, y, n_classes):
    pred = probs.argmax(axis=1)
    return {"missed_fault_rate": missed_fault_rate(y, pred), "false_alarm_rate": false_alarm_rate(y, pred),
            "macro_f1": macro_f1(y, pred, n_classes), "ece": ece(probs, y)}


def merge(files):
    """({method: {run: [environment entries]}}, {method: file}); later files replace earlier methods."""
    results, source = {}, {}
    for f in files:
        for m, runs in json.loads(Path(f).read_text(encoding="utf-8"))["results"].items():
            results[m], source[m] = runs, str(f)
    return results, source


def add_metrics(results, evaluator, stream):
    """Adds the four metrics to every environment entry from the cached test probabilities of its deployed model."""
    for runs in results.values():
        for envs in runs.values():
            for entry, env in zip(envs, stream.environments, strict=True):
                if entry["env"] != env.name:
                    raise ValueError(f"result environment {entry['env']} != stream environment {env.name}")
                _, pT = evaluator.cached(env, config_from_dict(entry["deployed"]), entry["deployed"]["seed"])
                entry.update(metrics(pT, env.T["y"], stream.dataset.n_classes))
    return results


def summarize(results):
    return {m: {"runs": len(runs), **{k: float(np.mean([np.mean([e[k] for e in envs]) for envs in runs.values()]))
                                      for k in KEYS}}
            for m, runs in results.items() if runs}


def compare_to_agent(results, ref="AGENT"):
    """run_level_tests of ref against every method for each key; lower-is-better keys are negated first."""
    out = {}
    for k in KEYS:
        sign = -1.0 if k in LOWER_IS_BETTER else 1.0
        flipped = {m: {r: [{k: sign * e[k]} for e in envs] for r, envs in runs.items()} for m, runs in results.items()}
        out[k] = run_level_tests(flipped, k, ref=ref)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("jnu", "pu"), required=True)
    ap.add_argument("--files", nargs="+", required=True, help="stream result files; later files replace methods")
    args = ap.parse_args()
    stream = STREAMS[args.dataset]()
    results, source = merge(args.files)
    add_metrics(results, RecipeEvaluator(stream), stream)
    out = {"dataset": args.dataset, "healthy_class": HEALTHY, "ece_bins": N_BINS, "source_file": source,
           "summary": summarize(results), "tests_vs_agent": compare_to_agent(results)}
    (OUT_DIR / f"devtable_{args.dataset}.json").write_text(json.dumps(out, indent=1))
    for m, s in out["summary"].items():
        print(f"{m:15s} acc {s['deployed_acc'] * 100:6.2f}  missed {s['missed_fault_rate'] * 100:5.2f}  "
              f"false alarm {s['false_alarm_rate'] * 100:5.2f}  F1 {s['macro_f1'] * 100:6.2f}  ECE {s['ece']:.3f}")


if __name__ == "__main__":
    main()
