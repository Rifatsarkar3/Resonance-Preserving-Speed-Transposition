"""Development gate for agent v2 (development data only; rule fixed 2026-10-10, before any v2 result existed).

Agent v1 passed the spec §5 gate (AGENT > GA-RESTART) but lost to WARM-TRANSFER on PU, and the Ottawa protocol v1
was withdrawn unseen. v2 must be the best label-free method before the protocol is re-frozen:
  PU  (discriminating): AGENT's 10-run mean deployed accuracy AND recovery AUC exceed those of every label-free
      method: STATIC-NONE, STATIC-H2, RANDOM, GA-RESTART, BO, WARM-TRANSFER, AGENT-B1.
  JNU (at ceiling): AGENT's means are >= GA-RESTART's (the spec §5 gate) and no label-free method beats AGENT at run
      level (one-sided exact sign-flip p < 0.05, uncorrected) on either metric.
ORACLE-FITNESS (fitness = test accuracy) is reported, never gated. Baseline results come only from the committed
baseline files, agent results only from the v2 file, so v1 agent rows never mix in.
    python -m tevc.probes.dev_gate   -> outputs/tevc/dev_gate_v2.json
"""
import json
from pathlib import Path

from tevc.run_stream import RUN_SEEDS, run_level_tests, summarize

OUT = Path("outputs/tevc/dev_gate_v2.json")
BASELINES = ("STATIC-NONE", "STATIC-H2", "RANDOM", "GA-RESTART", "BO", "WARM-TRANSFER", "ORACLE-FITNESS")
SOURCES = {"jnu": ("stream_jnu_r10.json", "stream_jnu_extra.json"), "pu": ("stream_pu_r10.json", "stream_pu_baselines.json")}
LABEL_FREE = ("STATIC-NONE", "STATIC-H2", "RANDOM", "GA-RESTART", "BO", "WARM-TRANSFER", "AGENT-B1")
METRICS = ("deployed_acc", "auc")


def merged(dataset, root=Path("outputs/tevc")):
    results = {}
    for name in SOURCES[dataset]:
        for m, runs in json.loads((root / name).read_text())["results"].items():
            if m in BASELINES:
                results.setdefault(m, {}).update(runs)
    for m, runs in json.loads((root / f"stream_{dataset}_v2.json").read_text())["results"].items():
        if m.startswith("AGENT"):
            results[m] = runs
    missing = [(m, s) for m in (*LABEL_FREE, "AGENT") for s in RUN_SEEDS if str(s) not in results.get(m, {})]
    if missing:
        raise SystemExit(f"{dataset}: {len(missing)} (method, run) cells missing, e.g. {missing[:3]}")
    return results


def beats_agent_p(results, m, key):
    """One-sided exact sign-flip p (run level) that method m beats AGENT."""
    return run_level_tests({m: results[m], "AGENT": results["AGENT"]}, key, ref=m)["AGENT"]["p"]


def gate(dataset, results, summary):
    """(passes, label-free methods that beat AGENT at run level with p < 0.05 on a metric)."""
    a = summary["AGENT"]
    worse = [(m, k) for m in LABEL_FREE for k in METRICS if beats_agent_p(results, m, k) < 0.05]
    if dataset == "pu":
        return all(a[f"{k}_mean"] > summary[m][f"{k}_mean"] for m in LABEL_FREE for k in ("acc", "auc")), worse
    g = summary["GA-RESTART"]
    return a["acc_mean"] >= g["acc_mean"] and a["auc_mean"] >= g["auc_mean"] and not worse, worse


def main():
    out = {}
    for dataset in SOURCES:
        results = merged(dataset)
        summary = summarize(results)
        tests = {k: run_level_tests(results, k) for k in METRICS}
        passes, worse = gate(dataset, results, summary)
        out[dataset] = {"summary": summary, "tests": tests, "passes": passes, "significantly_better_than_agent": worse}
        print(dataset, "gate passes:", passes, "| significantly better than AGENT:", worse or "none")
        for m, s in sorted(summary.items(), key=lambda e: -e[1]["acc_mean"]):
            t = [tests[k].get(m) for k in METRICS]
            diff = "" if m == "AGENT" else "  AGENT-it acc {:+.2f} ({}/{}/{}) auc {:+.2f} ({}/{}/{})".format(
                t[0]["mean_diff"] * 100, t[0]["wins"], t[0]["ties"], t[0]["losses"],
                t[1]["mean_diff"] * 100, t[1]["wins"], t[1]["ties"], t[1]["losses"])
            print(f"  {m:14s} acc {s['acc_mean'] * 100:6.2f}  auc {s['auc_mean'] * 100:6.2f}  evals {s['evals_per_stream']:4.0f}{diff}")
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
