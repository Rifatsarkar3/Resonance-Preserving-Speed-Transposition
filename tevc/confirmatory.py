"""Ottawa confirmatory protocol (spec §5, §8), frozen before any Ottawa model is trained.

Development is finished: on JNU and PU (10 independent runs each) AGENT passed the development gate (c568066,
787f889), and the author approved the hypotheses below on 2026-10-10. Every setting — the Ottawa stream (library
13-16 Hz; 16-19 -> 19-22 -> 22-25 -> 19-22 -> 16-19 Hz), budget B = 30 per change, cluster-balanced IM fitness, the
agent, the baselines, TICNN for 40 epochs, RUN_SEEDS — is the code at the commit that adds this file. The runner refuses
to start when tevc/ differs from a commit and records the commit it ran from.

Primary family: one-sided, unit = independent run (10), exact sign-flip test on the run-level mean difference, Holm
over the four, alpha = 0.05. Each hypothesis is confirmed iff its Holm-adjusted p < alpha; verdicts are reported
individually, whatever they are.
    H1  AGENT > GA-RESTART  deployed accuracy   memory + surrogate beat restarting the GA at every change
    H2  AGENT > GA-RESTART  recovery AUC        ... and recover faster
    H3  AGENT > AGENT-B1    deployed accuracy   resonance-preserving beats resonance-shifting synthesis
    H4  AGENT > STATIC-H2   deployed accuracy   search beats the fixed RPST recipe
Secondary, reported with bootstrap CIs and Holm within each family, no confirmatory claim:
    S1  AGENT vs RANDOM and vs STATIC-NONE, accuracy and AUC
    S2  DFID, the pre-registered secondary fitness: AGENT-DFID vs GA-RESTART, and AGENT vs AGENT-DFID, accuracy and AUC
    S3  secondary grid: AGENT vs BO, WARM-TRANSFER, ORACLE-FITNESS, AGENT-NOMEM, AGENT-NOSURR, AGENT-NOREEVAL
    S4  fitness validity: Spearman rho(fitness, true accuracy) over all evaluations per environment, IM and DFID
Each grid runs once; shards split the run seeds and may run in parallel (distinct seeds never share cache entries). A
re-run after a crash reproduces finished runs from the evaluation cache. Logs show progress only, never accuracy.
    python -m tevc.confirmatory run --grid core --shard 0/2        (and --shard 1/2)
    python -m tevc.confirmatory analyse --grid core                -> outputs/tevc/ottawa/verdict_core.json
    python -m tevc.confirmatory run --grid secondary --shard 0/2   (and 1/2), then analyse --grid secondary
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from tevc.evaluator import RecipeEvaluator
from tevc.run_stream import RUN_SEEDS, run_level_tests, run_method, score, summarize
from tevc.stream import assert_disjoint, ottawa_stream

OUT = Path("outputs/tevc/ottawa")
ALPHA = 0.05
GRIDS = {"core": ("STATIC-NONE", "STATIC-H2", "RANDOM", "GA-RESTART", "AGENT", "AGENT-B1", "AGENT-DFID"),
         "secondary": ("AGENT", "BO", "WARM-TRANSFER", "ORACLE-FITNESS", "AGENT-NOMEM", "AGENT-NOSURR",
                       "AGENT-NOREEVAL")}
PRIMARY = {"H1": ("AGENT", "GA-RESTART", "deployed_acc"), "H2": ("AGENT", "GA-RESTART", "auc"),
           "H3": ("AGENT", "AGENT-B1", "deployed_acc"), "H4": ("AGENT", "STATIC-H2", "deployed_acc")}
SECONDARY = {"S1": [("AGENT", m, k) for m in ("RANDOM", "STATIC-NONE") for k in ("deployed_acc", "auc")],
             "S2": [(a, b, k) for a, b in (("AGENT-DFID", "GA-RESTART"), ("AGENT", "AGENT-DFID"))
                    for k in ("deployed_acc", "auc")],
             "S3": [("AGENT", m, k) for m in GRIDS["secondary"][1:] for k in ("deployed_acc", "auc")]}
NOT_IM = ("AGENT-DFID", "ORACLE-FITNESS")       # methods whose trial fitness is not cluster-balanced IM


def holm(ps):
    """Holm step-down adjusted p-values, in the input order."""
    adjusted, running = [0.0] * len(ps), 0.0
    for i, j in enumerate(np.argsort(ps, kind="stable")):
        running = max(running, min(1.0, (len(ps) - i) * ps[j]))
        adjusted[j] = running
    return adjusted


def family(results, tests):
    """One-sided run-level comparisons first > second (run_stream.run_level_tests), Holm over this family."""
    rows = [{"first": a, "second": b, "metric": k, **run_level_tests({a: results[a], b: results[b]}, k, ref=a)[b]}
            for a, b, k in tests]
    for row, p in zip(rows, holm([row["p"] for row in rows])):
        row["p_holm"] = p
    return rows


def verdict(results):
    rows = family(results, list(PRIMARY.values()))
    return {h: {**row, "confirmed": bool(row["p_holm"] < ALPHA)} for h, row in zip(PRIMARY, rows)}


def fitness_validity(results, methods):
    """Spearman rho(fitness, true accuracy) over every evaluation by `methods`, per environment name."""
    pairs = {}
    for m in methods:
        for envs in results[m].values():
            for e in envs:
                f, a = pairs.setdefault(e["env"], ([], []))
                f += [t["fitness"] for t in e["trials"]]
                a += [t["acc"] for t in e["trials"]]
    return {env: {"n": len(f), "rho": float(spearmanr(f, a)[0])} for env, (f, a) in pairs.items()}


def _frozen_commit():
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "tevc"], capture_output=True, text=True,
                           check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"tevc/ differs from the last commit; the protocol runs only from committed code:\n{dirty}")
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def run(grid, shard, n_shards, out_dir=OUT):
    commit = _frozen_commit()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{grid}_shard{shard}of{n_shards}.json"
    stream = ottawa_stream()
    assert_disjoint(stream)
    evaluator = RecipeEvaluator(stream)
    results, t0 = {m: {} for m in GRIDS[grid]}, time.time()
    for run_seed in RUN_SEEDS[shard::n_shards]:
        for m in GRIDS[grid]:
            t = time.time()
            results[m][str(run_seed)] = [score(evaluator, stream, env, carried, trials) for env, (carried, trials)
                                         in zip(stream.environments, run_method(m, stream, evaluator, run_seed))]
            path.write_text(json.dumps({"commit": commit, "grid": grid, "shard": [shard, n_shards],
                                        "elapsed_s": round(time.time() - t0), "results": results}))
            evals = sum(e["n_evals"] for e in results[m][str(run_seed)])
            print(f"run {run_seed:5d} {m:14s} evals {evals:3d}  {time.time() - t:6.0f}s", flush=True)
    print(f"{grid} shard {shard}/{n_shards} done in {time.time() - t0:.0f}s", flush=True)


def analyse(grid, out_dir=OUT):
    path = out_dir / f"verdict_{grid}.json"
    if path.exists():
        raise SystemExit(f"{path} exists: the {grid} analysis runs once")
    results, commits = {m: {} for m in GRIDS[grid]}, set()
    for shard in sorted(out_dir.glob(f"{grid}_shard*.json")):
        d = json.loads(shard.read_text())
        commits.add(d["commit"])
        for m, runs in d["results"].items():
            results[m].update(runs)
    if len(commits) != 1:
        raise SystemExit(f"{grid} shards ran from {len(commits)} commits: {sorted(commits)}")
    missing = [(m, s) for m in GRIDS[grid] for s in RUN_SEEDS if str(s) not in results[m]]
    if missing:
        raise SystemExit(f"{grid} grid incomplete: {len(missing)} (method, run) cells missing, e.g. {missing[:3]}")
    out = {"grid": grid, "protocol_commit": commits.pop(), "run_seeds": list(RUN_SEEDS), "alpha": ALPHA,
           "summary": summarize(results)}
    if grid == "core":
        out["primary"] = verdict(results)
        out["S1"], out["S2"] = family(results, SECONDARY["S1"]), family(results, SECONDARY["S2"])
        out["S4"] = {"IM": fitness_validity(results, [m for m in GRIDS["core"] if m not in NOT_IM]),
                     "DFID": fitness_validity(results, ["AGENT-DFID"])}
    else:
        out["S3"] = family(results, SECONDARY["S3"])
    path.write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("run", "analyse"))
    ap.add_argument("--grid", choices=sorted(GRIDS), required=True)
    ap.add_argument("--shard", default="0/1", help="i/n: this process runs RUN_SEEDS[i::n]")
    args = ap.parse_args()
    if args.action == "run":
        shard, n_shards = map(int, args.shard.split("/"))
        run(args.grid, shard, n_shards)
    else:
        out = analyse(args.grid)
        for key in ("primary", "S1", "S2", "S3"):
            if key in out:
                print(key, json.dumps(out[key], indent=1))


if __name__ == "__main__":
    main()
