"""DIRG confirmatory protocol (RESS spec §5), frozen before any model is trained on a DIRG target-speed record.
The author approved this version, D6 included, on 2026-10-16.

Data. Politecnico di Torino rig (Daga et al., MSSP 120:252, 2019; Zenodo 10.5281/zenodo.3559553), bearing B1, nominal
load 1000 N: the single record per class and speed at load-cell 450-550 mV (1A exists only as repeat m = 2), 10 s at
51.2 kHz. Seven classes: 0A healthy; 1A-3A inner-ring indentation 450/250/150 µm; 4A-6A roller indentation
450/250/150 µm. Channel: radial z of accelerometer A1, chosen by the channel rule on the 300 Hz records only
(5-fold blocked CV, z 100.00 % vs y 98.93 %; e56ff2e) before any target-speed record was read.
Speed. The rig has no tachometer. Every speed is estimated label-free from the windows the method may see (library
windows, the unlabelled buffer U): tevc.dirg_data.estimate_speed_hz.
Stream (tevc.stream.dirg_stream). The library is the 300 Hz records (80 windows per class). The environments are
nominal 400 -> 500 -> 400 -> 200 -> 100 Hz; the revisit is the same environment. In each target record, U = the first
5 s and T = the last 5 s, 57 windows per class each. Windows are 5120 samples with hop 2560. Admissibility: every
estimated environment speed is >= 1.3x or <= 1/1.3x the estimated library speed (checked before training).
Methods (13), each with B = 30 evaluations per change, 10 independent runs (RUN_SEEDS), TICNN for 40 epochs, fitness =
cluster-balanced IM on U: STATIC-NONE, STATIC-H2, DSR, COT, DANN, DEEP-CORAL, CDAN, SHOT, RANDOM, GA-RESTART,
WARM-TRANSFER, AGENT (v2), AGENT-B1.

Primary family: one-sided, unit = independent run, exact sign-flip test on the run-level mean difference, Holm over
the family, alpha = 0.05. Each hypothesis is confirmed iff its Holm-adjusted p < alpha; all verdicts are reported.
    D1  AGENT > DSR           deployed accuracy    resonance-preserving adaptation beats published resampling
    D2  AGENT > AGENT-B1      deployed accuracy    resonance preservation (replicates Ottawa H3)
    D3  AGENT > STATIC-NONE   deployed accuracy    adaptation pays off under large real speed shifts
    D5  AGENT < STATIC-NONE   missed-fault rate    adaptation reduces missed faults
    D6  AGENT < CDAN          missed-fault rate    fewer missed faults than the strongest domain-adaptation baseline
                                                   (CDAN: best DA accuracy on the development sets)
D4 (AGENT-PCS > AGENT) is dropped: PCS failed its one-shot development gate (bc3843b). For a lower-is-better metric
the tested difference is second - first, so a positive mean difference favours AGENT.
Secondary families (bootstrap CIs, Holm within each family, no confirmatory claim):
    S1  AGENT vs DANN, DEEP-CORAL, CDAN, SHOT, COT; accuracy and AUC
    S2  AGENT vs RANDOM, GA-RESTART, WARM-TRANSFER; accuracy and AUC
    S3  D1-D3 on 3-class location accuracy (healthy / inner ring / roller), merging the 7-class predictions
    S4  Spearman rho(IM fitness, true accuracy) over all evaluations, per environment
    S5  missed-fault rate, false-alarm rate, macro-F1 and ECE for every method; AGENT vs each, Holm per metric
Exploratory after the verdict only: +-2 % speed sensitivity; Ottawa (spec §6).
Mechanics. The runner refuses a dirty tevc/ and records HEAD. Shards split the run seeds and run in parallel; cache
entries are published atomically. Logs show progress only, never accuracy. The analysis runs once, refuses mixed
commits and refuses gaps.
    python -m tevc.dirg_protocol run --shard 0/2        (and --shard 1/2)
    python -m tevc.dirg_protocol analyse                -> outputs/tevc/dirg/verdict_dirg.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from tevc.baselines import config_from_dict
from tevc.confirmatory import _frozen_commit, fitness_validity, holm
from tevc.evaluator import RecipeEvaluator
from tevc.reliability import LOWER_IS_BETTER, add_metrics, compare_to_agent
from tevc.reliability import summarize as summarize_reliability
from tevc.run_stream import RUN_SEEDS, run_level_tests, run_method, score, summarize
from tevc.stream import assert_disjoint, dirg_stream

OUT = Path("outputs/tevc/dirg")
ALPHA = 0.05
MIN_RATIO = 1.3
METHODS = ("STATIC-NONE", "STATIC-H2", "DSR", "COT", "DANN", "DEEP-CORAL", "CDAN", "SHOT", "RANDOM", "GA-RESTART",
           "WARM-TRANSFER", "AGENT", "AGENT-B1")
PRIMARY = {"D1": ("AGENT", "DSR", "deployed_acc"), "D2": ("AGENT", "AGENT-B1", "deployed_acc"),
           "D3": ("AGENT", "STATIC-NONE", "deployed_acc"), "D5": ("AGENT", "STATIC-NONE", "missed_fault_rate"),
           "D6": ("AGENT", "CDAN", "missed_fault_rate")}
SECONDARY = {"S1": [("AGENT", m, k) for m in ("DANN", "DEEP-CORAL", "CDAN", "SHOT", "COT") for k in ("deployed_acc", "auc")],
             "S2": [("AGENT", m, k) for m in ("RANDOM", "GA-RESTART", "WARM-TRANSFER") for k in ("deployed_acc", "auc")],
             "S3": [(a, b, "location_acc") for a, b, _ in (PRIMARY["D1"], PRIMARY["D2"], PRIMARY["D3"])]}
LOCATION = np.array([0, 1, 1, 1, 2, 2, 2])        # 0A healthy; 1A-3A inner ring; 4A-6A roller


def family(results, tests):
    """One-sided run-level comparisons of first against second, Holm over this family; lower-is-better metrics are
    negated first, so a positive mean difference always favours first."""
    rows = []
    for a, b, k in tests:
        sign = -1.0 if k in LOWER_IS_BETTER else 1.0
        flipped = {m: {r: [{k: sign * e[k]} for e in envs] for r, envs in results[m].items()} for m in (a, b)}
        rows.append({"first": a, "second": b, "metric": k, "better": "lower" if sign < 0 else "higher",
                     **run_level_tests(flipped, k, ref=a)[b]})
    for row, p in zip(rows, holm([row["p"] for row in rows])):
        row["p_holm"] = p
    return rows


def verdict(results):
    rows = family(results, list(PRIMARY.values()))
    return {h: {**row, "confirmed": bool(row["p_holm"] < ALPHA)} for h, row in zip(PRIMARY, rows)}


def location_accuracy(probs, y):
    """Accuracy after merging the 7 classes into healthy / inner ring / roller."""
    return float((LOCATION[probs.argmax(axis=1)] == LOCATION[y]).mean())


def add_location(results, evaluator, stream):
    for runs in results.values():
        for envs in runs.values():
            for entry, env in zip(envs, stream.environments, strict=True):
                _, pT = evaluator.cached(env, config_from_dict(entry["deployed"]), entry["deployed"]["seed"])
                entry["location_acc"] = location_accuracy(pT, env.T["y"])
    return results


def check_admissible(stream):
    for env in stream.environments:
        r = env.rpm / stream.library_rpm
        if 1 / MIN_RATIO < r < MIN_RATIO:
            raise SystemExit(f"{env.name}: speed ratio {r:.3f} to the library is inside (1/{MIN_RATIO}, {MIN_RATIO})")


def run(shard, n_shards, out_dir=OUT):
    commit = _frozen_commit()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"core_shard{shard}of{n_shards}.json"
    stream = dirg_stream()
    assert_disjoint(stream)
    check_admissible(stream)
    speeds = {"library": stream.library_rpm, **{e.name: e.rpm for e in stream.environments}}
    evaluator = RecipeEvaluator(stream)
    results, t0 = {m: {} for m in METHODS}, time.time()
    for run_seed in RUN_SEEDS[shard::n_shards]:
        for m in METHODS:
            t = time.time()
            results[m][str(run_seed)] = [score(evaluator, stream, env, carried, trials) for env, (carried, trials)
                                         in zip(stream.environments, run_method(m, stream, evaluator, run_seed))]
            path.write_text(json.dumps({"commit": commit, "shard": [shard, n_shards], "estimated_rpm": speeds,
                                        "elapsed_s": round(time.time() - t0), "results": results}))
            evals = sum(e["n_evals"] for e in results[m][str(run_seed)])
            print(f"run {run_seed:5d} {m:14s} evals {evals:3d}  {time.time() - t:6.0f}s", flush=True)
    print(f"DIRG shard {shard}/{n_shards} done in {time.time() - t0:.0f}s", flush=True)


def load_shards(out_dir):
    """(results, protocol commit) of a complete grid from one commit; SystemExit otherwise."""
    results, commits = {m: {} for m in METHODS}, set()
    for shard in sorted(out_dir.glob("core_shard*.json")):
        d = json.loads(shard.read_text())
        commits.add(d["commit"])
        for m, runs in d["results"].items():
            results[m].update(runs)
    if len(commits) != 1:
        raise SystemExit(f"DIRG shards ran from {len(commits)} commits: {sorted(commits)}")
    missing = [(m, s) for m in METHODS for s in RUN_SEEDS if str(s) not in results[m]]
    if missing:
        raise SystemExit(f"DIRG grid incomplete: {len(missing)} (method, run) cells missing, e.g. {missing[:3]}")
    return results, commits.pop()


def analyse(out_dir=OUT, stream=None, evaluator=None):
    path = out_dir / "verdict_dirg.json"
    if path.exists():
        raise SystemExit(f"{path} exists: the DIRG analysis runs once")
    results, commit = load_shards(out_dir)
    stream = stream or dirg_stream()
    evaluator = evaluator or RecipeEvaluator(stream)
    add_metrics(results, evaluator, stream)
    add_location(results, evaluator, stream)
    out = {"protocol_commit": commit, "run_seeds": list(RUN_SEEDS), "alpha": ALPHA, "summary": summarize(results),
           "primary": verdict(results), **{k: family(results, tests) for k, tests in SECONDARY.items()},
           "S4": fitness_validity(results, METHODS),
           "S5": {"summary": summarize_reliability(results), "tests_vs_agent": compare_to_agent(results)}}
    path.write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("run", "analyse"))
    ap.add_argument("--shard", default="0/1", help="i/n: this process runs RUN_SEEDS[i::n]")
    args = ap.parse_args()
    if args.action == "run":
        run(*map(int, args.shard.split("/")))
    else:
        out = analyse()
        print("primary", json.dumps(out["primary"], indent=1))


if __name__ == "__main__":
    main()
