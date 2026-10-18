"""Lee confirmatory protocol (MSSP spec docs/superpowers/specs/2026-10-18-mssp-rpst-da-design.md §4), frozen before any
Lee target-speed record is read. The author approved this version, as drafted, on 2026-10-18.

Data. Lee, Kim & Kim (2024), Data in Brief 57:110940, subset 1: deep groove ball bearing 6204 (8 balls, d 7.94 mm,
pitch diameter 34.57 mm, contact angle 0), Mendeley Data 10.17632/53vtnjy6c6.1 (CC BY 4.0), every file SHA-256
verified against the published manifest (tevc.lee_data). Healthy rotor, 16 kHz, 80 s per record, one uniaxial
accelerometer on the shaft-end housing. Four classes: bearing healthy, ball, inner race, outer race (one bearing per
class). Until this protocol is approved and frozen, the adapter refuses every record but the 800-rpm library.
Speed. Label-free (tevc.dirg_data.estimate_speed_hz) from the library windows and each U buffer, in the band frozen
from the 800-rpm library records alone (0.85-1.05 x nominal; library 1.001-1.004 x nominal; aeec018).
Stream (tevc.stream.lee_stream). Library = the 800-rpm records, 80 windows per class. Environments nominal
1200 -> 1600 -> 1200 -> 600 -> 1400 rpm; the revisit is the same environment. In each target record U = the first
40 s and T = the last 40 s, 80 windows per class each; windows 8000 samples (0.5 s), hop 4000. Admissibility: every
estimated environment speed >= 1.3x or <= 1/1.3x the estimated library speed (checked before training).
Operator frames in physical time. Phase-vocoder and WSOLA frames are a quarter of the development sample counts on
Lee (LEE.frame_scale = 1/4): the durations of PU's frames at 64 kHz. Reason, fixed before any target record: at
16 kHz a 256-sample frame (16 ms) is longer than the inner-race impact period at 800 rpm (~15 ms), and a phase vocoder
keeps the rate of impacts closer than its frame (synthetic probe, 2026-10-18). It applies to every RPST-based method
(AGENT, AGENT-B1, RPST-DA, RPST-DA-NC) and to the transposed validation set; it is the identity on PU, JNU and DIRG.
Methods (10), each with B = 30 evaluations per change, 10 independent runs (RUN_SEEDS), TICNN for 40 epochs, fitness =
cluster-balanced IM on U: STATIC-NONE, DSR, COT, DANN, CDAN, SHOT, AGENT (v2), AGENT-B1, RPST-DA (deployed by the mean
rank of TV accuracy and IM; development gate passed, adb6184), RPST-DA-NC (the same draws with copies = 0).

Primary family: one-sided, unit = independent run, exact sign-flip test on the run-level mean difference, Holm over
the five, alpha = 0.05. Each hypothesis is confirmed iff its Holm-adjusted p < alpha; all verdicts are reported.
    E1  RPST-DA > CDAN                       deployed accuracy    beats the strongest domain adaptation
    E2  RPST-DA > AGENT                      deployed accuracy    beats synthesis-only adaptation
    E3  AGENT > AGENT-B1                     deployed accuracy    resonance preservation (third held-out test)
    E4  AGENT > DSR                          deployed accuracy    beats published resampling (replicates DIRG D1)
    E5  RPST-DA > RPST-DA (IM-only)          deployed accuracy    transposed validation helps, same candidates
Secondary (no confirmatory claim; Holm within each family):
    S1  RPST-DA vs DANN, SHOT, COT, STATIC-NONE; accuracy and AUC
    S2  RPST-DA vs RPST-DA-NC; accuracy and AUC (contribution of the RPST copies)
    S3  missed-fault and false-alarm rates (reported as a pair), macro-F1 and ECE for every method; descriptive
    S4  Spearman rho of IM with test accuracy for every method; of TV accuracy for both RPST-DA arms; descriptive
    S5  best-of-30 test accuracy per method (best evaluated or carried model per visit): equal tuning budgets
Exploratory after the verdict only: Lee operator fidelity; RPST-DA on DIRG and Ottawa (spec §5 M7).
Limitations to state: one faulty bearing per class; 8 kHz bandwidth.
Mechanics. The runner refuses locked target records, a dirty tevc/, and records HEAD. Shards split the run seeds and
run in parallel; cache entries are published atomically. Logs show progress only, never accuracy. The analysis runs
once, refuses mixed commits and refuses gaps.
    python -m tevc.lee_protocol run --shard 0/2        (and --shard 1/2)
    python -m tevc.lee_protocol analyse                -> outputs/tevc/lee/verdict_lee.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from tevc import lee_data
from tevc.confirmatory import _frozen_commit, fitness_validity
from tevc.dirg_protocol import ALPHA, check_admissible, family
from tevc.evaluator import RecipeEvaluator
from tevc.probes.dirg_diagnostics import visit_headroom
from tevc.reliability import add_metrics
from tevc.reliability import summarize as summarize_reliability
from tevc.rpstda_gate import descriptive
from tevc.run_stream import RPSTDA_ARMS, RUN_SEEDS, run_method, score_visit, summarize
from tevc.stream import assert_disjoint, lee_stream

OUT = Path("outputs/tevc/lee")
METHODS = ("STATIC-NONE", "DSR", "COT", "DANN", "CDAN", "SHOT", "AGENT", "AGENT-B1", "RPST-DA", "RPST-DA-NC")
IM_ONLY = "RPST-DA (IM-only)"           # RPST-DA's own candidates deployed by IM alone (each visit's deployed_acc_im)
PRIMARY = {"E1": ("RPST-DA", "CDAN", "deployed_acc"), "E2": ("RPST-DA", "AGENT", "deployed_acc"),
           "E3": ("AGENT", "AGENT-B1", "deployed_acc"), "E4": ("AGENT", "DSR", "deployed_acc"),
           "E5": ("RPST-DA", IM_ONLY, "deployed_acc")}
SECONDARY = {"S1": [("RPST-DA", m, k) for m in ("DANN", "SHOT", "COT", "STATIC-NONE") for k in ("deployed_acc", "auc")],
             "S2": [("RPST-DA", "RPST-DA-NC", k) for k in ("deployed_acc", "auc")]}


def with_im_only(results):
    """results plus RPST-DA's IM-only deployments on the same candidates as a pseudo-method (for E5)."""
    im = {r: [{"deployed_acc": e["deployed_acc_im"]} for e in envs] for r, envs in results["RPST-DA"].items()}
    return {**results, IM_ONLY: im}


def verdict(results):
    rows = family(with_im_only(results), list(PRIMARY.values()))
    return {h: {**row, "confirmed": bool(row["p_holm"] < ALPHA)} for h, row in zip(PRIMARY, rows)}


def best_of(results):
    """Per method: mean over runs of the stream-mean best test accuracy per visit (best evaluated candidate, or the
    deployed model when it was carried from an earlier visit)."""
    return {m: float(np.mean([np.mean([visit_headroom(e)[1] for e in envs]) for envs in runs.values()]))
            for m, runs in results.items()}


def run(shard, n_shards, out_dir=OUT):
    if not lee_data.TARGETS_RELEASED:
        raise SystemExit("Lee target records are locked: this protocol has not been approved and frozen")
    commit = _frozen_commit()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"core_shard{shard}of{n_shards}.json"
    stream = lee_stream()
    assert_disjoint(stream)
    check_admissible(stream)
    speeds = {"library": stream.library_rpm, **{e.name: e.rpm for e in stream.environments}}
    evaluator = RecipeEvaluator(stream)
    results, t0 = {m: {} for m in METHODS}, time.time()
    for run_seed in RUN_SEEDS[shard::n_shards]:
        for m in METHODS:
            t = time.time()
            results[m][str(run_seed)] = [score_visit(m, evaluator, stream, env, carried, trials)
                                         for env, (carried, trials)
                                         in zip(stream.environments, run_method(m, stream, evaluator, run_seed))]
            path.write_text(json.dumps({"commit": commit, "shard": [shard, n_shards], "estimated_rpm": speeds,
                                        "elapsed_s": round(time.time() - t0), "results": results}))
            evals = sum(e["n_evals"] for e in results[m][str(run_seed)])
            print(f"run {run_seed:5d} {m:11s} evals {evals:3d}  {time.time() - t:6.0f}s", flush=True)
    print(f"Lee shard {shard}/{n_shards} done in {time.time() - t0:.0f}s", flush=True)


def load_shards(out_dir):
    """(results, protocol commit) of a complete grid from one commit; SystemExit otherwise."""
    results, commits = {m: {} for m in METHODS}, set()
    for shard in sorted(out_dir.glob("core_shard*.json")):
        d = json.loads(shard.read_text())
        commits.add(d["commit"])
        for m, runs in d["results"].items():
            results[m].update(runs)
    if len(commits) != 1:
        raise SystemExit(f"Lee shards ran from {len(commits)} commits: {sorted(commits)}")
    missing = [(m, s) for m in METHODS for s in RUN_SEEDS if str(s) not in results[m]]
    if missing:
        raise SystemExit(f"Lee grid incomplete: {len(missing)} (method, run) cells missing, e.g. {missing[:3]}")
    return results, commits.pop()


def analyse(out_dir=OUT, stream=None, evaluator=None):
    path = out_dir / "verdict_lee.json"
    if path.exists():
        raise SystemExit(f"{path} exists: the Lee analysis runs once")
    results, commit = load_shards(out_dir)
    stream = stream or lee_stream()
    evaluator = evaluator or RecipeEvaluator(stream)
    add_metrics(results, evaluator, stream)
    out = {"protocol_commit": commit, "run_seeds": list(RUN_SEEDS), "alpha": ALPHA, "summary": summarize(results),
           "primary": verdict(results), **{k: family(results, tests) for k, tests in SECONDARY.items()},
           "S3": summarize_reliability(results),
           "S4": {"im": {m: fitness_validity(results, [m]) for m in METHODS},
                  "tv": {m: descriptive(results[m]) for m in RPSTDA_ARMS}},
           "S5": best_of(results)}
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
