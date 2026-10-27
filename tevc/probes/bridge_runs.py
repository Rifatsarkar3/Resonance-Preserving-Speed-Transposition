"""EXPLORATORY bridge runs (MSSP reframe spec docs/superpowers/specs/2026-10-20-mssp-sp-first-reframe-design.md
§5.3-5.6, with the attack-audit amendments A1-A6). No confirmatory claim; the Lee verdict (3bdb9b9) stands as
reported; nothing feeds back into any method; the withdrawn Ottawa v1 shards are never read.

  lee, pu   RPST-DA arms on RPST-DA's own draws (tevc.rpstda.ARMS): B1 (copies by plain resampling), NCS (no copies,
            the copies twin's number of steps), W0 (copies, adversarial weight 0), B05 (beta 0.5; --optional); new
            baselines (tevc.baselines.EXPLORATORY_METHODS): COT-T, COT-CDAN, ENV-ORDER. Compared with the archived
            RPST-DA and comparators of the same stream and run seeds (lee: core shards of dc05747; pu: the
            development results stream_pu_rpstda.json and stream_pu_rbase.json).
  dirg      RPST-DA-FR: copies and transposed validation at the frame of the rule N <= rho* P (rpstda.frame_rule_n;
            rho* and the period convention from R6, outputs/tevc/probe_validity_interval.json), compared with the
            archived RPST-DA (6a4a040), CDAN and DANN (df3eccf).
  tests     two-sided exact sign-flip over the 10 runs, Holm within the family (lee, pu: RPST-DA against B1, NCS,
            W0, COT-CDAN, COT-T, ENV-ORDER and DSR; dirg: RPST-DA-FR against RPST-DA, CDAN and DANN); every arm also
            reported deployed by IM alone and by best-of-30.
  check     before any comparison: the archived RPST-DA and CDAN of run 42, rescored from the model cache alone (no
            training), must reproduce every stored candidate accuracy; a miss means a cache key changed.
  speed     (audit G7, optional) RPST-DA with every environment's estimated speed off by -3 % and +3 %: every
            speed-dependent part (copies, transposed validation, IM clusters) sees the wrong speed, the test set is
            unchanged; compared two-sided with the archived RPST-DA of the same runs.

    python -m tevc.probes.bridge_runs check   --dataset lee
    python -m tevc.probes.bridge_runs run     --dataset lee --shard 0/2 [--optional]   (resumable at the same commit)
    python -m tevc.probes.bridge_runs analyse --dataset lee
    python -m tevc.probes.bridge_runs run-speed --dataset pu [--shard i/n];  analyse-speed --dataset pu
Outputs: outputs/tevc/{lee,dirg}/exploratory_bridge*.json, outputs/tevc/pu_bridge/exploratory_bridge*.json,
exploratory_speed_error*.json beside them.
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from tevc import baselines, dirg_protocol, rpstda
from tevc.baselines import config_from_dict
from tevc.confirmatory import _frozen_commit
from tevc.dirg_protocol import family
from tevc.evaluator import RecipeEvaluator
from tevc.lee_protocol import best_of
from tevc.run_stream import RUN_SEEDS, score, score_rpstda, summarize
from tevc.stream import assert_disjoint, dirg_stream, lee_stream, pu_stream

STREAMS = {"lee": lee_stream, "pu": pu_stream, "dirg": dirg_stream}
OUT_DIRS = {"lee": Path("outputs/tevc/lee"), "pu": Path("outputs/tevc/pu_bridge"), "dirg": Path("outputs/tevc/dirg")}
ARMS = {"lee": ("RPST-DA-B1", "RPST-DA-NCS", "RPST-DA-W0"), "pu": ("RPST-DA-B1", "RPST-DA-NCS", "RPST-DA-W0"),
        "dirg": ("RPST-DA-FR",)}
OPTIONAL_ARMS = {"lee": ("RPST-DA-B05",), "pu": ("RPST-DA-B05",), "dirg": ()}
BASELINES = {"lee": baselines.EXPLORATORY_METHODS, "pu": baselines.EXPLORATORY_METHODS, "dirg": ()}
ARCHIVES = {"lee": ("outputs/tevc/lee/core_shard0of2.json", "outputs/tevc/lee/core_shard1of2.json"),
            "pu": ("outputs/tevc/stream_pu_rpstda.json", "outputs/tevc/stream_pu_rbase.json")}
CHECKED = ("RPST-DA", "CDAN")
INTERVAL = Path("outputs/tevc/probe_validity_interval.json")
SHARD_GLOB = "exploratory_bridge_shard*.json"


def frame_rule_parameters(path=INTERVAL):
    """(rho*, period convention) from the R6 output; SystemExit before R6 has run (spec §5.5: R8 waits for R6)."""
    if not path.exists():
        raise SystemExit(f"{path} is missing: run R6 (python -m tevc.probes.validity_interval) before RPST-DA-FR")
    d = json.loads(path.read_text())
    return float(d["lower_end"]), d["convention"]


def arm_transforms(dataset, stream):
    """arm -> (environment -> config transform). The frame-rule arm's frame may depend on the environment (target
    period convention); every other arm is environment-independent."""
    if dataset == "dirg":
        rho, convention = frame_rule_parameters()
        return {"RPST-DA-FR": lambda env: rpstda.frame_rule_arm(rpstda.frame_rule_n(stream, env, rho, convention),
                                                              stream.dataset)}
    return {a: (lambda env, a=a: rpstda.ARMS[a]) for a in ARMS[dataset] + OPTIONAL_ARMS[dataset]}


def _merge(paths):
    out = {}
    for p in paths:
        for m, runs in json.loads(Path(p).read_text(encoding="utf-8"))["results"].items():
            out.setdefault(m, {}).update(runs)
    return out


def archived(dataset):
    """The frozen results the new arms are compared with, on the same stream and run seeds."""
    if dataset == "dirg":
        from tevc.probes.rpstda_heldout import OUTS, load_own
        core, _ = dirg_protocol.load_shards(dirg_protocol.OUT)
        own, _ = load_own(OUTS["dirg"])
        return {**core, **own}
    return _merge(ARCHIVES[dataset])


def compare_cached(entries, envs, scorer):
    """Every stored candidate of one archived run against its accuracy rescored by scorer(env, config, seed), which
    reads the model cache only (KeyError when the entry is absent): {"checked", "mismatches", "misses"}."""
    report = {"checked": 0, "mismatches": [], "misses": []}
    for visit in entries:
        env = envs[visit["env"]]
        for trial in visit["trials"]:
            cfg, report["checked"] = config_from_dict(trial), report["checked"] + 1
            try:
                acc = scorer(env, cfg, trial["seed"])
            except KeyError:
                report["misses"].append([visit["env"], trial["seed"]])
                continue
            if abs(acc - trial["acc"]) > 1e-9:
                report["mismatches"].append([visit["env"], trial["seed"], trial["acc"], acc])
    return report


def check(dataset, run_seed=RUN_SEEDS[0]):
    stream = STREAMS[dataset]()
    evaluator, envs, old = RecipeEvaluator(stream), {e.name: e for e in stream.environments}, archived(dataset)

    def scorer(env, cfg, seed):
        _, pT = evaluator.cached(env, cfg, seed)
        return float((pT.argmax(axis=1) == env.T["y"]).mean())

    report = {m: compare_cached(old[m][str(run_seed)], envs, scorer) for m in CHECKED if m in old}
    print(json.dumps({m: {"checked": r["checked"], "mismatches": len(r["mismatches"]), "misses": len(r["misses"])}
                      for m, r in report.items()}, indent=1), flush=True)
    ok = bool(report) and all(not r["mismatches"] and not r["misses"] for r in report.values())
    print("cache check passed" if ok else "cache check FAILED: a cache key or a code path changed", flush=True)
    return ok


def run(dataset, shard, n_shards, optional=False):
    commit = _frozen_commit()
    out_dir = OUT_DIRS[dataset]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"exploratory_bridge_shard{shard}of{n_shards}.json"
    methods = ARMS[dataset] + (OPTIONAL_ARMS[dataset] if optional else ()) + BASELINES[dataset]
    results, frames = {m: {} for m in methods}, {}
    if path.exists():
        d = json.loads(path.read_text())
        if d["commit"] != commit:
            raise SystemExit(f"{path} was written at {d['commit'][:7]}, HEAD is {commit[:7]}: resume from that commit")
        results = {m: d["results"].get(m, {}) for m in methods}
        frames = d.get("frame_rule", {})
    stream = STREAMS[dataset]()
    assert_disjoint(stream)
    evaluator, transforms, t0 = RecipeEvaluator(stream), arm_transforms(dataset, stream), time.time()
    if dataset == "dirg":
        rho, convention = frame_rule_parameters()
        frames = {"rho": rho, "convention": convention,
                  "n_by_env": {e.name: rpstda.frame_rule_n(stream, e, rho, convention) for e in stream.environments}}
    for run_seed in RUN_SEEDS[shard::n_shards]:
        for m in methods:
            if str(run_seed) in results[m]:
                continue
            t, visits = time.time(), []
            for env in stream.environments:
                if m in transforms:
                    trials = rpstda.search(evaluator, env, run_seed, transform=transforms[m](env))
                    visits.append(score_rpstda(evaluator, stream, env, trials))
                else:
                    visits.append(score(evaluator, stream, env, [], baselines.search(m, evaluator, env, run_seed)))
            results[m][str(run_seed)] = visits
            path.write_text(json.dumps({"commit": commit, "dataset": dataset, "shard": [shard, n_shards],
                                        "frame_rule": frames, "results": results}))
            print(f"run {run_seed:5d} {m:12s} {time.time() - t:6.0f}s", flush=True)
    print(f"{dataset} bridge shard {shard}/{n_shards} done in {time.time() - t0:.0f}s", flush=True)


def load_own(dataset):
    """(results, commit) of the complete bridge grid of a dataset; SystemExit on gaps or mixed commits."""
    results, commits = {}, set()
    for f in sorted(OUT_DIRS[dataset].glob(SHARD_GLOB)):
        d = json.loads(f.read_text())
        commits.add(d["commit"])
        for m, runs in d["results"].items():
            results.setdefault(m, {}).update(runs)
    missing = [(m, s) for m in results for s in RUN_SEEDS if str(s) not in results[m]]
    if not results or missing or len(commits) != 1:
        raise SystemExit(f"incomplete or mixed bridge grid: {len(missing)} cells missing, commits {sorted(commits)}")
    return results, commits.pop()


def im_only(results, arm):
    """Run-level mean accuracy of an RPST-DA arm deployed by IM alone (stored per visit as deployed_acc_im)."""
    return {r: float(np.mean([v["deployed_acc_im"] for v in visits])) for r, visits in results[arm].items()}


def tests_for(dataset, methods):
    if dataset == "dirg":
        return [("RPST-DA-FR", m, "deployed_acc") for m in ("RPST-DA", "CDAN", "DANN")]
    return [("RPST-DA", m, "deployed_acc") for m in methods if m != "RPST-DA-B05"] + [("RPST-DA", "DSR", "deployed_acc")]


def analysis(dataset, results, own_methods):
    acc, best = summarize(results), best_of(results)
    arms = [m for m in results if m.startswith("RPST-DA")]
    return {"tests": family(results, tests_for(dataset, own_methods), alternative="two-sided"),
            "summary": acc, "best_of": best,
            "selection_loss": {m: best[m] - acc[m]["acc_mean"] for m in results},
            "im_only": {m: float(np.mean(list(im_only(results, m).values()))) for m in arms
                        if all("deployed_acc_im" in v for runs in results[m].values() for v in runs)}}


def analyse(dataset):
    own, commit = load_own(dataset)
    results = {**archived(dataset), **own}
    out = {"note": "exploratory bridge runs (reframe spec 2026-10-20 §5.3-5.6, amendments A1-A6); two-sided tests",
           "dataset": dataset, "bridge_commit": commit, **analysis(dataset, results, tuple(own))}
    path = OUT_DIRS[dataset] / "exploratory_bridge.json"
    path.write_text(json.dumps(out, indent=1))
    for row in out["tests"]:
        print(f"{row['first']} vs {row['second']:12s} {100 * row['mean_diff']:+6.2f} pp  W/L {row['wins']}/"
              f"{row['losses']}  p {row['p']:.4f}  p_holm {row['p_holm']:.4f}", flush=True)
    print(f"-> {path}", flush=True)


SPEED_ERRORS = (-0.03, 0.03)
SPEED_GLOB = "exploratory_speed_error_shard*.json"


def speed_label(error):
    return f"RPST-DA@{100 * error:+.0f}%"


def perturbed(stream, error):
    """The stream with every environment's estimated speed off by the fraction `error` (names marked, so that no
    cache entry is shared with the true speed); revisits stay the same environment object; library unchanged."""
    envs = {}
    for e in stream.environments:
        if e.name not in envs:
            envs[e.name] = replace(e, name=f"{e.name}~{100 * error:+.0f}%", rpm=e.rpm * (1.0 + error))
    return replace(stream, environments=tuple(envs[e.name] for e in stream.environments))


def run_speed(dataset, shard, n_shards):
    if dataset == "dirg":
        raise SystemExit("the speed-error runs are defined for lee and pu only")
    commit = _frozen_commit()
    out_dir = OUT_DIRS[dataset]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"exploratory_speed_error_shard{shard}of{n_shards}.json"
    labels = [speed_label(e) for e in SPEED_ERRORS]
    results = {m: {} for m in labels}
    if path.exists():
        d = json.loads(path.read_text())
        if d["commit"] != commit:
            raise SystemExit(f"{path} was written at {d['commit'][:7]}, HEAD is {commit[:7]}: resume from that commit")
        results = {m: d["results"].get(m, {}) for m in labels}
    base = STREAMS[dataset]()
    assert_disjoint(base)
    for error, label in zip(SPEED_ERRORS, labels):
        stream = perturbed(base, error)
        evaluator = RecipeEvaluator(stream)
        for run_seed in RUN_SEEDS[shard::n_shards]:
            if str(run_seed) in results[label]:
                continue
            t = time.time()
            results[label][str(run_seed)] = [score_rpstda(evaluator, stream, env, rpstda.search(evaluator, env, run_seed))
                                             for env in stream.environments]
            path.write_text(json.dumps({"commit": commit, "dataset": dataset, "shard": [shard, n_shards],
                                        "speed_errors": list(SPEED_ERRORS), "results": results}))
            print(f"run {run_seed:5d} {label:12s} {time.time() - t:6.0f}s", flush=True)


def analyse_speed(dataset):
    results, commits = {}, set()
    for f in sorted(OUT_DIRS[dataset].glob(SPEED_GLOB)):
        d = json.loads(f.read_text())
        commits.add(d["commit"])
        for m, runs in d["results"].items():
            results.setdefault(m, {}).update(runs)
    missing = [(m, s) for m in results for s in RUN_SEEDS if str(s) not in results[m]]
    if not results or missing or len(commits) != 1:
        raise SystemExit(f"incomplete or mixed speed-error grid: {len(missing)} missing, commits {sorted(commits)}")
    results["RPST-DA"] = archived(dataset)["RPST-DA"]
    tests = family(results, [("RPST-DA", speed_label(e), "deployed_acc") for e in SPEED_ERRORS], alternative="two-sided")
    out = {"note": "exploratory (audit G7): estimated target speed off by the stated fraction; two-sided tests",
           "dataset": dataset, "commit": commits.pop(), "summary": summarize(results), "tests": tests}
    path = OUT_DIRS[dataset] / "exploratory_speed_error.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"-> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("check", "run", "analyse", "run-speed", "analyse-speed"))
    ap.add_argument("--dataset", choices=tuple(STREAMS), required=True)
    ap.add_argument("--shard", default="0/1", help="i/n: this process runs RUN_SEEDS[i::n]")
    ap.add_argument("--optional", action="store_true", help="also run the optional arms (RPST-DA-B05)")
    args = ap.parse_args()
    if args.action == "check":
        sys.exit(0 if check(args.dataset) else 1)
    if args.action == "run":
        run(args.dataset, *map(int, args.shard.split("/")), optional=args.optional)
    elif args.action == "run-speed":
        run_speed(args.dataset, *map(int, args.shard.split("/")))
    elif args.action == "analyse-speed":
        analyse_speed(args.dataset)
    else:
        analyse(args.dataset)


if __name__ == "__main__":
    main()
