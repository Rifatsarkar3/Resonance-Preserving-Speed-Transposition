"""EXPLORATORY WP7 (RESS spec §6, run after the DIRG verdict): the six baselines (DSR, COT, DANN, DEEP-CORAL, CDAN,
SHOT) on the Ottawa stream, 10 runs, B = 30 per change, deployed by the same cluster-balanced IM as every method, and
compared with AGENT from the frozen Ottawa core grid (verdict 43baf64). No confirmatory claim; nothing feeds back into
any method; the withdrawn v1 shards (a subdirectory) are never read.

    python -m tevc.probes.ottawa_baselines run [--shard i/n]   (resumable)  -> exploratory_baselines_shard*.json
    python -m tevc.probes.ottawa_baselines analyse                          -> exploratory_baselines.json
in outputs/tevc/ottawa/.
"""
import argparse
import json
import time

from tevc import baselines
from tevc.confirmatory import OUT, _frozen_commit
from tevc.dirg_protocol import family
from tevc.evaluator import RecipeEvaluator
from tevc.reliability import add_metrics
from tevc.reliability import summarize as summarize_reliability
from tevc.run_stream import RUN_SEEDS, run_method, score, summarize
from tevc.stream import assert_disjoint, ottawa_stream

METHODS = baselines.METHODS


def run(shard, n_shards):
    commit = _frozen_commit()
    path = OUT / f"exploratory_baselines_shard{shard}of{n_shards}.json"
    results = {m: {} for m in METHODS}
    if path.exists():
        d = json.loads(path.read_text())
        if d["commit"] != commit:
            raise SystemExit(f"{path} was written at {d['commit'][:7]}, HEAD is {commit[:7]}: resume from that commit")
        results = d["results"]
    stream = ottawa_stream()
    assert_disjoint(stream)
    evaluator, t0 = RecipeEvaluator(stream), time.time()
    for run_seed in RUN_SEEDS[shard::n_shards]:
        for m in METHODS:
            if str(run_seed) in results[m]:
                continue
            t = time.time()
            results[m][str(run_seed)] = [score(evaluator, stream, env, carried, trials) for env, (carried, trials)
                                         in zip(stream.environments, run_method(m, stream, evaluator, run_seed))]
            path.write_text(json.dumps({"commit": commit, "shard": [shard, n_shards], "results": results}))
            print(f"run {run_seed:5d} {m:11s} {time.time() - t:6.0f}s", flush=True)
    print(f"Ottawa baselines shard {shard}/{n_shards} done in {time.time() - t0:.0f}s", flush=True)


def _merged(pattern):
    results, commits = {}, set()
    for f in sorted(OUT.glob(pattern)):
        d = json.loads(f.read_text())
        commits.add(d["commit"])
        for m, runs in d["results"].items():
            results.setdefault(m, {}).update(runs)
    return results, commits


def analyse():
    base, commits = _merged("exploratory_baselines_shard*.json")
    missing = [(m, s) for m in METHODS for s in RUN_SEEDS if str(s) not in base.get(m, {})]
    if missing or len(commits) != 1:
        raise SystemExit(f"incomplete or mixed baseline grid: {len(missing)} cells missing, commits {sorted(commits)}")
    core, _ = _merged("core_shard*.json")              # the frozen Ottawa core grid (top level only)
    results = {"AGENT": core["AGENT"], **{m: base[m] for m in METHODS}}
    stream = ottawa_stream()
    add_metrics(results, RecipeEvaluator(stream), stream)
    out = {"note": "exploratory WP7, after the DIRG verdict; AGENT from the frozen Ottawa core grid",
           "baseline_commit": commits.pop(), "summary": summarize(results),
           "reliability": summarize_reliability(results),
           "accuracy_tests": family(results, [("AGENT", m, k) for m in METHODS for k in ("deployed_acc", "auc")]),
           "reliability_tests": family(results, [("AGENT", m, k) for m in METHODS
                                                 for k in ("missed_fault_rate", "false_alarm_rate")])}
    (OUT / "exploratory_baselines.json").write_text(json.dumps(out, indent=1))
    for m, s in out["reliability"].items():
        print(f"{m:11s} acc {s['deployed_acc'] * 100:6.2f}  missed {s['missed_fault_rate'] * 100:5.2f}  "
              f"false alarm {s['false_alarm_rate'] * 100:5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("run", "analyse"))
    ap.add_argument("--shard", default="0/1", help="i/n: this process runs RUN_SEEDS[i::n]")
    args = ap.parse_args()
    if args.action == "run":
        run(*map(int, args.shard.split("/")))
    else:
        analyse()


if __name__ == "__main__":
    main()
