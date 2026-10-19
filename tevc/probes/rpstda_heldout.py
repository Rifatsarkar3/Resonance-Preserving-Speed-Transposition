"""EXPLORATORY (MSSP spec §5 M7, after the one-time Lee verdict 3bdb9b9): RPST-DA and its no-copies arm RPST-DA-NC
on the two used held-out streams, DIRG and Ottawa: 10 runs, B = 30 per change; candidates, transposed validation and
deployment exactly as on Lee (operator frames are the identity on both). Compared on the same streams and run seeds
with the frozen archives: DIRG's confirmatory grid (verdict df3eccf) and Ottawa's core grid (verdict 43baf64) plus the
WP7 baselines (d9c205e). No confirmatory claim; nothing feeds back into any method; the withdrawn Ottawa v1 shards (a
subdirectory) are never read.

    python -m tevc.probes.rpstda_heldout run --dataset dirg [--shard i/n]   (resumable at the same commit)
    python -m tevc.probes.rpstda_heldout analyse --dataset dirg             -> exploratory_rpstda.json
in outputs/tevc/<dataset>/ (dataset: dirg or ottawa).
"""
import argparse
import json
import time

from tevc import confirmatory, dirg_protocol
from tevc.confirmatory import _frozen_commit, fitness_validity
from tevc.dirg_protocol import family
from tevc.evaluator import RecipeEvaluator
from tevc.lee_protocol import best_of
from tevc.probes.ottawa_baselines import _merged
from tevc.reliability import add_metrics
from tevc.reliability import summarize as summarize_reliability
from tevc.rpstda_gate import descriptive
from tevc.run_stream import RPSTDA_ARMS, RUN_SEEDS, run_method, score_visit, summarize
from tevc.stream import assert_disjoint, dirg_stream, ottawa_stream

ARMS = tuple(RPSTDA_ARMS)
STREAMS = {"dirg": dirg_stream, "ottawa": ottawa_stream}
OUTS = {"dirg": dirg_protocol.OUT, "ottawa": confirmatory.OUT}
COMPARATORS = ("STATIC-NONE", "DSR", "COT", "DANN", "DEEP-CORAL", "CDAN", "SHOT", "AGENT", "AGENT-B1")
VERSUS = ("CDAN", "DANN", "DEEP-CORAL", "SHOT", "AGENT", "DSR", "STATIC-NONE", "RPST-DA-NC")


def run(dataset, shard, n_shards):
    commit = _frozen_commit()
    path = OUTS[dataset] / f"exploratory_rpstda_shard{shard}of{n_shards}.json"
    results = {m: {} for m in ARMS}
    if path.exists():
        d = json.loads(path.read_text())
        if d["commit"] != commit:
            raise SystemExit(f"{path} was written at {d['commit'][:7]}, HEAD is {commit[:7]}: resume from that commit")
        results = d["results"]
    stream = STREAMS[dataset]()
    assert_disjoint(stream)
    evaluator, t0 = RecipeEvaluator(stream), time.time()
    for run_seed in RUN_SEEDS[shard::n_shards]:
        for m in ARMS:
            if str(run_seed) in results[m]:
                continue
            t = time.time()
            results[m][str(run_seed)] = [score_visit(m, evaluator, stream, env, carried, trials)
                                         for env, (carried, trials)
                                         in zip(stream.environments, run_method(m, stream, evaluator, run_seed))]
            path.write_text(json.dumps({"commit": commit, "dataset": dataset, "shard": [shard, n_shards],
                                        "results": results}))
            print(f"run {run_seed:5d} {m:11s} {time.time() - t:6.0f}s", flush=True)
    print(f"{dataset} RPST-DA shard {shard}/{n_shards} done in {time.time() - t0:.0f}s", flush=True)


def load_own(out_dir):
    """(results, commit) of the complete RPST-DA grid in out_dir; SystemExit on gaps or mixed commits."""
    results, commits = {m: {} for m in ARMS}, set()
    for f in sorted(out_dir.glob("exploratory_rpstda_shard*.json")):
        d = json.loads(f.read_text())
        commits.add(d["commit"])
        for m, runs in d["results"].items():
            results[m].update(runs)
    missing = [(m, s) for m in ARMS for s in RUN_SEEDS if str(s) not in results[m]]
    if missing or len(commits) != 1:
        raise SystemExit(f"incomplete or mixed RPST-DA grid: {len(missing)} cells missing, commits {sorted(commits)}")
    return results, commits.pop()


def comparators(dataset):
    """COMPARATORS from the frozen archives of the same stream and run seeds."""
    if dataset == "dirg":
        archived, _ = dirg_protocol.load_shards(OUTS["dirg"])
    else:
        archived = {**_merged("core_shard*.json")[0], **_merged("exploratory_baselines_shard*.json")[0]}
    return {m: archived[m] for m in COMPARATORS}


def analysis(results):
    """Exploratory tables from complete results whose entries carry add_metrics' keys."""
    acc, best = summarize(results), best_of(results)
    return {"summary": acc, "reliability": summarize_reliability(results),
            "tests": family(results, [("RPST-DA", m, k) for m in VERSUS for k in ("deployed_acc", "auc")]),
            "tv": {m: descriptive(results[m]) for m in ARMS},
            "im_validity": {m: fitness_validity(results, [m]) for m in ARMS + ("CDAN", "DANN", "AGENT")},
            "best_of": best, "selection_loss": {m: best[m] - acc[m]["acc_mean"] for m in results}}


def analyse(dataset):
    own, commit = load_own(OUTS[dataset])
    results = {**comparators(dataset), **own}
    stream = STREAMS[dataset]()
    add_metrics(results, RecipeEvaluator(stream), stream)
    out = {"note": "exploratory M7, after the Lee verdict (3bdb9b9); comparators from the frozen archives",
           "dataset": dataset, "rpstda_commit": commit, **analysis(results)}
    (OUTS[dataset] / "exploratory_rpstda.json").write_text(json.dumps(out, indent=1))
    for m, s in out["summary"].items():
        print(f"{m:12s} acc {100 * s['acc_mean']:6.2f}  best-of-30 {100 * out['best_of'][m]:6.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("run", "analyse"))
    ap.add_argument("--dataset", choices=tuple(STREAMS), required=True)
    ap.add_argument("--shard", default="0/1", help="i/n: this process runs RUN_SEEDS[i::n]")
    args = ap.parse_args()
    if args.action == "run":
        run(args.dataset, *map(int, args.shard.split("/")))
    else:
        analyse(args.dataset)


if __name__ == "__main__":
    main()
