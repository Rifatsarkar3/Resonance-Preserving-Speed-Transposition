"""D1 experiment driver (spec §5): run each method over an environment stream and score its deployed models.

    python -m tevc.run_stream --dataset jnu             # development stream, all methods, 5 independent runs
    python -m tevc.run_stream --dataset pu --runs 1     # first run only
    python -m tevc.run_stream --dataset pu --methods DSR,COT,DANN,DEEP-CORAL,CDAN,SHOT --tag rbase   # RESS baselines
    python -m tevc.run_stream --dataset pu --methods RPST-DA --tag rpstda   # MSSP RPST-DA development grid
Writes outputs/tevc/stream_<dataset>.json after every (run, method), so partial results are readable early.
Methods see only the agent-visible fitness; true accuracy (evaluator.true_accuracy) is used here, for scoring only.
Ottawa is the confirmatory set: it runs only through the frozen protocol in tevc.confirmatory.
"""
import argparse
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tevc import baselines, rpstda
from tevc.agent import Agent
from tevc.evaluator import Evaluation, RecipeEvaluator
from tevc.optimizers import B, N_GENES, N_INIT, Trial, bo_search, encode, ga_search, random_search
from tevc.recipe import ANCHORS
from tevc.stream import assert_disjoint, jnu_stream, ottawa_stream, pu_stream

STREAMS = {"jnu": jnu_stream, "pu": pu_stream, "ottawa": ottawa_stream}
METHODS = ("STATIC-NONE", "STATIC-H2", "RANDOM", "GA-RESTART", "AGENT", "AGENT-B1")
EXTRA = ("BO", "WARM-TRANSFER", "ORACLE-FITNESS", "AGENT-NOMEM", "AGENT-NOSURR", "AGENT-NOEVO", "AGENT-NOPRIOR",
         "AGENT-DFID") + baselines.METHODS + ("RPST-DA",)
AGENT_VARIANTS = {"AGENT": {}, "AGENT-B1": {"beta": 1.0}, "AGENT-NOMEM": {"memory": False},
                  "AGENT-NOSURR": {"surrogate": False}, "AGENT-NOEVO": {"evolve": False},
                  "AGENT-NOPRIOR": {"prior": False}, "AGENT-DFID": {}}
RUN_SEEDS = (42, 1337, 2025, 999, 7, 3, 11, 101, 2718, 31415)   # 10 runs: exact run-level p can reach 1/1024
REEVAL_SEED_OFFSET = 10_000
OUT_DIR = Path("outputs/tevc")


class OracleEvaluator:
    """ORACLE-FITNESS: fitness = true test-set accuracy (optimistic bound: selection and scoring share T)."""

    def __init__(self, evaluator):
        self.evaluator = evaluator

    def evaluate(self, env, recipe, seed):
        e = self.evaluator.evaluate(env, recipe, seed)
        return Evaluation(self.evaluator.true_accuracy(env, recipe, seed), seed, e.cost_s, e.cached)


class DFIDEvaluator:
    """AGENT-DFID ablation (spec §5, IM vs DFID fitness): fitness = DFID, the secondary label-free fitness."""

    def __init__(self, evaluator):
        self.evaluator = evaluator

    def evaluate(self, env, recipe, seed):
        value, cached = self.evaluator.dfid(env, recipe, seed)        # trains and caches the model when needed
        e = self.evaluator.evaluate(env, recipe, seed)                # hidden log; model-cache hit
        return Evaluation(value, seed, e.cost_s, cached)


def _rng(run_seed, env_index, purpose):
    return np.random.default_rng([run_seed, env_index, purpose])


def _top_genes(trials, n):
    out = []
    for t in sorted(trials, key=lambda t: -t.fitness):
        if t.recipe not in [r for r, _ in out]:
            out.append((t.recipe, encode(t.recipe)))
    return np.array([g for _, g in out[:n]])


def run_method(method, stream, evaluator, run_seed):
    """[(carried, trials)] per environment for one method and one independent run."""
    seeds = (run_seed, run_seed + REEVAL_SEED_OFFSET)
    agent = (Agent(stream.library_rpm, seeds, run_seed=run_seed, **AGENT_VARIANTS[method])
             if method in AGENT_VARIANTS else None)
    past, previous = [], None             # WARM-TRANSFER: every earlier evaluation, previous environment's trials
    out = []
    for i, env in enumerate(stream.environments):
        init_rng, op_rng = _rng(run_seed, i, 0), _rng(run_seed, i, 1)   # shared starts across methods
        start = _rng(run_seed, i, 0).random((N_INIT, N_GENES))         # = the agent's first-change start
        d = float(np.log(env.rpm / stream.library_rpm))
        if method in ("STATIC-NONE", "STATIC-H2"):
            recipe = ANCHORS["NONE" if method == "STATIC-NONE" else "H2_LIKE"]
            out.append(([], [Trial(recipe, run_seed, evaluator.evaluate(env, recipe, run_seed).fitness)]))
        elif method == "RANDOM":
            out.append(([], random_search(evaluator, env, init_rng, run_seed)))
        elif method == "GA-RESTART":
            out.append(([], ga_search(evaluator, env, start, op_rng, seeds)))
        elif method == "ORACLE-FITNESS":
            out.append(([], ga_search(OracleEvaluator(evaluator), env, start, op_rng, seeds)))
        elif method == "BO":
            out.append(([], bo_search(evaluator, env, start, op_rng, run_seed, d=d)))
        elif method == "WARM-TRANSFER":
            init = _top_genes(previous, N_INIT) if previous else start
            trials = bo_search(evaluator, env, init, op_rng, run_seed, d=d, past=past)
            past += [(d, encode(t.recipe), t.fitness) for t in trials]
            previous = trials
            out.append(([], trials))
        elif method in baselines.METHODS:
            out.append(([], baselines.search(method, evaluator, env, run_seed)))
        elif method == "RPST-DA":
            out.append(([], rpstda.search(evaluator, env, run_seed)))
        else:
            fitness = DFIDEvaluator(evaluator) if method == "AGENT-DFID" else evaluator
            out.append(agent.handle(fitness, env, init_rng, op_rng))
    return out


def _highest_fitness(pool):
    return max(pool, key=lambda t: t.fitness)


def score(evaluator, stream, env, carried, trials, budget=B, deploy=_highest_fitness):
    """Recovery curve a_k, k = 1..budget: true accuracy of the model `deploy` picks from the pool so far (default:
    the highest fitness)."""
    pool, curve = list(carried), []
    for k in range(budget):
        if k < len(trials):
            pool.append(trials[k])
        best = deploy(pool)
        curve.append(evaluator.true_accuracy(env, best.recipe, best.seed))
    return {"env": env.name, "rpm": env.rpm, "d": float(np.log(env.rpm / stream.library_rpm)),
            "n_evals": len(trials), "n_carried": len(carried),
            "deployed": {**asdict(best.recipe), "seed": best.seed, "fitness": best.fitness},
            "deployed_acc": curve[-1], "auc": float(np.mean(curve)), "curve": curve,
            "trials": [{**asdict(t.recipe), "seed": t.seed, "fitness": t.fitness,
                        "acc": evaluator.true_accuracy(env, t.recipe, t.seed)} for t in trials]}


def score_rpstda(evaluator, stream, env, trials, budget=B):
    """An RPST-DA visit, deployed by rank-mixing TV accuracy and IM (rpstda.rank_deploy). Stored alongside,
    descriptively: every candidate's TV accuracy and the IM-only and TV-only deployments on the same candidates."""
    tv = {(t.recipe, t.seed): evaluator.tv_accuracy(env, t.recipe, t.seed) for t in trials}
    s = score(evaluator, stream, env, [], trials, budget, deploy=rpstda.rank_deploy(lambda t: tv[(t.recipe, t.seed)]))
    for row, t in zip(s["trials"], trials):
        row["tv_acc"] = tv[(t.recipe, t.seed)]
    for name, key in (("im", lambda t: t.fitness), ("tv", lambda t: tv[(t.recipe, t.seed)])):
        best = max(trials, key=key)                                   # ties -> earliest candidate
        s[f"deployed_acc_{name}"] = evaluator.true_accuracy(env, best.recipe, best.seed)
    return s


def score_visit(method, evaluator, stream, env, carried, trials):
    """Scores one visit the way its method deploys."""
    if method == "RPST-DA":
        return score_rpstda(evaluator, stream, env, trials)
    return score(evaluator, stream, env, carried, trials)


def summarize(results):
    out = {}
    for m, runs in results.items():
        acc = [np.mean([e["deployed_acc"] for e in envs]) for envs in runs.values()]
        auc = [np.mean([e["auc"] for e in envs]) for envs in runs.values()]
        out[m] = {"runs": len(acc), "acc_mean": float(np.mean(acc)), "acc_std": float(np.std(acc)),
                  "auc_mean": float(np.mean(auc)), "auc_std": float(np.std(auc)),
                  "evals_per_stream": float(np.mean([sum(e["n_evals"] for e in envs) for envs in runs.values()]))}
    return out


def run_level_tests(results, key, ref="AGENT", n_boot=10_000):
    """Dependence-aware comparison of ref against every other method. Environments within one stream run share
    the agent's memory and the same data, so the unit is the independent run: per run, the stream-mean difference
    ref - other; exact one-sided sign-flip permutation test over runs (smallest p = 2^-runs); Holm across the
    other methods; percentile bootstrap 95% CI of the mean difference over runs."""
    if ref not in results:
        return {}
    tests, rng = {}, np.random.default_rng(0)
    for m in results:
        common = sorted(set(results[ref]) & set(results[m]))
        if m == ref or not common:
            continue
        diff = np.array([np.mean([e[key] for e in results[ref][r]]) - np.mean([e[key] for e in results[m][r]])
                         for r in common])
        null = (np.array(list(itertools.product((1.0, -1.0), repeat=len(diff)))) * np.abs(diff)).mean(axis=1)
        boot = diff[rng.integers(len(diff), size=(n_boot, len(diff)))].mean(axis=1)
        tests[m] = {"runs": len(diff), "mean_diff": float(diff.mean()),
                    "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                    "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()), "losses": int((diff < 0).sum()),
                    "p": float(np.mean(null >= diff.mean() - 1e-12)), "p_holm": None}
    ranked = sorted((v["p"], m) for m, v in tests.items())
    running = 0.0
    for i, (p, m) in enumerate(ranked):
        running = max(running, min(1.0, (len(ranked) - i) * p))
        tests[m]["p_holm"] = running
    return tests


def report(dataset, methods, results, elapsed):
    done = {m: r for m, r in results.items() if r}
    summary = summarize(done)
    gate = None
    if "AGENT" in summary and "GA-RESTART" in summary:
        a, g = summary["AGENT"], summary["GA-RESTART"]
        gate = {"agent_acc_gt_ga_restart": a["acc_mean"] > g["acc_mean"],
                "agent_auc_gt_ga_restart": a["auc_mean"] > g["auc_mean"]}
    run_seeds = [s for s in RUN_SEEDS if any(str(s) in runs for runs in done.values())]
    return {"dataset": dataset, "budget_per_change": B, "run_seeds": run_seeds, "methods": list(methods),
            "fitness": "cluster-balanced IM on the unlabelled buffer", "elapsed_s": round(elapsed),
            "summary": summary, "dev_gate": gate,
            "tests_deployed_acc": run_level_tests(done, "deployed_acc"), "tests_auc": run_level_tests(done, "auc"),
            "results": done}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(STREAMS), required=True)
    ap.add_argument("--runs", type=int, default=len(RUN_SEEDS))
    ap.add_argument("--methods", default=",".join(METHODS), help=f"comma list from {METHODS + EXTRA}")
    ap.add_argument("--tag", default="", help="output stream_<dataset>_<tag>.json (keeps the main grid's file)")
    args = ap.parse_args()
    if args.dataset == "ottawa":
        raise SystemExit("Ottawa is the confirmatory set: it runs only through the frozen protocol, tevc.confirmatory")
    methods = args.methods.split(",")
    unknown = set(methods) - set(METHODS + EXTRA)
    if unknown:
        raise SystemExit(f"unknown methods {sorted(unknown)}")
    stream = STREAMS[args.dataset]()
    assert_disjoint(stream)
    evaluator = RecipeEvaluator(stream)
    out_path = OUT_DIR / f"stream_{args.dataset}{'_' + args.tag if args.tag else ''}.json"
    results, t0 = {m: {} for m in methods}, time.time()
    for run_seed in RUN_SEEDS[:args.runs]:
        for m in methods:
            t = time.time()
            envs = [score_visit(m, evaluator, stream, env, carried, trials) for env, (carried, trials)
                    in zip(stream.environments, run_method(m, stream, evaluator, run_seed))]
            results[m][str(run_seed)] = envs
            print(f"run {run_seed:4d} {m:11s} deployed acc {[round(e['deployed_acc'] * 100, 1) for e in envs]} "
                  f"mean {np.mean([e['deployed_acc'] for e in envs]) * 100:5.1f}  auc "
                  f"{np.mean([e['auc'] for e in envs]) * 100:5.1f}  evals {sum(e['n_evals'] for e in envs):3d}  "
                  f"{time.time() - t:5.0f}s", flush=True)
            out_path.write_text(json.dumps(report(args.dataset, methods, results, time.time() - t0), indent=1))
    print("summary:", json.dumps(report(args.dataset, methods, results, time.time() - t0)["summary"], indent=1))


if __name__ == "__main__":
    main()
