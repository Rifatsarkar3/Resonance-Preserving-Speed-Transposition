"""Offline: deploy the average of the top-k models by agent-visible fitness instead of the single best (dev data).

Motivation: on PU at 1500 rpm the single highest-IM model is often far from the best one (stream runs, Gate-1), and
model averaging lowers that variance. Every trained model's test-set probabilities are cached, so the deployed
ensemble's accuracy is computed without retraining: for each searching method, run and environment of
stream_<dataset>.json, the models available after k evaluations (carried + trials so far) are ranked by fitness,
the top-k probability vectors on T are averaged, and accuracy and recovery AUC are recomputed. k = 1 reproduces the
stored results (checked). Selection uses fitness only; T is used only for scoring.
Adoption rule, fixed before running (2026-10-09): top-k deployment (k in {3, 5}) replaces single-model deployment for
every method only if it raises AGENT's run-mean stream accuracy on BOTH JNU and PU and does not lower AGENT's
run-mean recovery AUC on either; if both k qualify, the smaller k. Ottawa is never touched.
Run from repo root:  python -m tevc.probes.deploy_ensemble   -> outputs/tevc/probe_deploy_ensemble.json
"""
import json
from pathlib import Path

import numpy as np

from tevc.evaluator import RecipeEvaluator
from tevc.recipe import Recipe
from tevc.stream import jnu_stream, pu_stream

OUT = Path("outputs/tevc/probe_deploy_ensemble.json")
STREAMS = {"jnu": jnu_stream, "pu": pu_stream}
SEARCHERS = ("RANDOM", "GA-RESTART", "AGENT", "AGENT-B1")
KS = (1, 3, 5)
B = 30


def _model(entry):
    fields = {k: entry[k] for k in ("n_targets", "q_lo", "q_hi", "n_fft", "beta", "copies", "keep_real")}
    return Recipe(**fields, tsm=entry.get("tsm", "pv")), entry["seed"], entry["fitness"]


def _env_curves(ev, env, carried, trials, k):
    pool, curve = list(carried), []
    for i in range(B):
        if i < len(trials):
            pool.append(trials[i])
        top = sorted(pool, key=lambda m: -m[2])[:k]
        pT = np.mean([ev.probabilities(env, r, s)[1] for r, s, _ in top], axis=0)
        curve.append(float((pT.argmax(axis=1) == env.T["y"]).mean()))
    return curve


def main():
    out = {}
    for name, make in STREAMS.items():
        stored = json.loads(Path(f"outputs/tevc/stream_{name}.json").read_text())["results"]
        stream = make()
        ev = RecipeEvaluator(stream)
        out[name] = {}
        for method in SEARCHERS:
            if method not in stored:
                continue
            per_k = {k: {"acc": [], "auc": []} for k in KS}
            for run, envs in stored[method].items():
                deployed = {}                               # env name -> deployed model of its last visit
                accs, aucs = {k: [] for k in KS}, {k: [] for k in KS}
                for env, e in zip(stream.environments, envs):
                    trials = [_model(t) for t in e["trials"]]
                    carried = [_model(deployed[env.name])] if e["n_carried"] and env.name in deployed else []
                    for k in KS:
                        c = _env_curves(ev, env, carried, trials, k)
                        accs[k].append(c[-1])
                        aucs[k].append(float(np.mean(c)))
                    if abs(accs[1][-1] - e["deployed_acc"]) > 1e-9:
                        raise AssertionError(f"k=1 does not reproduce {method} run {run} {env.name}")
                    deployed[env.name] = e["deployed"]
                for k in KS:
                    per_k[k]["acc"].append(float(np.mean(accs[k])))
                    per_k[k]["auc"].append(float(np.mean(aucs[k])))
            out[name][method] = {str(k): {"runs": len(v["acc"]), "acc_mean": float(np.mean(v["acc"])),
                                          "auc_mean": float(np.mean(v["auc"])), "acc_runs": v["acc"]}
                                 for k, v in per_k.items()}
            print(name, method, {k: (round(v["acc_mean"] * 100, 1), round(v["auc_mean"] * 100, 1))
                                 for k, v in out[name][method].items()}, flush=True)
    rule = {}
    for k in ("3", "5"):
        rule[k] = all(out[d]["AGENT"][k]["acc_mean"] > out[d]["AGENT"]["1"]["acc_mean"] and
                      out[d]["AGENT"][k]["auc_mean"] >= out[d]["AGENT"]["1"]["auc_mean"] for d in out)
    adopted = next((k for k in ("3", "5") if rule[k]), None)
    out["adoption"] = {"criterion": "AGENT acc up on JNU and PU, AUC not down; smaller k wins", "qualifies": rule,
                       "adopted_k": adopted}
    OUT.write_text(json.dumps(out, indent=1))
    print("adoption:", out["adoption"])


if __name__ == "__main__":
    main()
