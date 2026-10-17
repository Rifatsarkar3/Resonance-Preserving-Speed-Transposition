"""EXPLORATORY DIRG diagnostics, run after the one-time verdict (df3eccf); nothing here feeds back into any method or
confirmatory claim. They explain two results: D2 was not confirmed (AGENT = AGENT-B1), and every label-free method
stays at 26-41 % accuracy.

  fidelity  Is the RPST premise true on DIRG, i.e. do the resonances stay fixed when the speed changes? The 300 Hz
            library records are transposed to each environment's estimated speed with β = 0 (RPST) and β = 1
            (resampling) and compared class by class with the real U windows (tevc.probes.operator_fidelity metrics;
            band_error = resonance placement). "untransposed" = the library windows as recorded.
  headroom  Selection vs generation, without training: per visit of the frozen grid, the true accuracy of the
            deployed model against the best model among the visit's evaluated candidates.
  oracle    ORACLE-FITNESS (GA guided by true test accuracy; optimistic, since selection and scoring share T) on the
            same 10 runs, paired with AGENT.

    python -m tevc.probes.dirg_diagnostics fidelity      -> outputs/tevc/dirg/exploratory_fidelity.json
    python -m tevc.probes.dirg_diagnostics headroom      -> outputs/tevc/dirg/exploratory_headroom.json
    python -m tevc.probes.dirg_diagnostics oracle        -> outputs/tevc/dirg/exploratory_oracle.json
"""
import argparse
import json
import time

import numpy as np

from tevc import synthesis
from tevc.datasets import make_windows
from tevc.dirg_protocol import OUT, load_shards
from tevc.evaluator import RecipeEvaluator
from tevc.probes.operator_fidelity import CAP, _compare
from tevc.run_stream import RUN_SEEDS, run_level_tests, run_method, score
from tevc.stream import dirg_stream

OPERATORS = {"untransposed": None, "beta=0 n_fft=128 (RPST)": (0.0, 128), "beta=0 n_fft=256 (RPST)": (0.0, 256),
             "beta=1 (resampling)": (1.0, 256)}


def fidelity():
    stream = dirg_stream()
    ds, src = stream.dataset, stream.library_rpm
    out = {"note": "exploratory, after the DIRG verdict", "library_rpm": src, "environments": {}}
    for env in dict.fromkeys(stream.environments):
        rows = {}
        for label, op in OPERATORS.items():
            if op is None:
                synth = make_windows(stream.library_entries, ds.win, ds.hop, CAP)
            else:
                beta, n_fft = op
                synth = make_windows(stream.library_entries, ds.win, ds.hop, CAP,
                                     synth=lambda lab, rec, sig, b=beta, n=n_fft:
                                     (synthesis.transpose(sig, src, env.rpm, b, n), env.rpm / src))
            rows[label] = _compare(synth, env.U, ds, env.rpm / 60.0)
            print(f"{env.name} {label:26s} {json.dumps({k: round(v, 4) for k, v in rows[label].items()})}",
                  flush=True)
        out["environments"][env.name] = {"speed_ratio": env.rpm / src, "operators": rows}
        (OUT / "exploratory_fidelity.json").write_text(json.dumps(out, indent=1))


def visit_headroom(entry):
    """(deployed accuracy, best accuracy among the visit's evaluated candidates and the deployed model)."""
    return entry["deployed_acc"], max([entry["deployed_acc"]] + [t["acc"] for t in entry["trials"]])


def headroom():
    results, commit = load_shards(OUT)
    out = {"note": "exploratory, after the DIRG verdict", "protocol_commit": commit, "methods": {}}
    for m, runs in results.items():
        pairs = np.array([visit_headroom(e) for envs in runs.values() for e in envs])
        per_env = {}
        for envs in runs.values():
            for i, e in enumerate(envs):
                per_env.setdefault(f"{i}:{e['env']}", []).append(visit_headroom(e))
        out["methods"][m] = {"deployed": float(pairs[:, 0].mean()), "best_available": float(pairs[:, 1].mean()),
                             "gap": float((pairs[:, 1] - pairs[:, 0]).mean()),
                             "per_environment": {k: {"deployed": float(np.mean([a for a, _ in v])),
                                                     "best_available": float(np.mean([b for _, b in v]))}
                                                 for k, v in per_env.items()}}
        print(f"{m:14s} deployed {pairs[:, 0].mean() * 100:6.2f}  best available {pairs[:, 1].mean() * 100:6.2f}",
              flush=True)
    (OUT / "exploratory_headroom.json").write_text(json.dumps(out, indent=1))


def oracle():
    path = OUT / "exploratory_oracle.json"
    stream = dirg_stream()
    evaluator = RecipeEvaluator(stream)
    done = json.loads(path.read_text())["results"]["ORACLE-FITNESS"] if path.exists() else {}
    agent, _ = load_shards(OUT)
    for run_seed in RUN_SEEDS:
        if str(run_seed) in done:
            continue
        t = time.time()
        done[str(run_seed)] = [score(evaluator, stream, env, carried, trials) for env, (carried, trials)
                               in zip(stream.environments, run_method("ORACLE-FITNESS", stream, evaluator, run_seed))]
        res = {"ORACLE-FITNESS": done, "AGENT": {r: agent["AGENT"][r] for r in done}}
        path.write_text(json.dumps({"note": "exploratory, after the DIRG verdict; optimistic bound",
                                    "results": {"ORACLE-FITNESS": done},
                                    "oracle_minus_agent": run_level_tests(res, "deployed_acc", ref="ORACLE-FITNESS")}))
        print(f"run {run_seed:5d} ORACLE-FITNESS  {time.time() - t:6.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("fidelity", "headroom", "oracle"))
    {"fidelity": fidelity, "headroom": headroom, "oracle": oracle}[ap.parse_args().step]()


if __name__ == "__main__":
    main()
