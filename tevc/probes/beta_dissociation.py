"""Is the accuracy cost of shifting resonances independent of how the time scale is changed? (development data)

Dissociation design for the reviewer point that RPST and resampling differ in more than resonance placement: the
H2 recipe (3 targets, q 0.3..0.8, frame 128, keep_real, 80 copies) at resonance-shift exponent β = 0, 0.5, 1 with two
very different time-scale methods, the frequency-domain phase vocoder and time-domain WSOLA (β = 1 is pure
resampling for both, so it is trained once). Seeds 42, 1337, 2025, 999, 7; PU 900<->1500, JNU 600<->1000.
Prediction, fixed before running (2026-10-09): for BOTH methods, mean accuracy at β = 0 exceeds β = 1 in >= 3 of 4
directions and β = 0 >= β = 0.5 in >= 3 of 4 directions. Run from repo root:
    python -m tevc.probes.beta_dissociation   -> outputs/tevc/probe_beta_dissociation.json
"""
import json
from pathlib import Path

import numpy as np

from tevc.evaluator import RecipeEvaluator
from tevc.recipe import Recipe
from tevc.stream import jnu_stream, pu_stream

OUT = Path("outputs/tevc/probe_beta_dissociation.json")
DIRECTIONS = (("pu", 900, 1500), ("pu", 1500, 900), ("jnu", 600, 1000), ("jnu", 1000, 600))
STREAMS = {"pu": pu_stream, "jnu": jnu_stream}
SEEDS = (42, 1337, 2025, 999, 7)
RECIPES = {f"{tsm} beta={b:g}": Recipe(3, 0.3, 0.8, 128, b, 80, True, tsm=tsm)
           for tsm in ("pv", "wsola") for b in (0.0, 0.5)}
RECIPES["resampling beta=1"] = Recipe(3, 0.3, 0.8, 128, 1.0, 80, True)


def main():
    results = {}
    for name, src, tgt in DIRECTIONS:
        stream = STREAMS[name](library_rpm=src, sequence=(tgt,))
        env, ev = stream.environments[0], RecipeEvaluator(stream)
        task = {}
        for label, r in RECIPES.items():
            acc = [ev.true_accuracy(env, r, s) for s in SEEDS]
            task[label] = {"acc": acc, "acc_mean": float(np.mean(acc)), "acc_std": float(np.std(acc))}
            print(f"{name} {src}->{tgt} {label:18s} acc {np.mean(acc) * 100:5.1f} +- {np.std(acc) * 100:4.1f}", flush=True)
        results[f"{name}:{src}->{tgt}"] = task
        OUT.write_text(json.dumps(results, indent=1))
    verdict = {}
    for tsm in ("pv", "wsola"):
        d01 = [t[f"{tsm} beta=0"]["acc_mean"] - t["resampling beta=1"]["acc_mean"] for t in results.values()]
        d05 = [t[f"{tsm} beta=0"]["acc_mean"] - t[f"{tsm} beta=0.5"]["acc_mean"] for t in results.values()]
        verdict[tsm] = {"beta0_minus_beta1_pp": [round(d * 100, 1) for d in d01],
                        "beta0_minus_beta05_pp": [round(d * 100, 1) for d in d05],
                        "holds": bool(sum(d > 0 for d in d01) >= 3 and sum(d >= 0 for d in d05) >= 3)}
    results["prediction"] = {"criterion": "both methods: beta0 > beta1 in >=3/4 and beta0 >= beta0.5 in >=3/4",
                             **verdict, "holds_for_both": all(v["holds"] for v in verdict.values())}
    OUT.write_text(json.dumps(results, indent=1))
    print("prediction:", json.dumps(results["prediction"]))


if __name__ == "__main__":
    main()
