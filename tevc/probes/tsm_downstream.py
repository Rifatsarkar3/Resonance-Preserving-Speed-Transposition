"""Does transient-preserving RPST (WSOLA) train better diagnosers than phase-vocoder RPST? (development data only)

Motivation: the operator-fidelity probe shows the phase vocoder smears impact ring-down and impulsiveness, which
WSOLA keeps (probe_operator_fidelity*.json). Here the same recipes are trained with either time-scale method and
scored on the real target-speed test set T (and by the agent-visible cluster-balanced IM on U).
Recipes (β = 0, keep_real, 80 copies): H2 = 3 targets q 0.3..0.8; AT = 1 target at the target speed; frames 128 / 256.
Seeds 42, 1337, 2025, 999, 7. Directions: PU 900->1500, 1500->900; JNU 600->1000, 1000->600.
Decision rule, fixed before running (2026-10-09): WSOLA enters the agent's decision space if, for the H2 family,
its mean accuracy is >= the phase vocoder's at the same frame in >= 3 of the 4 directions and the 4-direction mean
difference is > -2 pp. Run from repo root:  python -m tevc.probes.tsm_downstream
  -> outputs/tevc/probe_tsm_downstream.json
"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from tevc.evaluator import RecipeEvaluator
from tevc.recipe import Recipe
from tevc.stream import jnu_stream, pu_stream

OUT = Path("outputs/tevc/probe_tsm_downstream.json")
DIRECTIONS = (("pu", 900, 1500), ("pu", 1500, 900), ("jnu", 600, 1000), ("jnu", 1000, 600))
STREAMS = {"pu": pu_stream, "jnu": jnu_stream}
SEEDS = (42, 1337, 2025, 999, 7)
FAMILIES = {"H2": Recipe(3, 0.3, 0.8, 256, 0.0, 80, True), "AT": Recipe(1, 1.0, 1.0, 256, 0.0, 80, True)}
FRAMES = (128, 256)


def main():
    results = {}
    for name, src, tgt in DIRECTIONS:
        stream = STREAMS[name](library_rpm=src, sequence=(tgt,))
        env, ev = stream.environments[0], RecipeEvaluator(stream)
        task = {}
        for fam, base in FAMILIES.items():
            for frame in FRAMES:
                for tsm in ("pv", "wsola"):
                    r = replace(base, n_fft=frame, tsm=tsm)
                    acc = [ev.true_accuracy(env, r, s) for s in SEEDS]
                    fit = [ev.evaluate(env, r, s).fitness for s in SEEDS]
                    task[f"{fam} frame={frame} {tsm}"] = {"acc": acc, "acc_mean": float(np.mean(acc)),
                                                          "acc_std": float(np.std(acc)), "fitness_mean": float(np.mean(fit))}
                    print(f"{name} {src}->{tgt} {fam} frame={frame} {tsm:5s} acc {np.mean(acc) * 100:5.1f} "
                          f"+- {np.std(acc) * 100:4.1f}  IM {np.mean(fit):.3f}", flush=True)
        results[f"{name}:{src}->{tgt}"] = task
        OUT.write_text(json.dumps(results, indent=1))
    diffs = {frame: [results[t][f"H2 frame={frame} wsola"]["acc_mean"] - results[t][f"H2 frame={frame} pv"]["acc_mean"]
                     for t in results] for frame in FRAMES}
    rule = {str(f): {"directions_wsola_ge_pv": int(sum(d >= 0 for d in v)), "mean_diff_pp": float(np.mean(v) * 100),
                     "passes": bool(sum(d >= 0 for d in v) >= 3 and np.mean(v) > -0.02)} for f, v in diffs.items()}
    results["decision_rule"] = {"criterion": "H2: WSOLA >= PV in >= 3/4 directions and mean diff > -2 pp", **rule}
    OUT.write_text(json.dumps(results, indent=1))
    print("decision:", json.dumps(results["decision_rule"]))


if __name__ == "__main__":
    main()
