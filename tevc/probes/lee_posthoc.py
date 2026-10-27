"""POST-HOC fairness analysis of committed results (attack audit MSSP/review/2026-10-20_attack_audit.md U1, U2, U4,
U5, U8, G4; reframe spec amendment A5). Reads the committed result files only, trains nothing, and is labelled post
hoc wherever it is reported: none of these comparisons was pre-registered.

  lee        RPST-DA against EVERY method, two-sided, Holm over the nine (the protocol's families never tested DSR,
             the strongest Lee comparator); RPST-DA and RPST-DA-NC deployed by IM alone, the comparators' rule,
             against CDAN and DSR; the decomposition of E1 (RPST-DA - CDAN = copies (RPST-DA - NC) + the rest
             (NC - CDAN)); the generation / selection split of the copies' effect (best-of-30 and selection loss)
  steps      with --steps (reads the Lee records): training steps of every RPST-DA draw over its no-copies twin
  dirg       RPST-DA against its exploratory comparators, two-sided (the archived tables are one-sided)
  ottawa     the same on the Ottawa v2 stream (the withdrawn v1 shards are never read)

    python -m tevc.probes.lee_posthoc [--steps]   -> outputs/tevc/{lee,dirg,ottawa}/posthoc_fairness.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tevc.dirg_protocol import family
from tevc.run_stream import RUN_SEEDS

LEE_FILES = ("outputs/tevc/lee/core_shard0of2.json", "outputs/tevc/lee/core_shard1of2.json")
OUT = {"lee": Path("outputs/tevc/lee/posthoc_fairness.json"), "dirg": Path("outputs/tevc/dirg/posthoc_fairness.json"),
       "ottawa": Path("outputs/tevc/ottawa/posthoc_fairness.json")}
IM_SUFFIX = " (IM only)"


def load(paths):
    out = {}
    for p in paths:
        for m, runs in json.loads(Path(p).read_text(encoding="utf-8"))["results"].items():
            out.setdefault(m, {}).update(runs)
    return out


def im_only(results, m):
    """results[m] with every visit's deployed accuracy replaced by its IM-only deployment (deployed_acc_im)."""
    return {r: [{**v, "deployed_acc": v["deployed_acc_im"]} for v in visits] for r, visits in results[m].items()}


def best30(visit):
    return max([visit["deployed_acc"]] + [t["acc"] for t in visit.get("trials", [])])


def run_mean(results, m, value):
    return float(np.mean([np.mean([value(v) for v in visits]) for visits in results[m].values()]))


def selection_split(results, copies="RPST-DA", none="RPST-DA-NC"):
    """Deployed, best-of-30 and selection loss of the two arms, and the copies' effect split into generation
    (best-of-30 gain) and selection (loss saved): generation + selection = deployed difference."""
    rows = {m: {"deployed": run_mean(results, m, lambda v: v["deployed_acc"]), "best_of_30": run_mean(results, m, best30)}
            for m in (copies, none)}
    for r in rows.values():
        r["selection_loss"] = r["best_of_30"] - r["deployed"]
    gen = rows[copies]["best_of_30"] - rows[none]["best_of_30"]
    sel = rows[none]["selection_loss"] - rows[copies]["selection_loss"]
    return {**rows, "generation_gain": gen, "selection_gain": sel,
            "deployed_gain": rows[copies]["deployed"] - rows[none]["deployed"]}


def posthoc(results, ref="RPST-DA"):
    """Every post-hoc table of the Lee section, all two-sided."""
    others = [m for m in results if m != ref]
    strongest = max(others, key=lambda m: run_mean(results, m, lambda v: v["deployed_acc"]))
    redeployed = {**results, ref + IM_SUFFIX: im_only(results, ref), "RPST-DA-NC" + IM_SUFFIX: im_only(results, "RPST-DA-NC")}
    im_tests = [(a + IM_SUFFIX, b, "deployed_acc") for a in (ref, "RPST-DA-NC") for b in dict.fromkeys(("CDAN", strongest))]
    return {"note": "post hoc (attack audit 2026-10-20); two-sided exact sign-flip over the 10 runs, Holm within each "
                    "family; not pre-registered",
            "strongest_comparator": strongest,
            "versus_every_method": family(results, [(ref, m, "deployed_acc") for m in others], alternative="two-sided"),
            "im_only": {"deployed": {m: run_mean(redeployed, m + IM_SUFFIX, lambda v: v["deployed_acc"])
                                     for m in (ref, "RPST-DA-NC")},
                        "tests": family(redeployed, im_tests, alternative="two-sided")},
            "e1_decomposition": family(results, [(ref, "CDAN", "deployed_acc"), (ref, "RPST-DA-NC", "deployed_acc"),
                                                 ("RPST-DA-NC", "CDAN", "deployed_acc")], alternative="two-sided"),
            "selection_split": selection_split(results)}


def steps_ratios():
    """Training steps of every RPST-DA draw over its no-copies twin on Lee (reads the Lee library records)."""
    from tevc import rpstda
    from tevc.stream import lee_stream
    stream = lee_stream()
    n_train = len(rpstda.train_windows(stream)["y"])
    k = stream.dataset.n_classes
    ratios = [(n_train + c.n_targets * c.copies * k) / n_train for s in RUN_SEEDS for c in rpstda.sample_configs(s)]
    first = rpstda.PUBLISHED
    return {"n_train": n_train, "median": float(np.median(ratios)), "min": float(min(ratios)),
            "max": float(max(ratios)), "candidate_0": (n_train + first.n_targets * first.copies * k) / n_train}


def exploratory(dataset):
    """RPST-DA against the exploratory comparators of DIRG or Ottawa, two-sided (rpstda_heldout tested one-sided)."""
    from tevc.probes.rpstda_heldout import OUTS, VERSUS, comparators, load_own
    own, _ = load_own(OUTS[dataset])
    results = {**comparators(dataset), **own}
    return {"note": "post hoc two-sided re-test of the exploratory comparisons (attack audit U8)",
            "tests": family(results, [("RPST-DA", m, "deployed_acc") for m in VERSUS], alternative="two-sided")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", action="store_true", help="also count training steps (reads the Lee records)")
    args = ap.parse_args()
    lee = posthoc(load(LEE_FILES))
    if args.steps:
        lee["steps_ratios"] = steps_ratios()
    OUT["lee"].write_text(json.dumps(lee, indent=1))
    for dataset in ("dirg", "ottawa"):
        OUT[dataset].write_text(json.dumps(exploratory(dataset), indent=1))
    row = next(r for r in lee["versus_every_method"] if r["second"] == lee["strongest_comparator"])
    print(f"strongest comparator {lee['strongest_comparator']}: RPST-DA {100 * row['mean_diff']:+.2f} pp, "
          f"W/L {row['wins']}/{row['losses']}, p {row['p']:.4f}; IM-only RPST-DA "
          f"{100 * lee['im_only']['deployed']['RPST-DA']:.2f} %", flush=True)


if __name__ == "__main__":
    main()
