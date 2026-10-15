"""Physics-consistency selection, PCS (RESS spec §3.3, §4): label-free choice among trained diagnosers by the agreement
between the fault types a model claims on the unlabelled buffer U and the envelope evidence of each fault frequency
in U.

Evidence (model-independent, once per environment): the envelope block of tevc.features at the environment's shaft
speed. Per fault type k (OR: BPFO, IR: BPFI, RE: BSF) it is the peak-to-floor log ratio averaged over harmonics 1-3,
taken in the full band (A-full) or maximised over the full and the 8 equal bands (A-max), and standardised over the
buffer.
Claim: q_ik = window i's probability summed over the classes whose signature contains k (SIGNATURES).
PCS = mean over the dataset's fault types of the correlation between q_.k and e_.k, weighted by the frozen
cluster-balanced weights of U (Spearman = weighted Pearson of the ranks). A fault type the model never claims
scores 0.
Deployment rules: IM (argmax fitness, the frozen D1 rule); V1 argmax PCS; V2 argmax of the mean of the IM and PCS
ranks; V3 argmax PCS among the TOP_IM candidates with the highest IM.

    python -m tevc.pcs select   # variant choice on the selection archives -> outputs/tevc/pcs_variant.json
    python -m tevc.pcs gate     # ONE shot, frozen variant, AGENT v2 archives -> outputs/tevc/pcs_gate.json
Commit pcs_variant.json before running the gate. The gate refuses to run twice or without the variant file, and
select refuses to run once the gate has. Archived searches are replayed, never re-run: a different deployment's
downstream effect on the agent's memory is measured on DIRG, where AGENT-PCS runs as a full method.
"""
import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
from scipy.stats import rankdata

from tevc import features
from tevc.agent import RECUR
from tevc.baselines import config_from_dict
from tevc.datasets import DATASETS, make_windows
from tevc.evaluator import RecipeEvaluator
from tevc.reliability import merge
from tevc.run_gate1 import OUT as GATE1_FILE
from tevc.run_gate1 import TASKS as GATE1_TASKS
from tevc.run_stream import OUT_DIR, STREAMS
from tevc.stream import Environment, Stream, assert_disjoint

FAULT_TARGET = {"OR": "BPFO", "IR": "BPFI", "RE": "BSF"}
SIGNATURES = {"jnu": (set(), {"IR"}, {"OR"}, {"RE"}),          # normal, inner race, outer race, roller
              "pu": (set(), {"OR"}, {"IR"}, {"IR", "OR"})}      # healthy, KA, KI, KB (combined inner + outer damage)
RULES, AGGREGATIONS, CORRELATIONS = ("V1", "V3", "V2"), ("A-full", "A-max"), ("spearman", "pearson")
TOP_IM = 5
CONSTANT = 1e-6         # a claim whose range is below this is constant (float32 probabilities)
PU_MIN_GAIN, PU_MIN_WINS, JNU_MAX_LOSS = 0.005, 7, 0.0025
SPEC = "docs/superpowers/specs/2026-10-13-ress-rpst-pcs-design.md"
DATASET_ORDER = ("pu", "jnu")
# Selection archives (spec §4). PU's BO and WARM-TRANSFER come from stream_pu_baselines (10 runs), which supersedes
# the partial stream_pu_extra. ORACLE-FITNESS is excluded: its fitness is test accuracy.
SELECTION_FILES = {"pu": ("stream_pu_r10.json", "stream_pu_baselines.json", "stream_pu_rbase.json"),
                   "jnu": ("stream_jnu_r10.json", "stream_jnu_extra.json", "stream_jnu_rbase.json")}
SELECTION_METHODS = ("RANDOM", "GA-RESTART", "BO", "WARM-TRANSFER", "DANN", "DEEP-CORAL", "CDAN", "SHOT")
GATE_FILES = {"pu": "stream_pu_v2.json", "jnu": "stream_jnu_v2.json"}
VARIANT_OUT, GATE_OUT = OUT_DIR / "pcs_variant.json", OUT_DIR / "pcs_gate.json"
_ENVELOPE = features.feature_names().index("env_full_f_sx1")     # first column of the envelope block


@dataclass(frozen=True)
class Variant:
    rule: str
    aggregation: str
    correlation: str


VARIANTS = tuple(Variant(r, a, c) for r in RULES for a in AGGREGATIONS for c in CORRELATIONS)    # tie-break order


def envelope_block(W, fs, rpm, geometry):
    """(windows, bands, targets, harmonics) envelope peak-to-floor log ratios of the z-normalised windows at shaft
    speed rpm; band 0 = full band, targets = features.ORDER_TARGETS."""
    X = features.extract(W, fs, rpm / 60.0, geometry, znorm=True)[:, _ENVELOPE:]
    return X.reshape(len(X), 1 + features.N_ENV_BANDS, len(features.ORDER_TARGETS), len(features.HARMONICS))


def evidence(block, aggregation):
    """{fault type k: standardised evidence per window}: harmonic mean, then the full band (A-full) or the best
    band (A-max)."""
    per_band = block.mean(axis=3)
    e = per_band[:, 0] if aggregation == "A-full" else per_band.max(axis=1)
    out = {}
    for k, target in FAULT_TARGET.items():
        x = e[:, features.ORDER_TARGETS.index(target)].astype(np.float64)
        out[k] = (x - x.mean()) / (x.std() + 1e-12)
    return out


def weighted_corr(x, y, w):
    """Weighted Pearson correlation; 0 when either variable is constant."""
    w = np.asarray(w, dtype=np.float64) / np.sum(w)
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    dx, dy = x - w @ x, y - w @ y
    sxx, syy = w @ (dx * dx), w @ (dy * dy)
    if sxx <= 1e-24 or syy <= 1e-24:
        return 0.0
    return float(w @ (dx * dy) / np.sqrt(sxx * syy))


def fault_types(signatures):
    return [k for k in FAULT_TARGET if any(k in s for s in signatures)]


def claims(p, signatures):
    """{fault type k: per-window probability that the model assigns a class carrying k}."""
    return {k: p[:, [c for c, s in enumerate(signatures) if k in s]].sum(axis=1) for k in fault_types(signatures)}


def pcs(p, ev, signatures, w, correlation):
    """Physics-consistency score of one model's probabilities p on U against the evidence ev = {k: e_.k}."""
    terms = []
    for k, q in claims(p, signatures).items():
        if np.ptp(q) < CONSTANT:
            terms.append(0.0)
            continue
        e = ev[k]
        if correlation == "spearman":
            q, e = rankdata(q), rankdata(e)
        terms.append(weighted_corr(q, e, w))
    return float(np.mean(terms))


def choose(rule, im, score):
    """Index of the deployed candidate; ties go to the earliest candidate, as max() over carried + trials does."""
    im, score = np.asarray(im, dtype=float), np.asarray(score, dtype=float)
    if rule == "IM":
        return int(np.argmax(im))
    if rule == "V1":
        return int(np.argmax(score))
    if rule == "V2":
        return int(np.argmax(rankdata(im) + rankdata(score)))
    if rule == "V3":
        top = np.argsort(-im, kind="stable")[:TOP_IM]
        return int(top[np.argmax(score[top])])
    raise ValueError(f"unknown rule {rule}")


def best_variant(scores):
    """The variant with the highest selection criterion; equal criteria go to the earlier variant in VARIANTS."""
    return max(VARIANTS, key=lambda v: scores[v])


def gate_verdict(pu_gains, jnu_gains):
    """Spec §4: PU mean gain >= +0.5 pp and gain > 0 in >= 7 runs; JNU mean loss <= 0.25 pp."""
    pu, jnu = np.asarray(pu_gains, dtype=float), np.asarray(jnu_gains, dtype=float)
    checks = {"pu_mean_gain": bool(pu.mean() >= PU_MIN_GAIN - 1e-12),
              "pu_runs_gained": bool((pu > 0).sum() >= PU_MIN_WINS),
              "jnu_mean_loss": bool(-jnu.mean() <= JNU_MAX_LOSS + 1e-12)}
    return {**checks, "pass": all(checks.values())}


class Scorer:
    """True test accuracy and PCS values of cached models on one stream's environments; never trains."""

    def __init__(self, stream, evaluator=None):
        self.stream, self.signatures = stream, SIGNATURES[stream.dataset.name]
        self.evaluator = evaluator or RecipeEvaluator(stream)
        self._evidence, self._stats = {}, {}

    def evidence(self, env):
        if env.name not in self._evidence:
            ds = self.stream.dataset
            block = envelope_block(env.U["W"], ds.fs, env.rpm, ds.geometry)
            self._evidence[env.name] = {a: evidence(block, a) for a in AGGREGATIONS}
        return self._evidence[env.name]

    def stats(self, env, cand):
        """(true test accuracy, {(aggregation, correlation): PCS}) of the cached model of a result-file entry."""
        config, seed = config_from_dict(cand), int(cand["seed"])
        key = (env.name, config, seed)
        if key not in self._stats:
            pU, pT = self.evaluator.cached(env, config, seed)
            ev, w = self.evidence(env), self.evaluator.weights(env)
            self._stats[key] = (float((pT.argmax(axis=1) == env.T["y"]).mean()),
                                {(a, c): pcs(pU, ev[a], self.signatures, w, c)
                                 for a in AGGREGATIONS for c in CORRELATIONS})
        return self._stats[key]

    def pick(self, env, cands, rule, aggregation=None, correlation=None):
        """Index of the candidate the rule deploys; IM uses the archived agent-visible fitness."""
        im = [c["fitness"] for c in cands]
        score = im if rule == "IM" else [self.stats(env, c)[1][(aggregation, correlation)] for c in cands]
        return choose(rule, im, score)


def same_model(a, b):
    """Whether two result-file entries name the same trained model (configuration and training seed)."""
    return config_from_dict(a) == config_from_dict(b) and int(a["seed"]) == int(b["seed"])


def replay(entries, envs, pick):
    """Deployed entry per environment of one archived run when pick(env, candidates) -> index deploys. The archived
    search stays as it was; at a recurrence the candidate set starts with the rule's own model deployed at the
    regime's previous visit, as tevc.agent.Agent.handle carries it."""
    deployed, out = [], []
    for env, entry in zip(envs, entries, strict=True):
        if entry["env"] != env.name:
            raise ValueError(f"archived environment {entry['env']} != stream environment {env.name}")
        if not entry["trials"]:                 # unchanged speed: the last deployed model stays
            out.append(deployed[-1][1])
            continue
        carried = [c for d, c in deployed if abs(d - entry["d"]) < RECUR][-1:] if entry["n_carried"] else []
        if len(carried) != entry["n_carried"]:
            raise ValueError(f"{env.name}: {entry['n_carried']} carried models archived, {len(carried)} rebuilt")
        cands = carried + entry["trials"]
        deployed.append((entry["d"], cands[pick(env, cands)]))
        out.append(deployed[-1][1])
    return out


def gate1_stream(ds_name, src, tgt):
    """A Gate-1 task (tevc.run_gate1) as a one-environment stream: its library, A windows as U, B windows as T."""
    ds = DATASETS[ds_name]
    lib_entries = ds.entries("library", src)
    env = Environment(f"{ds_name}@{tgt}", float(tgt), make_windows(ds.entries("A", tgt), ds.win, ds.hop, ds.cap_eval),
                      make_windows(ds.entries("B", tgt), ds.win, ds.hop, ds.cap_eval))
    stream = Stream(ds, lib_entries, make_windows(lib_entries, ds.win, ds.hop, ds.cap_fit), float(src), (env,))
    assert_disjoint(stream)
    return stream


def selection_visits(ds_name):
    """[(scorer, environment, candidates)]: every candidate set evaluated in one dataset's selection archives, plus
    its Gate-1 tasks, whose models are trained here when not cached (Gate-1 never cached its predictions)."""
    stream = STREAMS[ds_name]()
    scorer = Scorer(stream)
    results, _ = merge([OUT_DIR / f for f in SELECTION_FILES[ds_name]])
    visits = []
    for m in SELECTION_METHODS:
        for envs in results[m].values():
            for env, entry in zip(stream.environments, envs, strict=True):
                if entry["env"] != env.name or entry["n_carried"]:
                    raise ValueError(f"{m}: unexpected archive entry at {env.name}")
                cands = entry["trials"]
                if not same_model(cands[scorer.pick(env, cands, "IM")], entry["deployed"]):
                    raise ValueError(f"{m} {env.name}: the IM rule does not reproduce the archived deployment")
                visits.append((scorer, env, cands))
    rows = json.loads(GATE1_FILE.read_text())["rows"]
    for ds, src, tgt in GATE1_TASKS:
        if ds != ds_name:
            continue
        g = Scorer(gate1_stream(ds, src, tgt))
        (env,) = g.stream.environments
        cands = [{**r["recipe"], "seed": r["seed"],
                  "fitness": g.evaluator.evaluate(env, config_from_dict(r["recipe"]), r["seed"]).fitness}
                 for r in rows if r["task"] == f"{ds}:{src}->{tgt}"]
        print(f"Gate-1 {ds}:{src}->{tgt}: {len(cands)} models ready", flush=True)
        visits.append((g, env, cands))
    return visits


def rule_accuracy(visits, rule, aggregation=None, correlation=None):
    """Mean true accuracy over visits of the model the rule deploys from each visit's candidates."""
    return float(np.mean([s.stats(env, c[s.pick(env, c, rule, aggregation, correlation)])[0]
                          for s, env, c in visits]))


def _stamp():
    return datetime.now().isoformat(timespec="seconds")


def select():
    if GATE_OUT.exists():
        raise SystemExit(f"{GATE_OUT} exists: the gate is one shot, so the variant can no longer change (spec §4)")
    visits = {ds: selection_visits(ds) for ds in DATASET_ORDER}

    def row(rule, aggregation=None, correlation=None):
        acc = {ds: rule_accuracy(v, rule, aggregation, correlation) for ds, v in visits.items()}
        return {**acc, "criterion": float(np.mean(list(acc.values())))}

    table = {v: row(v.rule, v.aggregation, v.correlation) for v in VARIANTS}
    chosen = best_variant({v: r["criterion"] for v, r in table.items()})
    out = {"spec": f"{SPEC} §4 (variant selection)", "time": _stamp(), "files": SELECTION_FILES,
           "gate1_file": str(GATE1_FILE), "methods": SELECTION_METHODS,
           "n_visits": {ds: len(v) for ds, v in visits.items()},
           "n_candidates": {ds: sum(len(c) for _, _, c in v) for ds, v in visits.items()},
           "criterion": "mean over datasets of the mean deployed accuracy over visits",
           "im_rule": row("IM"), "variants": [{**asdict(v), **r} for v, r in table.items()], "chosen": asdict(chosen)}
    VARIANT_OUT.write_text(json.dumps(out, indent=1))
    print(f"IM rule                  pu {out['im_rule']['pu'] * 100:6.2f}  jnu {out['im_rule']['jnu'] * 100:6.2f}")
    for v, r in table.items():
        print(f"{v.rule} {v.aggregation:6s} {v.correlation:8s} pu {r['pu'] * 100:6.2f}  jnu {r['jnu'] * 100:6.2f}  "
              f"criterion {r['criterion'] * 100:6.2f}{'  <- chosen' if v == chosen else ''}")


def gate():
    if GATE_OUT.exists():
        raise SystemExit(f"{GATE_OUT} exists: the gate is one shot (spec §4)")
    if not VARIANT_OUT.exists():
        raise SystemExit(f"{VARIANT_OUT} missing: run `python -m tevc.pcs select` and commit it first")
    v = Variant(**json.loads(VARIANT_OUT.read_text())["chosen"])
    out = {"spec": f"{SPEC} §4 (one-shot gate)", "time": _stamp(), "variant": asdict(v),
           "variant_file": str(VARIANT_OUT), "files": GATE_FILES, "method": "AGENT", "datasets": {}}
    gains = {}
    for ds in DATASET_ORDER:
        stream = STREAMS[ds]()
        scorer, envs = Scorer(stream), stream.environments
        runs = json.loads((OUT_DIR / GATE_FILES[ds]).read_text())["results"]["AGENT"]
        per_run = {}
        for r, entries in runs.items():
            im = replay(entries, envs, lambda env, c: scorer.pick(env, c, "IM"))
            for env, chosen, entry in zip(envs, im, entries):
                if (not same_model(chosen, entry["deployed"])
                        or abs(scorer.stats(env, chosen)[0] - entry["deployed_acc"]) > 1e-9):
                    raise ValueError(f"{ds} run {r} {env.name}: the IM replay does not reproduce the archive")
            pc = replay(entries, envs, lambda env, c: scorer.pick(env, c, v.rule, v.aggregation, v.correlation))
            a_im = float(np.mean([scorer.stats(env, c)[0] for env, c in zip(envs, im)]))
            a_pc = float(np.mean([scorer.stats(env, c)[0] for env, c in zip(envs, pc)]))
            per_run[r] = {"im": a_im, "pcs": a_pc, "gain": a_pc - a_im,
                          "changed_deployments": int(sum(not same_model(a, b) for a, b in zip(im, pc)))}
        gains[ds] = [x["gain"] for x in per_run.values()]
        out["datasets"][ds] = {"runs": per_run, "mean_im": float(np.mean([x["im"] for x in per_run.values()])),
                               "mean_pcs": float(np.mean([x["pcs"] for x in per_run.values()])),
                               "mean_gain": float(np.mean(gains[ds])),
                               "runs_gained": int(sum(g > 0 for g in gains[ds])),
                               "runs_lost": int(sum(g < 0 for g in gains[ds]))}
        d = out["datasets"][ds]
        print(f"{ds}: IM {d['mean_im'] * 100:6.2f}  PCS {d['mean_pcs'] * 100:6.2f}  "
              f"gain {d['mean_gain'] * 100:+6.2f} pp  runs gained/lost {d['runs_gained']}/{d['runs_lost']}")
    out["verdict"] = gate_verdict(gains["pu"], gains["jnu"])
    GATE_OUT.write_text(json.dumps(out, indent=1))
    print("verdict:", out["verdict"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("select", "gate"))
    {"select": select, "gate": gate}[ap.parse_args().step]()


if __name__ == "__main__":
    main()
