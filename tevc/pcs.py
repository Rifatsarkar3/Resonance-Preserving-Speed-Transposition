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
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from tevc import features
from tevc.agent import RECUR
from tevc.baselines import config_from_dict
from tevc.evaluator import RecipeEvaluator

FAULT_TARGET = {"OR": "BPFO", "IR": "BPFI", "RE": "BSF"}
SIGNATURES = {"jnu": (set(), {"IR"}, {"OR"}, {"RE"}),          # normal, inner race, outer race, roller
              "pu": (set(), {"OR"}, {"IR"}, {"IR", "OR"})}      # healthy, KA, KI, KB (combined inner + outer damage)
RULES, AGGREGATIONS, CORRELATIONS = ("V1", "V3", "V2"), ("A-full", "A-max"), ("spearman", "pearson")
TOP_IM = 5
CONSTANT = 1e-6         # a claim whose range is below this is constant (float32 probabilities)
PU_MIN_GAIN, PU_MIN_WINS, JNU_MAX_LOSS = 0.005, 7, 0.0025
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
