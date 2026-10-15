from dataclasses import asdict

import numpy as np

from tevc.datasets import PU
from tevc.features import compute_fault_frequencies
from tevc.pcs import (AGGREGATIONS, CORRELATIONS, SIGNATURES, VARIANTS, Scorer, Variant, best_variant, choose,
                      claims, evidence, gate_verdict, pcs, weighted_corr)
from tevc.recipe import Recipe
from tevc.stream import Environment, Stream

RNG = np.random.default_rng(0)


def test_weighted_corr_is_pearson_under_uniform_weights_and_zero_for_a_constant():
    x, y = RNG.normal(size=50), RNG.normal(size=50)
    assert np.isclose(weighted_corr(x, y, np.ones(50)), np.corrcoef(x, y)[0, 1])
    assert weighted_corr(np.ones(50), y, np.ones(50)) == 0.0


def test_weights_decide_which_windows_count():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 3.0, 2.0])               # the last two windows disagree with x
    assert weighted_corr(x, y, [1, 1, 1e-9, 1e-9]) > weighted_corr(x, y, [1e-9, 1e-9, 1, 1])


def _probs(cls, n_classes=4):
    return np.eye(n_classes)[cls] * 0.9 + 0.1 / n_classes


def test_pu_combined_damage_class_claims_both_inner_and_outer_race():
    q = claims(_probs(np.array([0, 1, 2, 3])), SIGNATURES["pu"])
    assert set(q) == {"OR", "IR"}                    # PU has no rolling-element class
    assert np.allclose(q["OR"], [0.05, 0.95, 0.05, 0.95]) and np.allclose(q["IR"], [0.05, 0.05, 0.95, 0.95])


def test_pcs_rewards_claims_that_follow_the_evidence():
    y = np.repeat([0, 1, 2], 5)                      # PU: healthy, outer race (KA), inner race (KI)
    ev = {"OR": (y == 1) + 0.1 * RNG.normal(size=15), "IR": (y == 2) + 0.1 * RNG.normal(size=15)}
    w = np.ones(15)
    swapped = np.array([0, 2, 1])[y]
    for corr in ("spearman", "pearson"):
        assert pcs(_probs(y), ev, SIGNATURES["pu"], w, corr) > 0.8
        assert pcs(_probs(swapped), ev, SIGNATURES["pu"], w, corr) < -0.3


def test_a_fault_type_the_model_never_claims_scores_zero():
    p = np.zeros((6, 4))
    p[:3, 0], p[3:, 1] = 1.0, 1.0                    # never KI or KB: the inner-race claim is constant 0
    ev = {"OR": np.r_[np.zeros(3), np.ones(3)], "IR": RNG.normal(size=6)}
    assert np.isclose(pcs(p, ev, SIGNATURES["pu"], np.ones(6), "pearson"), 0.5)     # (1 + 0) / 2


def test_rules():
    im = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    score = np.array([0.1, 0.5, 0.2, 0.3, 0.0, 0.99])
    assert choose("IM", im, score) == 0
    assert choose("V1", im, score) == 5
    assert choose("V3", im, score) == 1              # candidate 5 is outside the top-5 by IM
    assert choose("V2", im, score) == 1              # mean ranks: candidate 1 = (5 + 5) / 2 is the best
    assert choose("V1", im, np.zeros(6)) == 0        # ties go to the earliest candidate


def test_variants_are_listed_in_tie_break_order_and_the_first_of_equals_wins():
    assert len(VARIANTS) == 12 and VARIANTS[0] == Variant("V1", "A-full", "spearman")
    assert [v.rule for v in VARIANTS[::4]] == ["V1", "V3", "V2"]
    scores = {v: 0.7 for v in VARIANTS}
    scores[VARIANTS[7]] = scores[VARIANTS[9]] = 0.8
    assert best_variant(scores) == VARIANTS[7]


def test_gate_verdict_thresholds():
    seven_wins = [0.01] * 7 + [-0.001] * 3
    assert gate_verdict(seven_wins, [-0.0025] * 4)["pass"]
    assert not gate_verdict([0.01] * 6 + [-0.001] * 4, [0.0] * 4)["pass"]           # 6 of 10 runs
    assert not gate_verdict([0.004] * 10, [0.0] * 4)["pass"]                        # mean gain below 0.5 pp
    assert not gate_verdict(seven_wins, [-0.003] * 4)["pass"]                       # JNU loses 0.3 pp


def _z(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std()


def test_evidence_averages_harmonics_then_takes_the_full_band_or_the_best_band():
    block = np.zeros((3, 9, 4, 3))                   # windows, bands (0 = full), (f_s, BPFO, BPFI, BSF), harmonics
    block[:, 0, 1, :] = [[1, 2, 3], [0, 0, 0], [2, 2, 2]]
    block[1, 5, 1, :] = 9.0                          # window 1: BPFO stands out in one narrow band only
    full, best = evidence(block, "A-full"), evidence(block, "A-max")
    assert set(full) == {"OR", "IR", "RE"}
    assert np.allclose(full["OR"], _z([2, 0, 2])) and np.allclose(best["OR"], _z([2, 9, 2]))
    assert np.allclose(full["IR"], 0.0)


def _bursts(rate, rng, fs=PU.fs, n=PU.win, resonance=3000.0):
    """Decaying 3 kHz resonance excited at `rate` Hz (random phase), plus noise; rate 0 = noise only."""
    t = np.arange(n) / fs
    x = 0.3 * rng.normal(size=n)
    if rate:
        for t0 in np.arange(rng.uniform(0, 1 / rate), t[-1], 1 / rate):
            m = t >= t0
            x[m] += np.exp(-(t[m] - t0) / 1e-3) * np.sin(2 * np.pi * resonance * (t[m] - t0))
    return x


class _Cache:
    """cached(): the probabilities registered for a seed, on U and on T alike; uniform buffer weights."""

    def __init__(self, probs):
        self.probs = probs

    def cached(self, env, config, seed):
        return self.probs[seed], self.probs[seed]

    def weights(self, env):
        return np.ones(len(env.U["W"]))


def test_pcs_prefers_the_model_whose_fault_claims_match_the_envelope_physics():
    f = compute_fault_frequencies(f_shaft_hz=1500 / 60, **PU.geometry)
    y = np.repeat([0, 1, 2], [4, 6, 6])              # healthy, outer race (KA), inner race (KI)
    rate = {0: 0.0, 1: f["BPFO"], 2: f["BPFI"]}
    rng = np.random.default_rng(1)
    W = np.stack([_bursts(rate[c], rng) for c in y])
    env = Environment("pu@1500", 1500.0, {"W": W}, {"y": y})
    scorer = Scorer(Stream(PU, [], {}, 900.0, (env,)),
                    _Cache({1: _probs(y), 2: _probs(np.array([0, 2, 1])[y])}))   # seed 2 swaps OR and IR
    cand = asdict(Recipe(0, 0.0, 0.0, 256, 0.0, 0, True))
    (acc_good, good), (acc_bad, bad) = scorer.stats(env, {**cand, "seed": 1}), scorer.stats(env, {**cand, "seed": 2})
    assert acc_good == 1.0 and acc_bad == 4 / 16
    for a in AGGREGATIONS:
        for c in CORRELATIONS:
            assert good[(a, c)] > 0.5 and bad[(a, c)] < 0.0
    cands = [{**cand, "seed": 2, "fitness": 0.9}, {**cand, "seed": 1, "fitness": 0.1}]
    assert scorer.pick(env, cands, "IM") == 0 and scorer.pick(env, cands, "V1", "A-max", "pearson") == 1
