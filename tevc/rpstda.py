"""RPST-DA (MSSP spec docs/superpowers/specs/2026-10-18-mssp-rpst-da-design.md §2): domain adaptation (DANN or CDAN)
whose labelled source is the first 80 % in time of every library record plus RPST copies at intermediate speeds,
deployed by transposed validation (TV): the last 20 % of every library record, RPST-transposed to the environment's
speed, keeps the library labels and scores every candidate without any target label.

B = 30 candidates per environment change, drawn once per run (rng stream [run_seed, 7], after the six baselines'
streams 0-5); candidate 0 = CDAN at its published setting with the earlier RPST recipe. Deployment: argmax of the mean
of rank(TV accuracy) and rank(IM) (rank_deploy). The comparators keep the full library.
"""
import numpy as np
from scipy.stats import rankdata

from tevc import synthesis, uda
from tevc.baselines import LR_RANGE, RPSTDA_Q_MAX, UDA_WEIGHT_RANGE, RPSTDAConfig, _log_uniform
from tevc.datasets import make_windows
from tevc.optimizers import B, Trial
from tevc.recipe import COPIES, N_FFTS, Recipe, training_set

METHODS = ("DANN", "CDAN")
TRAIN_FRACTION = 0.8
TV_N_FFT = 256              # beta = 0 phase vocoder at the earlier RPST frame
TV_HOP_DIVISOR = 4          # validation windows overlap (hop = win / 4): evaluation only
RNG_STREAM = 7
PUBLISHED = RPSTDAConfig("CDAN", 1e-3, 1.0, 3, 0.3, 0.8, 256, 80)


def _cut(sig, lo, hi, fraction=TRAIN_FRACTION):
    """Sample index splitting an entry's span [lo, hi) (fractions of the record) into its train and validation parts."""
    return int(len(sig) * (lo + fraction * (hi - lo)))


def split_entries(entries, fraction=TRAIN_FRACTION):
    """(train, validation) entries: every library entry (label, rec, signal, lo, hi) cut in time. Each part's signal
    is the segment itself (span 0-1), so no operator frame reads across the cut."""
    train, val = [], []
    for label, rec, sig, lo, hi in entries:
        a, c, b = int(len(sig) * lo), _cut(sig, lo, hi, fraction), int(len(sig) * hi)
        train.append((label, rec, sig[a:c], 0.0, 1.0))
        val.append((label, rec, sig[c:b], 0.0, 1.0))
    return train, val


def train_windows(stream, fraction=TRAIN_FRACTION):
    """The stream's library windows lying wholly in the train part of their record: the library's own thinning and
    class balance, minus the windows at or beyond the cut."""
    cut = {rec: _cut(sig, lo, hi, fraction) for _, rec, sig, lo, hi in stream.library_entries}
    lib = stream.library
    keep = np.array([hi <= cut[str(r)] for r, hi in zip(lib["rec"], lib["span_hi"])])
    return {k: np.asarray(v)[keep] for k, v in lib.items()}


def source_set(config, stream, env):
    """Labelled source windows (W, y): the train-part library windows plus, unless copies = 0, RPST copies (beta = 0)
    of the train-part records at the config's intermediate speeds between the library and env."""
    lib = train_windows(stream)
    if config.copies == 0:
        return lib["W"], lib["y"]
    train_entries, _ = split_entries(stream.library_entries)
    recipe = Recipe(config.n_targets, config.q_lo, config.q_hi, config.n_fft, 0.0, config.copies, True)
    return training_set(recipe, stream.dataset, train_entries, lib, stream.library_rpm, env.rpm, stream.src_rpm_of)


def train(config, stream, env, seed, model_name, epochs):
    """(trained model, source windows) for one candidate; baselines.train dispatches RPST-DA candidates here."""
    W, y = source_set(config, stream, env)
    return uda.train_uda(W, y, env.U["W"], config.method, config.weight, config.lr, model_name, epochs, seed), W


def tv_set(stream, env, cap=None):
    """Transposed validation windows (spec §2.3): the validation part of every library record, RPST-transposed
    (beta = 0, n_fft 256) from its source speed to env's speed, windowed with hop win / 4, at most `cap` (default
    cap_fit) per class. Labels are the library's; no target label is read."""
    ds = stream.dataset
    _, val = split_entries(stream.library_entries)

    def to_env(label, rec, seg):
        s0 = stream.src_rpm_of(rec) if stream.src_rpm_of else stream.library_rpm
        return synthesis.transpose(seg, s0, env.rpm, 0.0, TV_N_FFT), env.rpm / s0

    V = make_windows(val, ds.win, ds.win // TV_HOP_DIVISOR, cap or ds.cap_fit, synth=to_env)
    missing = sorted(set(range(ds.n_classes)) - set(V["y"].tolist()))
    if missing:
        raise ValueError(f"{env.name}: no transposed validation window for classes {missing}")
    return V


def sample_configs(run_seed, budget=B):
    """The run's `budget` candidates: PUBLISHED first, then draws (spec §2.4) from the run's own rng stream."""
    rng = np.random.default_rng([run_seed, RNG_STREAM])
    out = [PUBLISHED]
    for _ in range(budget - 1):
        method = str(rng.choice(METHODS))
        lr = round(_log_uniform(rng, *LR_RANGE), 6)
        weight = round(_log_uniform(rng, *UDA_WEIGHT_RANGE[method]), 4)
        n_targets = int(rng.integers(1, 5))
        q_lo = float(rng.uniform(0.2, RPSTDA_Q_MAX))
        q_hi = float(rng.uniform(q_lo, RPSTDA_Q_MAX))
        n_fft, copies = int(rng.choice(N_FFTS)), int(rng.choice(COPIES))
        out.append(RPSTDAConfig(method, lr, weight, n_targets, round(q_lo, 3), round(q_hi, 3), n_fft, copies))
    return out


def search(evaluator, env, run_seed, budget=B):
    """The run's candidates in order, candidate k trained with seed run_seed * 100 + k; fitness = IM on U."""
    return [Trial(c, run_seed * 100 + k, evaluator.evaluate(env, c, run_seed * 100 + k).fitness)
            for k, c in enumerate(sample_configs(run_seed, budget))]


def rank_deploy(tv):
    """Deployment rule (spec §2.4) on a pool of trials, tv(trial) = its TV accuracy: argmax of the mean of
    rank(TV accuracy) and rank(IM), average ranks for ties; remaining ties go to the earliest candidate."""
    def deploy(pool):
        mixed = (rankdata([tv(t) for t in pool]) + rankdata([t.fitness for t in pool])) / 2.0
        return pool[int(np.argmax(mixed))]
    return deploy
