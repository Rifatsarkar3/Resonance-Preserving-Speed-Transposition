"""RPST-DA (MSSP spec docs/superpowers/specs/2026-10-18-mssp-rpst-da-design.md §2): domain adaptation (DANN or CDAN)
whose labelled source is the first 80 % in time of every library record plus RPST copies at intermediate speeds,
deployed by transposed validation (TV): the last 20 % of every library record, RPST-transposed to the environment's
speed, keeps the library labels and scores every candidate without any target label.

B = 30 candidates per environment change, drawn once per run (rng stream [run_seed, 7], after the six baselines'
streams 0-5); candidate 0 = CDAN at its published setting with the earlier RPST recipe. Deployment: argmax of the mean
of rank(TV accuracy) and rank(IM) (rank_deploy). The comparators keep the full library.

Exploratory arms (MSSP reframe spec docs/superpowers/specs/2026-10-20-mssp-sp-first-reframe-design.md §5.3-5.4;
attack audit U2, G2, G6): ARMS maps an arm name to a transform of RPST-DA's own draws, so every arm is paired with
RPST-DA candidate by candidate. RPST-DA-B1 / -B05: copies with resonance shift beta = 1 (plain resampling) / 0.5;
RPST-DA-NCS: no copies but as many training steps as the copies twin; RPST-DA-W0: copies without adversarial
alignment (weight 0; target batches still share the batch statistics); frame_rule_arm: copies and transposed
validation at the frame of the rule N <= rho* P (RPST-DA-FR, DIRG).
"""
from dataclasses import replace

import numpy as np
from scipy.stats import rankdata

from tevc import synthesis, uda
from tevc.baselines import LR_RANGE, RPSTDA_Q_MAX, UDA_WEIGHT_RANGE, RPSTDAConfig, _log_uniform
from tevc.datasets import make_windows
from tevc.optimizers import B, Trial
from tevc.recipe import COPIES, N_FFTS, Recipe, frame, training_set

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


def _balanced(y):
    """Indices keeping every class at the smallest class's count, evenly thinned in order, so that every synthesised
    speed keeps its share of each class."""
    groups = [np.flatnonzero(y == c) for c in np.unique(y)]
    n = min(len(g) for g in groups)
    return np.sort(np.concatenate([g[np.linspace(0, len(g) - 1, n).round().astype(int)] for g in groups]))


def source_set(config, stream, env):
    """Labelled source windows (W, y): the train-part library windows plus, unless copies = 0, transposed copies
    (resonance shift config.copies_beta: 0 = RPST, 1 = plain resampling) of the train-part records at the config's
    intermediate speeds between the library and env. The copies are thinned to equal counts per class (a longer
    healthy record would otherwise give it more copies, as on JNU)."""
    lib = train_windows(stream)
    if config.copies == 0:
        return lib["W"], lib["y"]
    train_entries, _ = split_entries(stream.library_entries)
    recipe = Recipe(config.n_targets, config.q_lo, config.q_hi, config.n_fft, config.copies_beta, config.copies, True)
    W, y = training_set(recipe, stream.dataset, train_entries, lib, stream.library_rpm, env.rpm, stream.src_rpm_of)
    n_lib = len(lib["y"])                                   # training_set puts the real windows first
    keep = np.r_[np.arange(n_lib), n_lib + _balanced(y[n_lib:])]
    return W[keep], y[keep]


def twin_steps(config, n_train, n_classes, epochs, batch_size=uda.BATCH_SIZE):
    """Training steps of the copies twin of a steps-matched candidate (RPST-DA-NCS): the twin's source is the
    n_train real windows plus n_targets x steps_like_copies copies per class (exact when every class yields that many
    copy windows per synthesised speed, as on PU and Lee)."""
    n_twin = n_train + config.n_targets * config.steps_like_copies * n_classes
    return epochs * int(np.ceil(n_twin / batch_size))


def train(config, stream, env, seed, model_name, epochs):
    """(trained model, source windows) for one candidate; baselines.train dispatches RPST-DA candidates here."""
    W, y = source_set(config, stream, env)
    total = (twin_steps(config, len(y), stream.dataset.n_classes, epochs) if config.steps_like_copies else None)
    return uda.train_uda(W, y, env.U["W"], config.method, config.weight, config.lr, model_name, epochs, seed,
                         total_steps=total), W


def tv_set(stream, env, cap=None, n_fft=None):
    """Transposed validation windows (spec §2.3): the validation part of every library record, RPST-transposed
    (beta = 0, n_fft 256, or the frame-rule arm's n_fft, scaled by the dataset's frame_scale) from its source speed to
    env's speed, windowed with hop win / 4, at most `cap` (default cap_fit) per class. Labels are the library's; no
    target label is read."""
    ds = stream.dataset
    _, val = split_entries(stream.library_entries)

    n_fft = frame(n_fft or TV_N_FFT, ds)

    def to_env(label, rec, seg):
        s0 = stream.src_rpm_of(rec) if stream.src_rpm_of else stream.library_rpm
        return synthesis.transpose(seg, s0, env.rpm, 0.0, n_fft), env.rpm / s0

    V = make_windows(val, ds.win, ds.win // TV_HOP_DIVISOR, cap or ds.cap_fit, synth=to_env)
    missing = sorted(set(range(ds.n_classes)) - set(V["y"].tolist()))
    if missing:
        raise ValueError(f"{env.name}: no transposed validation window for classes {missing}")
    return V


def sample_configs(run_seed, budget=B, copies=True, transform=None):
    """The run's `budget` candidates: PUBLISHED first, then draws (spec §2.4) from the run's own rng stream; with
    copies=False the same draws with copies = 0, the no-copies arm (spec §4); with transform (an ARMS value or
    frame_rule_arm) every draw mapped by it, an exploratory arm paired with RPST-DA candidate by candidate."""
    rng = np.random.default_rng([run_seed, RNG_STREAM])
    out = [PUBLISHED]
    for _ in range(budget - 1):
        method = str(rng.choice(METHODS))
        lr = round(_log_uniform(rng, *LR_RANGE), 6)
        weight = round(_log_uniform(rng, *UDA_WEIGHT_RANGE[method]), 4)
        n_targets = int(rng.integers(1, 5))
        q_lo = float(rng.uniform(0.2, RPSTDA_Q_MAX))
        q_hi = float(rng.uniform(q_lo, RPSTDA_Q_MAX))
        n_fft, n_copies = int(rng.choice(N_FFTS)), int(rng.choice(COPIES))
        out.append(RPSTDAConfig(method, lr, weight, n_targets, round(q_lo, 3), round(q_hi, 3), n_fft, n_copies))
    if transform is not None:
        return [transform(c) for c in out]
    return out if copies else [replace(c, copies=0) for c in out]


ARMS = {
    "RPST-DA": lambda c: c,
    "RPST-DA-NC": lambda c: replace(c, copies=0),
    "RPST-DA-B1": lambda c: replace(c, copies_beta=1.0),
    "RPST-DA-B05": lambda c: replace(c, copies_beta=0.5),
    "RPST-DA-NCS": lambda c: replace(c, copies=0, steps_like_copies=c.copies),
    "RPST-DA-W0": lambda c: replace(c, weight=0.0),
}


def frame_rule_n(stream, env, rho, convention="source"):
    """Frame of the rule N <= rho x P (Eq. 4; spec §5.4), in samples at the dataset's own rate: the largest power of
    two not above rho x P, with P the shortest fault-impact period (BPFO, BPFI or 2 BSF) at the library speed
    ("source"), or at the environment speed when that is faster ("target": P_lib / max(1, s_env / s_lib)), the
    convention R6 selects."""
    from tevc.features import compute_fault_frequencies
    if convention not in ("source", "target"):
        raise ValueError(f"unknown period convention {convention}")
    ds = stream.dataset
    f = compute_fault_frequencies(f_shaft_hz=stream.library_rpm / 60.0, **ds.geometry)
    p = min(ds.fs / f["BPFO"], ds.fs / f["BPFI"], ds.fs / (2.0 * f["BSF"]))
    if convention == "target":
        p /= max(1.0, env.rpm / stream.library_rpm)
    if rho * p < 2:
        raise ValueError(f"the rule leaves no frame of at least 2 samples (rho x P = {rho * p:.2f})")
    return int(2 ** int(np.floor(np.log2(rho * p))))


def frame_rule_arm(n_samples, ds):
    """Transform for RPST-DA-FR: copies and transposed validation at the frame-rule frame n_samples (actual samples;
    stored in development counts, so that recipe.frame() maps it back on datasets with a frame_scale)."""
    dev = int(round(n_samples / getattr(ds, "frame_scale", 1.0)))
    return lambda c: replace(c, n_fft=dev, tv_n_fft=dev)


def search(evaluator, env, run_seed, budget=B, copies=True, transform=None):
    """The run's candidates in order, candidate k trained with seed run_seed * 100 + k; fitness = IM on U."""
    return [Trial(c, run_seed * 100 + k, evaluator.evaluate(env, c, run_seed * 100 + k).fitness)
            for k, c in enumerate(sample_configs(run_seed, budget, copies, transform))]


def rank_deploy(tv):
    """Deployment rule (spec §2.4) on a pool of trials, tv(trial) = its TV accuracy: argmax of the mean of
    rank(TV accuracy) and rank(IM), average ranks for ties; remaining ties go to the earliest candidate."""
    def deploy(pool):
        mixed = (rankdata([tv(t) for t in pool]) + rankdata([t.fitness for t in pool])) / 2.0
        return pool[int(np.argmax(mixed))]
    return deploy
