"""RESS baselines (spec §3.4): published speed handling (DSR, COT) and domain adaptation (DANN, DEEP-CORAL, CDAN,
SHOT). Each is searched like every other method: B = 30 (configuration, training seed) candidates per environment
change, deployed by the cluster-balanced IM on the unlabelled buffer, the label-free fitness every method uses.
Candidate 0 is the published setting; the other 29 are drawn once per run from ranges around it, so all
environments of a run see the same candidates. Candidates are frozen dataclasses with a `kind` field: the evaluator
caches them like recipes (asdict -> cache key) and result files rebuild them (config_from_dict). COT uses the same
speed information as every method (library speed and environment speed). RPSTDAConfig, the candidate of the MSSP
method (tevc.rpstda), is registered here with the other kinds so the evaluator and result files treat it alike; it is
not a baseline and not in METHODS.
"""
import hashlib
from collections import OrderedDict
from dataclasses import dataclass, fields

import numpy as np

from tevc import synthesis, uda
from tevc.datasets import make_windows
from tevc.deep_train import train_cnn, znorm
from tevc.optimizers import B, Trial
from tevc.recipe import Recipe

METHODS = ("DSR", "COT", "DANN", "DEEP-CORAL", "CDAN", "SHOT")
MAX_MODELS = 64        # trained environment-independent models kept in memory (reused across a run's environments)


@dataclass(frozen=True)
class DSRConfig:
    """Hu, Tang & Chen (2019): pseudo-parameter variances of (r, l, eps) and vicinal copies per original window."""
    var_r: float
    var_l: float
    var_e: float
    copies: int
    kind: str = "DSR"


@dataclass(frozen=True)
class COTConfig:
    """Computed order tracking: train and test on the library's angular sampling."""
    kind: str = "COT"


@dataclass(frozen=True)
class UDAConfig:
    method: str          # DANN | DEEP-CORAL | CDAN
    lr: float
    weight: float        # adversarial coefficient (DANN, CDAN) or CORAL weight
    kind: str = "UDA"


@dataclass(frozen=True)
class SHOTConfig:
    lr: float            # adaptation learning rate
    beta: float          # pseudo-label cross-entropy weight
    kind: str = "SHOT"


RPSTDA_Q_MAX = 0.8      # RPST-DA speed guard (MSSP spec §2.2): no training copy is synthesised at the target speed


@dataclass(frozen=True)
class RPSTDAConfig:
    """An RPST-DA candidate (tevc.rpstda; MSSP spec 2026-10-18 §2): DANN or CDAN whose labelled source is the
    library's train part plus RPST copies (beta = 0) at s_lib (s_env / s_lib) ** q for n_targets q evenly spaced in
    [q_lo, q_hi]; copies = 0 trains on the train part alone (the no-copies arm)."""
    method: str          # DANN | CDAN
    lr: float
    weight: float        # adversarial coefficient
    n_targets: int
    q_lo: float
    q_hi: float
    n_fft: int           # phase-vocoder frame of the copies
    copies: int          # windows per class per synthesised speed
    kind: str = "RPSTDA"

    def __post_init__(self):
        if self.method not in ("DANN", "CDAN"):
            raise ValueError(f"RPST-DA adapts with DANN or CDAN, not {self.method}")
        if not 0.0 <= self.q_lo <= self.q_hi <= RPSTDA_Q_MAX:
            raise ValueError(f"RPST-DA needs 0 <= q_lo <= q_hi <= {RPSTDA_Q_MAX}, got [{self.q_lo}, {self.q_hi}]")


KINDS = {"DSR": DSRConfig, "COT": COTConfig, "UDA": UDAConfig, "SHOT": SHOTConfig, "RPSTDA": RPSTDAConfig}
# Hu et al. 2019, Sec. IV-B: (var_r, var_l, var_e) = (0.02, 0.1, 0.02); their Table 4 enlarges the set 2x, 5x, 10x
DSR_PUBLISHED = DSRConfig(0.02, 0.1, 0.02, 4)
DSR_COPIES = (1, 4, 9)
UDA_PUBLISHED = {"DANN": UDAConfig("DANN", 1e-3, 1.0), "DEEP-CORAL": UDAConfig("DEEP-CORAL", 1e-3, 1.0),
                 "CDAN": UDAConfig("CDAN", 1e-3, 1.0)}
UDA_WEIGHT_RANGE = {"DANN": (0.1, 3.0), "DEEP-CORAL": (0.1, 10.0), "CDAN": (0.1, 3.0)}
SHOT_PUBLISHED = SHOTConfig(1e-3, 0.3)
LR_RANGE = (3e-4, 3e-3)


def config_from_dict(d):
    """Recipe or baseline configuration from a result-file entry; extra keys (seed, fitness) are ignored."""
    cls = KINDS[d["kind"]] if "kind" in d else Recipe
    return cls(**{f.name: d[f.name] for f in fields(cls) if f.name in d})


def _log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_configs(method, run_seed, budget=B):
    """The run's `budget` candidate configurations: the published setting first, then draws around it."""
    rng = np.random.default_rng([run_seed, METHODS.index(method)])
    if method == "COT":
        return [COTConfig()] * budget
    if method == "DSR":
        return [DSR_PUBLISHED] + [DSRConfig(round(_log_uniform(rng, 0.005, 0.08), 5),
                                            round(_log_uniform(rng, 0.025, 0.4), 5),
                                            round(_log_uniform(rng, 0.005, 0.08), 5), int(rng.choice(DSR_COPIES)))
                                  for _ in range(budget - 1)]
    if method == "SHOT":
        return [SHOT_PUBLISHED] + [SHOTConfig(round(_log_uniform(rng, *LR_RANGE), 6),
                                              round(float(rng.uniform(0.1, 0.5)), 3)) for _ in range(budget - 1)]
    return [UDA_PUBLISHED[method]] + [UDAConfig(method, round(_log_uniform(rng, *LR_RANGE), 6),
                                                round(_log_uniform(rng, *UDA_WEIGHT_RANGE[method]), 4))
                                      for _ in range(budget - 1)]


def search(method, evaluator, env, run_seed, budget=B):
    """The run's candidates in order, candidate k trained with seed run_seed * 100 + k; returns the trials."""
    return [Trial(c, run_seed * 100 + k, evaluator.evaluate(env, c, run_seed * 100 + k).fitness)
            for k, c in enumerate(sample_configs(method, run_seed, budget))]


def dsr_simulate(x, r, l, eps):
    """Hu, Tang & Chen (2019) Algorithm 1, steps 2-7, on one example x (n samples) for pseudo parameters r, l and
    noise eps (length >= N): resample at coordinates k r dt, k = 1..N with N r < n (their Eqs. 1-3); l b + eps
    (Eq. 4); FFT at resolution 1/(N dt); re-grid the complex spectrum to resolution 1/(n dt) by direct complex
    linear interpolation (their chosen option, Fig. 5); IFFT; real part x sqrt(2). Every frequency, structural
    resonances included, is multiplied by r."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    N = int(np.ceil(n / r)) - 1
    b = np.interp(np.arange(1, N + 1) * r, np.arange(1, n + 1), x)
    spec = np.fft.fft(l * b + eps[:N])
    old, new = np.arange(N) / N, np.arange(n) / n                   # cycles per sample
    regrid = np.interp(new, old, spec.real) + 1j * np.interp(new, old, spec.imag)
    return (np.real(np.fft.ifft(regrid)) * np.sqrt(2.0)).astype(np.float32)


def _positive_normal(rng, mean, var):
    """N(mean, var) redrawn until positive (DSR's boundary constraint r, l > 0)."""
    while True:
        v = rng.normal(mean, np.sqrt(var))
        if v > 0:
            return float(v)


def dsr_training_set(config, W, y, seed):
    """The z-normalised windows plus config.copies DSR vicinal examples of each, pseudo parameters drawn per example.
    Windows are z-normalised first, as every model input is, so var_e is relative to a window's variance."""
    rng = np.random.default_rng([seed, 2019])
    Z, y = znorm(W), np.asarray(y)
    sims, labels = [], []
    for x, label in zip(Z, y):
        for _ in range(config.copies):
            r, l = _positive_normal(rng, 1.0, config.var_r), _positive_normal(rng, 1.0, config.var_l)
            sims.append(dsr_simulate(x, r, l, rng.normal(0.0, np.sqrt(config.var_e), int(np.ceil(len(x) / r)) - 1)))
            labels.append(label)
    return np.concatenate([Z, np.stack(sims)]), np.concatenate([y, np.asarray(labels, dtype=y.dtype)])


def cot_library(stream):
    """Library windows on the library's angular sampling: an entry recorded at another speed (Ottawa chunks,
    src_rpm_of) is resampled to library_rpm before windowing; JNU and PU entries are at library speed already."""
    if stream.src_rpm_of is None:
        return stream.library["W"], stream.library["y"]
    ds = stream.dataset

    def to_library_speed(label, rec, sig):
        s0 = stream.src_rpm_of(rec)
        return synthesis.resample(sig, s0, stream.library_rpm), stream.library_rpm / s0

    w = make_windows(stream.library_entries, ds.win, ds.hop, ds.cap_fit, synth=to_library_speed)
    return w["W"], w["y"]


def _identity(W):
    return W


def view(config, stream, env):
    """Window transform before prediction: COT stretches target windows by env speed / library speed onto the
    library's angular sampling (every frequency moves, resonances included); other methods see windows unchanged."""
    if not isinstance(config, COTConfig):
        return _identity
    return lambda W: np.stack([synthesis.resample(w, env.rpm, stream.library_rpm) for w in W])


def validation(config, stream, env):
    """Labelled windows a candidate is validated on without any target label: RPST-DA's transposed validation set
    (tevc.rpstda.tv_set); None for every other kind."""
    if not isinstance(config, RPSTDAConfig):
        return None
    from tevc import rpstda                      # rpstda imports this module
    return rpstda.tv_set(stream, env)


_MODELS = OrderedDict()


def _memo(key, build):
    if key not in _MODELS:
        _MODELS[key] = build()
        if len(_MODELS) > MAX_MODELS:
            _MODELS.popitem(last=False)
    _MODELS.move_to_end(key)
    return _MODELS[key]


def _library_key(stream):
    h = hashlib.sha1(np.ascontiguousarray(stream.library["W"]).tobytes())
    h.update(np.asarray(stream.library["y"]).tobytes())
    return h.hexdigest()[:16]


def train(config, stream, env, seed, model_name, epochs):
    """(trained model, library windows) for one candidate; RecipeEvaluator calls it for non-recipe candidates.
    DSR, COT and SHOT's source model do not depend on the environment: trained once and reused (memo)."""
    lib, key = stream.library, (_library_key(stream), seed, model_name, epochs)
    if isinstance(config, DSRConfig):
        return _memo(("DSR", config) + key,
                     lambda: train_cnn(*dsr_training_set(config, lib["W"], lib["y"], seed), model_name, epochs,
                                       seed)), lib["W"]
    if isinstance(config, COTConfig):
        return _memo(("COT",) + key, lambda: train_cnn(*cot_library(stream), model_name, epochs, seed)), lib["W"]
    if isinstance(config, UDAConfig):
        return uda.train_uda(lib["W"], lib["y"], env.U["W"], config.method, config.weight, config.lr, model_name,
                             epochs, seed), lib["W"]
    if isinstance(config, SHOTConfig):
        source = _memo(("SHOT-source",) + key, lambda: uda.train_source(lib["W"], lib["y"], model_name, epochs, seed))
        return uda.train_shot(lib["W"], lib["y"], env.U["W"], config.lr, config.beta, model_name, epochs, seed,
                              source=source), lib["W"]
    if isinstance(config, RPSTDAConfig):
        from tevc import rpstda
        return rpstda.train(config, stream, env, seed, model_name, epochs)
    raise TypeError(f"not a baseline configuration: {config!r}")
