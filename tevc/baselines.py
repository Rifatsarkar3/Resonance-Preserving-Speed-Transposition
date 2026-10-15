"""RESS baselines (spec §3.4): published speed handling (DSR, COT) and domain adaptation (DANN, DEEP-CORAL, CDAN,
SHOT). Each is searched like every other method: B = 30 (configuration, training seed) candidates per environment
change, deployed by the cluster-balanced IM on the unlabelled buffer, the label-free fitness every method uses.
Candidate 0 is the published setting; the other 29 are drawn once per run from ranges around it, so all
environments of a run see the same candidates. Candidates are frozen dataclasses with a `kind` field: the evaluator
caches them like recipes (asdict -> cache key) and result files rebuild them (config_from_dict). COT uses the same
speed information as every method (library speed and environment speed).
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


KINDS = {"DSR": DSRConfig, "COT": COTConfig, "UDA": UDAConfig, "SHOT": SHOTConfig}
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
