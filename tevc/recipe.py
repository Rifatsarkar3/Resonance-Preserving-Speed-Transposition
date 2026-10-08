"""RPST synthesis recipes: the decision vector of the deep-path DOP (D1)."""
from dataclasses import dataclass

import numpy as np

from tevc import synthesis
from tevc.datasets import make_windows

N_FFTS = (64, 128, 256, 512)
COPIES = (20, 40, 80)


@dataclass(frozen=True)
class Recipe:
    n_targets: int      # synthesised speeds (0 = library only)
    q_lo: float         # synthesised speeds s_src * (s_tgt / s_src) ** q, q evenly spaced in [q_lo, q_hi]
    q_hi: float
    n_fft: int          # phase-vocoder frame: smaller = less transient smearing, coarser frequency resolution
    beta: float         # resonance shift: 0 = RPST (bands fixed) ... 1 = plain resampling
    copies: int         # windows per class per synthesised speed
    keep_real: bool     # also train on the real library windows

    def target_rpms(self, src, tgt):
        return [float(src * (tgt / src) ** q) for q in np.linspace(self.q_lo, self.q_hi, self.n_targets)]


ANCHORS = {"NONE": Recipe(0, 0.0, 0.0, 256, 0.0, 0, True),
           "H2_LIKE": Recipe(3, 0.3, 0.8, 256, 0.0, 80, True),
           "AT_TARGET": Recipe(1, 1.0, 1.0, 256, 0.0, 80, True),
           "RESAMPLE_H2": Recipe(3, 0.3, 0.8, 256, 1.0, 80, True)}


def sample(rng):
    q_lo = rng.uniform(0.2, 1.0)
    return Recipe(n_targets=int(rng.integers(1, 5)), q_lo=round(float(q_lo), 3),
                  q_hi=round(float(rng.uniform(q_lo, 1.2)), 3), n_fft=int(rng.choice(N_FFTS)),
                  beta=round(float(rng.uniform(0.0, 1.0)), 3), copies=int(rng.choice(COPIES)),
                  keep_real=bool(rng.integers(2)))


def training_set(recipe, ds, library_entries, library_windows, src, tgt):
    """Windows (N, L) and labels for training a diagnoser under `recipe` (perceived target speed tgt)."""
    parts = [library_windows] if recipe.keep_real else []
    for t in recipe.target_rpms(src, tgt):
        parts.append(make_windows(library_entries, ds.win, ds.hop, recipe.copies,
                                  transform=lambda y, t=t: synthesis.transpose(y, src, t, recipe.beta, recipe.n_fft),
                                  rate=t / src))
    return np.concatenate([p["W"] for p in parts]), np.concatenate([p["y"] for p in parts])
