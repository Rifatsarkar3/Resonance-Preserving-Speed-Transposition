"""Recipe encoding, GA operators and the memory-less searches of spec §5 (RANDOM, GA-RESTART).

A recipe is searched as 7 genes in [0, 1]; `decode` maps any gene vector to a valid Recipe and `encode` maps a
recipe to the centre of its discrete levels. Searches see only the agent-visible fitness (`Evaluation.fitness`).
Budget per environment change: B = 30 real evaluations = 12 initial + 3 generations x 5 offspring + top-3 recipes
re-evaluated with a second training seed.
"""
from dataclasses import dataclass

import numpy as np

from tevc.recipe import COPIES, N_FFTS, Recipe

N_GENES = 7
N_INIT, N_GEN, N_OFF, N_REEVAL = 12, 3, 5, 3
B = N_INIT + N_GEN * N_OFF + N_REEVAL


@dataclass(frozen=True)
class Trial:
    """One trained model: its recipe, training seed and agent-visible fitness."""
    recipe: Recipe
    seed: int
    fitness: float


def _level(g, n):
    return min(int(g * n), n - 1)


def decode(genes, beta=None):
    """Genes in [0, 1]^7 -> Recipe; beta overrides gene 4 (1.0 for the AGENT-β1 physics control)."""
    g = np.clip(np.asarray(genes, dtype=float), 0.0, 1.0)
    q_lo = round(float(0.2 + 0.8 * g[1]), 3)
    return Recipe(n_targets=_level(g[0], 4) + 1, q_lo=q_lo, q_hi=round(float(q_lo + (1.2 - q_lo) * g[2]), 3),
                  n_fft=N_FFTS[_level(g[3], len(N_FFTS))], beta=round(float(g[4] if beta is None else beta), 3),
                  copies=COPIES[_level(g[5], len(COPIES))], keep_real=bool(g[6] >= 0.5))


def encode(recipe):
    """Recipe (n_targets >= 1) -> genes; decode(encode(r)) == r for recipes on the 3-decimal grid."""
    span = 1.2 - recipe.q_lo
    return np.array([(recipe.n_targets - 0.5) / 4, (recipe.q_lo - 0.2) / 0.8,
                     (recipe.q_hi - recipe.q_lo) / span if span > 0 else 0.0,
                     (N_FFTS.index(recipe.n_fft) + 0.5) / len(N_FFTS), recipe.beta,
                     (COPIES.index(recipe.copies) + 0.5) / len(COPIES), 0.75 if recipe.keep_real else 0.25])


def sbx(p1, p2, rng, eta=15.0):
    """Simulated binary crossover on [0, 1] genes; each gene crosses with probability 0.5."""
    u = rng.random(len(p1))
    b = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (0.5 / (1 - u)) ** (1 / (eta + 1)))
    b = np.where(rng.random(len(p1)) < 0.5, b, 1.0)          # b = 1 copies the parents
    return (np.clip(0.5 * ((1 + b) * p1 + (1 - b) * p2), 0.0, 1.0),
            np.clip(0.5 * ((1 - b) * p1 + (1 + b) * p2), 0.0, 1.0))


def polynomial_mutation(x, rng, eta=20.0):
    """Polynomial mutation on [0, 1] genes, each gene mutated with probability 1 / len(x)."""
    y = np.array(x, dtype=float)
    for i in np.flatnonzero(rng.random(len(y)) < 1.0 / len(y)):
        u = rng.random()
        delta = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
        y[i] = np.clip(y[i] + delta, 0.0, 1.0)
    return y


def tournament(fitness, rng):
    i, j = rng.integers(len(fitness), size=2)
    return i if fitness[i] >= fitness[j] else j


def offspring(pop, fitness, n, rng):
    """n children by binary tournament, SBX and polynomial mutation."""
    kids = []
    while len(kids) < n:
        a, b = pop[tournament(fitness, rng)], pop[tournament(fitness, rng)]
        kids += [polynomial_mutation(c, rng) for c in sbx(a, b, rng)]
    return kids[:n]


class EnvSearch:
    """Evaluations inside one environment, in order, with budget accounting."""

    def __init__(self, evaluator, env, budget):
        self.evaluator, self.env, self.budget, self.trials = evaluator, env, budget, []

    def done(self, recipe, seed):
        return any(t.recipe == recipe and t.seed == seed for t in self.trials)

    def run(self, recipe, seed):
        if len(self.trials) >= self.budget:
            raise RuntimeError("evaluation budget exhausted")
        self.trials.append(Trial(recipe, seed, self.evaluator.evaluate(self.env, recipe, seed).fitness))
        return self.trials[-1].fitness

    def top_recipes(self, n, seed):
        out = []
        for t in sorted((t for t in self.trials if t.seed == seed), key=lambda t: -t.fitness):
            if t.recipe not in out:
                out.append(t.recipe)
        return out[:n]


def _new_kids(search, pop, fit, n, rng, seed, beta, screen, n_candidates=50, attempts=20):
    """n children whose recipes are not yet evaluated with `seed`; ranked by `screen` when given."""
    kids, recipes = [], []
    for _ in range(attempts):
        for c in offspring(pop, np.asarray(fit), n_candidates if screen else n, rng):
            r = decode(c, beta)
            if r not in recipes and not search.done(r, seed):
                kids.append(c)
                recipes.append(r)
        if len(kids) >= (n_candidates if screen else n):
            break
    if screen is not None and kids:
        order = np.argsort(-screen(np.array(kids), search.trials), kind="stable")
        kids = [kids[i] for i in order]
    return kids[:n]


def ga_search(evaluator, env, init_genes, rng, seeds, screen=None, beta=None,
              n_gen=N_GEN, n_off=N_OFF, n_reeval=N_REEVAL):
    """(mu + lambda) GA from `init_genes`; screen(candidate genes, trials so far) -> scores picks the offspring
    actually evaluated (surrogate pre-screening). Returns the trials in evaluation order."""
    s1, s2 = seeds
    pop, recipes = [], []
    for g in init_genes:
        r = decode(g, beta)
        if r not in recipes:
            pop.append(np.asarray(g, dtype=float))
            recipes.append(r)
    search = EnvSearch(evaluator, env, len(pop) + n_gen * n_off + n_reeval)
    fit = [search.run(r, s1) for r in recipes]
    mu = len(pop)
    for _ in range(n_gen):
        for c in _new_kids(search, pop, fit, n_off, rng, s1, beta, screen):
            pop.append(c)
            fit.append(search.run(decode(c, beta), s1))
        keep = np.argsort(-np.asarray(fit), kind="stable")[:mu]
        pop, fit = [pop[i] for i in keep], [fit[i] for i in keep]
    for r in search.top_recipes(n_reeval, s1):
        search.run(r, s2)
    return search.trials


def random_search(evaluator, env, rng, seed, budget=B):
    search = EnvSearch(evaluator, env, budget)
    for g in rng.random((budget, N_GENES)):
        search.run(decode(g), seed)
    return search.trials
