"""Bi-objective feature selection with NSGA-II (pymoo): f1 = k-NN balanced error, f2 = feature ratio."""
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class FeatureSelection(Problem):
    """Search over the `allowed` features only; masks and the feature ratio always refer to the full bank."""

    def __init__(self, evaluator, n_features, allowed=None):
        self.idx = np.arange(n_features) if allowed is None else np.flatnonzero(allowed)
        self.n_features = n_features
        super().__init__(n_var=len(self.idx), n_obj=2, xl=0, xu=1, vtype=bool)
        self.evaluator = evaluator

    def full_masks(self, X):
        masks = np.zeros((len(X), self.n_features), dtype=bool)
        masks[:, self.idx] = X.astype(bool)
        return masks

    def _evaluate(self, X, out, *args, **kwargs):
        masks = self.full_masks(X)
        out["F"] = np.column_stack([self.evaluator.fitness(masks), masks.sum(axis=1) / self.n_features])


class SparseSampling(Sampling):
    """Each individual draws its own inclusion probability in [0.02, 0.5] (sparse-biased start)."""

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        rho = random_state.uniform(0.02, 0.5, size=(n_samples, 1))
        X = random_state.random((n_samples, problem.n_var)) < rho
        empty = np.flatnonzero(X.sum(axis=1) == 0)
        X[empty, random_state.integers(problem.n_var, size=len(empty))] = True
        return X


def run_nsga2(evaluator, n_features, seed, pop_size=100, n_gen=100, allowed=None):
    """Returns the final non-dominated masks (P, D) bool over the full bank and their fitness (P, 2)."""
    problem = FeatureSelection(evaluator, n_features, allowed)
    algorithm = NSGA2(pop_size=pop_size, sampling=SparseSampling(), crossover=UniformCrossover(),
                      mutation=BitflipMutation(), eliminate_duplicates=True)
    res = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=False)
    return problem.full_masks(np.atleast_2d(res.X)), np.atleast_2d(res.F)


def nd_indices(F):
    return NonDominatedSorting().do(np.asarray(F), only_non_dominated_front=True)
