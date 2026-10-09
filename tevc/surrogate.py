"""Environment-conditioned random-forest surrogate S(genes ⊕ d) -> fitness (spec §3, Reasoning).

d = ln(s_t / s_lib) lets evaluations from every past environment inform the current one. The acquisition is
optimistic: forest mean + kappa * spread across trees.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

MIN_POINTS = 6


class Surrogate:
    def __init__(self, seed=0, n_trees=200):
        self.forest = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=2, random_state=seed, n_jobs=-1)
        self.ready = False

    def fit(self, X, y):
        """Fit on rows [genes, d]; with fewer than MIN_POINTS evaluations the surrogate stays unused."""
        self.ready = len(y) >= MIN_POINTS
        if self.ready:
            self.forest.fit(np.asarray(X), np.asarray(y))
        return self

    def acquisition(self, genes, d, kappa=1.0):
        X = np.column_stack([np.asarray(genes), np.full(len(genes), d)])
        per_tree = np.stack([tree.predict(X) for tree in self.forest.estimators_])
        return per_tree.mean(axis=0) + kappa * per_tree.std(axis=0)
