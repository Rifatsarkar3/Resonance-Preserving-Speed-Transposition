import pytest

from tevc.evaluator import Evaluation
from tevc.recipe import Recipe


class FakeEvaluator:
    """Agent-visible fitness only, no training: rewards small beta, many targets and q_hi near 0.9. Baseline
    candidates (non-recipes) get a seed-dependent fitness and true accuracy 0.5."""

    def __init__(self):
        self.calls = []

    def evaluate(self, env, recipe, seed):
        self.calls.append((env.name, recipe, seed))
        if not isinstance(recipe, Recipe):
            return Evaluation(0.5 + 1e-3 * (seed % 7), seed, 0.0, False)
        f = 1.0 - recipe.beta + 0.1 * recipe.n_targets - abs(recipe.q_hi - 0.9) + 1e-3 * (seed % 7)
        return Evaluation(f, seed, 0.0, False)

    def true_accuracy(self, env, recipe, seed):     # scoring side (driver / oracle), never the agent
        return 1.0 - recipe.beta if isinstance(recipe, Recipe) else 0.5

    def dfid(self, env, recipe, seed):              # secondary fitness, deliberately unlike evaluate()
        return -abs(recipe.q_lo - 0.5), False

    def tv_accuracy(self, env, config, seed):       # RPST-DA's label-free validation score, unlike fitness
        return (seed % 5) / 5


@pytest.fixture
def fake_evaluator():
    return FakeEvaluator()
