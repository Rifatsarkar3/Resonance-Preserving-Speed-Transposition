import pytest

from tevc.evaluator import Evaluation


class FakeEvaluator:
    """Agent-visible fitness only, no training: rewards small beta, many targets and q_hi near 0.9."""

    def __init__(self):
        self.calls = []

    def evaluate(self, env, recipe, seed):
        self.calls.append((env.name, recipe, seed))
        f = 1.0 - recipe.beta + 0.1 * recipe.n_targets - abs(recipe.q_hi - 0.9) + 1e-3 * (seed % 7)
        return Evaluation(f, seed, 0.0, False)


@pytest.fixture
def fake_evaluator():
    return FakeEvaluator()
