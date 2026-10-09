import numpy as np

from tevc.optimizers import Trial
from tevc.recipe import Recipe
from tevc.run_stream import OracleEvaluator, run_method, score
from tevc.stream import Environment, Stream

E800, E1000 = Environment("fake@800", 800.0, {}, {}), Environment("fake@1000", 1000.0, {}, {})
STREAM = Stream(None, [], {}, 600.0, (E800, E1000, E800))


def _trial(beta, fitness):
    return Trial(Recipe(2, 0.5, 0.9, 128, beta, 40, True), 1, fitness)


def test_score_deploys_the_highest_fitness_model_so_far(fake_evaluator):
    trials = [_trial(0.9, 0.1), _trial(0.2, 0.5), _trial(0.5, 0.3)]     # fake true accuracy = 1 - beta
    s = score(fake_evaluator, STREAM, E800, [], trials, budget=4)
    assert np.allclose(s["curve"], [0.1, 0.8, 0.8, 0.8]) and np.isclose(s["deployed_acc"], 0.8)
    assert np.isclose(s["auc"], np.mean([0.1, 0.8, 0.8, 0.8])) and s["n_evals"] == 3


def test_a_carried_model_counts_from_the_first_step(fake_evaluator):
    s = score(fake_evaluator, STREAM, E800, [_trial(0.0, 0.9)], [_trial(0.5, 0.2)], budget=2)
    assert np.allclose(s["curve"], [1.0, 1.0]) and s["n_carried"] == 1 and s["n_evals"] == 1


def test_oracle_fitness_is_true_accuracy(fake_evaluator):
    assert OracleEvaluator(fake_evaluator).evaluate(E800, Recipe(1, 1.0, 1.0, 256, 0.25, 20, True), 3).fitness == 0.75


def test_searches_share_their_first_change_start(fake_evaluator):
    first = {m: [t.recipe for t in run_method(m, STREAM, fake_evaluator, 42)[0][1][:12]]
             for m in ("RANDOM", "GA-RESTART", "AGENT", "BO", "ORACLE-FITNESS", "WARM-TRANSFER")}
    assert all(v == first["GA-RESTART"] for v in first.values())


def test_every_search_spends_the_budget_per_change(fake_evaluator):
    for m in ("RANDOM", "GA-RESTART", "BO", "WARM-TRANSFER", "ORACLE-FITNESS"):
        assert [len(trials) for _, trials in run_method(m, STREAM, fake_evaluator, 7)] == [30, 30, 30]
