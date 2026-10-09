import inspect

import numpy as np

from tevc import agent as agent_module
from tevc import optimizers
from tevc.agent import Agent
from tevc.stream import Environment

E800, E1000 = Environment("fake@800", 800.0, {}, {}), Environment("fake@1000", 1000.0, {}, {})
ORDER = (E800, E1000, E800)                       # the 800-rpm regime recurs


def _run(evaluator, beta=None):
    agent = Agent(600.0, (1, 2), run_seed=0, beta=beta)
    out = [agent.handle(evaluator, env, np.random.default_rng([0, i, 0]), np.random.default_rng([0, i, 1]))
           for i, env in enumerate(ORDER)]
    return agent, out


def test_full_budget_on_new_regimes_and_a_third_on_a_recurrence(fake_evaluator):
    agent, out = _run(fake_evaluator)
    assert [len(trials) for _, trials in out] == [30, 30, 10]
    assert out[2][0] == [agent.deployed[0][1]]        # the first-visit model is redeployed at once


def test_a_recurrence_trains_fresh_models_rather_than_cached_copies(fake_evaluator):
    _, out = _run(fake_evaluator)
    assert not {(t.recipe, t.seed) for t in out[0][1]} & {(t.recipe, t.seed) for t in out[2][1]}


def test_first_change_starts_from_the_random_population_shared_with_ga_restart(fake_evaluator):
    _, out = _run(fake_evaluator)
    ga = optimizers.ga_search(fake_evaluator, E800, np.random.default_rng([0, 0, 0]).random((12, 7)),
                              np.random.default_rng(9), (1, 2))
    assert [t.recipe for t in out[0][1][:12]] == [t.recipe for t in ga[:12]]


def test_second_change_is_seeded_from_memory(fake_evaluator):
    agent, out = _run(fake_evaluator)
    assert out[1][1][0].recipe == agent.deployed[0][1].recipe   # nearest past deployed recipe comes first


def test_beta1_control_stays_at_plain_resampling(fake_evaluator):
    _, out = _run(fake_evaluator, beta=1.0)
    assert all(t.recipe.beta == 1.0 for _, trials in out for t in trials)


def test_agent_code_never_touches_test_accuracy():
    for module in (agent_module, optimizers):
        src = inspect.getsource(module)
        assert "true_accuracy" not in src and "hidden_log" not in src and ".T[" not in src
