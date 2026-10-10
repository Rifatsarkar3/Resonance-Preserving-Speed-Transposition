import inspect

import numpy as np

from tevc import agent as agent_module
from tevc import optimizers
from tevc.agent import PRIOR, VISIT_SEED_STEP, Agent
from tevc.optimizers import B, ei_search
from tevc.stream import Environment

E800, E1000 = Environment("fake@800", 800.0, {}, {}), Environment("fake@1000", 1000.0, {}, {})
ORDER = (E800, E1000, E800)                       # the 800-rpm regime recurs


def _run(evaluator, **flags):
    agent = Agent(600.0, (1, 2), run_seed=0, **flags)
    out = [agent.handle(evaluator, env, np.random.default_rng([0, i, 0]), np.random.default_rng([0, i, 1]))
           for i, env in enumerate(ORDER)]
    return agent, out


def _ga_start(evaluator):
    return [t.recipe for t in optimizers.ga_search(evaluator, E800, np.random.default_rng([0, 0, 0]).random((12, 7)),
                                                   np.random.default_rng(9), (1, 2))[:12]]


def test_full_budget_at_every_change_and_the_stored_model_redeployed_on_a_recurrence(fake_evaluator):
    agent, out = _run(fake_evaluator)
    assert [len(trials) for _, trials in out] == [B, B, B]
    assert out[2][0] == [agent.deployed[0][1]]        # the first-visit model is redeployed at once


def test_a_recurrence_retrains_the_regimes_best_recipe_with_a_fresh_seed(fake_evaluator):
    _, out = _run(fake_evaluator)
    assert {t.seed for t in out[2][1]} == {1 + VISIT_SEED_STEP}
    assert out[2][1][0].recipe == max(out[0][1], key=lambda t: t.fitness).recipe


def test_first_change_starts_from_the_rpst_prior_then_the_start_shared_with_ga_restart(fake_evaluator):
    _, out = _run(fake_evaluator)
    assert out[0][1][0].recipe == PRIOR
    assert [t.recipe for t in out[0][1][1:12]] == _ga_start(fake_evaluator)[:11]


def test_without_the_prior_the_first_change_shares_the_whole_start(fake_evaluator):
    _, out = _run(fake_evaluator, prior=False)
    assert [t.recipe for t in out[0][1][:12]] == _ga_start(fake_evaluator)


def test_a_new_regime_is_seeded_from_memory(fake_evaluator):
    agent, out = _run(fake_evaluator)
    assert out[1][1][0].recipe == agent.deployed[0][1].recipe   # nearest past deployed recipe comes first


def test_the_search_improves_on_its_start(fake_evaluator):
    _, out = _run(fake_evaluator, prior=False)
    assert max(t.fitness for t in out[0][1][12:]) > max(t.fitness for t in out[0][1][:12])


def test_ablations_keep_the_budget(fake_evaluator):
    for flags in ({"memory": False}, {"surrogate": False}, {"evolve": False}, {"prior": False}):
        _, out = _run(fake_evaluator, **flags)
        assert [len(trials) for _, trials in out] == [B, B, B]
        if flags == {"memory": False}:
            assert all(carried == [] for carried, _ in out)


def test_beta1_control_stays_at_plain_resampling(fake_evaluator):
    _, out = _run(fake_evaluator, beta=1.0)
    assert all(t.recipe.beta == 1.0 for _, trials in out for t in trials)


def test_ei_search_never_retrains_a_recipe_with_the_same_seed(fake_evaluator):
    trials = ei_search(fake_evaluator, E800, np.random.default_rng(0).random((4, 7)), np.random.default_rng(1), 3, 0.3)
    assert len(trials) == B and len({t.recipe for t in trials}) == B


def test_agent_code_never_touches_test_accuracy():
    for module in (agent_module, optimizers):
        src = inspect.getsource(module)
        assert "true_accuracy" not in src and "hidden_log" not in src and ".T[" not in src
