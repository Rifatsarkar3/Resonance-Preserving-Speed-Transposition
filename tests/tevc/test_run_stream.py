import numpy as np
import pytest

from tevc.agent import PRIOR
from tevc.optimizers import Trial
from tevc.recipe import Recipe
from tevc.run_stream import AGENT_VARIANTS, DFIDEvaluator, OracleEvaluator, main, report, run_level_tests, run_method, score
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


def test_dfid_evaluator_reports_dfid_as_fitness(fake_evaluator):
    r = Recipe(1, 0.8, 1.0, 256, 0.25, 20, True)
    assert np.isclose(DFIDEvaluator(fake_evaluator).evaluate(E800, r, 3).fitness, -0.3)


def test_agent_dfid_searches_and_deploys_on_dfid(fake_evaluator):
    trials = [t for _, ts in run_method("AGENT-DFID", STREAM, fake_evaluator, 42) for t in ts]
    assert trials and all(t.fitness == fake_evaluator.dfid(None, t.recipe, t.seed)[0] for t in trials)


def test_searches_share_their_first_change_start(fake_evaluator):
    first = {m: [t.recipe for t in run_method(m, STREAM, fake_evaluator, 42)[0][1][:12]]
             for m in ("RANDOM", "GA-RESTART", "BO", "ORACLE-FITNESS", "WARM-TRANSFER", "AGENT-NOPRIOR",
                       "AGENT", "AGENT-DFID")}
    assert all(first[m] == first["GA-RESTART"] for m in ("RANDOM", "BO", "ORACLE-FITNESS", "WARM-TRANSFER",
                                                         "AGENT-NOPRIOR"))
    assert all(first[m] == [PRIOR] + first["GA-RESTART"][:11] for m in ("AGENT", "AGENT-DFID"))


def _results(diffs):
    """AGENT leads OTHER by diffs[r] in run r; two environments per run so pooling would double-count."""
    return {"AGENT": {str(r): [{"deployed_acc": 0.5 + d}] * 2 for r, d in enumerate(diffs)},
            "OTHER": {str(r): [{"deployed_acc": 0.5}] * 2 for r in range(len(diffs))}}


def test_run_level_test_counts_runs_not_environments():
    t = run_level_tests(_results([0.1] * 5), "deployed_acc")["OTHER"]
    assert t["runs"] == 5 and np.isclose(t["p"], 1 / 32) and t["wins"] == 5


def test_run_level_test_gives_no_credit_for_mixed_signs():
    assert run_level_tests(_results([0.1, -0.1, 0.1, -0.1]), "deployed_acc")["OTHER"]["p"] > 0.4


def test_report_lists_only_the_runs_done():
    env = {"deployed_acc": 0.9, "auc": 0.8, "n_evals": 30}
    results = {"AGENT": {"42": [env], "1337": [env]}, "GA-RESTART": {"42": [env]}, "RANDOM": {}}
    assert report("jnu", ["AGENT", "GA-RESTART", "RANDOM"], results, 0.0)["run_seeds"] == [42, 1337]


def test_every_search_spends_the_budget_per_change(fake_evaluator):
    for m in ("RANDOM", "GA-RESTART", "BO", "WARM-TRANSFER", "ORACLE-FITNESS", *AGENT_VARIANTS):
        assert [len(trials) for _, trials in run_method(m, STREAM, fake_evaluator, 7)] == [30, 30, 30]


def test_ottawa_runs_only_through_the_frozen_protocol(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_stream", "--dataset", "ottawa"])
    with pytest.raises(SystemExit, match="tevc.confirmatory"):
        main()
