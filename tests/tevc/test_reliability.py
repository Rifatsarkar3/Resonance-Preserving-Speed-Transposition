import json
from dataclasses import asdict

import numpy as np

from tevc.recipe import Recipe
from tevc.reliability import (add_metrics, ece, false_alarm_rate, macro_f1, merge, metrics, missed_fault_rate,
                              compare_to_agent)
from tevc.stream import Environment, Stream

Y = np.array([0, 0, 1, 1, 2, 2])
PRED = np.array([0, 1, 0, 1, 2, 0])


def test_missed_faults_are_faults_called_healthy():
    assert missed_fault_rate(Y, PRED) == 0.5          # windows 2 and 5 of the four faulty ones


def test_false_alarms_are_healthy_windows_called_faulty():
    assert false_alarm_rate(Y, PRED) == 0.5           # window 1 of the two healthy ones


def test_macro_f1_is_one_when_perfect_and_counts_never_predicted_classes_as_zero():
    assert macro_f1(Y, Y, 3) == 1.0
    assert np.isclose(macro_f1(Y, np.zeros_like(Y), 3), 0.5 / 3)    # class 0 F1 = 0.5, classes 1 and 2 F1 = 0


def test_ece_is_zero_when_confidence_matches_accuracy_and_measures_overconfidence():
    y = np.array([0] * 5 + [1] * 5)
    assert ece(np.eye(2)[y], y) == 0.0
    assert np.isclose(ece(np.tile([0.9, 0.1], (10, 1)), y), 0.4)    # 90 % confident, 50 % right


def test_metrics_bundle_uses_the_argmax_prediction():
    m = metrics(np.eye(3)[PRED] * 0.8 + 0.2 / 3, Y, 3)
    assert set(m) == {"missed_fault_rate", "false_alarm_rate", "macro_f1", "ece"}
    assert m["missed_fault_rate"] == 0.5 and m["false_alarm_rate"] == 0.5


class _DS:
    n_classes = 2


class _Cache:
    """cached(): perfect test predictions for seed 1, everything called healthy for any other seed."""

    def cached(self, env, config, seed):
        y = env.T["y"]
        return None, np.eye(2)[y if seed == 1 else np.zeros_like(y)]


def test_add_metrics_scores_the_deployed_model_from_the_cache():
    env = Environment("fake@1000", 1000.0, {}, {"y": np.array([0, 0, 1, 1])})
    r = asdict(Recipe(1, 1.0, 1.0, 256, 0.0, 2, True))
    results = {"A": {"42": [{"env": env.name, "deployed": {**r, "seed": 1}}]},
               "B": {"42": [{"env": env.name, "deployed": {**r, "seed": 2}}]}}
    add_metrics(results, _Cache(), Stream(_DS, [], {}, 600.0, (env,)))
    assert results["A"]["42"][0]["missed_fault_rate"] == 0.0 and results["B"]["42"][0]["missed_fault_rate"] == 1.0


def test_lower_is_better_metrics_are_tested_in_the_agents_favour():
    def entry(acc, rate):
        return {"deployed_acc": acc, "auc": acc, "macro_f1": acc, "missed_fault_rate": rate,
                "false_alarm_rate": rate, "ece": rate}

    results = {"AGENT": {str(r): [entry(0.9, 0.1)] for r in range(5)},
               "OTHER": {str(r): [entry(0.8, 0.3)] for r in range(5)}}
    t = compare_to_agent(results)
    assert t["missed_fault_rate"]["OTHER"]["mean_diff"] > 0 and t["missed_fault_rate"]["OTHER"]["wins"] == 5
    assert np.isclose(t["deployed_acc"]["OTHER"]["p"], 1 / 32)


def test_merge_lets_later_files_replace_methods(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"results": {"AGENT": {"1": []}, "GA": {"1": []}}}))
    b.write_text(json.dumps({"results": {"AGENT": {"2": []}}}))
    results, source = merge([a, b])
    assert results["AGENT"] == {"2": []} and source == {"AGENT": str(b), "GA": str(a)}
