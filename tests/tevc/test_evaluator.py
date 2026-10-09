from dataclasses import fields

import numpy as np

from tevc.evaluator import Evaluation, RecipeEvaluator
from tevc.recipe import Recipe
from tevc.stream import Environment, Stream


class _DS:
    name, win, hop = "fake", 1000, 500


def _windows(n, seed):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(2), n // 2)
    return {"W": rng.standard_normal((len(y), 1000)).astype(np.float32), "y": y,
            "rec": np.array([f"r{seed}"] * len(y)), "span_lo": np.arange(len(y)) * 1000.0,
            "span_hi": np.arange(len(y)) * 1000.0 + 1000.0}


def _stream():
    rng = np.random.default_rng(0)
    entries = [(c, f"lib{c}", rng.standard_normal(20000).astype(np.float32), 0.0, 1.0) for c in range(2)]
    env = Environment("fake@1000", 1000.0, _windows(8, 1), _windows(8, 2))
    return Stream(_DS, entries, _windows(8, 3), 600.0, (env,))


def _fake_train(W, y, model_name, epochs, seed):
    return {"seed": seed}


def _fake_predict(model, W):
    p = np.zeros((len(W), 2))
    p[: len(W) // 2, 0] = 1.0
    p[len(W) // 2:, 1] = 1.0          # perfect and confident on the balanced fake sets
    return p


def _uniform_weights(env):
    return np.ones(len(env.U["W"]))


def _evaluator(tmp_path, weights_fn=_uniform_weights):
    return RecipeEvaluator(_stream(), epochs=1, cache_dir=tmp_path, train_fn=_fake_train, predict_fn=_fake_predict,
                           weights_fn=weights_fn)


RECIPE = Recipe(1, 1.0, 1.0, 256, 0.0, 2, True)


def test_agent_facing_result_has_no_accuracy():
    assert {f.name for f in fields(Evaluation)} == {"fitness", "seed", "cost_s", "cached"}


def test_evaluate_logs_hidden_accuracy_and_caches(tmp_path):
    ev = _evaluator(tmp_path)
    env = ev.stream.environments[0]
    first, second = ev.evaluate(env, RECIPE, seed=42), ev.evaluate(env, RECIPE, seed=42)
    assert not first.cached and second.cached and first.fitness == second.fitness
    assert np.isclose(first.fitness, np.log(2))
    assert ev.hidden_log[0]["true_acc"] == 1.0 and len(ev.hidden_log) == 2


def test_different_seed_is_a_different_cache_entry(tmp_path):
    ev = _evaluator(tmp_path)
    env = ev.stream.environments[0]
    ev.evaluate(env, RECIPE, seed=1)
    assert not ev.evaluate(env, RECIPE, seed=2).cached


def test_fitness_uses_buffer_weights_computed_once_per_environment(tmp_path):
    calls = []

    def first_class_only(env):                 # all mass on the first half of U -> one predicted class -> IM = 0
        calls.append(env.name)
        return np.r_[np.ones(len(env.U["W"]) // 2), np.zeros(len(env.U["W"]) // 2)]

    ev = _evaluator(tmp_path, weights_fn=first_class_only)
    env = ev.stream.environments[0]
    assert np.isclose(ev.evaluate(env, RECIPE, seed=1).fitness, 0.0, atol=1e-9)
    ev.evaluate(env, RECIPE, seed=2)
    assert calls == [env.name]


def test_phase_vocoder_recipes_keep_their_pre_tsm_cache_keys(tmp_path):
    import hashlib
    import json
    from dataclasses import asdict, replace

    from tevc.evaluator import _fingerprint
    ev = _evaluator(tmp_path)
    env = ev.stream.environments[0]
    legacy = {k: v for k, v in asdict(RECIPE).items() if k != "tsm"}
    payload = json.dumps([ev.stream.dataset.name, env.name, _fingerprint(ev.stream.library, env.U, env.T), legacy,
                          42, ev.model_name, ev.epochs], sort_keys=True)
    assert ev._key(env, RECIPE, 42) == hashlib.sha1(payload.encode()).hexdigest()[:16]
    assert ev._key(env, replace(RECIPE, tsm="wsola"), 42) != ev._key(env, RECIPE, 42)


def test_true_accuracy_is_scored_on_the_test_set(tmp_path):
    ev = _evaluator(tmp_path)
    env = ev.stream.environments[0]
    assert ev.true_accuracy(env, RECIPE, seed=3) == 1.0
