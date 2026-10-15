from dataclasses import asdict

import numpy as np
import pytest

import tevc.baselines as bl
from tevc.baselines import (DSR_PUBLISHED, METHODS, COTConfig, DSRConfig, SHOTConfig, UDAConfig, config_from_dict,
                            dsr_simulate, dsr_training_set, sample_configs, search, train, view)
from tevc.optimizers import B
from tevc.recipe import Recipe
from tevc.stream import Environment, Stream


def test_candidates_start_from_the_published_setting_and_repeat_within_a_run():
    for m in METHODS:
        c = sample_configs(m, 42)
        assert len(c) == B and c == sample_configs(m, 42)
        assert (c != sample_configs(m, 43)) == (m != "COT")          # COT has no hyperparameters
    assert sample_configs("DSR", 42)[0] == DSRConfig(0.02, 0.1, 0.02, 4) == DSR_PUBLISHED
    assert sample_configs("CDAN", 7)[0] == UDAConfig("CDAN", 1e-3, 1.0)
    assert sample_configs("SHOT", 7)[0] == SHOTConfig(1e-3, 0.3)


def test_result_entries_rebuild_their_configuration():
    for c in (DSR_PUBLISHED, COTConfig(), UDAConfig("DANN", 1e-3, 0.5), SHOTConfig(1e-3, 0.3),
              Recipe(2, 0.3, 0.8, 128, 0.0, 40, True)):
        assert config_from_dict({**asdict(c), "seed": 1, "fitness": 0.2}) == c


def test_search_evaluates_the_candidates_in_order_with_their_own_seeds(fake_evaluator):
    trials = search("CDAN", fake_evaluator, Environment("fake@1000", 1000.0, {}, {}), 7)
    assert [t.recipe for t in trials] == sample_configs("CDAN", 7)
    assert [t.seed for t in trials] == list(range(700, 730))


def _dominant_bin(x):
    return int(np.argmax(np.abs(np.fft.rfft(x))[1:])) + 1


@pytest.mark.parametrize("r", [0.8, 1.25])
def test_dsr_moves_every_frequency_by_the_pseudo_speed_ratio(r):
    n = 4096
    out = dsr_simulate(np.sin(2 * np.pi * 0.01 * np.arange(n)), r, 1.0, np.zeros(2 * n))
    assert len(out) == n and abs(_dominant_bin(out) - 0.01 * r * n) <= 1.5


def test_dsr_training_set_is_seeded_and_keeps_the_window_length():
    rng = np.random.default_rng(0)
    W, y = rng.standard_normal((3, 512)).astype(np.float32), np.array([0, 1, 1])
    a, ya = dsr_training_set(DSR_PUBLISHED, W, y, seed=3)
    b, _ = dsr_training_set(DSR_PUBLISHED, W, y, seed=3)
    assert a.shape == (3 * (1 + DSR_PUBLISHED.copies), 512) and np.array_equal(a, b)
    assert list(ya) == [0, 1, 1] + [0] * 4 + [1] * 8


def _cot_stream():
    env = Environment("fake@1000", 1000.0, {}, {})
    return Stream(None, [], {}, 600.0, (env,)), env


def test_cot_puts_target_windows_on_the_library_angular_sampling():
    stream, env = _cot_stream()
    W = np.zeros((2, 600), dtype=np.float32)
    W[:, ::60] = 1.0                                          # an impact every 60 samples at 1000 rpm
    out = view(COTConfig(), stream, env)(W)
    ac = np.correlate(out[0], out[0], "full")[out.shape[1] - 1:]
    assert out.shape == (2, 1000) and abs(int(np.argmax(ac[50:150])) + 50 - 100) <= 2      # every 100 at 600 rpm


def test_other_methods_see_the_windows_unchanged():
    stream, env = _cot_stream()
    W = np.ones((2, 600), dtype=np.float32)
    assert view(DSR_PUBLISHED, stream, env)(W) is W
    assert view(Recipe(1, 1.0, 1.0, 256, 0.0, 2, True), stream, env)(W) is W


def test_environment_independent_models_train_once_and_adaptation_sees_only_unlabelled_windows(monkeypatch):
    trained, adapted = [], []
    monkeypatch.setattr(bl, "train_cnn", lambda W, y, *a, **k: trained.append(len(W)) or object())
    monkeypatch.setattr(bl.uda, "train_uda", lambda Ws, ys, Wt, *a: adapted.append(Wt) or object())
    monkeypatch.setattr(bl.uda, "train_source", lambda Ws, ys, *a: trained.append("source") or object())
    monkeypatch.setattr(bl.uda, "train_shot", lambda Ws, ys, Wt, *a, source=None: adapted.append(source) or object())
    rng = np.random.default_rng(0)
    lib = {"W": rng.standard_normal((4, 64)).astype(np.float32), "y": np.array([0, 0, 1, 1])}
    u1, u2 = {"W": np.ones((3, 64), np.float32)}, {"W": np.zeros((3, 64), np.float32)}   # no "y": labels unreadable
    e1, e2 = Environment("a@1", 1.0, u1, {}), Environment("b@2", 2.0, u2, {})
    stream = Stream(None, [], lib, 1.0, (e1, e2))
    bl._MODELS.clear()
    try:
        for env in (e1, e2):
            for config in (DSR_PUBLISHED, COTConfig(), UDAConfig("DANN", 1e-3, 1.0), SHOTConfig(1e-3, 0.3)):
                train(config, stream, env, 5, "TICNN", 1)
    finally:
        bl._MODELS.clear()
    assert trained == [4 * (1 + DSR_PUBLISHED.copies), 4, "source"]
    assert adapted[0] is u1["W"] and adapted[2] is u2["W"] and adapted[1] is adapted[3]    # one SHOT source model
