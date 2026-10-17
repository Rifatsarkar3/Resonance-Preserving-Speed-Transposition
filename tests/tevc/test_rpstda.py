from dataclasses import asdict

import numpy as np
import pytest
from scipy.signal import hilbert

import tevc.baselines as bl
import tevc.rpstda as rd
from tevc.baselines import RPSTDAConfig, config_from_dict
from tevc.datasets import make_windows
from tevc.optimizers import B, Trial
from tevc.stream import Environment, Stream


class _DS:
    name, fs, win, hop, cap_fit, n_classes = "fake", 8000.0, 4000, 2000, 80, 2


def _stream(signals, env_rpm=1200.0, U=None):
    """Two-class fake stream with a 600-rpm library; signals[c] is class c's library record (span 0-1)."""
    entries = [(c, f"lib{c}", s, 0.0, 1.0) for c, s in enumerate(signals)]
    env = Environment(f"fake@{env_rpm:g}", env_rpm,
                      U if U is not None else {"W": np.zeros((4, _DS.win), np.float32)}, {})
    return Stream(_DS, entries, make_windows(entries, _DS.win, _DS.hop, _DS.cap_fit), 600.0, (env,)), env


def _noise(seed, n=100_000):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def test_the_speed_guard_forbids_training_copies_at_the_target_speed():
    with pytest.raises(ValueError):
        RPSTDAConfig("CDAN", 1e-3, 1.0, 1, 0.5, 0.9, 256, 20)
    with pytest.raises(ValueError):
        RPSTDAConfig("SHOT", 1e-3, 1.0, 1, 0.5, 0.6, 256, 20)
    assert RPSTDAConfig("DANN", 1e-3, 1.0, 1, 0.8, 0.8, 256, 0).copies == 0          # the no-copies arm


def test_result_entries_rebuild_rpstda_candidates_and_the_baseline_list_is_unchanged():
    c = RPSTDAConfig("CDAN", 1e-3, 1.0, 3, 0.3, 0.8, 256, 80)
    assert config_from_dict({**asdict(c), "seed": 3, "fitness": 0.1}) == c
    assert bl.METHODS == ("DSR", "COT", "DANN", "DEEP-CORAL", "CDAN", "SHOT")


def test_split_cuts_every_library_record_at_80_percent_of_its_span():
    train, val = rd.split_entries([(1, "r", np.arange(10_000, dtype=np.float32), 0.0, 0.5)])
    (lt, rt, st, lot, hit), (lv, rv, sv, lov, hiv) = train[0], val[0]
    assert (lt, rt, lot, hit) == (1, "r", 0.0, 1.0) == (lv, rv, lov, hiv)
    assert (st[0], st[-1], sv[0], sv[-1]) == (0, 3999, 4000, 4999)


def test_train_windows_are_the_library_windows_before_the_cut():
    stream, _ = _stream([_noise(0), _noise(1)])
    w = rd.train_windows(stream)
    assert (w["span_hi"] <= 80_000).all() and np.bincount(w["y"]).tolist() == [39, 39]
    lib = {(r, lo) for r, lo in zip(stream.library["rec"], stream.library["span_lo"])}
    assert {(r, lo) for r, lo in zip(w["rec"], w["span_lo"])} <= lib


def test_without_copies_the_source_is_the_train_part_alone():
    stream, env = _stream([_noise(0), _noise(1)])
    W, y = rd.source_set(RPSTDAConfig("DANN", 1e-3, 1.0, 2, 0.3, 0.8, 256, 0), stream, env)
    w = rd.train_windows(stream)
    assert np.array_equal(W, w["W"]) and np.array_equal(y, w["y"])


def test_copies_come_from_the_train_part_at_intermediate_speeds(monkeypatch):
    t = np.arange(100_000) / 8000.0
    sig = np.where(t < 10.0, np.sin(2 * np.pi * 500 * t), np.sin(2 * np.pi * 2000 * t)).astype(np.float32)
    stream, env = _stream([sig, sig.copy()])
    W, y = rd.source_set(RPSTDAConfig("CDAN", 1e-3, 1.0, 2, 0.3, 0.8, 256, 20), stream, env)
    assert np.bincount(y).tolist() == [39 + 2 * 20] * 2
    spec = np.abs(np.fft.rfft(W, axis=1))                       # 2 Hz bins: 500 Hz -> 250, 2000 Hz -> 1000
    assert (spec[:, 1000] < 1e-2 * spec[:, 250]).all()         # nothing from the validation part's 2000 Hz tone
    calls = []
    monkeypatch.setattr("tevc.synthesis.transpose",
                        lambda y_, s0, tgt, beta=0.0, n_fft=256: calls.append((s0, tgt, beta, n_fft)) or y_)
    rd.source_set(RPSTDAConfig("CDAN", 1e-3, 1.0, 2, 0.3, 0.8, 128, 20), stream, env)
    assert sorted({c[1] for c in calls}) == pytest.approx([600 * 2 ** 0.3, 600 * 2 ** 0.8])
    assert {(c[0], c[2], c[3]) for c in calls} == {(600.0, 0.0, 128)}


def test_copies_are_balanced_across_classes_when_records_differ_in_length():
    stream, env = _stream([_noise(0), _noise(1, n=60_000)])           # class 1's record is shorter
    lib = rd.train_windows(stream)
    W, y = rd.source_set(RPSTDAConfig("CDAN", 1e-3, 1.0, 2, 0.3, 0.8, 256, 80), stream, env)
    n_lib = len(lib["y"])
    assert np.array_equal(W[:n_lib], lib["W"]) and np.bincount(y[n_lib:]).tolist() == [30, 30]   # 18 + 12 for class 1


def _ringing(n, start, period, f_res=1000.0, tau=32.0):
    """Impacts every `period` samples from `start` on, each ringing at f_res Hz (decay tau samples), over a 2 % noise
    floor (without one the phase vocoder's phase drifts through the silent gaps and every other impact cancels)."""
    k = np.arange(n)
    since = np.where(k >= start, np.mod(k - start, period), np.inf)
    noise = 0.02 * np.random.default_rng(0).standard_normal(n)
    return (np.exp(-since / tau) * np.sin(2 * np.pi * f_res * k / _DS.fs) + noise).astype(np.float32)


def test_tv_set_transposes_only_the_validation_part_to_the_environment_speed():
    sig = _ringing(100_000, start=80_000, period=1000)         # validation part only: an impact every 1000 samples
    stream, env = _stream([sig, sig.copy()], env_rpm=1200.0)
    V = rd.tv_set(stream, env)
    assert np.bincount(V["y"]).tolist() == [7, 7]          # 20 000 / 2 samples, windows of 4000 at hop 1000
    x = V["W"][3]
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    assert abs(np.argmax(spec) * _DS.fs / len(x) - 1000.0) < 20         # the resonance stays at 1000 Hz
    e = np.abs(hilbert(x))
    ac = np.correlate(e - e.mean(), e - e.mean(), "full")[len(x) - 1:]
    hop = rd.TV_N_FFT // 4                                               # impacts land on the vocoder's hop grid
    assert abs(int(np.argmax(ac[300:700])) + 300 - 500) <= hop // 2      # an impact every 500 samples at 1200 rpm


def test_tv_set_refuses_a_class_without_validation_windows():
    stream, env = _stream([_noise(0), _noise(1, n=20_000)])          # class 1: 4000-sample validation part
    with pytest.raises(ValueError, match="classes \\[1\\]"):
        rd.tv_set(stream, env)


def test_validation_hook_is_defined_for_rpstda_only():
    stream, env = _stream([_noise(0), _noise(1)])
    assert bl.validation(bl.COTConfig(), stream, env) is None
    assert set(bl.validation(rd.PUBLISHED, stream, env)) >= {"W", "y"}


def test_baselines_train_adapts_the_source_set_to_the_unlabelled_buffer(monkeypatch):
    seen = {}

    def fake_uda(Ws, ys, Wt, method, weight, lr, model_name, epochs, seed):
        seen.update(Ws=Ws, Wt=Wt, args=(method, weight, lr, model_name, epochs, seed))
        return "model"

    monkeypatch.setattr(rd.uda, "train_uda", fake_uda)
    U = {"W": np.ones((5, _DS.win), np.float32)}                       # no "y": target labels unreadable
    stream, env = _stream([_noise(0), _noise(1)], U=U)
    model, W = bl.train(RPSTDAConfig("CDAN", 2e-3, 0.5, 1, 0.5, 0.5, 128, 0), stream, env, 7, "TICNN", 3)
    assert model == "model" and seen["Wt"] is U["W"] and seen["args"] == ("CDAN", 0.5, 2e-3, "TICNN", 3, 7)
    assert np.array_equal(W, seen["Ws"]) and len(W) == 2 * 39


def test_candidates_start_from_cdan_with_the_earlier_recipe_and_repeat_within_a_run():
    c = rd.sample_configs(42)
    assert len(c) == B and c == rd.sample_configs(42) and c != rd.sample_configs(43)
    assert c[0] == RPSTDAConfig("CDAN", 1e-3, 1.0, 3, 0.3, 0.8, 256, 80) == rd.PUBLISHED
    assert {x.method for x in c} == {"DANN", "CDAN"} and all(0.2 <= x.q_lo <= x.q_hi <= 0.8 for x in c[1:])
    assert rd.RNG_STREAM not in range(len(bl.METHODS))                 # not one of the baselines' streams


def test_search_trains_candidate_k_with_seed_run_seed_times_100_plus_k(fake_evaluator):
    trials = rd.search(fake_evaluator, Environment("fake@1000", 1000.0, {}, {}), 7)
    assert [t.recipe for t in trials] == rd.sample_configs(7) and [t.seed for t in trials] == list(range(700, 730))


def _t(fitness, seed):
    return Trial(rd.PUBLISHED, seed, fitness)


def test_rank_rule_deploys_the_best_mean_rank_even_when_it_tops_neither_score():
    pool, tv = [_t(0.9, 0), _t(0.8, 1), _t(0.1, 2), _t(0.5, 3)], {0: 0.1, 1: 0.8, 2: 0.9, 3: 0.5}
    assert rd.rank_deploy(lambda t: tv[t.seed])(pool).seed == 1


def test_rank_rule_ties_go_to_the_earliest_candidate():
    pool, tv = [_t(0.9, 0), _t(0.5, 1), _t(0.7, 2)], {0: 0.1, 1: 0.9, 2: 0.7}     # mean ranks 2, 2, 2
    assert rd.rank_deploy(lambda t: tv[t.seed])(pool).seed == 0
