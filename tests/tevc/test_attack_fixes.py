"""Tests of the attack-audit fixes (plan docs/superpowers/plans/2026-10-20-mssp-attack-fixes-code.md): cache keys,
step matching, RPST-DA arms, the new baselines, two-sided tests, the phase-vocoder variants, the R6 and R7 probes,
the bridge runner and the post-hoc fairness analysis."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.signal import hilbert

import tevc.baselines as bl
import tevc.rpstda as rd
from tevc import synthesis, tsm
from tevc.baselines import COTConfig, COTUDAConfig, EnvOrderConfig, RPSTDAConfig, UDAConfig, config_from_dict
from tevc.datasets import make_windows
from tevc.deep_train import predict
from tevc.dirg_data import DIRG
from tevc.dirg_protocol import family
from tevc.evaluator import RecipeEvaluator, legacy_view
from tevc.features import envelope_orders
from tevc.optimizers import B
from tevc.probes import bridge_runs as br
from tevc.probes import fidelity_band as fb
from tevc.probes import lee_posthoc as ph
from tevc.probes import validity_interval as vi
from tevc.probes.frame_validity import impact_periods, recovery
from tevc.recipe import Recipe
from tevc.run_stream import RUN_SEEDS, run_level_tests
from tevc.stream import Environment, Stream
from tevc.uda import train_uda


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


# ---------------------------------------------------------------------------------------------------- cache keys
def test_legacy_view_keeps_archived_rpstda_cot_and_uda_payloads():
    c = RPSTDAConfig("CDAN", 1e-3, 1.0, 3, 0.3, 0.8, 256, 80)
    assert legacy_view(c) == {"method": "CDAN", "lr": 1e-3, "weight": 1.0, "n_targets": 3, "q_lo": 0.3, "q_hi": 0.8,
                              "n_fft": 256, "copies": 80, "kind": "RPSTDA"}
    assert legacy_view(COTConfig()) == {"kind": "COT"}
    u = UDAConfig("CDAN", 1e-3, 1.0)
    assert legacy_view(u) == asdict(u)                     # a UDA learning rate of 1e-3 stays in its key
    r = Recipe(3, 0.3, 0.8, 256, 0.0, 80, True)
    assert legacy_view(r) == {k: v for k, v in asdict(r).items() if k != "tsm"}


def test_legacy_view_keeps_changed_new_fields():
    c = RPSTDAConfig("CDAN", 1e-3, 1.0, 3, 0.3, 0.8, 256, 80)
    assert legacy_view(replace(c, copies_beta=1.0))["copies_beta"] == 1.0
    assert legacy_view(replace(c, copies=0, steps_like_copies=80))["steps_like_copies"] == 80
    assert legacy_view(replace(c, tv_n_fft=16))["tv_n_fft"] == 16
    assert legacy_view(COTConfig(lr=3e-4)) == {"kind": "COT", "lr": 3e-4}
    assert legacy_view(Recipe(3, 0.3, 0.8, 256, 0.0, 80, True, "wsola"))["tsm"] == "wsola"


def test_models_with_their_own_predictor_bypass_the_network_predictor():
    ev = RecipeEvaluator.__new__(RecipeEvaluator)
    ev.predict_fn = lambda model, W: "network"

    class Own:
        def predict_windows(self, W):
            return "own"

    assert ev._predict(Own(), np.zeros((1, 4))) == "own"
    assert ev._predict(object(), np.zeros((1, 4))) == "network"


# --------------------------------------------------------------------------------------------- step matching
def _uda_data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 256)).astype(np.float32), np.array([0, 1] * (n // 2))


def test_total_steps_default_is_bit_identical():
    (Ws, ys), (Wt, _) = _uda_data(), _uda_data(seed=1)
    a = predict(train_uda(Ws, ys, Wt, "CDAN", 1.0, 1e-3, epochs=2, seed=5), Wt)
    b = predict(train_uda(Ws, ys, Wt, "CDAN", 1.0, 1e-3, epochs=2, seed=5, total_steps=2 * int(np.ceil(40 / 32))), Wt)
    assert np.allclose(a, b)


def test_total_steps_caps_and_extends_training(monkeypatch):
    (Ws, ys), (Wt, _) = _uda_data(), _uda_data(seed=1)
    calls, real = [], torch.nn.utils.clip_grad_norm_
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda p, m: calls.append(1) or real(p, m))
    train_uda(Ws, ys, Wt, "DANN", 1.0, 1e-3, epochs=1, seed=5, total_steps=3)      # stops inside the 2nd pass
    assert len(calls) == 3
    calls.clear()
    train_uda(Ws, ys, Wt, "DANN", 1.0, 1e-3, epochs=1, seed=5, total_steps=7)      # 2 steps a pass: 3.5 passes
    assert len(calls) == 7
    calls.clear()
    train_uda(Ws, ys, Wt, "DANN", 1.0, 1e-3, epochs=2, seed=5)                     # default: epochs x passes
    assert len(calls) == 4
    with pytest.raises(ValueError):
        train_uda(Ws[:0], ys[:0], Wt, "DANN", 1.0, 1e-3, epochs=1, total_steps=3)


# ------------------------------------------------------------------------------------------------ RPST-DA arms
def test_arms_change_only_their_own_field():
    base = rd.sample_configs(42)
    for arm, changed in (("RPST-DA", set()), ("RPST-DA-NC", {"copies"}), ("RPST-DA-B1", {"copies_beta"}),
                         ("RPST-DA-B05", {"copies_beta"}), ("RPST-DA-NCS", {"copies", "steps_like_copies"}),
                         ("RPST-DA-W0", {"weight"})):
        for c, a in zip(base, rd.sample_configs(42, transform=rd.ARMS[arm]), strict=True):
            assert {k for k, v in asdict(c).items() if asdict(a)[k] != v} <= changed
    ncs = rd.sample_configs(42, transform=rd.ARMS["RPST-DA-NCS"])
    assert all(a.copies == 0 and a.steps_like_copies == c.copies for c, a in zip(base, ncs))
    assert all(a.copies_beta == 1.0 for a in rd.sample_configs(42, transform=rd.ARMS["RPST-DA-B1"]))
    assert all(a.weight == 0.0 for a in rd.sample_configs(42, transform=rd.ARMS["RPST-DA-W0"]))


def test_nc_arm_equals_the_archived_no_copies_draws():
    for s in (42, 1337):
        assert rd.sample_configs(s, copies=False) == rd.sample_configs(s, transform=rd.ARMS["RPST-DA-NC"])
        assert rd.sample_configs(s) == rd.sample_configs(s, transform=rd.ARMS["RPST-DA"])


def test_b1_copies_are_plain_resampling(monkeypatch):
    stream, env = _stream([_noise(0), _noise(1)])
    calls = []
    monkeypatch.setattr("tevc.synthesis.transpose",
                        lambda y_, s0, tgt, beta=0.0, n_fft=256: calls.append(beta) or y_)
    rd.source_set(RPSTDAConfig("CDAN", 1e-3, 1.0, 2, 0.3, 0.8, 256, 20, copies_beta=1.0), stream, env)
    assert calls and set(calls) == {1.0}
    calls.clear()
    rd.source_set(RPSTDAConfig("CDAN", 1e-3, 1.0, 2, 0.3, 0.8, 256, 20), stream, env)
    assert set(calls) == {0.0}


def test_ncs_matches_the_twin_step_count(monkeypatch):
    stream, env = _stream([_noise(0), _noise(1)])
    seen = {}
    monkeypatch.setattr(rd.uda, "train_uda", lambda *a, **k: seen.update(k) or "model")
    rd.train(RPSTDAConfig("DANN", 1e-3, 1.0, 2, 0.3, 0.8, 256, 0, steps_like_copies=40), stream, env, 1, "TICNN", 40)
    n_train = len(rd.train_windows(stream)["y"])
    assert seen["total_steps"] == 40 * int(np.ceil((n_train + 2 * 40 * _DS.n_classes) / 32))
    seen.clear()
    rd.train(RPSTDAConfig("DANN", 1e-3, 1.0, 2, 0.3, 0.8, 256, 0), stream, env, 1, "TICNN", 40)
    assert seen["total_steps"] is None


def test_tv_frame_override(monkeypatch):
    stream, env = _stream([_noise(0), _noise(1)])
    calls = []
    monkeypatch.setattr("tevc.synthesis.transpose",
                        lambda y_, s0, tgt, beta=0.0, n_fft=256: calls.append(n_fft) or y_)
    rd.tv_set(stream, env)
    assert set(calls) == {rd.TV_N_FFT}
    calls.clear()
    bl.validation(RPSTDAConfig("CDAN", 1e-3, 1.0, 2, 0.3, 0.8, 16, 20, tv_n_fft=16), stream, env)
    assert set(calls) == {16}


def test_new_fields_are_validated():
    with pytest.raises(ValueError):
        RPSTDAConfig("CDAN", 1e-3, 1.0, 1, 0.3, 0.8, 256, 20, copies_beta=1.5)
    with pytest.raises(ValueError):
        RPSTDAConfig("CDAN", 1e-3, 1.0, 1, 0.3, 0.8, 256, 20, steps_like_copies=20)     # needs copies = 0
    with pytest.raises(ValueError):
        RPSTDAConfig("CDAN", 1e-3, 1.0, 1, 0.3, 0.8, 256, 0, tv_n_fft=-1)


def test_frame_rule_n_on_dirg_is_16():
    lib = 16690.6                                          # DIRG's label-free library speed (Table 3)
    stream = SimpleNamespace(dataset=DIRG, library_rpm=lib)
    p = min(impact_periods(DIRG, lib).values())
    assert 25 < p < 35                                     # inner-race impacts about 30 samples apart
    assert rd.frame_rule_n(stream, SimpleNamespace(rpm=lib), 0.64) == 16
    fast, slow = SimpleNamespace(rpm=24_000.0), SimpleNamespace(rpm=6_000.0)
    assert rd.frame_rule_n(stream, fast, 0.64, "target") == 2 ** int(np.floor(np.log2(0.64 * p / (24_000.0 / lib))))
    assert rd.frame_rule_n(stream, slow, 0.64, "target") == 16       # slowing down never lengthens the frame
    with pytest.raises(ValueError):
        rd.frame_rule_n(stream, slow, 0.64, "angle")


def test_new_fields_round_trip_through_result_files():
    for c in (RPSTDAConfig("CDAN", 1e-3, 1.0, 3, 0.3, 0.8, 256, 80, copies_beta=1.0),
              RPSTDAConfig("DANN", 1e-3, 0.0, 2, 0.3, 0.8, 16, 0, steps_like_copies=40, tv_n_fft=16),
              COTConfig(lr=3e-4), COTUDAConfig("CDAN", 1e-3, 1.0), EnvOrderConfig(0.5)):
        assert config_from_dict({**asdict(c), "seed": 1, "fitness": 0.2, "acc": 0.5}) == c


# --------------------------------------------------------------------------------------------- new baselines
GEOM = {"N_balls": 8, "d_mm": 6.75, "D_mm": 28.5, "alpha_deg": 0.0}


def _bearing_impacts(f_shaft, fs=20_000.0, seconds=1.0, f_res=3000.0, seed=0):
    """Outer-race impacts at BPFO of GEOM for shaft frequency f_shaft, ringing at f_res, over light noise."""
    from tevc.features import compute_fault_frequencies
    rate = compute_fault_frequencies(f_shaft_hz=f_shaft, **GEOM)["BPFO"]
    t = np.arange(int(fs * seconds)) / fs
    since = np.mod(t, 1.0 / rate)
    x = np.exp(-since * 400.0) * np.sin(2 * np.pi * f_res * since)
    return (x + 0.05 * np.random.default_rng(seed).standard_normal(len(t))).astype(np.float32)


def test_cot_cdan_draws_pair_with_cdan():
    for s in (42, 7):
        cot, cdan = bl.sample_configs("COT-CDAN", s), bl.sample_configs("CDAN", s)
        assert len(cot) == B and all(isinstance(c, COTUDAConfig) for c in cot)
        assert [(c.method, c.lr, c.weight) for c in cot] == [(c.method, c.lr, c.weight) for c in cdan]


def test_cot_t_starts_at_the_published_rate():
    cands = bl.sample_configs("COT-T", 42)
    assert cands[0] == COTConfig() and legacy_view(cands[0]) == {"kind": "COT"}
    assert len(cands) == B and all(bl.LR_RANGE[0] <= c.lr <= bl.LR_RANGE[1] for c in cands[1:])
    assert bl.sample_configs("COT-T", 42) == cands and bl.sample_configs("COT-T", 43) != cands
    env_c = bl.sample_configs("ENV-ORDER", 42)
    assert env_c[0] == EnvOrderConfig(1.0)
    assert all(bl.ENV_ORDER_C_RANGE[0] <= c.C <= bl.ENV_ORDER_C_RANGE[1] for c in env_c)


def test_cot_views_share_one_length():
    stream, _ = _stream([_noise(0), _noise(1)])                       # library 600 rpm, windows of 4000
    W = np.random.default_rng(0).standard_normal((3, _DS.win)).astype(np.float32)
    fast, slow = Environment("f", 1200.0, {"W": W}, {}), Environment("s", 400.0, {"W": W}, {})
    assert bl.cot_length(stream, fast) == _DS.win
    assert abs(bl.cot_length(stream, slow) - _DS.win * 400 / 600) <= 2
    for env in (fast, slow):
        V = bl.view(COTUDAConfig("CDAN", 1e-3, 1.0), stream, env)(W)
        assert V.shape == (3, bl.cot_length(stream, env))


def test_envelope_orders_follow_the_shaft_speed():
    fs, f_shaft = 20_000.0, 25.0
    W = np.stack([_bearing_impacts(f_shaft, seed=s) for s in range(2)])
    X = envelope_orders(W, fs, f_shaft, GEOM)
    assert X.shape == (2, 108) and np.isfinite(X).all()
    bpfo_full = 3                    # full band, BPFO, first harmonic (columns: bands x targets x harmonics)
    wrong = envelope_orders(W, fs, 1.3 * f_shaft, GEOM)
    assert (X[:, bpfo_full] > wrong[:, bpfo_full] + 1.0).all()


def test_env_order_model_returns_class_probabilities():
    from tevc import env_order
    fs, lib_shaft, env_shaft = 20_000.0, 25.0, 40.0
    W = np.stack([_noise(100 + s, 20_000) for s in range(6)] + [_bearing_impacts(lib_shaft, seed=s) for s in range(6)])
    ds = SimpleNamespace(fs=fs, geometry=GEOM, n_classes=2)
    stream = SimpleNamespace(dataset=ds, library={"W": W, "y": np.array([0] * 6 + [1] * 6),
                                                  "rec": np.array(["h"] * 6 + ["f"] * 6)},
                             library_rpm=60.0 * lib_shaft, src_rpm_of=None)
    model = env_order.train(EnvOrderConfig(1.0), stream, SimpleNamespace(rpm=60.0 * env_shaft))
    p = model.predict_windows(np.stack([_noise(150, 20_000), _bearing_impacts(env_shaft, seed=50)]))
    assert p.shape == (2, 2) and np.allclose(p.sum(axis=1), 1.0)
    assert p.argmax(axis=1).tolist() == [0, 1]             # the fault is recognised at a speed the library lacks


def test_exploratory_methods_leave_the_archived_list_alone():
    assert bl.METHODS == ("DSR", "COT", "DANN", "DEEP-CORAL", "CDAN", "SHOT")
    assert not set(bl.EXPLORATORY_METHODS) & set(bl.METHODS)
    assert min(bl.COT_T_STREAM, bl.ENV_ORDER_STREAM) >= len(bl.METHODS)
    assert rd.RNG_STREAM not in (bl.COT_T_STREAM, bl.ENV_ORDER_STREAM)


# ----------------------------------------------------------------------------------------------- two-sided tests
def _two_methods(diffs):
    return {"A": {str(i): [{"deployed_acc": 0.5 + d}] for i, d in enumerate(diffs)},
            "B": {str(i): [{"deployed_acc": 0.5}] for i, _ in enumerate(diffs)}}


def test_two_sided_p_doubles_the_extreme_one_sided_p():
    res = _two_methods([0.1] * 10)
    one = run_level_tests(res, "deployed_acc", ref="A")["B"]
    two = run_level_tests(res, "deployed_acc", ref="A", alternative="two-sided")["B"]
    assert np.isclose(one["p"], 1 / 1024) and np.isclose(two["p"], 2 / 1024) and two["alternative"] == "two-sided"
    worse = run_level_tests(_two_methods([-0.1] * 10), "deployed_acc", ref="A", alternative="two-sided")["B"]
    assert np.isclose(worse["p"], 2 / 1024)                # a consistent loss shows two-sided, never one-sided
    assert np.isclose(family(res, [("A", "B", "deployed_acc")], alternative="two-sided")[0]["p"], 2 / 1024)


def test_greater_is_the_archived_default():
    res = _two_methods([0.1, -0.05, 0.2, 0.1, 0.0, 0.1, 0.3, -0.1, 0.1, 0.1])
    default = run_level_tests(res, "deployed_acc", ref="A")["B"]["p"]
    assert default == run_level_tests(res, "deployed_acc", ref="A", alternative="greater")["B"]["p"]
    with pytest.raises(ValueError):
        run_level_tests(res, "deployed_acc", ref="A", alternative="less")


# ------------------------------------------------------------------------------------------ phase-vocoder variants
def _signal(n=8192, seed=0):
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    return (np.sin(2 * np.pi * 0.05 * t) + 0.5 * np.sin(2 * np.pi * 0.13 * t)
            + 0.1 * rng.standard_normal(n)).astype(np.float32)


@pytest.mark.parametrize("locked", [False, True])
def test_pv_is_the_identity_at_rate_one(locked):
    y = _signal()
    out = tsm.pv_tsm(y, 1.0, 256, locked=locked)
    assert len(out) == len(y)
    assert np.max(np.abs(out[512:-512] - y[512:-512])) < 1e-3 * np.max(np.abs(y))


def test_pv_output_length():
    y = _signal()
    for rate in (0.6, 1.5, 2.0):
        assert len(tsm.pv_tsm(y, rate, 128)) == int(len(y) / rate)
        assert len(tsm.pv_tsm(y, rate, 128, locked=True)) == int(len(y) / rate)


@pytest.mark.parametrize("locked", [False, True])
@pytest.mark.parametrize("rate", [0.7, 1.5])
def test_pv_keeps_a_sinusoid_frequency(locked, rate):
    y = np.sin(2 * np.pi * 0.05 * np.arange(16_384)).astype(np.float32)
    out = tsm.pv_tsm(y, rate, 256, locked=locked)
    spec = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    assert abs(np.fft.rfftfreq(len(out))[np.argmax(spec)] - 0.05) < 0.002


@pytest.mark.parametrize("locked", [False, True])
def test_short_frames_respace_impacts_in_both_variants(locked):
    x = vi.impact_train(400, 10.0, seed=0, n=40_000)
    out = tsm.pv_tsm(x, 2.0, 64, locked=locked)             # frame / period 0.16: impacts re-spaced to 200 samples
    found, spacing = recovery(out, 200, edge=1000)
    assert abs(found - 1) < 0.1 and abs(spacing - 1) < 0.1


def test_unlocked_pv_matches_librosa():
    y = _signal()
    for rate in (0.75, 1.5):
        ours, ref = tsm.pv_tsm(y, rate, 256), synthesis.transpose(y, 600.0, 600.0 * rate, 0.0, 256)
        assert ours.shape == ref.shape and np.max(np.abs(ours - ref)) < 1e-3 * np.max(np.abs(ref))
    assert np.array_equal(tsm.transpose_pv(y, 600.0, 900.0, beta=1.0), synthesis.resample(y, 600.0, 900.0))


# ------------------------------------------------------------------------------------------------ R6 probe
def test_decay_is_fixed_in_samples_across_periods():
    for period in (400, 800):
        e = np.abs(hilbert(vi.impact_train(period, 20.0, seed=0, n=8000, noise=0.0)))
        assert 0.08 < e[period + 42] / e[period + 2] < 0.2      # e^-2 = 0.135 after 40 samples, both periods


def test_an_impact_that_rounds_to_the_end_is_dropped():
    x = vi.impact_train(24.9, 2.0, seed=0, n=100, noise=0.0)     # 4 x 24.9 = 99.6 rounds to index 100
    assert x.shape == (100,)


def test_jitter_moves_impacts_but_keeps_the_mean_period():
    x0 = vi.impact_train(400, 20.0, seed=1, n=40_000, noise=0.0)
    xj = vi.impact_train(400, 20.0, seed=1, n=40_000, jitter=0.02, noise=0.0)
    found, spacing = recovery(xj, 400, edge=1000)
    assert abs(found - 1) < 0.1 and abs(spacing - 1) < 0.05 and not np.allclose(x0, xj)


def test_interval_reads_the_last_all_pass_and_first_none_pass_ratio():
    def cell(fr, ok, direction="up", r=1.5):
        v = 1.0 if ok else 0.5
        return {"frame_ratio": fr, "frame_ratio_target": fr * r, "found": v, "spacing": v, "direction": direction}

    cells = [cell(0.5, True), cell(0.5, True, "down", 0.8), cell(0.64, True), cell(0.64, True, "down", 0.8),
             cell(0.85, True), cell(0.85, False, "down", 0.8), cell(1.28, False), cell(1.28, False, "down", 0.8)]
    assert vi.interval(cells) == {"pass": 0.64, "fail": 1.28}
    assert vi.interval([c for c in cells if c["direction"] == "up"])["pass"] == 0.85
    both = vi.intervals(cells)
    assert both["source|0.1|all"] == {"pass": 0.64, "fail": 1.28} and "target|0.05|down" in both


def test_conditions_decouple_decay_from_period():
    conds = vi.conditions()
    assert conds["base"]["kappa"] == 5.0
    assert {"kappa=2.5", "kappa=10", "kappa=20", "tsm=pv_locked", "tsm=wsola", "hop_div=8"} <= set(conds)
    for name, c in conds.items():
        if name != "base":
            assert len({k for k in vi.BASE if k != "n_ffts" and c[k] != vi.BASE[k]}) == 1
    params = [vi.cell_params(256, fr, 5.0) for fr in vi.FRAME_RATIOS]
    assert {tau for _, tau in params} == {256 / 5.0} and len({p for p, _ in params}) == len(vi.FRAME_RATIOS)


# ------------------------------------------------------------------------------------------------ R7 probe
def _impulsive(fs=20_000.0, n=20_000, f_res=2500.0, rate=50.0, tone=6000.0, seed=0, noise=0.3):
    # noise as on a real rig: near noise-free, the bins beside the resonance carry only the impact onsets, which are
    # rarer than the ringing, so SK peaks there instead (0.05 gave 1250 Hz)
    t = np.arange(n) / fs
    since = np.mod(t, 1.0 / rate)
    x = np.exp(-since * 800.0) * np.sin(2 * np.pi * f_res * since) + 0.5 * np.sin(2 * np.pi * tone * t)
    return (x + noise * np.random.default_rng(seed).standard_normal(n)).astype(np.float32)


def test_sk_band_finds_the_impulsive_resonance():
    lo, hi, sk = fb.sk_band(np.stack([_impulsive(seed=s) for s in range(4)]), 20_000.0)
    assert lo <= 2500.0 <= hi and sk > 1.0                  # not the stationary tone at 6 kHz


def test_sk_band_skips_the_nyquist_bin():
    # the Nyquist bin is real-valued, so plain Gaussian noise gives it SK ~ 1 against ~0 for the complex bins
    W = np.random.default_rng(0).standard_normal((4, 20_000))
    lo, hi, _ = fb.sk_band(W, 20_000.0, windows=(32,))
    assert np.isclose(hi - lo, 2 * 20_000.0 / 32)            # an interior bin +- one bin, not clipped at fs / 2


def test_band_envelope_shows_the_impact_rate():
    fs = 20_000.0
    env = fb.band_envelope(np.stack([_impulsive(seed=s) for s in range(2)]), fs, 2000.0, 3000.0)
    spec = np.abs(np.fft.rfft(env[0] * np.hanning(env.shape[1])))
    f = np.fft.rfftfreq(env.shape[1], 1.0 / fs)
    sel = (f > 20) & (f < 200)
    assert abs(f[sel][np.argmax(spec[sel])] - 50.0) < 2.0


def test_energy_fraction_outside_the_band_is_small():
    tone = np.sin(2 * np.pi * 6000.0 * np.arange(20_000) / 20_000.0)[None, :]
    assert fb.band_energy_fraction(tone, 20_000.0, 2000.0, 3000.0) < 1e-3
    assert fb.band_energy_fraction(tone, 20_000.0, 5500.0, 6500.0) > 0.99


# ------------------------------------------------------------------------------------------------ bridge runner
def test_compare_cached_flags_mismatches_and_misses():
    entries = [{"env": "e", "trials": [{"kind": "COT", "seed": s, "acc": a} for s, a in ((1, 0.5), (2, 0.7), (3, 0.9))]}]

    def scorer(env, cfg, seed):
        if seed == 3:
            raise KeyError(seed)
        return {1: 0.5, 2: 0.6}[seed]

    r = br.compare_cached(entries, {"e": "ENV"}, scorer)
    assert r["checked"] == 3 and r["misses"] == [["e", 3]] and r["mismatches"] == [["e", 2, 0.7, 0.6]]


def test_frame_rule_arm_sets_both_frames():
    a = rd.frame_rule_arm(16, SimpleNamespace())(rd.PUBLISHED)
    assert (a.n_fft, a.tv_n_fft, a.copies) == (16, 16, rd.PUBLISHED.copies)
    b = rd.frame_rule_arm(16, SimpleNamespace(frame_scale=0.25))(rd.PUBLISHED)
    assert (b.n_fft, b.tv_n_fft) == (64, 64)               # recipe.frame maps 64 back to 16 samples at 16 kHz


def test_speed_error_streams_move_only_the_estimated_speed():
    stream, env = _stream([_noise(0), _noise(1)])
    p = br.perturbed(replace(stream, environments=(env, env)), 0.03)
    e1, e2 = p.environments
    assert e1 is e2                                        # a revisit stays the same environment
    assert e1.rpm == pytest.approx(env.rpm * 1.03) and e1.name != env.name and e1.T is env.T and e1.U is env.U
    assert p.library is stream.library and p.library_rpm == stream.library_rpm
    assert br.speed_label(-0.03) == "RPST-DA@-3%" and br.speed_label(0.03) == "RPST-DA@+3%"


def _visit(acc):
    trials = [{"acc": acc, "fitness": acc}, {"acc": acc + 0.05, "fitness": acc - 0.1}]
    return {"env": "e1", "deployed_acc": acc, "auc": acc, "n_evals": 2, "trials": trials, "deployed_acc_im": acc}


def test_bridge_family_is_two_sided():
    own = ("RPST-DA-B1", "RPST-DA-NCS", "RPST-DA-W0", "RPST-DA-B05", "COT-T", "COT-CDAN", "ENV-ORDER")
    assert [t[1] for t in br.tests_for("lee", own)] == ["RPST-DA-B1", "RPST-DA-NCS", "RPST-DA-W0", "COT-T",
                                                        "COT-CDAN", "ENV-ORDER", "DSR"]
    assert [t[1] for t in br.tests_for("dirg", ("RPST-DA-FR",))] == ["RPST-DA", "CDAN", "DANN"]
    results = {m: {str(s): [_visit(0.9 if m == "RPST-DA" else 0.7)] for s in RUN_SEEDS}
               for m in ("RPST-DA", "DSR") + own}
    out = br.analysis("lee", results, own)
    assert len(out["tests"]) == 7 and all(r["alternative"] == "two-sided" for r in out["tests"])
    assert np.isclose(out["tests"][0]["p"], 2 / 1024) and "RPST-DA-B1" in out["im_only"]


# --------------------------------------------------------------------------------------------- post-hoc fairness
def _pvisit(dep, im, best):
    return {"deployed_acc": dep, "deployed_acc_im": im, "trials": [{"acc": best}], "auc": dep, "n_evals": 30}


def _fake_lee():
    runs = range(10)
    return {"RPST-DA": {str(r): [_pvisit(0.9, 0.88, 1.0)] for r in runs},
            "RPST-DA-NC": {str(r): [_pvisit(0.74, 0.73, 0.95)] for r in runs},
            "CDAN": {str(r): [_pvisit(0.70, 0.70, 0.93)] for r in runs},
            "DSR": {str(r): [_pvisit(0.80 + 0.001 * r, 0.8, 0.88)] for r in runs}}


def test_posthoc_family_includes_dsr_and_is_two_sided():
    out = ph.posthoc(_fake_lee())
    assert out["strongest_comparator"] == "DSR"
    rows = {r["second"]: r for r in out["versus_every_method"]}
    assert set(rows) == {"RPST-DA-NC", "CDAN", "DSR"} and all(r["alternative"] == "two-sided" for r in rows.values())
    assert np.isclose(rows["DSR"]["p"], 2 / 1024)


def test_im_only_redeploy_reads_the_stored_deployment():
    out = ph.posthoc(_fake_lee())
    assert np.isclose(out["im_only"]["deployed"]["RPST-DA"], 0.88)
    pairs = {(r["first"], r["second"]) for r in out["im_only"]["tests"]}
    assert ("RPST-DA (IM only)", "DSR") in pairs and ("RPST-DA-NC (IM only)", "CDAN") in pairs


def test_selection_split_adds_up():
    s = ph.selection_split(_fake_lee())
    assert np.isclose(s["generation_gain"] + s["selection_gain"], s["deployed_gain"])
    assert np.isclose(s["generation_gain"], 0.05) and np.isclose(s["deployed_gain"], 0.16)
