import numpy as np

from tevc.probes.operator_fidelity import _envelopes, _ringdown

FS = 50000.0


def _ringing_bursts(tau_s, period_s=0.05, seconds=0.5, f_res=3000.0):
    """Impacts every period_s exciting a resonance at f_res that rings down with time constant tau_s."""
    t = np.arange(int(seconds * FS)) / FS
    phase = np.mod(t, period_s)
    return (np.exp(-phase / tau_s) * np.sin(2 * np.pi * f_res * t))[None, :]


def test_ringdown_metric_recovers_the_decay_time_constant():
    for tau in (0.001, 0.002, 0.004):
        measured = _ringdown(_envelopes(_ringing_bursts(tau)), FS)
        assert abs(measured - tau) / tau < 0.25


def test_fidelity_by_env_transposes_each_entry_from_its_own_source_speed(monkeypatch):
    import tevc.probes.operator_fidelity as of
    from tevc.datasets import make_windows
    from tevc.stream import Environment, Stream

    class _DS:
        name, fs, win, hop, n_classes = "fake", 8000.0, 1000, 1000, 2

    rng = np.random.default_rng(0)
    entries = [(c, f"lib{c}", rng.standard_normal(8000).astype(np.float32), 0.0, 1.0) for c in range(2)]
    U = make_windows(entries, _DS.win, _DS.hop, 4)
    env = Environment("fake@1200", 1200.0, U, {})
    stream = Stream(_DS, entries, U, 600.0, (env, env), src_rpm_of=lambda rec: 500.0 + 100 * int(rec[-1]))
    calls = []
    monkeypatch.setattr(of.synthesis, "transpose", lambda sig, s0, s1, b, n: calls.append((s0, s1, b, n)) or sig)
    monkeypatch.setattr(of, "_compare", lambda synth, real, ds, f: {"n": float(len(synth["y"])), "f": f})
    out = of.fidelity_by_env(stream, {"none": None, "rpst": (0.0, 64)}, cap=3)
    assert list(out) == ["fake@1200"] and out["fake@1200"]["speed_ratio"] == 2.0
    assert out["fake@1200"]["operators"]["none"] == {"n": 6.0, "f": 20.0}
    assert calls == [(500.0, 1200.0, 0.0, 64), (600.0, 1200.0, 0.0, 64)]
