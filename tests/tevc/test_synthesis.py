import numpy as np
from scipy.signal import hilbert

from tevc import synthesis


def _signal(n=40000, seed=0):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def test_transpose_beta0_equals_rpst():
    y = _signal()
    assert np.array_equal(synthesis.transpose(y, 600, 1000, beta=0.0, n_fft=256), synthesis.rpst(y, 600, 1000))


def test_transpose_beta1_equals_resample():
    y = _signal()
    assert np.array_equal(synthesis.transpose(y, 600, 1000, beta=1.0), synthesis.resample(y, 600, 1000))


def test_transpose_length_scales_inversely_with_speed_ratio():
    y = _signal()
    out = synthesis.transpose(y, 900, 1500, beta=0.3, n_fft=128)
    assert abs(len(out) / len(y) - 900 / 1500) < 0.005


FS = 50000.0


def _ringing_impacts(period_s=0.01, tau_s=0.0005, f_res=4000.0, seconds=1.0):
    t = np.arange(int(seconds * FS)) / FS
    return (np.exp(-np.mod(t, period_s) / tau_s) * np.sin(2 * np.pi * f_res * t)).astype(np.float32)


def _envelope_rate_hz(y, lo=50.0, hi=400.0):
    env = np.abs(hilbert(y))
    spec = np.abs(np.fft.rfft((env - env.mean()) * np.hanning(len(env))))
    f = np.fft.rfftfreq(len(env), 1 / FS)
    band = (f >= lo) & (f <= hi)
    return f[band][np.argmax(spec[band])]


def test_wsola_keeps_the_resonance_and_scales_the_impulse_rate():
    out = synthesis.wsola(_ringing_impacts(), 1.5, frame=128)
    spec = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    assert abs(np.fft.rfftfreq(len(out), 1 / FS)[np.argmax(spec)] - 4000.0) < 50       # resonance stays
    assert abs(_envelope_rate_hz(out) - 150.0) < 5                                      # 100 -> 150 impacts/s
    assert abs(len(out) - int(FS / 1.5)) <= 1


def test_wsola_keeps_the_ring_down_of_jittered_noisy_impacts():
    from tevc.probes.operator_fidelity import _envelopes, _ringdown
    rng = np.random.default_rng(0)
    t = np.arange(int(FS)) / FS
    impacts = np.cumsum(0.01 * (1 + 0.02 * rng.standard_normal(110)))             # +-2 % slip
    since = t - impacts[np.clip(np.searchsorted(impacts, t) - 1, 0, None)]
    y = (np.exp(-np.clip(since, 0, None) / 0.0005) * np.sin(2 * np.pi * 4000 * t)
         + 0.05 * rng.standard_normal(len(t))).astype(np.float32)
    tau = _ringdown(_envelopes(y[None, :]), FS)
    assert abs(_ringdown(_envelopes(synthesis.wsola(y, 1.5, frame=128)[None, :]), FS) - tau) / tau < 0.3
