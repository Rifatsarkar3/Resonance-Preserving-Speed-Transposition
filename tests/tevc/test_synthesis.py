import numpy as np

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
