import numpy as np

from tevc.proxies import information_maximization


def test_im_is_log_c_for_confident_balanced_predictions():
    p = np.eye(4)[np.repeat(np.arange(4), 10)]
    assert np.isclose(information_maximization(p), np.log(4), atol=1e-6)


def test_im_is_zero_for_collapsed_predictions():
    p = np.tile([1.0, 0.0, 0.0, 0.0], (40, 1))
    assert np.isclose(information_maximization(p), 0.0, atol=1e-6)


def test_im_is_zero_for_uniform_predictions():
    p = np.full((40, 4), 0.25)
    assert np.isclose(information_maximization(p), 0.0, atol=1e-6)
