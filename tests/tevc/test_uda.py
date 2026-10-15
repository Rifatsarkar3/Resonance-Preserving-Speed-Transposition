import numpy as np
import pytest
import torch

from tevc.deep_train import predict, train_cnn

L = 256


def _data(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, L)).astype(np.float32), np.array([0, 1] * (n // 2))


def test_train_cnn_default_is_unchanged_by_the_label_smoothing_option():
    W, y = _data()
    a = predict(train_cnn(W, y, "TICNN", 1, 3), W)
    b = predict(train_cnn(W, y, "TICNN", 1, 3, label_smoothing=0.0), W)
    assert np.array_equal(a, b)
