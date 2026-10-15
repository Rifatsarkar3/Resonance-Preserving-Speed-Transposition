import numpy as np
import pytest
import torch

from tevc.deep_train import predict, train_cnn
from tevc.models import TICNN
from tevc.uda import (centroid_labels, coral, entropy_weights, features_and_logits, grad_reverse, grl_coefficient,
                      multilinear)

L = 256


def _data(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, L)).astype(np.float32), np.array([0, 1] * (n // 2))


def test_train_cnn_default_is_unchanged_by_the_label_smoothing_option():
    W, y = _data()
    a = predict(train_cnn(W, y, "TICNN", 1, 3), W)
    b = predict(train_cnn(W, y, "TICNN", 1, 3, label_smoothing=0.0), W)
    assert np.array_equal(a, b)


def test_features_and_logits_reproduce_the_model_output():
    model = TICNN(signal_length=L, n_classes=3).eval()
    x = torch.randn(4, L)
    _, logits = features_and_logits(model, x)
    assert torch.allclose(logits, model(x))


def test_gradient_reversal_flips_and_scales_the_gradient():
    x = torch.ones(3, requires_grad=True)
    grad_reverse(x, 0.5).sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -0.5))


def test_grl_coefficient_rises_from_zero_towards_one():
    assert grl_coefficient(0.0) == 0.0 and 0.99 < grl_coefficient(1.0) < 1.0


def test_coral_is_zero_for_one_batch_and_positive_between_different_ones():
    torch.manual_seed(0)
    a, b = torch.randn(16, 8), 3.0 * torch.randn(16, 8)
    assert coral(a, a).item() == 0.0 and coral(a, b).item() > 0.0


def test_cdan_conditioning_and_entropy_weights():
    f, p = torch.randn(2, 256), torch.tensor([[0.98, 0.01, 0.01], [1 / 3, 1 / 3, 1 / 3]])
    assert multilinear(f, p).shape == (2, 768)
    w = entropy_weights(p)
    assert torch.isclose(w.mean(), torch.tensor(1.0)) and w[0] > w[1]       # the confident sample weighs more


def test_centroid_pseudo_labels_follow_the_feature_clusters_not_the_noisy_predictions():
    torch.manual_seed(0)
    f = torch.cat([torch.randn(10, 4) * 0.1 + torch.tensor([5.0, 0.0, 0.0, 0.0]),
                   torch.randn(10, 4) * 0.1 + torch.tensor([0.0, 5.0, 0.0, 0.0])])
    p = torch.full((20, 2), 0.5)
    p[:3], p[10:13] = torch.tensor([0.9, 0.1]), torch.tensor([0.1, 0.9])
    labels = centroid_labels(f, p)
    assert (labels[:10] == labels[0]).all() and (labels[10:] == labels[10]).all() and labels[0] != labels[10]
