import numpy as np

from tevc.proxies import cluster_balance_weights, information_maximization, weighted_information_maximization


def _skewed_blobs():
    """70 'healthy' + 15 + 15 'faulty' windows in three well-separated feature clusters."""
    rng = np.random.default_rng(0)
    return np.vstack([rng.normal(0, 0.1, (70, 3)), rng.normal(5, 0.1, (15, 3)), rng.normal(-5, 0.1, (15, 3))])


def test_weighted_im_with_equal_weights_equals_im():
    p = np.random.default_rng(0).dirichlet(np.ones(4), size=50)
    assert np.isclose(weighted_information_maximization(p, np.ones(50)), information_maximization(p))


def test_cluster_balance_weights_give_each_cluster_equal_mass():
    w, _ = cluster_balance_weights(_skewed_blobs(), k=3)
    assert np.allclose(sorted([w[:70].sum(), w[70:85].sum(), w[85:].sum()]), [1.0, 1.0, 1.0])


def test_cluster_balanced_im_recovers_log_c_on_a_skewed_buffer():
    p = np.eye(3)[np.repeat([0, 1, 2], [70, 15, 15])]          # perfect predictions on a 70/15/15 buffer
    w, _ = cluster_balance_weights(_skewed_blobs(), k=3)
    assert information_maximization(p) < np.log(3) - 0.2       # plain IM penalises the correct model
    assert np.isclose(weighted_information_maximization(p, w), np.log(3), atol=1e-6)


def test_im_is_log_c_for_confident_balanced_predictions():
    p = np.eye(4)[np.repeat(np.arange(4), 10)]
    assert np.isclose(information_maximization(p), np.log(4), atol=1e-6)


def test_im_is_zero_for_collapsed_predictions():
    p = np.tile([1.0, 0.0, 0.0, 0.0], (40, 1))
    assert np.isclose(information_maximization(p), 0.0, atol=1e-6)


def test_im_is_zero_for_uniform_predictions():
    p = np.full((40, 4), 0.25)
    assert np.isclose(information_maximization(p), 0.0, atol=1e-6)
