import numpy as np

from tevc.probes.im_imbalance import gate1_recipes, skewed_indices
from tevc.recipe import ANCHORS


def test_skewed_indices_hit_requested_healthy_share_and_keep_all_classes():
    y = np.repeat(np.arange(4), 80)
    for f in (0.5, 0.7):
        idx = skewed_indices(y, f)
        assert abs((y[idx] == 0).mean() - f) < 0.03
        assert set(np.unique(y[idx])) == {0, 1, 2, 3}


def test_gate1_recipes_reproduce_gate1_order():
    r = gate1_recipes()
    assert list(r)[:4] == list(ANCHORS) and list(r)[4] == "R00" and len(r) == 16
