import numpy as np

from tevc.recipe import Recipe, sample, training_set


class _DS:
    win, hop = 2000, 1000


def _entries():
    rng = np.random.default_rng(0)
    return [(c, f"r{c}", rng.standard_normal(40000).astype(np.float32), 0.0, 1.0) for c in range(2)]


def _library():
    return {"W": np.zeros((4, 2000), np.float32), "y": np.array([0, 0, 1, 1])}


def test_target_rpms_are_geometric_between_library_and_target():
    r = Recipe(3, 0.0, 1.0, 256, 0.0, 10, True)
    assert np.allclose(r.target_rpms(600, 1000), [600, 600 * (1000 / 600) ** 0.5, 1000])


def test_sample_respects_bounds():
    rng = np.random.default_rng(1)
    for _ in range(200):
        r = sample(rng)
        assert 1 <= r.n_targets <= 4 and 0.2 <= r.q_lo <= 1.0 and r.q_lo <= r.q_hi <= 1.2 and 0.0 <= r.beta <= 1.0


def test_training_set_size_is_real_plus_targets_times_classes_times_copies():
    r = Recipe(2, 0.5, 1.0, 256, 0.0, 3, True)
    W, y = training_set(r, _DS, _entries(), _library(), 600, 1000)
    assert len(y) == 4 + 2 * 2 * 3 and W.shape == (16, 2000)


def test_training_set_uses_per_entry_source_speed():
    r = Recipe(2, 0.5, 1.0, 256, 0.0, 3, True)
    seen = []

    def src_rpm_of(rec):
        seen.append(rec)
        return 650.0

    W, y = training_set(r, _DS, _entries(), _library(), 600, 1000, src_rpm_of=src_rpm_of)
    assert len(y) == 16 and set(seen) == {"r0", "r1"}
