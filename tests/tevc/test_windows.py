import numpy as np

from tevc.datasets import make_windows


def _entries():
    rng = np.random.default_rng(0)
    return [(0, "a", rng.standard_normal(10000).astype(np.float32), 0.0, 1.0),
            (0, "b", rng.standard_normal(10000).astype(np.float32), 0.0, 1.0),
            (1, "c", rng.standard_normal(10000).astype(np.float32), 0.0, 1.0)]


def test_cap_is_per_class_and_spans_are_source_coordinates():
    w = make_windows(_entries(), win=1000, hop=500, cap=5)
    assert (w["y"] == 0).sum() == 5 and (w["y"] == 1).sum() == 5
    assert np.allclose(w["span_hi"] - w["span_lo"], 1000)


def test_synth_uses_per_entry_rate_for_spans():
    rates = {"a": 2.0, "b": 0.5, "c": 1.0}

    def synth(label, rec, sig):
        r = rates[rec]
        return sig[: int(len(sig) / r)], r          # stand-in for a transposition by ratio r

    w = make_windows(_entries(), win=1000, hop=500, cap=100, synth=synth)
    for rec, lo, hi in zip(w["rec"], w["span_lo"], w["span_hi"]):
        assert np.isclose(hi - lo, 1000 * rates[rec])


def test_synth_path_matches_transform_path():
    def flip(sig):
        return sig[::-1].copy()

    a = make_windows(_entries(), 1000, 500, 7, transform=flip, rate=1.5)
    b = make_windows(_entries(), 1000, 500, 7, synth=lambda label, rec, sig: (flip(sig), 1.5))
    for k in ("W", "y", "rec", "span_lo", "span_hi"):
        assert np.array_equal(a[k], b[k])
