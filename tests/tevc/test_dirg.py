from pathlib import Path

import numpy as np
import pytest

import tevc.dirg_data as dd
from tevc.dirg_data import DIRG, estimate_speed_hz, record_path
from tevc.probes.dirg_channel import blocked_folds


def test_record_path_takes_the_single_1000_N_record_even_when_it_is_a_repeat(tmp_path):
    for name in ("C1A_300_502_2", "C1A_300_702_2", "C1A_300_898_2", "C0A_300_505_1", "C0A_100_000_1"):
        (tmp_path / f"{name}.mat").touch()
    assert record_path(1, 300, tmp_path).stem == "C1A_300_502_2"
    assert record_path(0, 300, tmp_path).stem == "C0A_300_505_1"
    with pytest.raises(FileNotFoundError):
        record_path(0, 100, tmp_path)                 # only a 0 N record at 100 Hz


def test_speed_estimate_finds_the_shaft_line_below_nominal():
    rng = np.random.default_rng(0)
    t = np.arange(DIRG.win) / DIRG.fs
    for nominal, actual in ((100, 96.3), (300, 291.7), (500, 488.2)):
        W = np.stack([np.sin(2 * np.pi * actual * t + rng.uniform(0, 2 * np.pi))
                      + 3.0 * np.sin(2 * np.pi * 2 * actual * t)       # stronger 2x line, outside the band
                      + 0.5 * rng.normal(size=t.size) for _ in range(40)])
        assert abs(estimate_speed_hz(W, DIRG.fs, nominal) - actual) < 0.002 * actual


def test_entries_refuse_until_the_channel_is_frozen(monkeypatch):
    monkeypatch.setattr(dd, "CHANNEL", None)
    with pytest.raises(RuntimeError, match="channel"):
        DIRG.entries("library", 300)


def _windows(n_per_rec=20, hop=2560, win=5120):
    rec = np.repeat(["a", "b"], n_per_rec)
    lo = np.tile(np.arange(n_per_rec) * hop, 2).astype(float)
    return {"rec": rec, "span_lo": lo, "span_hi": lo + win}


def test_blocked_folds_test_every_window_once_and_purge_overlapping_training_windows():
    w = _windows()
    folds = blocked_folds(w, 5)
    tested = np.concatenate([test for _, test in folds])
    assert sorted(tested) == list(range(len(w["rec"])))
    for train, test in folds:
        for i in test:
            same = w["rec"][train] == w["rec"][i]
            assert not np.any(same & (w["span_lo"][train] < w["span_hi"][i]) & (w["span_hi"][train] > w["span_lo"][i]))
        assert len(train) >= len(w["rec"]) - len(test) - 4      # at most one purged neighbour per block edge


@pytest.mark.skipif(dd.CHANNEL is None or not Path("data/raw/DIRG").exists(), reason="DIRG data or channel missing")
def test_dirg_stream_layout_and_disjointness():
    from tevc.stream import DIRG_SEQUENCE, assert_disjoint, dirg_stream
    s = dirg_stream()
    assert [e.name for e in s.environments] == [f"dirg@{h}Hz" for h in DIRG_SEQUENCE]
    assert s.environments[0] is s.environments[2]
    assert np.bincount(s.library["y"]).tolist() == [DIRG.cap_fit] * 7
    for env, hz in zip(s.environments, DIRG_SEQUENCE):
        assert np.bincount(env.U["y"]).tolist() == np.bincount(env.T["y"]).tolist() == [DIRG.cap_eval] * 7
        assert 0.85 * hz * 60 <= env.rpm <= hz * 60
    assert_disjoint(s)


def test_headroom_compares_the_deployed_model_with_the_best_evaluated_candidate():
    from tevc.probes.dirg_diagnostics import visit_headroom
    entry = {"deployed_acc": 0.3, "trials": [{"acc": 0.2}, {"acc": 0.6}, {"acc": 0.3}]}
    assert visit_headroom(entry) == (0.3, 0.6)
    assert visit_headroom({"deployed_acc": 0.7, "trials": [{"acc": 0.2}]}) == (0.7, 0.7)     # carried model deployed
