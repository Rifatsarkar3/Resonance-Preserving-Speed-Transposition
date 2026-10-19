from tevc import synthesis
from tevc.datasets import PU
from tevc.probes import frame_validity as fv


def test_impact_periods_follow_the_bearing_geometry():
    p = fv.impact_periods(PU, 900.0)               # 6203: BPFI 4.947 orders at 15 Hz
    assert abs(p["BPFI"] - 64000.0 / (4.947 * 15.0)) < 1.0 and p["BPFO"] > p["BPFI"]


def test_an_untouched_impact_train_is_recovered_exactly():
    found, spacing = fv.recovery(fv.impact_train(400, seed=0), 400)
    assert abs(found - 1) < 0.02 and abs(spacing - 1) < 0.02


def test_the_vocoder_respaces_impacts_only_with_frames_shorter_than_the_period():
    short = fv.recovery(synthesis.transpose(fv.impact_train(800, seed=0), 600.0, 1200.0, 0.0, 128), 400)
    long_ = fv.recovery(synthesis.transpose(fv.impact_train(100, seed=0), 600.0, 1200.0, 0.0, 512), 50)
    assert abs(short[0] - 1) < 0.1 and abs(short[1] - 1) < 0.1
    assert abs(long_[0] - 1) > 0.1 or abs(long_[1] - 1) > 0.1


def test_envelope_line_sits_at_the_target_rate_inside_the_regime_and_at_the_source_rate_beyond():
    inside = fv.envelope_line(synthesis.transpose(fv.impact_train(800, seed=0), 600.0, 1200.0, 0.0, 128), 400)
    beyond = fv.envelope_line(synthesis.transpose(fv.impact_train(100, seed=0), 600.0, 1200.0, 0.0, 512), 50)
    assert abs(inside[0] - 1.0) < 0.05 and abs(beyond[0] - 0.5) < 0.1 and inside[1] > beyond[1]


def test_boundary_is_the_largest_ratio_up_to_which_every_cell_recovers():
    cells = [{"frame_ratio": r, "ok": ok} for r, ok in ((0.1, True), (0.1, True), (0.5, True), (0.5, False),
                                                         (1.0, True))]
    assert fv.boundary(cells) == 0.1 and fv.boundary([{"frame_ratio": 0.1, "ok": False}]) == 0.0


def test_lee_frames_lie_inside_the_regime_and_dirg_frames_outside():
    rows = fv.table(limit=0.64)
    assert all(f["in_regime"] for f in rows["lee"]["frames"].values())
    assert not any(f["in_regime"] for f in rows["dirg"]["frames"].values())


def test_lee_fidelity_frames_straddle_the_impact_period():
    from tevc.lee_data import LEE
    from tevc.probes import lee_fidelity
    bpfi = fv.impact_periods(LEE, 801.7)["BPFI"]
    assert lee_fidelity.PROTOCOL_FRAMES == (16, 32, 64, 128)
    assert all(n / bpfi < 0.64 for n in lee_fidelity.PROTOCOL_FRAMES)
    assert all(n / bpfi > 1.0 for n in lee_fidelity.LONG_FRAMES)
