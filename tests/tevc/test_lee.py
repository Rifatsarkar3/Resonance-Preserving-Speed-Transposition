import pytest

from tevc import lee_data


def test_target_records_stay_unread_until_the_protocol_is_approved():
    assert not lee_data.TARGETS_RELEASED
    with pytest.raises(PermissionError, match="approves"):
        lee_data.record(0, 1200)


def test_file_names_follow_the_published_convention():
    assert lee_data.record_path(2, 800).name == "H_IR_16_6204_800.mat"
    assert len(lee_data.SHA256) == 24
    assert {lee_data.record_path(c, r).name for c in range(4) for r in lee_data.SPEEDS_RPM} == set(lee_data.SHA256)


def test_verify_reports_missing_and_altered_files(tmp_path, monkeypatch):
    (tmp_path / "H_H_16_6204_800.mat").write_bytes(b"x")
    monkeypatch.setattr(lee_data, "SHA256", {"H_H_16_6204_800.mat": "0" * 64, "H_B_16_6204_800.mat": "0" * 64})
    assert lee_data.verify(tmp_path) == ["H_B_16_6204_800.mat", "H_H_16_6204_800.mat"]


def test_bearing_frequencies_follow_the_6204_geometry():
    g = lee_data.GEOMETRY
    ratio = g["d_mm"] / g["D_mm"]
    assert abs(g["N_balls"] / 2 * (1 - ratio) - 3.08) < 0.01 and abs(g["N_balls"] / 2 * (1 + ratio) - 4.92) < 0.01


def test_speed_band_rule_keeps_the_default_only_with_margin_and_otherwise_widens():
    from tevc.probes.lee_speed_band import DEFAULT, choose_band
    assert choose_band([0.95, 0.99, 1.0]) == DEFAULT
    assert choose_band([0.95, 1.03]) == (0.88, 1.1)                  # a library peak too close to the default edge
    assert choose_band([0.80, 0.99]) == (0.73, 1.06)


def test_prominence_finds_a_clean_shaft_tone():
    import numpy as np
    from tevc.probes.lee_speed_band import prominence
    t = np.arange(128_000) / 16000.0                                   # 8 s segments: main lobe +-0.25 Hz
    W = np.stack([np.sin(2 * np.pi * 13.0 * t + p) for p in np.linspace(0, 3, 5)])
    assert prominence(W, 16000.0, 800 / 60.0) > 10
