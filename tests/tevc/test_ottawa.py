from pathlib import Path

import numpy as np
import pytest

from tevc.ottawa_data import OTTAWA, in_bin_chunks, pulse_edges, recording, speed_hz

FS = 200000.0


def test_geometry_reproduces_the_published_defect_frequencies():
    from tevc.features import compute_fault_frequencies
    f = compute_fault_frequencies(f_shaft_hz=1.0, **OTTAWA.geometry)
    assert round(f["BPFO"], 2) == 3.57 and round(f["BPFI"], 2) == 5.43


def _encoder(speed_of_t, seconds):
    """Synthetic 0/5 V encoder: 1024 cycles per revolution, shaft speed speed_of_t(t) in Hz."""
    t = np.arange(int(seconds * FS)) / FS
    phase = np.cumsum(speed_of_t(t) * 1024 / FS)
    return np.where((phase % 1.0) < 0.5, 5.0, 0.0)


def test_speed_decoding_at_constant_speed():
    edges = pulse_edges(_encoder(lambda t: np.full_like(t, 20.0), 1.0))
    assert abs(speed_hz(edges, 0, int(FS)) - 20.0) < 0.01


def test_chunks_follow_speed_bin_on_a_ramp():
    edges = pulse_edges(_encoder(lambda t: 10.0 + 2.0 * t, 10.0))   # inside 16-19 Hz for t in [3.0, 4.5) s
    assert in_bin_chunks(edges, int(10 * FS), (16.0, 19.0)) == [(600000, 800000), (800000, 900000)]


@pytest.mark.skipif(not Path("data/raw/OTTAWA").exists(), reason="Ottawa data not downloaded")
def test_real_recording_and_library_bin_chunks():
    vib, edges = recording("H", "A", 1)
    assert len(vib) == 2_000_000 and 10.0 < speed_hz(edges, 0, len(vib)) < 35.0
    entries = OTTAWA.entries(1, (13.0, 16.0))
    assert {label for label, *_ in entries} == set(range(5))
    for label, rec, sig, lo, hi in entries:
        assert 13.0 * 60 <= OTTAWA.chunk_rpm(rec) < 16.0 * 60
