from pathlib import Path

import pytest

from tevc.stream import (OTTAWA_SEQUENCE, Environment, Stream, assert_disjoint, jnu_stream, ottawa_stream,
                         pu_stream)


def test_jnu_stream_order_revisit_and_disjointness():
    s = jnu_stream()
    assert [e.name for e in s.environments] == ["jnu@800", "jnu@1000", "jnu@800", "jnu@600"]
    assert s.environments[0] is s.environments[2]
    assert_disjoint(s)


def test_pu_stream_disjointness():
    s = pu_stream()
    assert [e.rpm for e in s.environments] == [1500.0, 900.0, 1500.0]
    assert_disjoint(s)


def test_assert_disjoint_detects_overlap():
    s = jnu_stream()
    env = s.environments[0]
    leaky = Stream(s.dataset, s.library_entries, s.library, s.library_rpm, (Environment("leaky", env.rpm, env.U, env.U),))
    with pytest.raises(AssertionError):
        assert_disjoint(leaky)


@pytest.mark.skipif(not Path("data/raw/OTTAWA").exists(), reason="Ottawa data not downloaded")
def test_ottawa_stream_speeds_inside_bins_and_disjoint():
    s = ottawa_stream()
    for env, (lo, hi) in zip(s.environments, OTTAWA_SEQUENCE):
        assert lo * 60 <= env.rpm < hi * 60
    assert 13.0 * 60 <= s.library_rpm < 16.0 * 60
    assert s.src_rpm_of is not None
    assert_disjoint(s)
