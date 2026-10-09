from tevc.datasets import JNU, PU
from tevc.features import compute_fault_frequencies


def _orders(geometry):
    return compute_fault_frequencies(f_shaft_hz=1.0, **geometry)


def test_jnu_follows_li_2013_table1():
    g = JNU.geometry
    assert (g["N_balls"], g["N_balls_inner"], g["d_mm"], g["alpha_deg"]) == (10, 11, 7.0, 0.0)
    f = _orders(g)
    assert round(f["BPFO"], 3) == 4.091 and round(f["BPFI"], 3) == 6.5     # BPFI on the 11-roller NU205


def test_pu_is_a_standard_6203():
    f = _orders(PU.geometry)
    assert PU.geometry["N_balls"] == 8 and round(f["BPFO"], 2) == 3.05 and round(f["BPFI"], 2) == 4.95
