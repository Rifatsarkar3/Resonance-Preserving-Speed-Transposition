"""Environment streams (spec §4): a labelled library plus an ordered sequence of speed regimes, each with an
unlabelled buffer U (windows the agent may use) and a disjoint labelled test set T (scoring only).
"""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from tevc.datasets import JNU, PU, make_windows
from tevc.dirg_data import DIRG, estimate_speed_hz
from tevc.ottawa_data import OTTAWA

JNU_LIBRARY_CAP = 40      # first half of a 600-rpm fault recording yields ~40 windows: keep classes balanced
OTTAWA_LIBRARY_BIN = (13.0, 16.0)
# Trial 2 has almost no outer-race data above 25 Hz (2 windows), so the stream tops out at 22-25 Hz (user, 2026-10-08)
OTTAWA_SEQUENCE = ((16.0, 19.0), (19.0, 22.0), (22.0, 25.0), (19.0, 22.0), (16.0, 19.0))
MIN_PER_CLASS = 10
DIRG_LIBRARY_HZ = 300
DIRG_SEQUENCE = (400, 500, 400, 200, 100)      # nominal Hz; every ratio to the library is >= 1.3 or <= 1 / 1.3


@dataclass(frozen=True, eq=False)
class Environment:
    name: str
    rpm: float      # measured speed s_t
    U: dict         # unlabelled buffer; U["y"] exists for scoring and stress tests only, never for the agent
    T: dict         # labelled test set, scoring only


@dataclass(frozen=True, eq=False)
class Stream:
    dataset: type
    library_entries: list
    library: dict
    library_rpm: float
    environments: tuple                     # visiting order; a revisit is the same Environment object
    src_rpm_of: Optional[Callable] = None   # per-entry measured source speed (Ottawa chunks)


def _windows(ds, entries, cap):
    w = make_windows(entries, ds.win, ds.hop, cap)
    counts = np.bincount(w["y"], minlength=ds.n_classes)
    if counts.min() < MIN_PER_CLASS:
        raise ValueError(f"{ds.name}: too few windows per class {counts.tolist()}")
    return w


def jnu_stream(library_rpm=600, sequence=(800, 1000, 800, 600)):
    lib_entries = JNU.entries("lib_half", library_rpm)
    envs = {}
    for rpm in dict.fromkeys(sequence):
        u_role, t_role = ("U_lib", "T_lib") if rpm == library_rpm else ("A", "B")
        envs[rpm] = Environment(f"jnu@{rpm}", float(rpm), _windows(JNU, JNU.entries(u_role, rpm), JNU.cap_eval),
                                _windows(JNU, JNU.entries(t_role, rpm), JNU.cap_eval))
    return Stream(JNU, lib_entries, _windows(JNU, lib_entries, JNU_LIBRARY_CAP), float(library_rpm),
                  tuple(envs[r] for r in sequence))


def pu_stream(library_rpm=900, sequence=(1500, 900, 1500)):
    lib_entries = PU.entries("library", library_rpm)
    envs = {rpm: Environment(f"pu@{rpm}", float(rpm), _windows(PU, PU.entries("A", rpm), PU.cap_eval),
                             _windows(PU, PU.entries("B", rpm), PU.cap_eval)) for rpm in dict.fromkeys(sequence)}
    return Stream(PU, lib_entries, _windows(PU, lib_entries, PU.cap_fit), float(library_rpm),
                  tuple(envs[r] for r in sequence))


def _median_rpm(w):
    return float(np.median([OTTAWA.window_rpm(r, lo, hi) for r, lo, hi in zip(w["rec"], w["span_lo"], w["span_hi"])]))


def ottawa_stream(library_bin=OTTAWA_LIBRARY_BIN, sequence=OTTAWA_SEQUENCE):
    """Library = trial 1, U = trial 2, T = trial 3; speed bins in Hz; speeds measured from the encoder."""
    lib_entries = OTTAWA.entries(1, library_bin)
    library = _windows(OTTAWA, lib_entries, OTTAWA.cap_fit)
    envs = {}
    for b in dict.fromkeys(sequence):
        U = _windows(OTTAWA, OTTAWA.entries(2, b), OTTAWA.cap_eval)
        envs[b] = Environment(f"ottawa@{b[0]:g}-{b[1]:g}Hz", _median_rpm(U), U,
                              _windows(OTTAWA, OTTAWA.entries(3, b), OTTAWA.cap_eval))
    return Stream(OTTAWA, lib_entries, library, _median_rpm(library), tuple(envs[b] for b in sequence),
                  src_rpm_of=OTTAWA.chunk_rpm)


def dirg_stream(library_hz=DIRG_LIBRARY_HZ, sequence=DIRG_SEQUENCE):
    """RESS spec §5.4: library = the whole library-speed records; U = first 5 s, T = last 5 s of each target record.
    No tachometer: speeds (rpm) are estimated label-free from the library windows and from each U buffer."""
    lib_entries = DIRG.entries("library", library_hz)
    library = _windows(DIRG, lib_entries, DIRG.cap_fit)
    envs = {}
    for hz in dict.fromkeys(sequence):
        U = _windows(DIRG, DIRG.entries("U", hz), DIRG.cap_eval)
        envs[hz] = Environment(f"dirg@{hz}Hz", 60.0 * estimate_speed_hz(U["W"], DIRG.fs, hz), U,
                               _windows(DIRG, DIRG.entries("T", hz), DIRG.cap_eval))
    return Stream(DIRG, lib_entries, library, 60.0 * estimate_speed_hz(library["W"], DIRG.fs, library_hz),
                  tuple(envs[h] for h in sequence))


def _intervals(w):
    """(source recording id, absolute start, absolute end) per window; Ottawa chunk ids map back to their file."""
    out = []
    for rec, lo, hi in zip(w["rec"], w["span_lo"], w["span_hi"]):
        src, _, span = str(rec).partition("#")
        offset = int(span.split(":")[0]) if span else 0
        out.append((src, offset + lo, offset + hi))
    return out


def overlapping(a, b):
    """Number of window pairs (one from a, one from b) that share samples of the same source recording."""
    by_src = {}
    for src, lo, hi in _intervals(b):
        by_src.setdefault(src, []).append((lo, hi))
    return sum(lo < h and l < hi for src, lo, hi in _intervals(a) for l, h in by_src.get(src, ()))


def assert_disjoint(stream):
    for env in stream.environments:
        for x, y, what in ((env.U, env.T, "U/T"), (stream.library, env.U, "library/U"),
                           (stream.library, env.T, "library/T")):
            n = overlapping(x, y)
            if n:
                raise AssertionError(f"{env.name}: {n} overlapping {what} window pairs")
