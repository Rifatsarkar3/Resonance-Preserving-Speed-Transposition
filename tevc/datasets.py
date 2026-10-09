"""Dataset adapters for Gate-0 v2: one view of JNU and PU recordings.

entries(role, rpm) -> [(label, rec_id, signal, lo, hi)]; windows are cut fully inside [lo, hi) of a recording.
Roles: "library" (labelled, library speed), "A" (target speed; unlabelled for the agent, labelled only for
the oracle), "B" (target speed, test only). A and B never share samples.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io as sio

from tevc import jnu_data


class JNU:
    name, fs, win, hop = "jnu", jnu_data.FS, 12000, 6000
    cap_fit, cap_eval = 80, 40           # same windows per class as Gate-0 v1
    geometry = jnu_data.GEOMETRY
    tasks = ((600, 800), (600, 1000), (1000, 800), (1000, 600))
    n_classes = 4
    _span = {"library": (0.0, 1.0), "A": (0.0, 0.5), "B": (0.5, 1.0),
             # D1 streams revisit the library speed: library = first half, revisit U/T = remaining quarters
             "lib_half": (0.0, 0.5), "U_lib": (0.5, 0.75), "T_lib": (0.75, 1.0)}

    @staticmethod
    @lru_cache(maxsize=None)
    def _recordings():
        return jnu_data.load_recordings(jnu_data.ROOT)

    @classmethod
    def entries(cls, role, rpm):
        lo, hi = cls._span[role]
        return [(label, f"{label}@{rpm}", cls._recordings()[(label, rpm)], lo, hi) for label in range(4)]


class PU:
    """Paderborn vibration_1 (64 kHz), bearing 6203, M07_F10 so only speed differs; bearings as in G1."""
    name, fs, win, hop = "pu", 64000.0, 16000, 8000
    root = Path("data/raw/PU")
    cap_fit, cap_eval = 80, 80
    n_classes = 4
    # 6203 deep-groove ball bearing: 8 balls of 6.747 mm; pitch diameter = bore/OD mean (17 + 40) / 2 = 28.5 mm,
    # which equals the catalogue pitch circle
    geometry = {"N_balls": 8, "d_mm": 6.747, "D_mm": 28.5, "alpha_deg": 0.0}
    tasks = ((900, 1500), (1500, 900))
    condition = {900: "N09_M07_F10", 1500: "N15_M07_F10"}
    class_bearings = {0: ("K001", "K002", "K003", "K004"), 1: ("KA01", "KA03", "KA04", "KA05"),
                      2: ("KI01", "KI03", "KI04", "KI05"), 3: ("KB23", "KB24", "KB27")}
    reps = {"library": (1, 2), "A": (3, 4), "B": (5, 6)}

    @staticmethod
    @lru_cache(maxsize=None)
    def _vibration(path):
        d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        key = next(k for k in d if not k.startswith("__"))
        channel = next(c for c in d[key].Y if getattr(c, "Name", "") == "vibration_1")
        return np.asarray(channel.Data, dtype=np.float32).ravel()

    @classmethod
    def entries(cls, role, rpm):
        root, cond = cls.root, cls.condition[rpm]
        return [(label, f"{b}_{cond}_{rep}", cls._vibration(str(root / b / f"{cond}_{b}_{rep}.mat")), 0.0, 1.0)
                for label, bearings in cls.class_bearings.items() for b in bearings for rep in cls.reps[role]]


DATASETS = {"jnu": JNU, "pu": PU}


def make_windows(entries, win, hop, cap, transform=None, rate=1.0, synth=None):
    """Windows from entries, at most `cap` per class, evenly thinned across all of a class's recordings.

    transform: optional signal -> synthesised signal at speed ratio `rate` (= target/source rpm). Synthesised
    sample t corresponds to source sample t * rate; spans are stored in source coordinates so a library
    window and its transposed copy can be recognised as overlapping.
    synth: optional per-entry alternative, (label, rec, signal) -> (synthesised signal, rate), for sources whose
    speed differs between entries (Ottawa chunks). Overrides transform/rate.
    """
    candidates = {}
    for label, rec, sig, lo, hi in entries:
        if synth is not None:
            s, r = synth(label, rec, sig)
        else:
            s, r = (transform(sig) if transform else sig), rate
        for st in jnu_data.window_starts(len(s), win, hop, lo, hi):
            candidates.setdefault(label, []).append((s, rec, st, r))
    W, y, recs, span_lo, span_hi = [], [], [], [], []
    for label in sorted(candidates):
        items = candidates[label]
        for i in np.linspace(0, len(items) - 1, min(cap, len(items))).round().astype(int):
            s, rec, st, r = items[i]
            W.append(s[st:st + win])
            y.append(label)
            recs.append(rec)
            span_lo.append(st * r)
            span_hi.append((st + win) * r)
    return {"W": np.stack(W), "y": np.array(y), "rec": np.array(recs),
            "span_lo": np.array(span_lo), "span_hi": np.array(span_hi)}
