"""Library-only rule that freezes the Lee speed search band (MSSP spec §4: the band is fixed from the 800-rpm library
records only, before any target record is read). Stated before it runs:

Per bearing class, the shaft speed of its 800-rpm library record is estimated (tevc.dirg_data.estimate_speed_hz) on
its library windows (8000 samples, hop 4000, at most 80) with a wide band, 0.70-1.30 x nominal; r_c = estimate /
nominal. SPEED_BAND = (0.85, 1.05), the spec default, if 0.88 <= min r_c and max r_c <= 1.02 (every library peak at
least 0.03 inside it); otherwise (min r_c - 0.07, max r_c + 0.07). Reported alongside: each class's estimate with the
chosen band (it must equal the wide-band one), and, on ten 8 s segments of the record (a 0.5 s Hann window's main
lobe, +-4 Hz, is as wide as the band at 13.3 Hz), a cross-check estimate and the peak prominence (peak / median
amplitude in the wide band).

    python -m tevc.probes.lee_speed_band     -> outputs/tevc/lee_speed_band.json
"""
import json
from pathlib import Path

import numpy as np

from tevc.datasets import make_windows
from tevc.dirg_data import ZERO_PAD, estimate_speed_hz
from tevc.lee_data import BEARING, LEE, LIBRARY_RPM

OUT = Path("outputs/tevc/lee_speed_band.json")
WIDE, DEFAULT, MARGIN, WIDEN = (0.70, 1.30), (0.85, 1.05), 0.03, 0.07
SEGMENT = 128_000           # 8 s at 16 kHz: Hann main lobe +-0.25 Hz


def choose_band(ratios):
    """The frozen band for library speed ratios r_c (rule in the module docstring)."""
    lo, hi = min(ratios), max(ratios)
    if lo >= DEFAULT[0] + MARGIN and hi <= DEFAULT[1] - MARGIN:
        return DEFAULT
    return (round(lo - WIDEN, 3), round(hi + WIDEN, 3))


def prominence(W, fs, nominal_hz, band=WIDE, pad=ZERO_PAD):
    """Peak / median of the mean Hann amplitude spectrum (as in estimate_speed_hz) inside band x nominal."""
    W = np.asarray(W, dtype=np.float64)
    n = W.shape[1] * pad
    A = np.abs(np.fft.rfft((W - W.mean(axis=1, keepdims=True)) * np.hanning(W.shape[1]), n=n, axis=1)).mean(axis=0)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    a = A[(f >= band[0] * nominal_hz) & (f <= band[1] * nominal_hz)]
    return float(a.max() / np.median(a))


def main():
    nominal = LIBRARY_RPM / 60.0
    entries = LEE.entries("library", LIBRARY_RPM)
    lib = make_windows(entries, LEE.win, LEE.hop, LEE.cap_fit)
    per_class = {}
    for (c, _, sig, _, _), name in zip(entries, BEARING):
        W = lib["W"][lib["y"] == c]
        segments = sig[: len(sig) // SEGMENT * SEGMENT].reshape(-1, SEGMENT)
        per_class[name] = {"windows": int(len(W)), "r_wide": estimate_speed_hz(W, LEE.fs, nominal, WIDE) / nominal,
                           "r_8s": estimate_speed_hz(segments, LEE.fs, nominal, WIDE) / nominal,
                           "prominence_8s": prominence(segments, LEE.fs, nominal)}
    band = choose_band([v["r_wide"] for v in per_class.values()])
    for c, name in enumerate(BEARING):
        per_class[name]["r_band"] = estimate_speed_hz(lib["W"][lib["y"] == c], LEE.fs, nominal, band) / nominal
    out = {"note": "800-rpm library records only; no target record read", "rule": __doc__.split("\n\n")[1],
           "per_class": per_class, "band": list(band),
           "library_rpm": 60.0 * estimate_speed_hz(lib["W"], LEE.fs, nominal, band)}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
