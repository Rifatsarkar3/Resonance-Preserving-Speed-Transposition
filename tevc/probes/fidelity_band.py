"""EXPLORATORY R7 (MSSP reframe spec docs/superpowers/specs/2026-10-20-mssp-sp-first-reframe-design.md §5.2; attack
audit O4): fidelity of RPST against real target-speed recordings with the envelope taken in the resonance band that
carries the impacts, instead of the broadband envelope of tevc.probes.operator_fidelity.

  band         the band of maximum spectral kurtosis (STFT-based, Hann windows of 32-512 samples, hop a quarter;
               Antoni & Randall 2006): centre bin +- one bin of the best window length, chosen ONCE per dataset from
               the pooled library windows of the fault classes (no target recording is read to choose it)
  metrics      per environment and operator, class by class against the real U windows: envelope_corr_band (Pearson
               r of the class-mean envelope order spectra over 0.25-12 shaft orders, envelope in the band),
               envelope_corr_broadband (the archived metric, for reference), energy_fraction_synth / _real (share
               of the window's energy inside the band)
  hypothesis   on DIRG, speeding up by resampling moves the impact-carrying resonance out of the band (above the
               Nyquist frequency for the largest ratios), so the resampled copies have little energy where the real
               target shows its impacts: energy_fraction_synth of "resampling" against energy_fraction_real
  operators    untransposed library; RPST (beta = 0) at every frame the methods use (recipe.N_FFTS scaled by the
               dataset's frame_scale) and at longer frames; plain resampling (beta = 1)

Labels are used only to validate the operator; nothing feeds back into a method. Ottawa: v2 stream only.
    python -m tevc.probes.fidelity_band --dataset {lee,dirg,ottawa}  -> outputs/tevc/<dataset>/exploratory_fidelity_band.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import stft

from tevc.datasets import make_windows
from tevc.probes.operator_fidelity import _envelope_order_spectrum, _envelopes, _to_env, _znorm
from tevc.recipe import N_FFTS, frame
from tevc.stream import dirg_stream, lee_stream, ottawa_stream

SK_WINDOWS = (32, 64, 128, 256, 512)
CAP = 40
HEALTHY = 0                      # class 0 is the healthy bearing in every stream used here
STREAMS = {"lee": lee_stream, "dirg": dirg_stream, "ottawa": ottawa_stream}
OUTS = {"lee": Path("outputs/tevc/lee"), "dirg": Path("outputs/tevc/dirg"), "ottawa": Path("outputs/tevc/ottawa")}
EXTRA_FRAMES = {"lee": (256, 512), "dirg": (256,), "ottawa": ()}     # frames beyond the methods' own (samples)


def spectral_kurtosis(W, n_win):
    """SK(f) = <|X|^4> / <|X|^2>^2 - 2 of the STFT (Hann, n_win samples, hop n_win // 4) over every frame of every
    window of W (N, L): about 0 for stationary Gaussian noise, large where impacts ring."""
    _, _, Z = stft(np.asarray(W, dtype=np.float64), nperseg=n_win, noverlap=n_win - n_win // 4, window="hann",
                   axis=-1, boundary=None, padded=False)
    P = np.abs(Z) ** 2                                  # (N, n_freq, n_frames)
    m2, m4 = P.mean(axis=(0, 2)), (P ** 2).mean(axis=(0, 2))
    return m4 / (m2 ** 2 + 1e-30) - 2.0


def sk_band(W, fs, windows=SK_WINDOWS):
    """(lo, hi, sk): the band of maximum spectral kurtosis over the window lengths that fit in W, the bin of the
    maximum +- one bin (DC and Nyquist excluded: those bins are real-valued, so Gaussian noise gives them SK ~ 1)."""
    best = None
    for n in windows:
        if n > W.shape[1]:
            continue
        sk = spectral_kurtosis(W, n)
        sk[[0, -1]] = -np.inf                           # windows are even, so the last bin is Nyquist
        k, df = int(np.argmax(sk)), fs / n
        cand = (max(0.0, (k - 1) * df), min(fs / 2, (k + 1) * df), float(sk[k]))
        if best is None or cand[2] > best[2]:
            best = cand
    if best is None:
        raise ValueError("no spectral-kurtosis window fits in the windows")
    return best


def band_envelope(W, fs, lo, hi):
    """Envelope of every window of W in the band [lo, hi) Hz (analytic signal of the band-passed window), mean
    removed."""
    n = W.shape[1]
    f = np.fft.fftfreq(n, 1.0 / fs)
    gain = np.where((f > 0) & (f >= lo) & (f < hi), 2.0, 0.0)
    env = np.abs(np.fft.ifft(np.fft.fft(W, axis=1) * gain, axis=1))
    return env - env.mean(axis=1, keepdims=True)


def band_energy_fraction(W, fs, lo, hi):
    """Mean over the windows of W of the share of their energy inside [lo, hi) Hz."""
    P = np.abs(np.fft.rfft(np.asarray(W, dtype=np.float64), axis=1)) ** 2
    f = np.fft.rfftfreq(W.shape[1], 1.0 / fs)
    sel = (f >= lo) & (f < hi)
    return float(np.mean(P[:, sel].sum(axis=1) / (P.sum(axis=1) + 1e-30)))


def _stats(W, fs, f_shaft, band):
    W = _znorm(W)
    return {"band": _envelope_order_spectrum(band_envelope(W, fs, *band), fs, f_shaft),
            "broad": _envelope_order_spectrum(_envelopes(W), fs, f_shaft),
            "energy": band_energy_fraction(W, fs, *band)}


def compare_band(synth, real, fs, f_shaft, band):
    """Class-averaged band and broadband envelope correlations and band-energy fractions of synth against real."""
    rows = {}
    for c in np.unique(real["y"]):
        S, R = _stats(synth["W"][synth["y"] == c], fs, f_shaft, band), _stats(real["W"][real["y"] == c], fs, f_shaft, band)
        rows[int(c)] = {"envelope_corr_band": float(np.corrcoef(S["band"], R["band"])[0, 1]),
                        "envelope_corr_broadband": float(np.corrcoef(S["broad"], R["broad"])[0, 1]),
                        "energy_fraction_synth": S["energy"], "energy_fraction_real": R["energy"]}
    keys = next(iter(rows.values())).keys()
    return {"mean": {k: float(np.mean([r[k] for r in rows.values()])) for k in keys},
            "fault_classes": {k: float(np.mean([r[k] for c, r in rows.items() if c != HEALTHY])) for k in keys},
            "by_class": rows}


def library_band(stream, cap=CAP):
    """The dataset's band, from the pooled library windows of the fault classes alone."""
    ds = stream.dataset
    lib = make_windows(stream.library_entries, ds.win, ds.hop, cap)
    return sk_band(_znorm(lib["W"][lib["y"] != HEALTHY]), ds.fs)


def operators(dataset, ds):
    frames = sorted({frame(n, ds) for n in N_FFTS} | set(EXTRA_FRAMES[dataset]))
    return {"untransposed": None, **{f"RPST n_fft={n}": (0.0, n) for n in frames}, "resampling": (1.0, 256)}


def fidelity_band_by_env(stream, ops, band, cap=CAP, log=None):
    ds, out = stream.dataset, {}
    for env in dict.fromkeys(stream.environments):
        rows = {}
        for label, op in ops.items():
            synth = None if op is None else _to_env(stream, env, *op)
            rows[label] = compare_band(make_windows(stream.library_entries, ds.win, ds.hop, cap, synth=synth),
                                       env.U, ds.fs, env.rpm / 60.0, band)
            if log:
                log(f"{env.name} {label:22s} {json.dumps({k: round(v, 3) for k, v in rows[label]['mean'].items()})}")
        out[env.name] = {"speed_ratio": env.rpm / stream.library_rpm, "operators": rows}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=tuple(STREAMS), required=True)
    args = ap.parse_args()
    stream = STREAMS[args.dataset]()
    band = library_band(stream)
    out = {"note": "exploratory R7 (reframe spec 2026-10-20 §5.2); band chosen from the library alone",
           "dataset": args.dataset, "band_hz": {"lo": band[0], "hi": band[1], "sk": band[2]},
           "environments": fidelity_band_by_env(stream, operators(args.dataset, stream.dataset), band[:2],
                                                log=lambda s: print(s, flush=True))}
    path = OUTS[args.dataset] / "exploratory_fidelity_band.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"band {band[0]:.0f}-{band[1]:.0f} Hz (SK {band[2]:.1f}) -> {path}", flush=True)


if __name__ == "__main__":
    main()
