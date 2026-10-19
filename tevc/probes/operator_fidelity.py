"""Operator fidelity against REAL target-speed recordings (reviewer point on the earlier manuscript: RPST and
resampling differ in more than resonance placement, and agreement with real target-speed signals, including
transient-envelope / ring-down behaviour, was never measured).

For each library -> target direction the library recordings are transposed exactly to the target speed with the
generalised operator transpose(β, n_fft) (β = 0: RPST, β = 1: plain resampling; n_fft sets phase-vocoder smearing)
and compared, class by class, with real windows recorded at the target speed. Labels are used: this is a physics
validation, not something the agent sees. Metrics (class-averaged; "error" lower is better):
  band_error      mean |Δ log| of the 64 relative FFT band energies of the class means   (resonance placement)
  envelope_corr   Pearson r of the class-mean envelope spectra over 0.25..12 shaft orders  (impulse rate)
  ringdown_error  |Δ ln τ|, τ = first lag where the envelope autocorrelation < 1/e        (transient decay)
  kurtosis_error  |Δ ln| of the median envelope kurtosis + 3                               (impulsiveness)
  feature_mmd     MMD² between the 561-feature banks, standardised by the real windows     (overall)
"untransposed" = real library windows at the source speed (how far apart the two speeds are to begin with).
Ottawa is never touched. Run from repo root:  python -m tevc.probes.operator_fidelity [--family wsola]
  -> outputs/tevc/probe_operator_fidelity.json (phase vocoder) / probe_operator_fidelity_wsola.json (WSOLA, β = 0)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis

from tevc import features, synthesis
from tevc.datasets import make_windows
from tevc.proxies import neg_mmd2
from tevc.stream import jnu_stream, pu_stream

OUT = Path("outputs/tevc/probe_operator_fidelity.json")
DIRECTIONS = {"pu": ((900, 1500), (1500, 900)), "jnu": ((600, 1000), (1000, 600))}
STREAMS = {"pu": pu_stream, "jnu": jnu_stream}
BETAS = (0.0, 0.25, 0.5, 0.75)
N_FFTS = (64, 128, 256, 512)
ORDERS = np.arange(0.25, 12.0, 0.05)
CAP = 40


def _znorm(W):
    W = np.asarray(W, dtype=np.float64)
    return (W - W.mean(axis=1, keepdims=True)) / (W.std(axis=1, keepdims=True) + 1e-12)


def _envelopes(W):
    env = np.abs(hilbert(W, axis=1))
    return env - env.mean(axis=1, keepdims=True)


def _envelope_order_spectrum(env, fs, f_shaft):
    spec = np.abs(np.fft.rfft(env * np.hanning(env.shape[1]), axis=1))
    orders = np.fft.rfftfreq(env.shape[1], 1.0 / fs) / f_shaft
    s = np.array([np.interp(ORDERS, orders, row) for row in spec]).mean(axis=0)
    return s / s.sum()


def _ringdown(env, fs):
    """Median first lag (s) at which the envelope autocorrelation falls below 1/e."""
    n = env.shape[1]
    f = np.fft.rfft(env, n=2 * n, axis=1)
    acf = np.fft.irfft(f * np.conj(f), axis=1)[:, :n]
    acf /= acf[:, :1]
    lags = np.argmax(acf < np.exp(-1), axis=1)
    return float(np.median(lags)) / fs


def _class_stats(W, fs, f_shaft):
    W = _znorm(W)
    env = _envelopes(W)
    return {"bands": features._fft_block(W).mean(axis=0),
            "envelope": _envelope_order_spectrum(env, fs, f_shaft),
            "tau": _ringdown(env, fs),
            "kurt": float(np.median(kurtosis(env, axis=1)))}


def _compare(synth, real, ds, f_shaft):
    rows = []
    for c in np.unique(real["y"]):
        S = _class_stats(synth["W"][synth["y"] == c], ds.fs, f_shaft)
        R = _class_stats(real["W"][real["y"] == c], ds.fs, f_shaft)
        rows.append({"band_error": float(np.mean(np.abs(S["bands"] - R["bands"]))),
                     "envelope_corr": float(np.corrcoef(S["envelope"], R["envelope"])[0, 1]),
                     "ringdown_error": float(abs(np.log(S["tau"] + 1e-9) - np.log(R["tau"] + 1e-9))),
                     "kurtosis_error": float(abs(np.log(S["kurt"] + 3) - np.log(R["kurt"] + 3)))})
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    Xr = features.extract(real["W"], ds.fs, f_shaft, ds.geometry, znorm=True)
    Xs = features.extract(synth["W"], ds.fs, f_shaft, ds.geometry, znorm=True)
    mu, sd = Xr.mean(axis=0), Xr.std(axis=0) + 1e-8
    out["feature_mmd"] = float(-neg_mmd2((Xs - mu) / sd, (Xr - mu) / sd))
    return out


def _to_env(stream, env, beta, n_fft):
    """make_windows synth: an entry transposed from its source speed (stream.src_rpm_of, else the library speed)."""
    def synth(label, rec, sig):
        s0 = stream.src_rpm_of(rec) if stream.src_rpm_of else stream.library_rpm
        return synthesis.transpose(sig, s0, env.rpm, beta, n_fft), env.rpm / s0
    return synth


def fidelity_by_env(stream, operators, cap=CAP, log=None):
    """{environment: {"speed_ratio", "operators": {label: metrics}}} for every distinct environment of stream.
    operators: label -> None (library windows as recorded) or (β, n_fft) for transpose; compared class by class with
    the environment's real U windows (_compare)."""
    ds, out = stream.dataset, {}
    for env in dict.fromkeys(stream.environments):
        rows = {}
        for label, op in operators.items():
            synth = None if op is None else _to_env(stream, env, *op)
            rows[label] = _compare(make_windows(stream.library_entries, ds.win, ds.hop, cap, synth=synth),
                                   env.U, ds, env.rpm / 60.0)
            if log:
                log(f"{env.name} {label:44s} {json.dumps({k: round(v, 4) for k, v in rows[label].items()})}")
        out[env.name] = {"speed_ratio": env.rpm / stream.library_rpm, "operators": rows}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=("pv", "wsola"), default="pv")
    args = ap.parse_args()
    out = OUT if args.family == "pv" else OUT.with_name("probe_operator_fidelity_wsola.json")
    operator = synthesis.transpose if args.family == "pv" else synthesis.transpose_wsola
    results = {}
    for name, directions in DIRECTIONS.items():
        for src, tgt in directions:
            stream = STREAMS[name](library_rpm=src, sequence=(tgt,))
            ds, env = stream.dataset, stream.environments[0]
            f_tgt = tgt / 60.0
            if args.family == "pv":
                ops = {"untransposed": None, "beta=1 (resampling)": (1.0, 256)}
                ops.update({f"beta={b:g} n_fft={n}": (b, n) for b in BETAS for n in N_FFTS})
            else:
                ops = {"untransposed": None}
                ops.update({f"wsola beta=0 frame={n}": (0.0, n) for n in N_FFTS})
            task = {}
            for label, op in ops.items():
                if op is None:
                    synth = make_windows(stream.library_entries, ds.win, ds.hop, CAP)
                else:
                    beta, size = op
                    synth = make_windows(stream.library_entries, ds.win, ds.hop, CAP,
                                         synth=lambda lab, rec, sig, b=beta, n=size:
                                         (operator(sig, src, tgt, b, n), tgt / src))
                task[label] = _compare(synth, env.U, ds, f_tgt)
                print(f"{name} {src}->{tgt} {label:24s} {json.dumps({k: round(v, 4) for k, v in task[label].items()})}",
                      flush=True)
            results[f"{name}:{src}->{tgt}"] = task
            out.write_text(json.dumps(results, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
