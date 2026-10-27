"""EXPLORATORY R6 (MSSP reframe spec docs/superpowers/specs/2026-10-20-mssp-sp-first-reframe-design.md §5.1, with
attack-audit amendment A1): the validity interval of phase-vocoder speed transposition on synthetic impact trains.

The first probe (tevc.probes.frame_validity, 39d1183) tied the ring-down to the impact period (decay constant P/8),
so frame/P and frame/decay always moved together and its boundary (0.64) could be a ring-down condition. Here the
decay tau is set per cell through kappa = N / tau while P = N / (frame/P) runs along the grid: a boundary that stays
put across kappa is an impact-period condition; one that moves with kappa is not.

  signal     ringing impacts every P samples (resonance f_res cycles/sample, decay tau samples, random carrier phase
             per impact, optional slip jitter in fractions of P), superposed, peak-normalised, plus a noise floor
             (without one the vocoder cancels alternate impacts, 39d1183)
  grid       frame/P in FRAME_RATIOS, frames N per condition, speed ratios down (0.6, 0.8) and up (1.25, 1.5, 2.0),
             5 seeds; a cell passes when the envelope peak count and their median spacing are within the tolerance
             of the target impact rate (tevc.probes.frame_validity.recovery)
  conditions the base (kappa 5, f_res 1/8, no jitter, noise 2 %, hop N/4, standard vocoder, N 64-512) and one factor
             at a time: kappa 2.5 / 10 / 20; f_res 1/16, 1/4; jitter 1 %, 2 %; noise 1 %, 5 %; hop N/8; the identity
             phase-locked vocoder (tevc.tsm) and WSOLA (tevc.synthesis.wsola), at N 128 and 256
  interval   per condition, tolerance (5, 10, 15 %), direction (all, down, up) and period convention (source P, or
             target P / r): rho*_pass = largest frame ratio up to which every cell passes; rho*_fail = smallest at
             which none does. The convention with the smaller down/up gap at the base is the paper's; lower_end =
             the smallest rho*_pass of the standard vocoder (both hops) at 10 % in that convention, the frame rule's
             rho* (spec §5.4).

    python -m tevc.probes.validity_interval [--quick]      -> outputs/tevc/probe_validity_interval.json
Synthetic only; no dataset is read.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from tevc import synthesis, tsm
from tevc.probes.frame_validity import envelope_line, recovery

OUT = Path("outputs/tevc/probe_validity_interval.json")
N_SAMPLES = 200_000
SEEDS = 5
FRAME_RATIOS = (0.40, 0.50, 0.55, 0.60, 0.64, 0.68, 0.72, 0.76, 0.80, 0.85, 0.90, 1.00, 1.10, 1.28, 1.50, 2.00)
SPEED_RATIOS = {"down": (0.6, 0.8), "up": (1.25, 1.5, 2.0)}
TOLERANCES = (0.05, 0.10, 0.15)
TOLERANCE = 0.10
BASE = {"kappa": 5.0, "f_res": 1 / 8, "jitter": 0.0, "noise": 0.02, "hop_div": 4, "tsm": "pv",
        "n_ffts": (64, 128, 256, 512)}
FACTOR_N_FFTS = (128, 256)
FACTORS = {"kappa": (2.5, 10.0, 20.0), "f_res": (1 / 16, 1 / 4), "jitter": (0.01, 0.02), "noise": (0.01, 0.05),
           "hop_div": (8,), "tsm": ("pv_locked", "wsola")}


def conditions():
    """{name: condition}: the base and every factor level, one factor at a time around the base."""
    out = {"base": dict(BASE)}
    for name, levels in FACTORS.items():
        for v in levels:
            out[f"{name}={v:g}" if isinstance(v, float) else f"{name}={v}"] = {**BASE, name: v,
                                                                              "n_ffts": FACTOR_N_FFTS}
    return out


def impact_train(period, tau, seed, n=N_SAMPLES, f_res=1 / 8, jitter=0.0, noise=0.02):
    """Ringing impacts every `period` samples (float allowed), each a resonance at f_res cycles/sample decaying with
    time constant tau samples (independent of the period) and a random carrier phase; impact times move by up to
    +-jitter x period; ring-downs superpose; peak-normalised, plus white noise of standard deviation `noise`."""
    rng = np.random.default_rng(seed)
    n_imp = int(n / period) + 2
    t = np.arange(n_imp) * period + jitter * period * rng.uniform(-1.0, 1.0, n_imp)
    t = np.round(t[(t >= 0) & (t < n)]).astype(int)
    pulses = np.zeros(n, dtype=np.complex128)
    np.add.at(pulses, t, np.exp(1j * rng.uniform(0.0, 2 * np.pi, len(t))))
    k = np.arange(int(np.ceil(8 * tau)) + 1)
    x = np.imag(fftconvolve(pulses, np.exp(-k / tau) * np.exp(2j * np.pi * f_res * k))[:n])
    x /= np.max(np.abs(x)) + 1e-12
    return (x + noise * rng.standard_normal(n)).astype(np.float32)


def transpose(x, r, n_fft, cond):
    """x transposed by speed ratio r (impact rate x r) with the condition's time-scale operator: the archived
    librosa vocoder (tevc.synthesis.transpose) at hop N/4, tevc.tsm.pv_tsm at other hops or phase-locked, WSOLA with
    segment N."""
    if cond["tsm"] == "wsola":
        return synthesis.wsola(x, r, n_fft)
    if cond["tsm"] == "pv" and cond["hop_div"] == 4:
        return synthesis.transpose(x, 600.0, 600.0 * r, 0.0, n_fft)
    return tsm.pv_tsm(x, r, n_fft, hop=n_fft // cond["hop_div"], locked=cond["tsm"] == "pv_locked")


def cell_params(n_fft, frame_ratio, kappa):
    """(impact period P, decay tau) of a cell, in samples: P follows the frame ratio, tau only the frame and kappa,
    so along the frame-ratio axis the decay stays fixed while the impact period changes."""
    return n_fft / frame_ratio, n_fft / kappa


def run_condition(name, cond, seeds=SEEDS, log=None):
    """Every cell of one condition: direction, speed ratio, frame, frame/P (source and target conventions), the
    recovery statistics and the dominant envelope line."""
    cells = []
    for direction, ratios in SPEED_RATIOS.items():
        for r in ratios:
            for n_fft in cond["n_ffts"]:
                for fr in FRAME_RATIOS:
                    period, tau = cell_params(n_fft, fr, cond["kappa"])
                    outs = [transpose(impact_train(period, tau, s, f_res=cond["f_res"], jitter=cond["jitter"],
                                                   noise=cond["noise"]), r, n_fft, cond) for s in range(seeds)]
                    found, spacing = np.nanmean([recovery(y, period / r) for y in outs], axis=0)
                    line, prominence = np.median([envelope_line(y, period / r) for y in outs], axis=0)
                    cells.append({"direction": direction, "speed_ratio": r, "n_fft": n_fft, "period": period,
                                  "tau": tau, "frame_ratio": fr, "frame_ratio_target": fr * r,
                                  "found": float(found), "spacing": float(spacing),
                                  "envelope_line": float(line), "line_prominence": float(prominence)})
                    if log:
                        log(f"{name:14s} {direction:4s} r {r:4.2f} N {n_fft:3d} N/P {fr:4.2f} found {found:5.2f} "
                            f"spacing {spacing:5.2f} line {line:4.2f}")
    return cells


def passes(cell, tol):
    return abs(cell["found"] - 1) <= tol and abs(cell["spacing"] - 1) <= tol


def interval(cells, tol=TOLERANCE, convention="source"):
    """{"pass": rho*_pass, "fail": rho*_fail} of a set of cells: the largest frame ratio up to which every cell
    passes (0 if the smallest fails), the smallest ratio at which no cell passes (None if there is none)."""
    key = "frame_ratio" if convention == "source" else "frame_ratio_target"
    ratios = sorted({round(c[key], 9) for c in cells})
    upto = 0.0
    for x in ratios:
        if not all(passes(c, tol) for c in cells if round(c[key], 9) <= x):
            break
        upto = x
    fail = next((x for x in ratios if not any(passes(c, tol) for c in cells if round(c[key], 9) == x)), None)
    return {"pass": upto, "fail": fail}


def intervals(cells):
    """interval() for every tolerance, convention and direction ("all", "down", "up")."""
    out = {}
    for convention in ("source", "target"):
        for tol in TOLERANCES:
            for direction in ("all", "down", "up"):
                sub = [c for c in cells if direction == "all" or c["direction"] == direction]
                out[f"{convention}|{tol:g}|{direction}"] = interval(sub, tol, convention)
    return out


def choose_convention(base_intervals, tol=TOLERANCE):
    """The period convention whose rho*_pass differs least between slowing down and speeding up at the base."""
    gap = {c: abs(base_intervals[f"{c}|{tol:g}|down"]["pass"] - base_intervals[f"{c}|{tol:g}|up"]["pass"])
           for c in ("source", "target")}
    return min(gap, key=lambda c: (gap[c], c != "source"))


def lower_end(results, convention, tol=TOLERANCE):
    """Smallest rho*_pass over the standard-vocoder conditions (every condition except the phase-locked vocoder and
    WSOLA), all directions, in the chosen convention: the frame rule's rho*."""
    return min(r["intervals"][f"{convention}|{tol:g}|all"]["pass"] for r in results.values()
               if r["condition"]["tsm"] == "pv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="base condition, 2 seeds (smoke test)")
    args = ap.parse_args()
    conds = {"base": dict(BASE)} if args.quick else conditions()
    seeds = 2 if args.quick else SEEDS
    results = {}
    for name, cond in conds.items():
        cells = run_condition(name, cond, seeds, log=lambda s: print(s, flush=True))
        results[name] = {"condition": {**cond, "n_ffts": list(cond["n_ffts"])}, "cells": cells,
                         "intervals": intervals(cells)}
    convention = choose_convention(results["base"]["intervals"])
    out = {"note": "exploratory R6 (reframe spec 2026-10-20 §5.1, amendment A1); synthetic only",
           "seeds": seeds, "quick": args.quick, "convention": convention,
           "lower_end": lower_end(results, convention), "conditions": results}
    if not args.quick:
        OUT.write_text(json.dumps(out, indent=1))
    print(f"convention {convention}, lower end {out['lower_end']:.2f}" + ("" if args.quick else f" -> {OUT}"),
          flush=True)


if __name__ == "__main__":
    main()
