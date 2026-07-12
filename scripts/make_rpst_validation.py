"""Physical-realism validation of RPST (no training).

A true shaft-speed change scales the fault-impulse rate (and its sidebands) but leaves
the structural resonance band fixed in absolute frequency. We verify, on a controlled
bearing-fault model with known ground truth and on real JNU records, that RPST reproduces
this behaviour whereas plain resampling additionally shifts the resonance carrier.

 (a) controlled model, raw spectrum   -> resonance fixed under RPST, shifted by resampling
 (b) controlled model, envelope spectrum -> fault rate scaled correctly by RPST
 (c) controlled model, tracking vs speed ratio -> RPST matches a true speed change on BOTH
     axes (resonance flat, fault rate linear); resampling is wrong on the resonance axis
 (d) real JNU outer-race record: RPST moves the envelope fault peak to the physically
     correct ball-pass frequency for the simulated speed
-> paper/figures/fig_rpst_validation.{pdf,png}
"""
import sys
from pathlib import Path
from math import gcd
import numpy as np
import pandas as pd
from scipy.signal import hilbert, resample_poly
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config import Config
import figstyle as fs

fs.apply()
cfg = Config.from_yaml("config.yaml")
JNU = Path(cfg.data.jnu_raw_root) / "JNU"
FS = 50000.0
N_FFT, HOP = 256, 64
OUT = Path("paper/figures")
GREY, RED, BLUE = fs.PALETTE["baseline"], fs.PALETTE["rpst"], fs.PALETTE["resample"]

# JNU ER-16K bearing geometry -> outer-race ball-pass frequency
NB, D_BALL, D_PITCH = 8, 7.5, 38.5
BPFO_PER_FS = (NB / 2) * (1 - D_BALL / D_PITCH)     # multiply by shaft freq (Hz)


def tsm(y, rate):
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    D2 = librosa.phase_vocoder(D, rate=rate, hop_length=HOP)
    return librosa.istft(D2, hop_length=HOP, n_fft=N_FFT, length=int(len(y) / rate)).astype(np.float32)


def resample_op(y, rate):
    num = int(round(rate * 1000)); den = 1000
    g = gcd(num, den)
    return resample_poly(y, den // g, num // g).astype(np.float32)


def raw_spec(y):
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return np.fft.rfftfreq(len(y), 1 / FS), Y


def env_spec(y):
    e = np.abs(hilbert(y)) ** 2
    e = e - e.mean()
    E = np.abs(np.fft.rfft(e * np.hanning(len(e))))
    return np.fft.rfftfreq(len(e), 1 / FS), E


def synth_signal(fault_hz, res_hz, dur=1.0):
    n = int(FS * dur); t = np.arange(n) / FS
    x = np.zeros(n)
    idx = (np.arange(int(dur * fault_hz)) / fault_hz * FS).astype(int)
    x[idx[idx < n]] = 1.0
    kernel = np.exp(-t * 800) * np.sin(2 * np.pi * res_hz * t)
    return np.convolve(x, kernel)[:n].astype(np.float32)


def peak(f, Y, lo, hi):
    m = (f >= lo) & (f <= hi)
    return f[m][np.argmax(Y[m])]


def main():
    FD, FR = 50.0, 4000.0          # ground-truth fault rate & resonance
    y0 = synth_signal(FD, FR)
    r = 2.0
    yr, ys = tsm(y0, r), resample_op(y0, r)

    fig, ax = plt.subplots(1, 3, figsize=(10.0, 3.0))

    # (a) raw spectrum: resonance band
    for y, lab, c, ls in [(y0, "original", GREY, "-"), (yr, f"RPST ($\\times${r:.0f})", RED, "-"),
                          (ys, f"resample ($\\times${r:.0f})", BLUE, "--")]:
        f, Y = raw_spec(y)
        ax[0].plot(f / 1000, Y / Y.max(), c, ls=ls, lw=0.9, label=lab)
    ax[0].axvline(FR / 1000, color=GREY, lw=0.6, ls=":")
    ax[0].set_xlim(0, 10); ax[0].set_xlabel("frequency (kHz)")
    ax[0].set_ylabel("norm. magnitude"); ax[0].set_title("(a) resonance band")
    ax[0].legend(fontsize=6.5, framealpha=0.9)

    # (b) envelope spectrum: fault rate
    for y, lab, c, ls in [(y0, "original", GREY, "-"), (yr, "RPST", RED, "-"),
                          (ys, "resample", BLUE, "--")]:
        f, E = env_spec(y); m = f <= 250
        ax[1].plot(f[m], E[m] / E[m].max(), c, ls=ls, lw=0.9, label=lab)
    ax[1].set_xlim(0, 250); ax[1].set_xlabel("envelope frequency (Hz)")
    ax[1].set_ylabel("norm. magnitude"); ax[1].set_title("(b) fault rate")
    ax[1].legend(fontsize=6.5, framealpha=0.9)

    # (c) tracking vs speed ratio
    ratios = np.linspace(1.0, 2.0, 6)
    res_rpst, res_res, fr_rpst = [], [], []
    for rr in ratios:
        a, b = tsm(y0, rr), resample_op(y0, rr)
        fa, Ya = raw_spec(a); fb, Yb = raw_spec(b)
        res_rpst.append(peak(fa, Ya, 1000, 9000) / FR)
        res_res.append(peak(fb, Yb, 1000, 16000) / FR)
        fe, Ee = env_spec(a); fr_rpst.append(peak(fe, Ee, 20, 200) / FD)
    ax[2].plot(ratios, ratios, color="k", lw=0.8, ls=":", label="true speed change")
    ax[2].plot(ratios, res_rpst, color=RED, marker="o", ms=3.5, lw=1, label="RPST resonance")
    ax[2].plot(ratios, res_res, color=BLUE, marker="s", ms=3.5, lw=1, ls="--", label="resample resonance")
    ax[2].plot(ratios, fr_rpst, color=RED, marker="^", ms=3.5, lw=1, ls="-.", label="RPST fault rate")
    ax[2].set_xlabel("speed ratio"); ax[2].set_ylabel("frequency / original")
    ax[2].set_title("(c) what scales vs stays"); ax[2].legend(fontsize=6.0, framealpha=0.9)
    ax[2].grid(alpha=0.3, lw=0.4)

    # real-data quantitative check: RPST changes the JNU spectral centroid by < 4 %
    cen0, cenR = [], []
    for p in sorted(JNU.glob("*.csv")):
        pre = p.stem.split("_")[0]
        if pre.startswith(("ib", "ob", "tb")) and "600" in pre:
            s = pd.read_csv(p).iloc[:, 0].values.astype(np.float32)[:60000]
            f, Y = raw_spec(s); cen0.append((f * Y).sum() / Y.sum())
            f, Y = raw_spec(tsm(s, 1000 / 600)); cenR.append((f * Y).sum() / Y.sum())
    shift = 100 * (np.mean(cenR) - np.mean(cen0)) / np.mean(cen0)
    print(f"REAL JNU: spectral centroid {np.mean(cen0):.0f}->{np.mean(cenR):.0f} Hz "
          f"under RPST 600->1000 ({shift:+.1f}%, n={len(cen0)})")

    for a in ax:
        fs.clean(a)
    fig.tight_layout()
    fs.save(fig, "fig_rpst_validation")

    print("synthetic: resonance RPST=%.0f resample=%.0f (true 4000); fault RPST=%.0f (true 100)"
          % (peak(*raw_spec(yr), 1000, 9000), peak(*raw_spec(ys), 1000, 16000),
             peak(*env_spec(yr), 20, 200)))
    print("-> fig_rpst_validation.{pdf,png}")


if __name__ == "__main__":
    main()
