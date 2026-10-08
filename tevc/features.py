"""Physics feature bank for evolutionary feature selection (D = 561).

Blocks, in order:
  time-domain statistics                         11
  relative FFT band energies (64 linear bands)   64
  wavelet-packet node stats, levels 1-6          126 nodes x (log rel. energy, kurtosis, crest) = 378
  envelope fault-order energies                  9 bands x {f_s, BPFO, BPFI, BSF} x harmonics 1-3 = 108
Fixed-frequency blocks (FFT/WPT) see the resonance bands; order blocks use the environment's shaft
speed, so they follow the fault frequencies.
"""
import numpy as np
import pywt
from scipy.stats import kurtosis, skew

from src.utils.fault_frequencies import compute_fault_frequencies

EPS = 1e-12
N_FFT_BANDS = 64
WPT_WAVELET, WPT_LEVELS = "db4", 6
N_ENV_BANDS = 8
ORDER_TARGETS = ("f_s", "BPFO", "BPFI", "BSF")
HARMONICS = (1, 2, 3)
TIME_NAMES = ["t_log_rms", "t_log_std", "t_log_mabs", "t_log_peak", "t_log_p2p", "t_crest",
              "t_kurt", "t_skew", "t_impulse", "t_shape", "t_clearance"]
WPT_STATS = ("loge", "kurt", "crest")


def feature_names():
    names = list(TIME_NAMES)
    names += [f"fft_b{i:02d}" for i in range(N_FFT_BANDS)]
    for level in range(1, WPT_LEVELS + 1):
        names += [f"wpt_L{level}_n{j:02d}_{s}" for j in range(2 ** level) for s in WPT_STATS]
    for band in ["full"] + [f"b{i}" for i in range(N_ENV_BANDS)]:
        names += [f"env_{band}_{t}x{h}" for t in ORDER_TARGETS for h in HARMONICS]
    return names


def _time_block(W):
    a = np.abs(W)
    rms = np.sqrt(np.mean(W ** 2, axis=1)) + EPS
    peak = a.max(axis=1) + EPS
    mabs = a.mean(axis=1) + EPS
    return np.column_stack([np.log(rms), np.log(W.std(axis=1) + EPS), np.log(mabs), np.log(peak),
                            np.log(np.ptp(W, axis=1) + EPS), peak / rms, kurtosis(W, axis=1),
                            skew(W, axis=1), peak / mabs, rms / mabs,
                            peak / (np.mean(np.sqrt(a), axis=1) ** 2 + EPS)])


def _fft_block(W):
    p = np.abs(np.fft.rfft(W * np.hanning(W.shape[1]), axis=1)) ** 2
    e = np.column_stack([b.sum(axis=1) for b in np.array_split(p[:, 1:], N_FFT_BANDS, axis=1)])
    return np.log(e / e.sum(axis=1, keepdims=True) + EPS)


def _wpt_block(W):
    wp = pywt.WaveletPacket(data=W, wavelet=WPT_WAVELET, mode="symmetric", maxlevel=WPT_LEVELS, axis=-1)
    cols = []
    for level in range(1, WPT_LEVELS + 1):
        coeffs = [node.data for node in wp.get_level(level, order="freq")]
        energies = np.column_stack([np.sum(c ** 2, axis=1) for c in coeffs])
        total = energies.sum(axis=1) + EPS
        for c, e in zip(coeffs, energies.T):
            rms = np.sqrt(np.mean(c ** 2, axis=1)) + EPS
            cols += [np.log(e / total + EPS), kurtosis(c, axis=1), np.abs(c).max(axis=1) / rms]
    return np.column_stack(cols)


def _envelope_block(W, fs, freqs):
    n = W.shape[1]
    spec = np.fft.fft(W, axis=1)
    f = np.fft.fftfreq(n, 1.0 / fs)
    ef = np.fft.rfftfreq(n, 1.0 / fs)
    df = fs / n
    edges = np.linspace(0.0, fs / 2, N_ENV_BANDS + 1)
    bands = [(0.0, fs / 2)] + list(zip(edges[:-1], edges[1:]))
    targets = [h * freqs[t] for t in ORDER_TARGETS for h in HARMONICS]
    floor_sel = (ef > 0) & (ef <= 1.2 * max(targets))
    hann = np.hanning(n)
    cols = []
    for lo, hi in bands:
        analytic_gain = np.where((f > 0) & (f >= lo) & (f < hi), 2.0, 0.0)  # one-sided band -> analytic signal
        env = np.abs(np.fft.ifft(spec * analytic_gain, axis=1))
        env -= env.mean(axis=1, keepdims=True)
        es = np.abs(np.fft.rfft(env * hann, axis=1))
        floor = np.median(es[:, floor_sel], axis=1) + EPS
        for target in targets:
            sel = np.abs(ef - target) <= max(df, 0.02 * target)
            cols.append(np.log(es[:, sel].max(axis=1) / floor + EPS))
    return np.column_stack(cols)


def extract(W, fs, f_shaft, geometry, znorm=False):
    """W: windows (N, L) recorded at shaft frequency f_shaft (Hz). Returns (N, 561) float32.

    znorm: z-score each window first, as in every deep-learning RPST experiment. RPST transposes time
    scale only, so absolute amplitude (which grows with speed in real data) is not transposable.
    """
    W = np.asarray(W, dtype=np.float64)
    if znorm:
        W = (W - W.mean(axis=1, keepdims=True)) / (W.std(axis=1, keepdims=True) + EPS)
    freqs = compute_fault_frequencies(f_shaft_hz=f_shaft, **geometry)
    X = np.column_stack([_time_block(W), _fft_block(W), _wpt_block(W), _envelope_block(W, fs, freqs)])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
