"""Phase-vocoder time-scale modification with a free hop and optional identity phase locking (MSSP reframe spec
docs/superpowers/specs/2026-10-20-mssp-sp-first-reframe-design.md §5.1, R6: is the validity boundary a property of
the standard vocoder only?).

pv_tsm(locked=False) is the algorithm of librosa.phase_vocoder (horizontal phase coherence: every bin propagates its
own phase with its instantaneous frequency), here with any hop. pv_tsm(locked=True) adds the identity phase locking
of Laroche & Dolson (1999): in each synthesis frame, every bin takes the propagated phase of the spectral peak whose
region of influence holds it, plus its analysis phase offset from that peak, which keeps the vertical phase
relations of a transient and reduces its smearing. transpose_pv mirrors tevc.synthesis.transpose (resampling by
r ** beta, then time-scale modification by r ** (1 - beta)).
"""
import librosa
import numpy as np

from tevc import synthesis


def _peaks(mag):
    """Indices of the local maxima of one frame's magnitude (a bin above its left and at least its right neighbour)."""
    inner = (mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:])
    return np.flatnonzero(inner) + 1


def _owners(n_bins, peaks):
    """For every bin, the peak whose region of influence holds it; regions split halfway between adjacent peaks."""
    mids = (peaks[:-1] + peaks[1:]) / 2.0
    return peaks[np.searchsorted(mids, np.arange(n_bins), side="right")]


def pv_tsm(y, rate, n_fft, hop=None, locked=False):
    """Time-scale modification of y by `rate` (> 1 shortens; output length int(len(y) / rate)) with a Hann frame of
    n_fft samples and hop (default n_fft // 4, as in tevc.synthesis). locked: identity phase locking."""
    hop = hop or n_fft // 4
    y = np.asarray(y, dtype=np.float32)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    n_bins = D.shape[0]
    steps = np.arange(0, D.shape[1], rate, dtype=np.float64)
    out = np.zeros((n_bins, len(steps)), dtype=D.dtype)
    advance = 2.0 * np.pi * hop * np.arange(n_bins) / n_fft      # expected phase advance per hop, bin by bin
    acc = np.angle(D[:, 0]).astype(np.float64)
    D = np.pad(D, [(0, 0), (0, 2)])
    for t, step in enumerate(steps):
        cols = D[:, int(step):int(step) + 2]
        alpha = step % 1.0
        mag = (1.0 - alpha) * np.abs(cols[:, 0]) + alpha * np.abs(cols[:, 1])
        phase = acc
        if locked:
            peaks = _peaks(mag)
            if len(peaks):
                own = _owners(n_bins, peaks)
                analysis = np.angle(cols[:, 0])
                phase = acc[own] + analysis - analysis[own]
        out[:, t] = mag * np.exp(1j * phase)
        dphase = np.angle(cols[:, 1]) - np.angle(cols[:, 0]) - advance
        dphase -= 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))
        acc = phase + advance + dphase
    return librosa.istft(out, hop_length=hop, n_fft=n_fft, length=int(len(y) / rate)).astype(np.float32)


def transpose_pv(y, src_rpm, tgt_rpm, beta=0.0, n_fft=256, hop=None, locked=False):
    """Generalised speed transposition with r = tgt / src (impact rate x r, resonance bands x r ** beta), with the
    time-scale part done by pv_tsm (free hop, optional phase locking)."""
    r = tgt_rpm / src_rpm
    if beta > 0:
        y = synthesis.resample(y, src_rpm, src_rpm * r ** beta)
    if beta < 1:
        y = pv_tsm(y, r ** (1 - beta), n_fft, hop, locked)
    return y
