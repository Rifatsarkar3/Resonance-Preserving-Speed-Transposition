"""Environment-synthesis operators: RPST and the resampling control.

Parameters are identical to the RPST used in all earlier RPST experiments
(librosa phase vocoder, n_fft 256, hop 64).
"""
from math import gcd

import librosa
import numpy as np
from scipy.signal import resample_poly

N_FFT, HOP = 256, 64


def rpst(y, src_rpm, tgt_rpm):
    """Resonance-preserving speed transposition: impulse rate scales by tgt/src, resonance bands stay fixed."""
    rate = tgt_rpm / src_rpm
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    D2 = librosa.phase_vocoder(D, rate=rate, hop_length=HOP)
    return librosa.istft(D2, hop_length=HOP, n_fft=N_FFT, length=int(len(y) / rate)).astype(np.float32)


def resample(y, src_rpm, tgt_rpm):
    """Plain resampling control: every frequency scales, so resonance bands move too."""
    rate = tgt_rpm / src_rpm
    num, den = int(round(rate * 1000)), 1000
    g = gcd(num, den)
    return resample_poly(y, den // g, num // g).astype(np.float32)


def transpose(y, src_rpm, tgt_rpm, beta=0.0, n_fft=N_FFT):
    """Generalised speed transposition with r = tgt/src: impulse rate x r, resonance bands x r**beta.

    beta = 0 with n_fft 256 is exactly rpst(); beta = 1 is exactly resample(). Implemented as resampling
    by r**beta (moves bands and impulse rate) then a phase vocoder by r**(1 - beta) (impulse rate only).
    """
    r = tgt_rpm / src_rpm
    if beta > 0:
        y = resample(y, src_rpm, src_rpm * r ** beta)
    if beta < 1:
        hop = n_fft // 4
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop)
        D2 = librosa.phase_vocoder(D, rate=r ** (1 - beta), hop_length=hop)
        y = librosa.istft(D2, hop_length=hop, n_fft=n_fft, length=int(len(y) / r ** (1 - beta))).astype(np.float32)
    return y
