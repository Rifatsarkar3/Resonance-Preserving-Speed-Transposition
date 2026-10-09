import numpy as np

from tevc.probes.operator_fidelity import _envelopes, _ringdown

FS = 50000.0


def _ringing_bursts(tau_s, period_s=0.05, seconds=0.5, f_res=3000.0):
    """Impacts every period_s exciting a resonance at f_res that rings down with time constant tau_s."""
    t = np.arange(int(seconds * FS)) / FS
    phase = np.mod(t, period_s)
    return (np.exp(-phase / tau_s) * np.sin(2 * np.pi * f_res * t))[None, :]


def test_ringdown_metric_recovers_the_decay_time_constant():
    for tau in (0.001, 0.002, 0.004):
        measured = _ringdown(_envelopes(_ringing_bursts(tau)), FS)
        assert abs(measured - tau) / tau < 0.25
