"""EXPLORATORY (RESS spec §6, after the DIRG freeze and verdict): operator fidelity on Ottawa, the fourth dataset.
The trial-1 library chunks (13-16 Hz) are transposed from their own measured speed to each environment's measured
speed with β = 0 (RPST) and β = 1 (resampling), and compared class by class with the real trial-2 windows (U) of
that environment (tevc.probes.operator_fidelity metrics; band_error = resonance placement). "untransposed" = the
library windows as recorded. Labels are used: physics validation only, nothing feeds back into any method, and the
withdrawn v1 shards are never read.

    python -m tevc.probes.ottawa_fidelity      -> outputs/tevc/ottawa/exploratory_fidelity.json
"""
import json
from pathlib import Path

from tevc import synthesis
from tevc.datasets import make_windows
from tevc.probes.dirg_diagnostics import OPERATORS
from tevc.probes.operator_fidelity import CAP, _compare
from tevc.stream import ottawa_stream

OUT = Path("outputs/tevc/ottawa/exploratory_fidelity.json")


def main():
    stream = ottawa_stream()
    ds = stream.dataset
    out = {"note": "exploratory, after the DIRG verdict", "library_rpm": stream.library_rpm, "environments": {}}
    for env in dict.fromkeys(stream.environments):
        rows = {}
        for label, op in OPERATORS.items():
            if op is None:
                synth = make_windows(stream.library_entries, ds.win, ds.hop, CAP)
            else:
                beta, n_fft = op

                def to_env(lab, rec, sig, b=beta, n=n_fft):
                    s0 = stream.src_rpm_of(rec)
                    return synthesis.transpose(sig, s0, env.rpm, b, n), env.rpm / s0

                synth = make_windows(stream.library_entries, ds.win, ds.hop, CAP, synth=to_env)
            rows[label] = _compare(synth, env.U, ds, env.rpm / 60.0)
            print(f"{env.name} {label:26s} {json.dumps({k: round(v, 4) for k, v in rows[label].items()})}",
                  flush=True)
        out["environments"][env.name] = {"speed_ratio": env.rpm / stream.library_rpm, "operators": rows}
        OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
