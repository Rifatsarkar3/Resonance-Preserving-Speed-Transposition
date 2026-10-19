"""EXPLORATORY (MSSP spec §4 secondary and §5 M7, after the one-time Lee verdict 3bdb9b9): operator fidelity on Lee.
The 800-rpm library records are transposed to each environment's estimated speed and compared class by class with
the real U windows (tevc.probes.operator_fidelity: band_error = resonance placement, envelope_corr = impulse rate in
shaft orders). Operators: untransposed; RPST (β = 0) at the protocol's frames, frame(64..512) = 16-128 samples
(1-8 ms, PU's durations); RPST at 256 and 512 samples (16 and 32 ms), longer than the inner-race impact period at the
library speed (~15 ms), i.e. outside the vocoder's validity regime (tevc.probes.frame_validity); resampling (β = 1).
Labels are used: physics validation only.

    python -m tevc.probes.lee_fidelity      -> outputs/tevc/lee/exploratory_fidelity.json
"""
import json
from pathlib import Path

from tevc.lee_data import LEE
from tevc.probes.operator_fidelity import fidelity_by_env
from tevc.recipe import N_FFTS, frame
from tevc.stream import lee_stream

OUT = Path("outputs/tevc/lee/exploratory_fidelity.json")
PROTOCOL_FRAMES = tuple(frame(n, LEE) for n in N_FFTS)
LONG_FRAMES = (256, 512)
OPERATORS = {"untransposed": None,
             **{f"beta=0 n_fft={n} (RPST)": (0.0, n) for n in PROTOCOL_FRAMES + LONG_FRAMES},
             "beta=1 (resampling)": (1.0, 256)}


def main():
    stream = lee_stream()
    out = {"note": "exploratory, after the Lee verdict (3bdb9b9)", "library_rpm": stream.library_rpm,
           "environments": fidelity_by_env(stream, OPERATORS, log=lambda s: print(s, flush=True))}
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
