"""DIRG smoke test before the freeze (RESS spec §5.8): training only. It builds the DIRG stream, checks
disjointness, prints the label-free speed estimates and the buffer sizes, then trains candidate 0 of every method
family on the first environment and prints training times. No model predicts on U or T, nothing is cached, and no
accuracy is computed.

    python -m tevc.probes.dirg_smoke
"""
import time

import numpy as np

from tevc import baselines
from tevc.deep_train import train_cnn
from tevc.recipe import ANCHORS, training_set
from tevc.stream import assert_disjoint, dirg_stream

MODEL, EPOCHS, SEED = "TICNN", 40, 4200


def main():
    t0 = time.time()
    stream = dirg_stream()
    assert_disjoint(stream)
    ds = stream.dataset
    print(f"stream built in {time.time() - t0:.0f}s; library {len(stream.library['y'])} windows "
          f"{np.bincount(stream.library['y']).tolist()}, estimated speed {stream.library_rpm / 60:.2f} Hz")
    for env in dict.fromkeys(stream.environments):
        print(f"{env.name}: estimated speed {env.rpm / 60:.2f} Hz, U {len(env.U['y'])} windows, "
              f"T {len(env.T['y'])} windows")
    env = stream.environments[0]
    for name in ("NONE", "H2_LIKE", "RESAMPLE_H2"):
        t = time.time()
        W, y = training_set(ANCHORS[name], ds, stream.library_entries, stream.library, stream.library_rpm, env.rpm)
        train_cnn(W, y, MODEL, EPOCHS, SEED)
        print(f"recipe {name:12s} {len(y):5d} training windows  {time.time() - t:6.1f}s", flush=True)
    for m in baselines.METHODS:
        t = time.time()
        baselines.train(baselines.sample_configs(m, 42)[0], stream, env, SEED, MODEL, EPOCHS)
        print(f"baseline {m:10s} trained  {time.time() - t:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
