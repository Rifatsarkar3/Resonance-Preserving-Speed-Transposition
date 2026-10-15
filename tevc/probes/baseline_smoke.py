"""Smoke run of the RESS baselines on development data: candidate 0 (the published setting) of each baseline on the
first environment of the JNU and PU streams, trained with run 42's candidate-0 seed, so the 10-run grids reuse these
cache entries. Prints the agent-visible fitness, the true test accuracy (development data) and the time per
candidate, to check the pipeline end to end and to size the grids. Writes only cache entries.

    python -m tevc.probes.baseline_smoke
"""
import time

from tevc import baselines
from tevc.evaluator import RecipeEvaluator
from tevc.stream import assert_disjoint, jnu_stream, pu_stream


def main():
    for make in (jnu_stream, pu_stream):
        stream = make()
        assert_disjoint(stream)
        ev, env = RecipeEvaluator(stream), stream.environments[0]
        for m in baselines.METHODS:
            config, seed, t = baselines.sample_configs(m, 42)[0], 42 * 100, time.time()
            e = ev.evaluate(env, config, seed)
            print(f"{stream.dataset.name:4s} {env.name:9s} {m:10s} fitness {e.fitness:6.3f}  acc "
                  f"{ev.true_accuracy(env, config, seed) * 100:5.1f}  {time.time() - t:6.1f}s  cached {e.cached}",
                  flush=True)


if __name__ == "__main__":
    main()
