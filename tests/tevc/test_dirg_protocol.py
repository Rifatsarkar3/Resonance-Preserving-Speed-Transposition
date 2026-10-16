import json
from dataclasses import asdict

import numpy as np
import pytest

from tevc.dirg_protocol import (METHODS, PRIMARY, SECONDARY, analyse, check_admissible, family, location_accuracy,
                                verdict)
from tevc.recipe import Recipe
from tevc.run_stream import EXTRA, RUN_SEEDS
from tevc.run_stream import METHODS as STREAM_METHODS
from tevc.stream import Environment, Stream

ENV = Environment("dirg@400Hz", 22500.0, {}, {"y": np.repeat(np.arange(7), 2)})
DEPLOYED = {**asdict(Recipe(0, 0.0, 0.0, 256, 0.0, 0, True)), "fitness": 1.0}


def _results(lead):
    """AGENT leads every method by `lead` in accuracy and AUC and misses `lead` fewer faults, in every run."""
    def envs(m, s):
        a = m == "AGENT"
        return [{"env": ENV.name, "deployed_acc": 0.5 + lead * a, "auc": 0.5 + lead * a,
                 "missed_fault_rate": 0.2 - lead * a, "n_evals": 30, "trials": [{"fitness": 1.0, "acc": 0.5}],
                 "deployed": {**DEPLOYED, "seed": 1 if a else 2}}]
    return {m: {str(s): envs(m, s) for s in RUN_SEEDS} for m in METHODS}


def test_protocol_names_known_methods_and_drops_the_pcs_hypothesis():
    assert set(METHODS) <= set(STREAM_METHODS + EXTRA) and len(RUN_SEEDS) == 10
    assert list(PRIMARY) == ["D1", "D2", "D3", "D5", "D6"]
    assert all({a, b} <= set(METHODS) for rows in [PRIMARY.values(), *SECONDARY.values()] for a, b, _ in rows)


def test_a_lead_in_every_run_confirms_every_hypothesis_including_the_lower_is_better_ones():
    v = verdict(_results(0.05))
    assert all(r["confirmed"] and r["wins"] == 10 and r["mean_diff"] > 0 for r in v.values())
    assert v["D5"]["better"] == "lower" and np.isclose(v["D5"]["p_holm"], 5 / 1024)


def test_more_missed_faults_never_confirm_d5():
    res = _results(0.05)
    for envs in res["AGENT"].values():
        envs[0]["missed_fault_rate"] = 0.3
    assert not verdict(res)["D5"]["confirmed"] and verdict(res)["D1"]["confirmed"]


def test_family_holm_counts_only_its_own_rows():
    rows = family(_results(0.05), [("AGENT", "DSR", "deployed_acc")])
    assert np.isclose(rows[0]["p_holm"], 1 / 1024)


def test_location_accuracy_merges_fault_sizes():
    y = np.array([0, 1, 2, 3, 4, 5, 6])
    pred = np.array([0, 3, 1, 2, 6, 4, 1])            # sizes confused everywhere, location wrong once (6 -> 1)
    assert np.isclose(location_accuracy(np.eye(7)[pred], y), 6 / 7)


def test_admissibility_rejects_a_speed_shift_below_1_3():
    lib = 278.0 * 60
    ok = Stream(None, [], {}, lib, (Environment("a", lib * 1.35, {}, {}), Environment("b", lib / 1.4, {}, {})))
    check_admissible(ok)
    with pytest.raises(SystemExit, match="ratio"):
        check_admissible(Stream(None, [], {}, lib, (Environment("c", lib * 1.2, {}, {}),)))


class _DS:
    n_classes = 7


class _Cache:
    """cached(): perfect test predictions for seed 1, everything called healthy otherwise."""

    def cached(self, env, config, seed):
        y = env.T["y"]
        return None, np.eye(7)[y if seed == 1 else np.zeros_like(y)]


def test_analysis_runs_once_on_a_complete_single_commit_grid(tmp_path):
    res, stream = _results(0.05), Stream(_DS, [], {}, 16800.0, (ENV,))
    for i, part in enumerate((RUN_SEEDS[0::2], RUN_SEEDS[1::2])):
        if i == 1:
            with pytest.raises(SystemExit, match="incomplete"):
                analyse(tmp_path, stream, _Cache())
        shard = {m: {str(s): runs[str(s)] for s in part} for m, runs in res.items()}
        (tmp_path / f"core_shard{i}of2.json").write_text(json.dumps({"commit": "abc", "results": shard}))
    out = analyse(tmp_path, stream, _Cache())
    assert out["protocol_commit"] == "abc" and all(r["confirmed"] for r in out["primary"].values())
    assert out["S5"]["summary"]["AGENT"]["missed_fault_rate"] == 0.0
    assert out["S5"]["summary"]["DSR"]["missed_fault_rate"] == 1.0           # recomputed from the cache
    assert [r["metric"] for r in out["S3"]] == ["location_acc"] * 3 and out["S3"][0]["wins"] == 10
    with pytest.raises(SystemExit, match="runs once"):
        analyse(tmp_path, stream, _Cache())


def test_analysis_refuses_shards_from_different_commits(tmp_path):
    for i, commit in enumerate(("abc", "def")):
        (tmp_path / f"core_shard{i}of2.json").write_text(json.dumps({"commit": commit, "results": _results(0.05)}))
    with pytest.raises(SystemExit, match="2 commits"):
        analyse(tmp_path, Stream(_DS, [], {}, 16800.0, (ENV,)), _Cache())
