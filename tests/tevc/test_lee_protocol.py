import json
from dataclasses import asdict

import numpy as np
import pytest

from tevc import lee_data
from tevc.lee_protocol import IM_ONLY, METHODS, PRIMARY, SECONDARY, analyse, best_of, run, verdict
from tevc.recipe import Recipe
from tevc.run_stream import EXTRA, RUN_SEEDS
from tevc.run_stream import METHODS as STREAM_METHODS
from tevc.stream import Environment, Stream

ENV = Environment("lee@1200", 1203.0, {}, {"y": np.repeat(np.arange(4), 2)})
DEPLOYED = {**asdict(Recipe(0, 0.0, 0.0, 256, 0.0, 0, True)), "fitness": 1.0}
LEADS = {"RPST-DA": 0.3, "AGENT": 0.2}                   # RPST-DA > AGENT > every other method, in every run


def _results(im_only_gap=0.05):
    def envs(m):
        acc = 0.5 + LEADS.get(m, 0.0)
        return [{"env": ENV.name, "deployed_acc": acc, "auc": acc, "n_evals": 30,
                 "trials": [{"fitness": 1.0, "acc": 0.5, "tv_acc": 0.5}, {"fitness": 0.5, "acc": acc + 0.1, "tv_acc": 0.6}],
                 "deployed": {**DEPLOYED, "seed": 1 if m in LEADS else 2},
                 "deployed_acc_im": acc - im_only_gap, "deployed_acc_tv": acc}]
    return {m: {str(s): envs(m) for s in RUN_SEEDS} for m in METHODS}


def test_protocol_names_known_methods_and_the_five_primary_hypotheses():
    assert set(METHODS) <= set(STREAM_METHODS + EXTRA) and len(RUN_SEEDS) == 10
    assert list(PRIMARY) == ["E1", "E2", "E3", "E4", "E5"]
    assert all({a, b} <= set(METHODS) | {IM_ONLY} for rows in [PRIMARY.values(), *SECONDARY.values()]
               for a, b, _ in rows)


def test_a_lead_in_every_run_confirms_all_five():
    v = verdict(_results())
    assert all(r["confirmed"] and r["wins"] == 10 for r in v.values()) and np.isclose(v["E1"]["p_holm"], 5 / 1024)
    assert np.isclose(v["E5"]["mean_diff"], 0.05)                  # the rule over IM-only on the same candidates


def test_e5_needs_transposed_validation_to_beat_im_only_selection():
    v = verdict(_results(im_only_gap=0.0))
    assert not v["E5"]["confirmed"] and v["E1"]["confirmed"]


def test_best_of_counts_the_best_evaluated_or_deployed_model():
    assert np.isclose(best_of(_results())["RPST-DA"], 0.9) and np.isclose(best_of(_results())["DSR"], 0.6)


def test_the_runner_refuses_while_target_records_are_locked(monkeypatch):
    monkeypatch.setattr(lee_data, "TARGETS_RELEASED", False)
    with pytest.raises(SystemExit, match="locked"):
        run(0, 2)


class _DS:
    n_classes = 4


class _Cache:
    """cached(): perfect test predictions for seed 1, everything called healthy otherwise."""

    def cached(self, env, config, seed):
        y = env.T["y"]
        return None, np.eye(4)[y if seed == 1 else np.zeros_like(y)]


def test_analysis_runs_once_on_a_complete_single_commit_grid(tmp_path):
    res, stream = _results(), Stream(_DS, [], {}, 801.7, (ENV,))
    for i, part in enumerate((RUN_SEEDS[0::2], RUN_SEEDS[1::2])):
        if i == 1:
            with pytest.raises(SystemExit, match="incomplete"):
                analyse(tmp_path, stream, _Cache())
        shard = {m: {str(s): runs[str(s)] for s in part} for m, runs in res.items()}
        (tmp_path / f"core_shard{i}of2.json").write_text(json.dumps({"commit": "abc", "results": shard}))
    out = analyse(tmp_path, stream, _Cache())
    assert out["protocol_commit"] == "abc" and all(r["confirmed"] for r in out["primary"].values())
    assert out["S3"]["RPST-DA"]["missed_fault_rate"] == 0.0 and out["S3"]["DSR"]["missed_fault_rate"] == 1.0
    assert set(out["S4"]["tv"]) == {"RPST-DA", "RPST-DA-NC"} and np.isclose(out["S5"]["RPST-DA"], 0.9)
    with pytest.raises(SystemExit, match="runs once"):
        analyse(tmp_path, stream, _Cache())


def test_analysis_refuses_shards_from_different_commits(tmp_path):
    for i, commit in enumerate(("abc", "def")):
        (tmp_path / f"core_shard{i}of2.json").write_text(json.dumps({"commit": commit, "results": _results()}))
    with pytest.raises(SystemExit, match="2 commits"):
        analyse(tmp_path, Stream(_DS, [], {}, 801.7, (ENV,)), _Cache())
