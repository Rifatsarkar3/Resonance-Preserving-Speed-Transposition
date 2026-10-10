import json

import numpy as np
import pytest

from tevc.confirmatory import GRIDS, PRIMARY, SECONDARY, analyse, fitness_validity, holm, verdict
from tevc.run_stream import EXTRA, METHODS, RUN_SEEDS


def _results(lead):
    """AGENT leads every other core method by `lead` in every run, on accuracy and AUC."""
    def envs(v):
        return [{"env": "e", "deployed_acc": v, "auc": v, "n_evals": 30, "trials": [{"fitness": v, "acc": v}]}]
    return {m: {str(s): envs(0.5 + (lead if m == "AGENT" else 0.0)) for s in RUN_SEEDS} for m in GRIDS["core"]}


def test_protocol_names_known_methods_and_five_primary_hypotheses():
    assert all(set(g) <= set(METHODS + EXTRA) for g in GRIDS.values()) and len(RUN_SEEDS) == 10
    assert len(PRIMARY) == 5 and all({a, b} <= set(GRIDS["core"]) for a, b, _ in PRIMARY.values())
    for name, rows in SECONDARY.items():
        grid = GRIDS["secondary" if name == "S3" else "core"]
        assert all({a, b} <= set(grid) for a, b, _ in rows)


def test_holm_steps_down_and_stays_monotone():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_a_lead_in_every_run_confirms_all_five():
    v = verdict(_results(0.05))
    assert all(r["confirmed"] and r["wins"] == 10 for r in v.values())
    assert np.isclose(v["H1"]["p"], 1 / 1024) and np.isclose(v["H1"]["p_holm"], 5 / 1024)


def test_no_lead_confirms_nothing():
    assert not any(r["confirmed"] for r in verdict(_results(0.0)).values())


def test_fitness_validity_pools_evaluations_per_environment():
    res = {"A": {"1": [{"env": "e", "trials": [{"fitness": f, "acc": f / 10} for f in (1, 2, 3)]}]}}
    v = fitness_validity(res, ["A"])["e"]
    assert v["n"] == 3 and np.isclose(v["rho"], 1.0)


def test_analysis_runs_once_on_a_complete_single_commit_grid(tmp_path):
    res = _results(0.05)
    for i, part in enumerate((RUN_SEEDS[0::2], RUN_SEEDS[1::2])):
        if i == 1:
            with pytest.raises(SystemExit, match="incomplete"):
                analyse("core", tmp_path)
        shard = {m: {str(s): runs[str(s)] for s in part} for m, runs in res.items()}
        (tmp_path / f"core_shard{i}of2.json").write_text(json.dumps({"commit": "abc", "results": shard}))
    out = analyse("core", tmp_path)
    assert out["protocol_commit"] == "abc" and all(r["confirmed"] for r in out["primary"].values())
    with pytest.raises(SystemExit, match="runs once"):
        analyse("core", tmp_path)


def test_analysis_refuses_shards_from_different_commits(tmp_path):
    for i, commit in enumerate(("abc", "def")):
        (tmp_path / f"core_shard{i}of2.json").write_text(json.dumps({"commit": commit, "results": _results(0.05)}))
    with pytest.raises(SystemExit, match="2 commits"):
        analyse("core", tmp_path)
