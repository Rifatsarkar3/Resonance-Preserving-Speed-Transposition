import json

import pytest

from tevc.probes import rpstda_heldout as rh
from tevc.run_stream import RUN_SEEDS


def _visit(acc, env):
    trials = [{"acc": acc, "fitness": acc, "tv_acc": 0.5}, {"acc": acc + 0.05, "fitness": acc - 0.1, "tv_acc": 0.5}]
    return {"env": env, "deployed_acc": acc, "auc": acc, "n_evals": 2, "trials": trials, "deployed_acc_im": acc,
            "deployed_acc_tv": acc, "macro_f1": acc, "missed_fault_rate": 1 - acc, "false_alarm_rate": 0.1,
            "ece": 0.1}


def _results(levels):
    return {m: {str(s): [_visit(a, "e1"), _visit(a, "e2")] for s in RUN_SEEDS} for m, a in levels.items()}


def test_load_own_refuses_gaps_and_mixed_commits(tmp_path):
    (tmp_path / "exploratory_rpstda_shard0of1.json").write_text(
        json.dumps({"commit": "abc", "results": _results({m: 0.5 for m in rh.ARMS})}))
    assert rh.load_own(tmp_path)[1] == "abc"
    (tmp_path / "exploratory_rpstda_shard1of2.json").write_text(json.dumps({"commit": "def", "results": {}}))
    with pytest.raises(SystemExit, match="mixed"):
        rh.load_own(tmp_path)


def test_analysis_tests_rpstda_against_every_comparator_and_reports_selection_loss():
    out = rh.analysis(_results({m: 0.5 for m in rh.COMPARATORS} | {"RPST-DA": 0.9, "RPST-DA-NC": 0.6}))
    rows = {(r["second"], r["metric"]): r for r in out["tests"]}
    assert set(rows) == {(m, k) for m in rh.VERSUS for k in ("deployed_acc", "auc")}
    assert rows[("CDAN", "deployed_acc")]["mean_diff"] == pytest.approx(0.4)
    assert rows[("CDAN", "deployed_acc")]["wins"] == len(RUN_SEEDS)
    assert out["selection_loss"]["RPST-DA"] == pytest.approx(0.05)
