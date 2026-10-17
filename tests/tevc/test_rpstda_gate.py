import json

import numpy as np
import pytest

import tevc.rpstda_gate as g
from tevc.run_stream import RUN_SEEDS


def _means(values):
    return dict(zip(RUN_SEEDS, values))


def _verdict(pu_rpstda, jnu_rpstda, pu_cdan=0.79, pu_agent=0.795, jnu_cdan=1.0, jnu_agent=0.999):
    pu = {"RPST-DA": _means(pu_rpstda), "CDAN": _means([pu_cdan] * 10), "AGENT": _means([pu_agent] * 10)}
    jnu = {"RPST-DA": _means([jnu_rpstda] * 10), "CDAN": _means([jnu_cdan] * 10), "AGENT": _means([jnu_agent] * 10)}
    return g.gate_verdict(pu, jnu)


def test_gate_needs_the_margin_the_paired_wins_and_no_harm_on_jnu():
    win7 = [0.82] * 7 + [0.78] * 3                                    # mean 0.808 >= 0.795 + 0.010; 7 wins
    assert _verdict(win7, 0.998)["passed"]
    assert not _verdict([0.83] * 6 + [0.785] * 4, 0.998)["passed"]   # margin met, only 6 wins
    assert not _verdict([0.80] * 10, 0.998)["passed"]                # 10 wins, below the margin
    assert not _verdict(win7, 0.996)["passed"]                       # JNU 0.4 pp below the best comparator


def _entry(acc):
    return {"env": "e", "deployed_acc": acc, "n_evals": 30, "deployed_acc_im": acc - 0.01, "deployed_acc_tv": acc,
            "trials": [{"acc": a, "tv_acc": a, "fitness": -a} for a in np.linspace(0.5, 0.9, 30)]}


def _archives(tmp_path, monkeypatch, rpstda_runs=RUN_SEEDS):
    monkeypatch.setattr(g, "OUT_DIR", tmp_path)
    monkeypatch.setattr(g, "GATE_OUT", tmp_path / "rpstda_gate.json")
    monkeypatch.setattr(g, "_head", lambda: "abc123")
    acc = {"pu": {"RPST-DA": 0.9, "CDAN": 0.8, "AGENT": 0.8}, "jnu": {"RPST-DA": 1.0, "CDAN": 1.0, "AGENT": 0.999}}
    for ds, files in g.GATE_FILES.items():
        for method, name in files.items():
            runs = rpstda_runs if method == "RPST-DA" else RUN_SEEDS
            (tmp_path / name).write_text(json.dumps(
                {"results": {method: {str(r): [_entry(acc[ds][method])] * 2 for r in runs}}}))


def test_gate_runs_once_and_writes_the_verdict_with_descriptives(tmp_path, monkeypatch):
    _archives(tmp_path, monkeypatch)
    g.gate()
    out = json.loads((tmp_path / "rpstda_gate.json").read_text())
    assert out["verdict"]["passed"] and out["commit"] == "abc123"
    d = out["descriptive"]["pu"]
    assert np.isclose(d["run_level_means"]["im_only"], 0.89)
    assert d["spearman_vs_test_acc"]["e"] == {"tv": 1.0, "im": -1.0}
    with pytest.raises(SystemExit, match="one shot"):
        g.gate()


def test_gate_refuses_an_incomplete_rpstda_grid(tmp_path, monkeypatch):
    _archives(tmp_path, monkeypatch, rpstda_runs=RUN_SEEDS[:9])
    with pytest.raises(SystemExit, match="lacks runs"):
        g.gate()
    assert not (tmp_path / "rpstda_gate.json").exists()
