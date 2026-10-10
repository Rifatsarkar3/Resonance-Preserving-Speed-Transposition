import json

from tevc.probes import dev_gate
from tevc.run_stream import RUN_SEEDS, summarize

PU_BASE = ("STATIC-NONE", "STATIC-H2", "RANDOM", "GA-RESTART", "AGENT", "AGENT-B1")
PU_EXTRA = ("BO", "WARM-TRANSFER", "ORACLE-FITNESS", "AGENT")


def _write(root, name, methods, value):
    res = {m: {str(s): [{"env": "e", "deployed_acc": value(m), "auc": value(m), "n_evals": 30}] for s in RUN_SEEDS}
           for m in methods}
    (root / name).write_text(json.dumps({"results": res}))


def _merged(tmp_path, monkeypatch, dataset, base, v2):
    monkeypatch.setitem(dev_gate.SOURCES, dataset, ("base.json", "extra.json"))
    _write(tmp_path, "base.json", PU_BASE, base)
    _write(tmp_path, "extra.json", PU_EXTRA, base)
    _write(tmp_path, f"stream_{dataset}_v2.json", ("AGENT", "AGENT-B1"), v2)
    results = dev_gate.merged(dataset, tmp_path)
    return results, summarize(results)


def test_agent_rows_come_only_from_the_v2_file(tmp_path, monkeypatch):
    results, summary = _merged(tmp_path, monkeypatch, "pu", lambda m: 0.5, lambda m: 0.9 if m == "AGENT" else 0.4)
    assert results["AGENT"]["42"][0]["deployed_acc"] == 0.9 and results["AGENT-B1"]["42"][0]["deployed_acc"] == 0.4
    assert dev_gate.gate("pu", results, summary) == (True, [])


def test_a_baseline_beating_the_agent_in_every_run_fails_the_jnu_gate(tmp_path, monkeypatch):
    results, summary = _merged(tmp_path, monkeypatch, "jnu", lambda m: 0.6 if m == "STATIC-H2" else 0.5,
                               lambda m: 0.55)
    passes, worse = dev_gate.gate("jnu", results, summary)
    assert not passes and set(worse) == {("STATIC-H2", "deployed_acc"), ("STATIC-H2", "auc")}
