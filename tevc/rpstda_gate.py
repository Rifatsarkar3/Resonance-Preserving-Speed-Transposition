"""One-shot development gate of RPST-DA (MSSP spec 2026-10-18 §3), on the PU and JNU development grids against the
archived CDAN (stream_*_rbase.json) and AGENT (stream_*_v2.json) grids:

    python -m tevc.rpstda_gate        -> outputs/tevc/rpstda_gate.json (refuses to run twice)

"Mean" is the run-level mean: per run, the mean deployed accuracy over the stream's environments, then the mean over
the 10 runs. Pass iff (1) PU: mean RPST-DA >= max(mean CDAN, mean AGENT) + 0.010; (2) PU: RPST-DA > CDAN in >= 7 of
10 runs, paired by run seed; (3) JNU: mean RPST-DA >= max(mean CDAN, mean AGENT) - 0.0025. Reported alongside, not
part of the gate: rule vs IM-only vs TV-only deployments on the same candidates, and the Spearman rho of TV accuracy
and of IM with test accuracy over each visit's candidates, averaged per environment.
"""
import hashlib
import json
import subprocess
import time

import numpy as np
from scipy.stats import spearmanr

from tevc.optimizers import B
from tevc.run_stream import OUT_DIR, RUN_SEEDS

SPEC = "docs/superpowers/specs/2026-10-18-mssp-rpst-da-design.md"
GATE_OUT = OUT_DIR / "rpstda_gate.json"
GATE_FILES = {ds: {"RPST-DA": f"stream_{ds}_rpstda.json", "CDAN": f"stream_{ds}_rbase.json",
                   "AGENT": f"stream_{ds}_v2.json"} for ds in ("pu", "jnu")}
PU_MARGIN, PU_MIN_WINS, JNU_TOLERANCE = 0.010, 7, 0.0025


def run_means(runs, key="deployed_acc"):
    """{run seed: stream mean of `key`} from one method's archived runs {run seed: [environment entries]}."""
    return {int(r): float(np.mean([e[key] for e in envs])) for r, envs in runs.items()}


def _mean(per_run):
    return float(np.mean([per_run[r] for r in RUN_SEEDS]))


def gate_verdict(pu, jnu):
    """pu, jnu: {method: {run seed: run-level mean deployed accuracy}} for RPST-DA, CDAN and AGENT."""
    pu_bar = max(_mean(pu["CDAN"]), _mean(pu["AGENT"])) + PU_MARGIN
    jnu_bar = max(_mean(jnu["CDAN"]), _mean(jnu["AGENT"])) - JNU_TOLERANCE
    wins = int(sum(pu["RPST-DA"][r] > pu["CDAN"][r] for r in RUN_SEEDS))
    checks = {"pu_mean": {"rpstda": _mean(pu["RPST-DA"]), "bar": pu_bar, "pass": _mean(pu["RPST-DA"]) >= pu_bar},
              "pu_wins_vs_cdan": {"wins": wins, "needed": PU_MIN_WINS, "pass": wins >= PU_MIN_WINS},
              "jnu_no_harm": {"rpstda": _mean(jnu["RPST-DA"]), "bar": jnu_bar,
                              "pass": _mean(jnu["RPST-DA"]) >= jnu_bar}}
    return {"checks": checks, "passed": all(c["pass"] for c in checks.values())}


def _rho(a, b):
    """Spearman rho, or None when either side is constant (e.g. every candidate at 100 %)."""
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b)[0])


def descriptive(runs):
    """RPST-DA's rule vs IM-only vs TV-only deployments (run-level means), and the mean Spearman rho per environment
    of TV accuracy and of IM with test accuracy over each visit's candidates."""
    means = {name: _mean(run_means(runs, key)) for name, key in
             (("rule", "deployed_acc"), ("im_only", "deployed_acc_im"), ("tv_only", "deployed_acc_tv"))}
    rho = {}
    for envs in runs.values():
        for e in envs:
            acc = [t["acc"] for t in e["trials"]]
            for name, key in (("tv", "tv_acc"), ("im", "fitness")):
                rho.setdefault(e["env"], {}).setdefault(name, []).append(_rho([t[key] for t in e["trials"]], acc))
    per_env = {env: {name: (float(np.mean([v for v in vals if v is not None]))
                            if any(v is not None for v in vals) else None) for name, vals in d.items()}
               for env, d in rho.items()}
    return {"run_level_means": means, "spearman_vs_test_acc": per_env}


def _load(ds):
    """Archived runs per method for ds, and each file's SHA-1; the RPST-DA grid must be complete."""
    runs, digests = {}, {}
    for method, name in GATE_FILES[ds].items():
        path = OUT_DIR / name
        if not path.exists():
            raise SystemExit(f"{path} missing")
        raw = path.read_bytes()
        digests[name] = hashlib.sha1(raw).hexdigest()
        runs[method] = json.loads(raw)["results"][method]
        missing = sorted(set(RUN_SEEDS) - {int(r) for r in runs[method]})
        if missing:
            raise SystemExit(f"{name}: {method} lacks runs {missing}")
    if any(e["n_evals"] != B for envs in runs["RPST-DA"].values() for e in envs):
        raise SystemExit(f"{ds}: an RPST-DA visit did not spend the budget of {B}")
    return runs, digests


def _head():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def gate():
    if GATE_OUT.exists():
        raise SystemExit(f"{GATE_OUT} exists: the gate is one shot and RPST-DA is never revised and re-gated")
    loaded = {ds: _load(ds) for ds in GATE_FILES}
    means = {ds: {m: run_means(r) for m, r in runs.items()} for ds, (runs, _) in loaded.items()}
    out = {"spec": f"{SPEC} §3 (one-shot development gate)", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "commit": _head(), "files": GATE_FILES, "sha1": {ds: d for ds, (_, d) in loaded.items()},
           "run_level_means": {ds: {m: _mean(v) for m, v in mm.items()} for ds, mm in means.items()},
           "per_run": {ds: {m: {str(r): v[r] for r in RUN_SEEDS} for m, v in mm.items()} for ds, mm in means.items()},
           "verdict": gate_verdict(means["pu"], means["jnu"]),
           "descriptive": {ds: descriptive(runs["RPST-DA"]) for ds, (runs, _) in loaded.items()}}
    GATE_OUT.write_text(json.dumps(out, indent=1))
    for ds, mm in out["run_level_means"].items():
        print(ds, {m: round(v * 100, 2) for m, v in mm.items()})
    print("verdict:", json.dumps(out["verdict"], indent=1))


if __name__ == "__main__":
    gate()
