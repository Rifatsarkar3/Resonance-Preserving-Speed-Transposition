"""Re-score Gate-0 deployed accuracy under the default and the user-defined deployment rules."""
import json
from pathlib import Path

import numpy as np

from tevc.deploy_rule import choose_deployed, default_rule
from tevc.run_gate0 import ARMS

RESULTS = Path("outputs/tevc/gate0_results.json")


def deployed_acc(run, rule, n_features):
    i = rule(np.array(run["fit_err"]), np.array(run["ratio"]))
    return 1.0 - run["test_err"][i], round(run["ratio"][i] * n_features)


def main():
    results = json.loads(RESULTS.read_text())
    runs, n_features = results["runs"], results["config"]["D"]
    print(f"{'task':10s} {'arm':14s} {'default acc / nfeat':>22s} {'your rule acc / nfeat':>24s}")
    for task in dict.fromkeys(r["task"] for r in runs):
        for arm in ARMS:
            rs = [r for r in runs if r["task"] == task and r["arm"] == arm]
            cols = []
            for rule in (default_rule, choose_deployed):
                a, n = zip(*(deployed_acc(r, rule, n_features) for r in rs))
                cols.append(f"{np.mean(a) * 100:5.1f}+/-{np.std(a) * 100:4.1f} / {np.mean(n):5.1f}")
            print(f"{task:10s} {arm:14s} {cols[0]:>22s} {cols[1]:>24s}")


if __name__ == "__main__":
    main()
