"""Reliability metrics of deployed diagnosers (RESS spec §3.5) and the development comparison table.

Per environment, from the cached test-set probabilities of the deployed model (never retrained): missed-fault rate
P(predicted healthy | true fault), false-alarm rate P(predicted fault | true healthy), macro-F1, and expected
calibration error over 15 equal-width bins of the maximum probability. Class 0 is the healthy class in every
dataset (JNU n, PU K00x, Ottawa H, DIRG 0A).
"""
import numpy as np
from sklearn.metrics import f1_score

HEALTHY = 0
N_BINS = 15
LOWER_IS_BETTER = ("missed_fault_rate", "false_alarm_rate", "ece")
KEYS = ("deployed_acc", "auc", "macro_f1") + LOWER_IS_BETTER


def missed_fault_rate(y, pred, healthy=HEALTHY):
    fault = y != healthy
    return float(np.mean(pred[fault] == healthy)) if fault.any() else 0.0


def false_alarm_rate(y, pred, healthy=HEALTHY):
    ok = y == healthy
    return float(np.mean(pred[ok] != healthy)) if ok.any() else 0.0


def macro_f1(y, pred, n_classes):
    return float(f1_score(y, pred, labels=np.arange(n_classes), average="macro", zero_division=0))


def ece(probs, y, n_bins=N_BINS):
    conf, correct = probs.max(axis=1), probs.argmax(axis=1) == y
    b = np.clip(np.digitize(conf, np.linspace(0.0, 1.0, n_bins + 1)[1:-1], right=True), 0, n_bins - 1)
    return float(sum((b == k).mean() * abs(correct[b == k].mean() - conf[b == k].mean())
                     for k in range(n_bins) if (b == k).any()))


def metrics(probs, y, n_classes):
    pred = probs.argmax(axis=1)
    return {"missed_fault_rate": missed_fault_rate(y, pred), "false_alarm_rate": false_alarm_rate(y, pred),
            "macro_f1": macro_f1(y, pred, n_classes), "ece": ece(probs, y)}
