import numpy as np

from tevc.reliability import ece, false_alarm_rate, macro_f1, metrics, missed_fault_rate

Y = np.array([0, 0, 1, 1, 2, 2])
PRED = np.array([0, 1, 0, 1, 2, 0])


def test_missed_faults_are_faults_called_healthy():
    assert missed_fault_rate(Y, PRED) == 0.5          # windows 2 and 5 of the four faulty ones


def test_false_alarms_are_healthy_windows_called_faulty():
    assert false_alarm_rate(Y, PRED) == 0.5           # window 1 of the two healthy ones


def test_macro_f1_is_one_when_perfect_and_counts_never_predicted_classes_as_zero():
    assert macro_f1(Y, Y, 3) == 1.0
    assert np.isclose(macro_f1(Y, np.zeros_like(Y), 3), 0.5 / 3)    # class 0 F1 = 0.5, classes 1 and 2 F1 = 0


def test_ece_is_zero_when_confidence_matches_accuracy_and_measures_overconfidence():
    y = np.array([0] * 5 + [1] * 5)
    assert ece(np.eye(2)[y], y) == 0.0
    assert np.isclose(ece(np.tile([0.9, 0.1], (10, 1)), y), 0.4)    # 90 % confident, 50 % right


def test_metrics_bundle_uses_the_argmax_prediction():
    m = metrics(np.eye(3)[PRED] * 0.8 + 0.2 / 3, Y, 3)
    assert set(m) == {"missed_fault_rate", "false_alarm_rate", "macro_f1", "ece"}
    assert m["missed_fault_rate"] == 0.5 and m["false_alarm_rate"] == 0.5
