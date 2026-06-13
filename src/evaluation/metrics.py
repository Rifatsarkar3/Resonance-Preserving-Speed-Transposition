"""Evaluation metrics: Accuracy, F1, AUROC, Kappa, Wilcoxon, Cohen's d."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix,
)
from scipy.stats import wilcoxon
from typing import Dict, Tuple


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray = None) -> Dict:
    """
    Compute primary metrics for classification task.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        y_score: Prediction scores (N, n_classes) for AUROC

    Returns:
        Dict with accuracy, macro-F1, weighted-F1, kappa, and AUROC if y_score provided
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }

    # AUROC (one-vs-rest for multiclass)
    if y_score is not None:
        n_classes = y_score.shape[1]
        if n_classes > 2:
            metrics["auroc"] = roc_auc_score(
                y_true, y_score, multi_class="ovr", average="macro", labels=np.arange(n_classes)
            )
        else:
            metrics["auroc"] = roc_auc_score(y_true, y_score[:, 1])

    return metrics


def wilcoxon_signed_rank_test(scores1: np.ndarray, scores2: np.ndarray) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test for paired comparisons.

    Args:
        scores1: Baseline scores (n_tasks,)
        scores2: Proposed method scores (n_tasks,)

    Returns:
        (statistic, p_value)
    """
    statistic, p_value = wilcoxon(scores1, scores2, alternative="two-sided")
    return float(statistic), float(p_value)


def cohens_d(scores1: np.ndarray, scores2: np.ndarray) -> float:
    """
    Cohen's d effect size for two groups.

    Args:
        scores1: Baseline scores
        scores2: Proposed method scores

    Returns:
        Cohen's d value (positive means scores2 > scores1)
    """
    mean1 = np.mean(scores1)
    mean2 = np.mean(scores2)
    var1 = np.var(scores1, ddof=1)
    var2 = np.var(scores2, ddof=1)
    n1, n2 = len(scores1), len(scores2)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    d = (mean2 - mean1) / (pooled_std + 1e-8)

    return float(d)


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[int, Dict]:
    """
    Compute per-class metrics using confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dict mapping class_id to {precision, recall, f1}
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    results = {}
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        results[c] = {"precision": precision, "recall": recall, "f1": f1}

    return results


def latency_stats(latencies_ms: np.ndarray) -> Dict[str, float]:
    """
    Compute latency statistics.

    Args:
        latencies_ms: Array of inference latencies in milliseconds

    Returns:
        Dict with mean, std, min, max, p50, p95, p99
    """
    return {
        "mean_ms": float(np.mean(latencies_ms)),
        "std_ms": float(np.std(latencies_ms)),
        "min_ms": float(np.min(latencies_ms)),
        "max_ms": float(np.max(latencies_ms)),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
    }
