"""Shared evaluation: logits capture + primary metrics (MF1, macro-AUROC, kappa)."""
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score


@torch.no_grad()
def evaluate_full(model, loader, device, is_wapigt=False, fs=50000.0, reorg_bp=None):
    """Returns (metrics_dict, logits, labels)."""
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        sigs = batch["signal"].to(device)
        labs = batch["label"]
        if is_wapigt:
            bp = reorg_bp(batch["bearing_params"], sigs.shape[0])
            logits, *_ = model(sigs, bp, fs_sampling=fs)
        else:
            out = model(sigs)
            logits = out[0] if isinstance(out, tuple) else out
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(labs.numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    return compute_metrics(logits, labels), logits, labels


def compute_metrics(logits, labels):
    preds = logits.argmax(1)
    acc = float((preds == labels).mean())
    mf1 = float(f1_score(labels, preds, average="macro"))
    kappa = float(cohen_kappa_score(labels, preds))
    # softmax probabilities for AUROC
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(1, keepdims=True)
    try:
        present = np.unique(labels)
        if len(present) == p.shape[1]:
            auroc = float(roc_auc_score(labels, p, multi_class="ovr", average="macro"))
        else:
            auroc = float(roc_auc_score(labels, p[:, present],
                                        multi_class="ovr", average="macro",
                                        labels=present))
    except Exception:
        auroc = float("nan")
    return {"acc": acc, "mf1": mf1, "auroc": auroc, "kappa": kappa}
