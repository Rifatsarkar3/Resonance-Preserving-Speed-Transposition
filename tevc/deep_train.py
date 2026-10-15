"""Fast CNN training / inference for recipe evaluation (the expensive fitness of the deep path).

Same models (tevc/models.py), optimiser and per-window z-normalisation as the earlier RPST baseline runs;
fixed epoch budget and the last checkpoint, because no labelled validation data exist at the target speed.
"""
import random

import numpy as np
import torch
import torch.nn.functional as F

from tevc.models import MODELS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_all_seeds(seed):
    """Seed python, numpy and torch; deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def znorm(W):
    W = np.asarray(W, dtype=np.float32)
    return (W - W.mean(axis=1, keepdims=True)) / (W.std(axis=1, keepdims=True) + 1e-6)


def _logits(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


def train_cnn(W, y, model_name="WDCNN", epochs=30, seed=42, batch_size=32, lr=1e-3, label_smoothing=0.0):
    set_all_seeds(seed)
    model = MODELS[model_name](signal_length=W.shape[1], n_classes=int(y.max()) + 1, dropout=0.3).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    X = torch.tensor(znorm(W), device=DEVICE)
    Y = torch.tensor(y, dtype=torch.long, device=DEVICE)
    g = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(X), generator=g).split(batch_size):
            idx = idx.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(_logits(model, X[idx]), Y[idx], label_smoothing=label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model.eval()


@torch.no_grad()
def predict(model, W, batch_size=256, return_embedding=False):
    """Class probabilities (N, C); optionally the input of the final linear layer as an embedding."""
    head = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1]
    captured = []
    hook = head.register_forward_hook(lambda mod, inp, out: captured.append(inp[0].detach())) if return_embedding else None
    X = torch.tensor(znorm(W), device=DEVICE)
    probs = torch.cat([F.softmax(_logits(model, xb), dim=1) for xb in X.split(batch_size)]).cpu().numpy()
    if hook:
        hook.remove()
        return probs, torch.cat(captured).cpu().numpy()
    return probs
