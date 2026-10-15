"""Domain-adaptation baselines on the shared 1-D Inception CNN backbone (RESS spec §3.4).

DANN (Ganin et al., JMLR 2016), Deep CORAL (Sun & Saenko, ECCV Workshops 2016), CDAN+E (Long et al., NeurIPS 2018)
and SHOT (Liang et al., ICML 2020). Every method keeps the backbone, the optimiser (Adam, weight decay 1e-5),
batch size 32, the epoch budget and the per-window z-normalisation of the recipe evaluations, so differences come
from the adaptation objective alone. Features = the 256-d input of the final linear layer (the embedding that
`deep_train.predict(return_embedding=True)` returns). Source = the labelled library windows; target = the
environment's unlabelled buffer U; no target label is ever read.
Stated departures from the originals: Adam for every method (the originals use SGD); domain discriminators with
256-unit hidden layers (the originals use 1024 for 2048-d ResNet features); CDAN conditions on detached
predictions; Deep CORAL acts on the 256-d features; SHOT keeps the shared head (no weight-normalised classifier or
bottleneck batch normalisation) and runs epochs // 2 adaptation epochs.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tevc.deep_train import DEVICE, set_all_seeds, train_cnn, znorm
from tevc.models import MODELS, TICNN

FEAT_DIM = 256
UDA_METHODS = ("DANN", "DEEP-CORAL", "CDAN")


def features_and_logits(model, x):
    """TICNN.forward split at the head: (256-d features, logits); the logits equal model(x)."""
    if not isinstance(model, TICNN):
        raise TypeError("the domain-adaptation baselines use the TICNN backbone")
    if x.dim() == 2:
        x = x.unsqueeze(1)
    x = model.pool2(model.inception2(model.pool1(model.inception1(model.conv_init(x)))))
    x = model.gap(x).view(x.size(0), -1)
    f = model.relu_fc(model.fc1(model.dropout(x)))
    return f, model.fc2(model.dropout(f))


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def grad_reverse(x, lam):
    """Identity forward; gradient multiplied by -lam backward (DANN's gradient reversal layer)."""
    return _GradReverse.apply(x, lam)


def grl_coefficient(progress):
    """DANN schedule 2 / (1 + exp(-10 p)) - 1 for training progress p in [0, 1]."""
    return 2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0


class Discriminator(nn.Module):
    """Domain classifier: two hidden layers of 256 units (ReLU, dropout 0.5) and one logit (source = 1)."""

    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.5), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def coral(fs, ft):
    """Deep CORAL loss ||C_s - C_t||_F^2 / (4 d^2) between the feature covariances of two batches."""
    def cov(x):
        x = x - x.mean(dim=0, keepdim=True)
        return x.t() @ x / max(x.size(0) - 1, 1)

    d = fs.size(1)
    return ((cov(fs) - cov(ft)) ** 2).sum() / (4.0 * d * d)


def multilinear(f, p):
    """CDAN conditioning: flattened outer product p ⊗ f, (N, C x 256); small enough to need no random projection."""
    return torch.bmm(p.unsqueeze(2), f.unsqueeze(1)).view(f.size(0), -1)


def entropy_weights(p):
    """CDAN+E sample weights 1 + exp(-H(p)), normalised to mean 1 within the batch."""
    w = 1.0 + torch.exp((p * torch.log(p + 1e-12)).sum(dim=1))
    return w / w.mean()


def centroid_labels(f, p, rounds=1):
    """SHOT self-supervised pseudo-labels: features with a constant 1 appended and L2-normalised; centroids weighted
    by the predictions p; nearest centroid by cosine similarity; `rounds` refinements with hard labels."""
    f = torch.cat([f, torch.ones(len(f), 1, device=f.device, dtype=f.dtype)], dim=1)
    f = f / f.norm(dim=1, keepdim=True)
    weights = p
    for _ in range(rounds + 1):
        c = weights.t() @ f / (weights.sum(dim=0)[:, None] + 1e-8)
        labels = (f @ (c / (c.norm(dim=1, keepdim=True) + 1e-8)).t()).argmax(dim=1)
        weights = F.one_hot(labels, p.size(1)).to(f.dtype)
    return labels
