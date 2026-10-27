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
BATCH_SIZE = 32


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


def train_uda(Ws, ys, Wt, method, weight, lr, model_name="TICNN", epochs=40, seed=42, batch_size=BATCH_SIZE,
              total_steps=None):
    """DANN | DEEP-CORAL | CDAN on labelled source windows (Ws, ys) and unlabelled target windows Wt. Each step: one
    source batch and an equal-size random target batch in one forward pass (shared batch statistics, as in the CDAN
    reference code). DANN / CDAN: discriminator behind a gradient reversal layer with coefficient
    weight x grl_coefficient(progress); DEEP-CORAL: weight x coral(features).

    total_steps: train for exactly this many steps instead of `epochs` passes over the source (the steps-matched
    no-copies arm of the MSSP reframe spec, §5.3); passes over the source continue, or stop inside one, until it is
    reached. The default (None) takes epochs x ceil(n / batch_size) steps with the same random draws in the same
    order as before the option existed."""
    if method not in UDA_METHODS:
        raise ValueError(f"unknown domain-adaptation method {method}")
    set_all_seeds(seed)
    n_classes = int(ys.max()) + 1
    model = MODELS[model_name](signal_length=Ws.shape[1], n_classes=n_classes, dropout=0.3).to(DEVICE)
    disc = (Discriminator(FEAT_DIM * (n_classes if method == "CDAN" else 1)).to(DEVICE)
            if method != "DEEP-CORAL" else None)
    params = list(model.parameters()) + (list(disc.parameters()) if disc is not None else [])
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    Xs, Xt = torch.tensor(znorm(Ws), device=DEVICE), torch.tensor(znorm(Wt), device=DEVICE)
    Ys = torch.tensor(ys, dtype=torch.long, device=DEVICE)
    g = torch.Generator(device="cpu").manual_seed(seed)
    if total_steps and len(Xs) == 0:
        raise ValueError("total_steps needs at least one source window")
    total = int(total_steps) if total_steps else epochs * int(np.ceil(len(Xs) / batch_size))
    step = 0
    model.train()
    if disc is not None:
        disc.train()
    while step < total:
        for idx in torch.randperm(len(Xs), generator=g).split(batch_size):
            if step == total:
                break
            tidx = torch.randint(len(Xt), (len(idx),), generator=g).to(DEVICE)
            idx, n = idx.to(DEVICE), len(idx)
            f, logits = features_and_logits(model, torch.cat([Xs[idx], Xt[tidx]]))
            loss = F.cross_entropy(logits[:n], Ys[idx])
            if method == "DEEP-CORAL":
                loss = loss + weight * coral(f[:n], f[n:])
            else:
                h, w = f, None
                if method == "CDAN":
                    p = F.softmax(logits, dim=1).detach()
                    h, w = multilinear(f, p), torch.cat([entropy_weights(p[:n]), entropy_weights(p[n:])])
                d = disc(grad_reverse(h, weight * grl_coefficient(step / total)))
                domain = torch.cat([torch.ones(n, device=DEVICE), torch.zeros(n, device=DEVICE)])
                loss = loss + F.binary_cross_entropy_with_logits(d, domain, weight=w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
    return model.eval()


def train_source(Ws, ys, model_name="TICNN", epochs=40, seed=42):
    """SHOT's source model: the shared training loop with label smoothing 0.1."""
    return train_cnn(Ws, ys, model_name, epochs, seed, label_smoothing=0.1)


@torch.no_grad()
def pseudo_labels(model, X, batch_size=256):
    model.eval()
    parts = [features_and_logits(model, xb) for xb in X.split(batch_size)]
    f = torch.cat([a for a, _ in parts])
    return centroid_labels(f, F.softmax(torch.cat([b for _, b in parts]), dim=1))


def train_shot(Ws, ys, Wt, lr, beta, model_name="TICNN", epochs=40, seed=42, batch_size=32, source=None):
    """SHOT: a copy of the source model (train_source, or `source` to reuse one) adapted on the unlabelled Wt for
    epochs // 2 epochs with the classifier head fc2 frozen; loss = mean prediction entropy - entropy of the mean
    prediction + beta x cross-entropy to centroid pseudo-labels recomputed at the start of every epoch."""
    model = copy.deepcopy(source if source is not None else train_source(Ws, ys, model_name, epochs, seed))
    set_all_seeds(seed)
    for p in model.fc2.parameters():
        p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    Xt = torch.tensor(znorm(Wt), device=DEVICE)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(max(1, epochs // 2)):
        labels = pseudo_labels(model, Xt)
        model.train()
        for idx in torch.randperm(len(Xt), generator=g).split(batch_size):
            idx = idx.to(DEVICE)
            _, logits = features_and_logits(model, Xt[idx])
            p = F.softmax(logits, dim=1)
            mean_p = p.mean(dim=0)
            loss = (-(p * torch.log(p + 1e-5)).sum(dim=1).mean() + (mean_p * torch.log(mean_p + 1e-5)).sum()
                    + beta * F.cross_entropy(logits, labels[idx]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
    return model.eval()
