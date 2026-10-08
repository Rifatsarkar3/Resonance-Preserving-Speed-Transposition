"""GPU-batched k-NN wrapper fitness: scores a whole population of feature masks at once."""
import numpy as np
import torch


class KNNEvaluator:
    """k-NN fitted on one dataset (the environment's fitness data).

    fitness():    blocked leave-one-out balanced error on the fit set. Windows whose sample spans overlap
                  in the same source recording are never each other's neighbours, so overlap cannot leak
                  (spans of synthesised windows are mapped back to source-recording coordinates).
    test_error(): balanced error on an external labelled set, the fit set acting as reference.
    Features are z-scored with the fit set's statistics.
    """

    def __init__(self, X, y, rec, span_lo, span_hi, k=5, device="cuda"):
        self.device, self.k = device, k
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-8] = 1.0
        self.X = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.long, device=device)
        self.n_classes = int(y.max()) + 1
        rec = np.unique(np.asarray(rec), return_inverse=True)[1]
        lo, hi = np.asarray(span_lo, dtype=float), np.asarray(span_hi, dtype=float)
        overlap = (rec[:, None] == rec[None, :]) & (lo[:, None] < hi[None, :]) & (lo[None, :] < hi[:, None])
        self.excluded = torch.tensor(overlap, device=device)  # includes the diagonal
        # nearer neighbour wins vote ties
        self.vote_w = 1.0 + 1e-3 * torch.arange(k, 0, -1, device=device, dtype=torch.float32)

    @staticmethod
    def _sqdist(A, B, masks):
        """Masked squared Euclidean distances: A (m, D), B (n, D), masks (P, D) -> (P, m, n)."""
        a2 = (A ** 2) @ masks.T
        b2 = (B ** 2) @ masks.T
        cross = torch.matmul(A.unsqueeze(0) * masks.unsqueeze(1), B.T)
        return a2.T.unsqueeze(2) + b2.T.unsqueeze(1) - 2.0 * cross

    def _predict(self, dist):
        idx = dist.topk(self.k, dim=2, largest=False).indices
        lab = self.y[idx]
        votes = torch.zeros(*lab.shape[:2], self.n_classes, device=self.device)
        votes.scatter_add_(2, lab, self.vote_w.expand_as(lab).contiguous())
        return votes.argmax(dim=2)

    def _balanced_error(self, pred, y_true):
        recalls = [(pred[:, y_true == c] == c).float().mean(dim=1) for c in range(self.n_classes)]
        return 1.0 - torch.stack(recalls, dim=1).mean(dim=1)

    def _masks(self, masks_np):
        return torch.tensor(np.asarray(masks_np), dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def fitness(self, masks_np, chunk=50):
        masks = self._masks(masks_np)
        errs = []
        for m in masks.split(chunk):
            d = self._sqdist(self.X, self.X, m)
            d.masked_fill_(self.excluded, float("inf"))
            errs.append(self._balanced_error(self._predict(d), self.y))
        err = torch.cat(errs)
        err[masks.sum(dim=1) == 0] = 1.0
        return err.cpu().numpy()

    @torch.no_grad()
    def test_error(self, masks_np, X_test, y_test, chunk=50):
        masks = self._masks(masks_np)
        Xt = torch.tensor((X_test - self.mu) / self.sd, dtype=torch.float32, device=self.device)
        yt = torch.tensor(y_test, dtype=torch.long, device=self.device)
        errs = [self._balanced_error(self._predict(self._sqdist(Xt, self.X, m)), yt) for m in masks.split(chunk)]
        err = torch.cat(errs)
        err[masks.sum(dim=1) == 0] = 1.0
        return err.cpu().numpy()
