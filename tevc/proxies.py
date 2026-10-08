"""Label-free fitness proxies, computed on real UNLABELLED windows from the new environment."""
import numpy as np

EPS = 1e-12


def _entropy(p):
    return -(p * np.log(p + EPS)).sum(axis=1)


def information_maximization(p):
    """H(mean prediction) - mean H(prediction): confident AND class-diverse predictions score high."""
    return float(_entropy(p.mean(axis=0, keepdims=True))[0] - _entropy(p).mean())


def neg_entropy(p):
    return float(-_entropy(p).mean())


def _js_agreement(p, q):
    m = 0.5 * (p + q)
    js = 0.5 * ((p * np.log2((p + EPS) / (m + EPS))).sum(axis=1) + (q * np.log2((q + EPS) / (m + EPS))).sum(axis=1))
    return float(1.0 - js.mean())


def transposition_consistency(p, view_probs):
    """Mean agreement between predictions on real windows and on their RPST views at nearby speeds."""
    return float(np.mean([_js_agreement(p, q) for q in view_probs]))


def neg_mmd2(A, B, max_n=400, seed=0):
    """-MMD^2 (RBF kernel, median heuristic) between two embedding sets; higher = more alike."""
    rng = np.random.default_rng(seed)
    A = A[rng.permutation(len(A))[:max_n]]
    B = B[rng.permutation(len(B))[:max_n]]
    Z = np.vstack([A, B]).astype(np.float64)
    sq = (Z ** 2).sum(axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * Z @ Z.T, 0.0)
    K = np.exp(-d2 / np.median(d2[d2 > 0]))
    n = len(A)
    return float(-(K[:n, :n].mean() + K[n:, n:].mean() - 2.0 * K[:n, n:].mean()))
