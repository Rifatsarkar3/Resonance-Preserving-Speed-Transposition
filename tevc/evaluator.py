"""Recipe evaluation: the expensive, noisy fitness of D1 (spec §2).

The agent receives only `Evaluation` (fitness = cluster-balanced IM on the environment's UNLABELLED buffer, plus
cost). True accuracy on the environment's test set is for scoring only: `true_accuracy` (stream driver) and
`hidden_log`, neither of which agent code reads. Predicted probabilities are cached on disk, so re-scoring never
retrains. Non-recipe candidates (the RESS baselines, tevc.baselines) train through baseline_fn; view_fn transforms
windows before prediction (COT only).
"""
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tevc import baselines, features
from tevc.deep_train import predict, train_cnn
from tevc.proxies import cluster_balance_weights, information_maximization, neg_mmd2, weighted_information_maximization
from tevc.recipe import Recipe, training_set


def cluster_weights(env, ds):
    """Frozen D1 buffer weights (spec §4): k-means on the standardised physics features of U, 1 / cluster size."""
    X = features.extract(env.U["W"], ds.fs, env.rpm / 60.0, ds.geometry, znorm=True)
    return cluster_balance_weights(X, ds.n_classes)[0]


@dataclass(frozen=True)
class Evaluation:
    fitness: float      # cluster-balanced IM on U_t (agent-visible)
    seed: int
    cost_s: float
    cached: bool


def _fingerprint(*window_sets):
    h = hashlib.sha1()
    for w in window_sets:
        h.update(np.asarray(w["rec"]).astype(str).tobytes())
        h.update(np.asarray(w["span_lo"], dtype=np.float64).tobytes())
    return h.hexdigest()[:12]


class RecipeEvaluator:
    def __init__(self, stream, model_name="TICNN", epochs=40, cache_dir=Path("outputs/tevc/cache/evals"),
                 train_fn=train_cnn, predict_fn=predict, weights_fn=None, baseline_fn=None, view_fn=None):
        self.stream, self.model_name, self.epochs = stream, model_name, epochs
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.train_fn, self.predict_fn = train_fn, predict_fn
        self.weights_fn = weights_fn or (lambda env: cluster_weights(env, stream.dataset))
        self.baseline_fn = baseline_fn or baselines.train
        self.view_fn = view_fn or baselines.view
        self._weights = {}
        self.hidden_log = []

    def weights(self, env):
        """Label-free window weights of env's buffer, computed once per environment."""
        if env.name not in self._weights:
            self._weights[env.name] = self.weights_fn(env)
        return self._weights[env.name]

    def _key(self, env, recipe, seed):
        r = asdict(recipe)
        if r.get("tsm") == "pv":
            del r["tsm"]                # phase-vocoder recipes keep the keys they had before the tsm field
        payload = json.dumps([self.stream.dataset.name, env.name, _fingerprint(self.stream.library, env.U, env.T),
                              r, seed, self.model_name, self.epochs], sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    def probabilities(self, env, recipe, seed):
        """(probabilities on U, probabilities on T, cost in seconds, cached?) for one trained model."""
        path = self.cache_dir / f"{self._key(env, recipe, seed)}.npz"
        if path.exists():
            z = np.load(path)
            return z["pU"], z["pT"], float(z["cost_s"]), True
        t0 = time.time()
        model, _ = self._train(env, recipe, seed)
        view = self.view_fn(recipe, self.stream, env)
        pU, pT = self.predict_fn(model, view(env.U["W"])), self.predict_fn(model, view(env.T["W"]))
        cost = time.time() - t0
        np.savez(path, pU=pU, pT=pT, cost_s=cost)
        return pU, pT, cost, False

    def cached(self, env, recipe, seed):
        """(pU, pT) of a model already in the cache, never training (reliability re-scoring); KeyError if absent."""
        path = self.cache_dir / f"{self._key(env, recipe, seed)}.npz"
        if not path.exists():
            raise KeyError(f"{env.name}: no cached model for {recipe} with seed {seed}")
        with np.load(path) as z:
            return z["pU"], z["pT"]

    def _train(self, env, recipe, seed):
        if not isinstance(recipe, Recipe):
            return self.baseline_fn(recipe, self.stream, env, seed, self.model_name, self.epochs)
        W, y = training_set(recipe, self.stream.dataset, self.stream.library_entries, self.stream.library,
                            self.stream.library_rpm, env.rpm, self.stream.src_rpm_of)
        return self.train_fn(W, y, self.model_name, self.epochs, seed), W

    def dfid(self, env, recipe, seed):
        """(DFID, cached?): the secondary label-free fitness (spec §8; Gate-1 Spearman ρ 0.84 / 0.65 / 0.88), -MMD²
        between the model's embeddings of 400 of its training windows and of U. Embeddings are not in the model
        cache, so a cached model is retrained (seeded training is deterministic; the largest |ΔpU| against the cache
        is stored as pU_drift). The value is kept beside the model cache, which is never rewritten."""
        if not isinstance(recipe, Recipe):
            raise TypeError("DFID is defined for synthesis recipes only")
        key = self._key(env, recipe, seed)
        side = self.cache_dir / f"{key}.dfid.json"
        if side.exists():
            return json.loads(side.read_text())["dfid"], True
        t0 = time.time()
        model, W = self._train(env, recipe, seed)
        pU, eU = self.predict_fn(model, env.U["W"], return_embedding=True)
        _, eW = self.predict_fn(model, W[np.random.default_rng(0).permutation(len(W))[:400]], return_embedding=True)
        value = neg_mmd2(eW, eU)
        path = self.cache_dir / f"{key}.npz"
        if path.exists():
            with np.load(path) as z:
                drift = float(np.abs(z["pU"] - pU).max())
        else:
            np.savez(path, pU=pU, pT=self.predict_fn(model, env.T["W"]), cost_s=time.time() - t0)
            drift = 0.0
        side.write_text(json.dumps({"dfid": value, "pU_drift": drift}))
        return value, False

    def evaluate(self, env, recipe, seed):
        pU, pT, cost, cached = self.probabilities(env, recipe, seed)
        fitness = weighted_information_maximization(pU, self.weights(env))
        self.hidden_log.append({"env": env.name, "recipe": asdict(recipe), "seed": seed, "fitness": fitness,
                                "plain_im": information_maximization(pU),
                                "true_acc": float((pT.argmax(axis=1) == env.T["y"]).mean()),
                                "cost_s": cost, "cached": cached})
        return Evaluation(fitness, seed, cost, cached)

    def true_accuracy(self, env, recipe, seed):
        """Scoring only (stream driver, never the agent): accuracy of that trained model on env's test set."""
        _, pT, _, _ = self.probabilities(env, recipe, seed)
        return float((pT.argmax(axis=1) == env.T["y"]).mean())
