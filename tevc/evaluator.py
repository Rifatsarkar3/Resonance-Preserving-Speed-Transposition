"""Recipe evaluation: the expensive, noisy fitness of D1 (spec §2).

The agent receives only `Evaluation` (fitness = IM on the environment's UNLABELLED buffer, plus cost). True accuracy
on the environment's test set is computed for scoring and appended to `hidden_log`, which agent code never reads.
Predicted probabilities are cached on disk, so re-scoring (e.g. imbalance stress tests) never retrains.
"""
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tevc.deep_train import predict, train_cnn
from tevc.proxies import information_maximization
from tevc.recipe import training_set


@dataclass(frozen=True)
class Evaluation:
    fitness: float      # IM on U_t (agent-visible)
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
                 train_fn=train_cnn, predict_fn=predict):
        self.stream, self.model_name, self.epochs = stream, model_name, epochs
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.train_fn, self.predict_fn = train_fn, predict_fn
        self.hidden_log = []

    def _key(self, env, recipe, seed):
        payload = json.dumps([self.stream.dataset.name, env.name, _fingerprint(self.stream.library, env.U, env.T),
                              asdict(recipe), seed, self.model_name, self.epochs], sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    def probabilities(self, env, recipe, seed):
        """(probabilities on U, probabilities on T, cost in seconds, cached?) for one trained model."""
        path = self.cache_dir / f"{self._key(env, recipe, seed)}.npz"
        if path.exists():
            z = np.load(path)
            return z["pU"], z["pT"], float(z["cost_s"]), True
        t0 = time.time()
        W, y = training_set(recipe, self.stream.dataset, self.stream.library_entries, self.stream.library,
                            self.stream.library_rpm, env.rpm, self.stream.src_rpm_of)
        model = self.train_fn(W, y, self.model_name, self.epochs, seed)
        pU, pT = self.predict_fn(model, env.U["W"]), self.predict_fn(model, env.T["W"])
        cost = time.time() - t0
        np.savez(path, pU=pU, pT=pT, cost_s=cost)
        return pU, pT, cost, False

    def evaluate(self, env, recipe, seed):
        pU, pT, cost, cached = self.probabilities(env, recipe, seed)
        fitness = information_maximization(pU)
        self.hidden_log.append({"env": env.name, "recipe": asdict(recipe), "seed": seed, "fitness": fitness,
                                "true_acc": float((pT.argmax(axis=1) == env.T["y"]).mean()),
                                "cost_s": cost, "cached": cached})
        return Evaluation(fitness, seed, cost, cached)
