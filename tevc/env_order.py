"""ENV-ORDER: classical envelope fault-order diagnosis, the baseline a signal-processing reviewer asks for first (MSSP
reframe spec docs/superpowers/specs/2026-10-20-mssp-sp-first-reframe-design.md §5.3; attack audit G1).

Envelope analysis in fixed bands with the fault orders taken at each window's own shaft speed is speed-normalised by
construction: no training data at the target speed is needed. Features: tevc.features.envelope_orders (108 columns)
of the z-normalised windows, the library windows at the library speed (per entry when the stream records one), the
environment's windows at its estimated speed. Classifier: standardisation fitted on the library, then a multinomial
logistic regression with inverse regularisation C (EnvOrderConfig; candidate 0 at C = 1). Deployed like every
comparator, by the cluster-balanced IM on the unlabelled buffer; no target label is read.
"""
import hashlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from tevc import features

MAX_ITER = 5000
_LIBRARY = {}                  # library content key -> (features, labels): computed once per library


def _library_key(stream):
    """Content key of a stream's library (windows, labels, records, speed), so that the feature cache never serves
    one library's features for another."""
    lib = stream.library
    h = hashlib.sha1(np.ascontiguousarray(lib["W"]).tobytes())
    h.update(np.asarray(lib["y"]).tobytes())
    h.update(np.asarray(lib["rec"]).astype(str).tobytes())
    h.update(repr(float(stream.library_rpm)).encode())
    return h.hexdigest()


class EnvOrderModel:
    """A fitted ENV-ORDER classifier bound to one environment's shaft speed. predict_windows(W) gives class
    probabilities (N, n_classes); RecipeEvaluator uses it in place of the network predictor."""

    def __init__(self, scaler, clf, fs, f_shaft, geometry, n_classes):
        self.scaler, self.clf = scaler, clf
        self.fs, self.f_shaft, self.geometry, self.n_classes = fs, f_shaft, geometry, n_classes

    def predict_windows(self, W):
        X = features.envelope_orders(W, self.fs, self.f_shaft, self.geometry)
        p = self.clf.predict_proba(self.scaler.transform(X))
        out = np.zeros((len(X), self.n_classes), dtype=np.float32)
        out[:, self.clf.classes_] = p                 # columns in class order, zero for an absent class
        return out


def library_features(stream):
    """(features, labels) of the stream's library windows, each at its own shaft speed (src_rpm_of per entry, as on
    Ottawa; otherwise the library speed)."""
    key = _library_key(stream)
    if key not in _LIBRARY:
        ds, lib = stream.dataset, stream.library
        if stream.src_rpm_of is None:
            X = features.envelope_orders(lib["W"], ds.fs, stream.library_rpm / 60.0, ds.geometry)
        else:
            X = np.vstack([features.envelope_orders(lib["W"][i:i + 1], ds.fs, stream.src_rpm_of(str(r)) / 60.0,
                                                    ds.geometry) for i, r in enumerate(lib["rec"])])
        _LIBRARY[key] = (X, np.asarray(lib["y"]))
    return _LIBRARY[key]


def train(config, stream, env):
    """The fitted ENV-ORDER model of one candidate for environment env (the fit itself is environment-independent;
    the model carries env's estimated shaft speed for its predictions)."""
    X, y = library_features(stream)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=config.C, max_iter=MAX_ITER).fit(scaler.transform(X), y)
    ds = stream.dataset
    return EnvOrderModel(scaler, clf, ds.fs, env.rpm / 60.0, ds.geometry, ds.n_classes)
