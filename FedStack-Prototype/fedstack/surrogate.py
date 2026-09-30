"""Surrogate node models — a lightweight stand-in for the fine-tuned ResNet-50 nodes.

Each node owns a private feature extractor (standardisation + PCA fitted only on the
node's own images, playing the role of the frozen backbone) and a small neural
classification head trained incrementally with Adam, so that federated rounds are
*real* local training steps with a visible learning curve -- the same
"frozen backbone + trained head" structure as the released code with freeze_backbone: true.  Nothing here
is used once the ResNet checkpoints are configured; it exists so the complete
system (nodes -> server -> meta-learner -> verdict, payload accounting, zero-day
flagging) can be demonstrated on a CPU-only machine.
"""
from __future__ import annotations

import pickle
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class SurrogateNode:
    def __init__(self, name: str, classes: list[str], n_components: int = 96, lr: float = 0.0005, seed: int = 42, hidden: int = 256):
        self.name = name
        self.classes = list(classes)                       # local class names (global vocabulary strings)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.n_components = n_components
        self.lr = lr
        self.seed = seed
        self.hidden = hidden
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.clf: MLPClassifier | None = None
        self.epochs_done = 0
        self.rounds_done = 0

    # ---- features ---------------------------------------------------------
    @staticmethod
    def _flat(X: np.ndarray) -> np.ndarray:
        return X.reshape(len(X), -1).astype(np.float32) / 255.0

    def fit_features(self, X: np.ndarray) -> None:
        F = self._flat(X)
        self.scaler = StandardScaler().fit(F)
        n_comp = min(self.n_components, F.shape[0] - 1, F.shape[1])
        self.pca = PCA(n_components=n_comp, random_state=self.seed).fit(self.scaler.transform(F))

    def features(self, X: np.ndarray) -> np.ndarray:
        assert self.scaler is not None and self.pca is not None, "node feature extractor not fitted"
        return self.pca.transform(self.scaler.transform(self._flat(X))).astype(np.float32)

    # ---- local training (one federated round = `epochs` local passes) --------
    def train_round(self, X: np.ndarray, y_local: np.ndarray, epochs: int = 5) -> None:
        if self.scaler is None:
            self.fit_features(X)
        Z = self.features(X)
        if self.clf is None:
            self.clf = MLPClassifier(hidden_layer_sizes=(self.hidden,), learning_rate_init=self.lr, alpha=1e-4,
                                     random_state=self.seed)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(epochs):                       # one partial_fit = one pass over the node's data
                self.clf.partial_fit(Z, y_local, classes=np.arange(len(self.classes)))
                self.epochs_done += 1
        self.rounds_done += 1

    # ---- inference ------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.clf is not None, "node not trained"
        return self.clf.predict_proba(self.features(X)).astype(np.float32)

    def accuracy(self, X: np.ndarray, y_local: np.ndarray) -> float:
        if self.clf is None or len(X) == 0:
            return 0.0
        return float((self.predict_proba(X).argmax(1) == y_local).mean())

    # ---- payload accounting -----------------------------------------------------
    def n_params(self) -> int:
        if self.pca is None or self.clf is None:
            return 0
        head = sum(w.size for w in self.clf.coefs_) + sum(b.size for b in self.clf.intercepts_)
        return int(self.pca.components_.size + self.pca.mean_.size + self.scaler.mean_.size + self.scaler.scale_.size + head)

    def payload(self) -> dict[str, int]:
        p = self.n_params()
        return {"params": p, "bytes_fp32": 4 * p, "bytes_fp16": 2 * p, "bytes_int8": p + 4 * 8}

    def info(self) -> dict[str, Any]:
        return {"type": "surrogate", "feature_extractor": f"StandardScaler + PCA({self.pca.n_components_ if self.pca else self.n_components})",
                "classifier": f"MLP head {self.hidden} hidden units, Adam", "epochs_done": self.epochs_done, "rounds_done": self.rounds_done, **self.payload()}

    # ---- persistence ---------------------------------------------------------------
    def dumps(self) -> bytes:
        return pickle.dumps(self.__dict__)

    @classmethod
    def loads(cls, b: bytes) -> "SurrogateNode":
        obj = cls.__new__(cls)
        obj.__dict__.update(pickle.loads(b))
        return obj
