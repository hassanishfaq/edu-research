"""Encoder back-ends -- the model slots of the Chapter 4 prototype (Section 4.4.2, encoder service).

  * RobertaSlot     the thesis model: a fine-tuned `roberta-base` sequence classifier saved with
                    save_pretrained() plus labels.json, loaded from the folder given in config.yaml.
  * SurrogateSlot   CPU stand-in used while that folder is missing: hashing TF-IDF over the SAME
                    serialised text projected into a shared latent space (truncated SVD, playing the
                    frozen encoder) plus a small classification head per slot -- the TF-IDF/linear
                    baseline family of Table 4.12.  Clearly labelled as a stand-in in the UI.

Both expose the same interface: classes, fit(rows, y), predict_proba(rows), predict_proba_texts(texts),
payload(), info().  `rows` are indices into the corpus so the surrogate can reuse one feature matrix.
"""
from __future__ import annotations

import io
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize

# Table 4.14 constants (analytical)
ROBERTA_BASE = {"layers": 12, "hidden": 768, "heads": 12, "params": 124_645_632, "ffn": 3072}


def roberta_gmacs(n: int, L: int = 12, d: int = 768) -> float:
    """Per-forward multiply-accumulates in G: L * (12 n d^2 + 2 n^2 d)  (Table 4.14 notation)."""
    return L * (12 * n * d * d + 2 * n * n * d) / 1e9


def attention_map_bytes(n: int, L: int = 12, h: int = 12) -> int:
    return L * h * n * n * 4


class Slot:
    kind = "abstract"

    def __init__(self, name: str, display: str):
        self.name, self.display = name, display
        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}
        self.last_ms = 0.0
        self.trained = False

    def set_classes(self, classes: list[str]) -> None:
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def fit(self, rows: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def predict_proba(self, rows: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def payload(self) -> dict[str, Any]:
        raise NotImplementedError

    def info(self) -> dict[str, Any]:
        raise NotImplementedError


# =============================================================================
# Surrogate: hashing TF-IDF -> shared SVD space -> per-slot head
# =============================================================================
class SurrogateFeaturizer:
    """Shared feature extractor of the surrogate path (fitted once per corpus, cached on disk)."""

    def __init__(self, n_features: int = 2 ** 18, ngram_max: int = 2, n_components: int = 256, seed: int = 42):
        self.n_features, self.ngram_max, self.n_components, self.seed = n_features, ngram_max, n_components, seed
        self.vec = HashingVectorizer(n_features=n_features, ngram_range=(1, ngram_max), alternate_sign=False, norm=None,
                                     token_pattern=r"(?u)[A-Za-z_][A-Za-z0-9_\-]{1,}|\d{2,}|<[a-z/]+>|%[A-Z0-9]+%", lowercase=True, dtype=np.float32)
        self.tfidf: TfidfTransformer | None = None
        self.svd: TruncatedSVD | None = None
        self.F: np.ndarray | None = None       # (n_corpus, n_components) float32, L2-normalised
        self.build_ms = 0.0

    def build(self, texts: list[str], progress=None) -> None:
        t = time.perf_counter()
        X = self.vec.transform(texts).tocsr()
        if progress:
            progress("hashed")
        self.tfidf = TfidfTransformer(sublinear_tf=True).fit(X)
        Z = self.tfidf.transform(X)
        self.svd = TruncatedSVD(n_components=min(self.n_components, Z.shape[1] - 1, Z.shape[0] - 1), algorithm="randomized", n_iter=4, random_state=self.seed)
        self.F = normalize(self.svd.fit_transform(Z)).astype(np.float32)
        self.svd.components_ = np.ascontiguousarray(self.svd.components_, dtype=np.float32)
        self.build_ms = (time.perf_counter() - t) * 1000

    def transform(self, texts: list[str]) -> np.ndarray:
        """Project new rows into the latent space.  A sparse-row gather against the (k, n_features)
        component matrix costs microseconds per report, where TruncatedSVD.transform would multiply
        by a transposed copy of the whole matrix (hundreds of ms)."""
        Z = self.tfidf.transform(self.vec.transform(texts)).tocsr()
        comps = self.svd.components_
        out = np.zeros((Z.shape[0], comps.shape[0]), dtype=np.float32)
        for i in range(Z.shape[0]):
            a, b = Z.indptr[i], Z.indptr[i + 1]
            if b > a:
                out[i] = Z.data[a:b].astype(np.float32) @ comps[:, Z.indices[a:b]].T
        return normalize(out).astype(np.float32)

    def save(self, path: Path, stamp: float) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"tfidf": self.tfidf, "svd": self.svd, "F": self.F, "stamp": stamp, "n_features": self.n_features, "ngram_max": self.ngram_max,
                         "n_components": self.n_components, "build_ms": self.build_ms}, f, protocol=5)

    def load(self, path: Path, stamp: float) -> bool:
        import pickle
        try:
            with open(path, "rb") as f:
                d = pickle.load(f)
            if d.get("stamp") != stamp or d.get("n_features") != self.n_features or d.get("ngram_max") != self.ngram_max or d.get("n_components") != self.n_components:
                return False
            self.tfidf, self.svd, self.F, self.build_ms = d["tfidf"], d["svd"], d["F"], d.get("build_ms", 0.0)
            self.svd.components_ = np.ascontiguousarray(self.svd.components_, dtype=np.float32)
            return True
        except Exception:
            return False

    @property
    def ready(self) -> bool:
        return self.F is not None

    def info(self) -> dict[str, Any]:
        return {"hashing": f"2^{int(np.log2(self.n_features))} features, 1-{self.ngram_max}-grams", "tfidf": "sublinear TF-IDF",
                "latent": f"truncated SVD {self.svd.n_components if self.svd else self.n_components} dims", "explained_variance": float(self.svd.explained_variance_ratio_.sum()) if self.svd is not None else None,
                "build_ms": round(self.build_ms)}


class SurrogateSlot(Slot):
    kind = "surrogate"

    def __init__(self, name: str, display: str, feat: SurrogateFeaturizer, head: str = "logreg", C: float = 4.0, hidden: int = 256, seed: int = 42):
        super().__init__(name, display)
        self.feat, self.head, self.C, self.hidden, self.seed = feat, head, C, hidden, seed
        self.clf = None
        self.train_ms = 0.0
        self.n_train = 0

    def _make(self):
        if self.head == "mlp":
            return MLPClassifier(hidden_layer_sizes=(self.hidden,), max_iter=60, random_state=self.seed, alpha=1e-4)
        return LogisticRegression(max_iter=400, C=self.C)

    def fit(self, rows: np.ndarray, y: np.ndarray) -> None:
        t = time.perf_counter()
        clf = self._make()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(self.feat.F[rows], y)
        self.clf = clf                      # swapped in only once fitted, so concurrent status / analyse calls never see a half-built head
        self.train_ms = (time.perf_counter() - t) * 1000
        self.n_train = int(len(rows))
        self.trained = True

    def _proba(self, F: np.ndarray) -> np.ndarray:
        t = time.perf_counter()
        p = self.clf.predict_proba(F).astype(np.float32)
        self.last_ms = (time.perf_counter() - t) * 1000
        seen = np.asarray(self.clf.classes_, dtype=int)
        if len(seen) != len(self.classes) or not np.array_equal(seen, np.arange(len(self.classes))):
            out = np.zeros((p.shape[0], len(self.classes)), dtype=np.float32)
            out[:, seen] = p
            return out
        return p

    def predict_proba(self, rows: np.ndarray) -> np.ndarray:
        return self._proba(self.feat.F[rows])

    def predict_proba_texts(self, texts: list[str]) -> np.ndarray:
        return self._proba(self.feat.transform(texts))

    def predict_proba_features(self, F: np.ndarray) -> np.ndarray:
        """Score already-projected rows (the analysis path projects a report once and shares it across slots)."""
        return self._proba(F)

    def payload(self) -> dict[str, Any]:
        clf = self.clf
        if clf is None or not (hasattr(clf, "coefs_") or hasattr(clf, "coef_")):     # None, or being re-fitted right now
            return {"params": 0, "bytes_fp32": 0, "bytes_fp16": 0, "bytes_int8": 0}
        if hasattr(clf, "coefs_"):
            p = int(sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_))
        else:
            p = int(clf.coef_.size + clf.intercept_.size)
        return {"params": p, "bytes_fp32": 4 * p, "bytes_fp16": 2 * p, "bytes_int8": p + 8}

    def info(self) -> dict[str, Any]:
        return {"kind": self.kind, "feature_extractor": "shared TF-IDF + SVD latent space (stand-in for the transformer encoder)",
                "classifier": f"{'MLP head ' + str(self.hidden) + ' hidden' if self.head == 'mlp' else 'logistic-regression head'}", "n_classes": len(self.classes),
                "n_train": self.n_train, "train_ms": round(self.train_ms), **self.payload()}

    # persistence
    def dumps(self) -> dict[str, Any]:
        return {"clf": self.clf, "classes": self.classes, "train_ms": self.train_ms, "n_train": self.n_train}

    def loads(self, d: dict[str, Any]) -> None:
        self.clf, self.train_ms, self.n_train = d["clf"], d.get("train_ms", 0.0), d.get("n_train", 0)
        self.set_classes(d["classes"])
        self.trained = self.clf is not None


# =============================================================================
# RoBERTa: fine-tuned Hugging Face folder
# =============================================================================
class RobertaSlot(Slot):
    kind = "roberta"

    def __init__(self, name: str, display: str, model_dir: Path, max_length: int = 512, batch_size: int = 8, int8: bool = False, texts: list[str] | None = None):
        super().__init__(name, display)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.dir, self.max_length, self.batch_size, self.int8 = Path(model_dir), max_length, batch_size, int8
        self.corpus_texts = texts
        t = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tok = AutoTokenizer.from_pretrained(str(self.dir))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.dir)).eval()
        labels_file = self.dir / "labels.json"
        if labels_file.exists():
            classes = json.loads(labels_file.read_text(encoding="utf-8"))
        else:
            id2label = self.model.config.id2label
            classes = [id2label[i] for i in range(len(id2label))]
        self.set_classes(classes)
        sd = self.model.state_dict()
        self.meta = {"params": int(sum(v.numel() for v in sd.values() if v.dtype.is_floating_point)),
                     "bytes_fp32": _sd_bytes(sd), "bytes_fp16": _sd_bytes(sd, dtype=torch.float16), "bytes_int8": _sd_bytes(sd, int8=True),
                     "layers": int(self.model.config.num_hidden_layers), "hidden": int(self.model.config.hidden_size), "heads": int(self.model.config.num_attention_heads)}
        if int8:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.load_ms = (time.perf_counter() - t) * 1000
        self.trained = True

    def fit(self, rows: np.ndarray, y: np.ndarray) -> None:  # fine-tuning happens offline on GPU (scripts/colab_finetune_roberta.py)
        pass

    def predict_proba_texts(self, texts: list[str]) -> np.ndarray:
        import torch
        t = time.perf_counter()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tok(texts[i:i + self.batch_size], truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
                logits = self.model(**enc).logits
                out.append(torch.softmax(logits, dim=1).numpy())
        self.last_ms = (time.perf_counter() - t) * 1000
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, len(self.classes)), dtype=np.float32)

    def predict_proba(self, rows: np.ndarray) -> np.ndarray:
        return self.predict_proba_texts([self.corpus_texts[int(i)] for i in rows])

    def payload(self) -> dict[str, Any]:
        return {k: self.meta[k] for k in ("params", "bytes_fp32", "bytes_fp16", "bytes_int8")}

    def info(self) -> dict[str, Any]:
        n = self.max_length
        return {"kind": self.kind, "dir": str(self.dir), "n_classes": len(self.classes), "layers": self.meta["layers"], "hidden": self.meta["hidden"], "heads": self.meta["heads"],
                "gmacs_forward": round(roberta_gmacs(n, self.meta["layers"], self.meta["hidden"]), 1), "attention_map_bytes": attention_map_bytes(n, self.meta["layers"], self.meta["heads"]),
                "int8_dynamic": self.int8, "load_ms": round(self.load_ms), **self.payload()}


def _sd_bytes(sd: dict, dtype=None, int8: bool = False) -> int:
    """Serialised size of a state-dict in the given precision (what an export would occupy)."""
    import torch
    buf = io.BytesIO()
    out = {}
    for k, v in sd.items():
        if not torch.is_floating_point(v):
            out[k] = v
        elif int8:
            scale = v.abs().max() / 127.0 if v.numel() else torch.tensor(1.0)
            out[k] = (v / (scale + 1e-12)).round().clamp(-127, 127).to(torch.int8)
            out[k + ".scale"] = scale.to(torch.float32)
        else:
            out[k] = v.to(dtype) if dtype else v
    torch.save(out, buf)
    return buf.tell()


def analytical_cost_table(max_length: int) -> list[dict[str, Any]]:
    """Table 4.14 / 4.17 rows for the UI (closed-form, independent of what is loaded)."""
    rows = []
    for name, L, d, h, n, params in [("DistilRoBERTa-base", 6, 768, 12, max_length, 82_118_400), ("RoBERTa-base, n = 128", 12, 768, 12, 128, 124_645_632),
                                     ("RoBERTa-base, n = 256", 12, 768, 12, 256, 124_645_632), (f"RoBERTa-base, n = {max_length} (used)", 12, 768, 12, max_length, 124_645_632),
                                     ("RoBERTa-large", 24, 1024, 16, max_length, 355_359_744)]:
        g = roberta_gmacs(n, L, d)
        rows.append({"model": name, "layers": L, "hidden": d, "heads": h, "n": n, "params": params, "weights_fp32_mb": round(params * 4 / 1e6), "weights_fp16_mb": round(params * 2 / 1e6),
                     "weights_int8_mb": round(params / 1e6), "gmacs": round(g, 1), "train_gmacs": round(3 * g), "attention_mb": round(attention_map_bytes(n, L, h) / 1e6, 1),
                     "used": "(used)" in name})
    return rows
