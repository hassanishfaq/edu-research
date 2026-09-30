"""FedStack engine — nodes, server-side stacking, federated rounds, scanning, zero-day evaluation.

Two node back-ends share one interface:
  * ResNetNode   — the thesis model, loaded from checkpoints written by train_node.py
  * SurrogateNode — CPU stand-in used until those checkpoints are placed in ./models/

The server side (accuracy-aware weighting + logistic-regression meta-learner, class
alignment, payload accounting) is the same in both cases and mirrors server_stack.py /
stacking.py of the released code.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from . import imaging
from .config import Config, ROOT, _norm_class
from .models import BACKBONE_GMACS, BACKBONE_PARAMS
from .surrogate import SurrogateNode

try:
    import psutil
except ImportError:  # optional
    psutil = None


def pretty(name: str) -> str:
    return name.replace("_", " ")


# =============================================================================
# Node runtimes
# =============================================================================
class NodeRuntime:
    kind = "abstract"

    def __init__(self, name: str, label: str, classes: list[str]):
        self.name, self.label = name, label
        self.classes = classes                      # local class names in local index order
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.val_accuracy = 0.0
        self.last_ms = 0.0

    def predict_proba(self, batch: np.ndarray) -> np.ndarray:  # (n, n_local)
        raise NotImplementedError

    def payload(self) -> dict[str, int]:
        raise NotImplementedError

    def info(self) -> dict[str, Any]:
        raise NotImplementedError


class SurrogateRuntime(NodeRuntime):
    kind = "surrogate"

    def __init__(self, name: str, label: str, classes: list[str], cfg: Config):
        super().__init__(name, label, classes)
        self.model = SurrogateNode(name, classes, n_components=int(cfg.get("surrogate.pca_components", 96)),
                                   lr=float(cfg.get("surrogate.learning_rate", 0.0005)), seed=int(cfg.get("surrogate.seed", 42)),
                                   hidden=int(cfg.get("surrogate.hidden_units", 256)))

    def predict_proba(self, batch: np.ndarray) -> np.ndarray:
        t = time.perf_counter()
        p = self.model.predict_proba(batch)
        self.last_ms = (time.perf_counter() - t) * 1000
        return p

    def payload(self) -> dict[str, int]:
        return self.model.payload()

    def info(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.model.info()}


class ResNetRuntime(NodeRuntime):
    kind = "resnet"

    def __init__(self, name: str, label: str, ckpt_path: Path, image_size: int):
        from .models import load_node_checkpoint
        model, c2i, meta = load_node_checkpoint(ckpt_path)
        classes = [None] * len(c2i)
        for c, i in c2i.items():
            classes[i] = _norm_class(c)
        super().__init__(name, label, classes)
        self.model, self.meta, self.image_size, self.path = model, meta, image_size, ckpt_path

    def predict_proba(self, batch: np.ndarray) -> np.ndarray:
        import torch
        t = time.perf_counter()
        with torch.no_grad():
            arr = batch.astype(np.float32) / 255.0 if batch.dtype == np.uint8 else np.ascontiguousarray(batch, dtype=np.float32)
            logits, _ = self.model(torch.from_numpy(arr))
            p = torch.softmax(logits, dim=1).numpy()
        self.last_ms = (time.perf_counter() - t) * 1000
        return p

    def payload(self) -> dict[str, int]:
        m = self.meta
        return {"params": m["params"], "bytes_fp32": m["bytes_fp32"], "bytes_fp16": m["bytes_fp16"], "bytes_int8": m["bytes_int8"]}

    def info(self) -> dict[str, Any]:
        return {"kind": self.kind, "checkpoint": str(self.path), "backbone": self.meta["backbone"], "n_classes": self.meta["n_classes"],
                "head_bytes_fp32": self.meta["head_bytes_fp32"], **self.payload()}


# =============================================================================
# Engine
# =============================================================================
class Engine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.lock = threading.RLock()
        self.events: deque[dict[str, Any]] = deque(maxlen=2000)
        self._event_id = 0
        self.started_at = time.time()
        self.mode = "surrogate"
        self.nodes: list[NodeRuntime] = []
        self.vocab: dict[str, int] = {}
        self.classes: list[str] = []
        self.meta: LogisticRegression | None = None
        self.acc_weights: np.ndarray | None = None
        self.meta_metrics: dict[str, Any] = {}
        self.threshold: float | None = None          # resolved open-set threshold (config value or auto-calibrated)
        self.history: list[dict[str, Any]] = []
        self.federation_running = False
        self.federation_progress: dict[str, Any] = {}
        self.zero_day_result: dict[str, Any] | None = None
        self.scan_history: deque[dict[str, Any]] = deque(maxlen=50)
        self.cache: dict[str, Any] | None = None
        self._torch_cache: dict[str, np.ndarray] = {}
        self.watch_enabled = False
        self._watch_seen: set[str] = set()
        self._watch_thread: threading.Thread | None = None
        self._proc = psutil.Process(os.getpid()) if psutil else None
        if self._proc:
            self._proc.cpu_percent(None)
        self._boot()

    # ------------------------------------------------------------------ events
    def log(self, msg: str, level: str = "info", **data: Any) -> None:
        with self.lock:
            self._event_id += 1
            self.events.append({"id": self._event_id, "t": time.time(), "level": level, "msg": msg, **({"data": data} if data else {})})

    def events_since(self, since: int) -> list[dict[str, Any]]:
        with self.lock:
            return [e for e in self.events if e["id"] > since]

    # ------------------------------------------------------------------ boot
    def _boot(self) -> None:
        self.log("FedStack engine starting", "sys")
        self._load_cache()
        req = self.cfg.mode
        have_all = all((self.cfg.node_checkpoint(n) or Path("/nonexistent")).exists() for n in self.cfg.nodes) and bool(self.cfg.nodes)
        if req == "resnet" or (req == "auto" and have_all):
            try:
                self._boot_resnet()
                self.mode = "resnet"
            except Exception as e:
                self.log(f"ResNet checkpoints could not be loaded ({e}); falling back to surrogate nodes", "warn")
                self._boot_surrogate()
        else:
            if req == "resnet":
                self.log("mode: resnet requested but checkpoints are missing; using surrogate nodes", "warn")
            self._boot_surrogate()
        self.log(f"engine ready in {self.mode.upper()} mode with {len(self.nodes)} nodes and {len(self.classes)} families", "sys")

    def _load_cache(self) -> None:
        p = self.cfg.cache_path
        if not p.exists():
            self.log(f"image cache not found ({p.name}); run  python scripts/prepare_data.py", "warn")
            return
        z = np.load(p, allow_pickle=False)
        self.cache = {k: z[k] for k in z.files}
        self.cache["classes"] = [str(c) for c in self.cache["classes"]]
        self.cache["zd_classes"] = [str(c) for c in self.cache["zd_classes"]]
        for k in ("files", "zd_files"):
            if k in self.cache:
                self.cache[k] = [str(f).replace("\\", "/") for f in self.cache[k]]
        self.log(f"image cache loaded: {len(self.cache['X'])} known images / {len(self.cache['X_zd'])} unseen-family images at {int(self.cache['size'])}px", "sys")

    def _boot_surrogate(self) -> None:
        self.mode = "surrogate"
        if not self.cache:
            raise RuntimeError("Surrogate mode needs the image cache. Run: python scripts/prepare_data.py")
        self.classes = list(self.cache["classes"])
        self.vocab = {c: i for i, c in enumerate(self.classes)}
        part = self.cfg.partition
        self.nodes = []
        for n in self.cfg.nodes:
            fams = [f for f in part.get(n["name"], []) if f in self.vocab]
            missing = [f for f in part.get(n["name"], []) if f not in self.vocab]
            if missing:
                self.log(f"{n['name']}: families not in dataset: {missing}", "warn")
            self.nodes.append(SurrogateRuntime(n["name"], n.get("label", ""), fams, self.cfg))
        state = ROOT / "data" / "cache" / "state_surrogate.pkl"
        if state.exists():
            try:
                with open(state, "rb") as f:
                    st = pickle.load(f)
                for rt, blob in zip(self.nodes, st["nodes"]):
                    rt.model = SurrogateNode.loads(blob)
                    rt.val_accuracy = rt.model.accuracy(*self._node_split(rt, 1))
                self.meta, self.acc_weights, self.meta_metrics, self.history = st["meta"], st["acc_weights"], st["meta_metrics"], st["history"]
                self.threshold = st.get("threshold")
                self.log(f"restored trained surrogate state ({len(self.history)} rounds)", "sys")
            except Exception as e:
                self.log(f"could not restore saved state: {e}", "warn")

    def _boot_resnet(self) -> None:
        size = int(self.cfg.get("model.image_size", 224))
        self.nodes = []
        for n in self.cfg.nodes:
            ck = self.cfg.node_checkpoint(n)
            t = time.perf_counter()
            rt = ResNetRuntime(n["name"], n.get("label", ""), ck, size)
            self.nodes.append(rt)
            self.log(f"{n['name']}: loaded {rt.meta['backbone']} checkpoint ({rt.meta['params'] / 1e6:.1f} M params, {rt.meta['bytes_int8'] / 1e6:.1f} MB INT8) in {(time.perf_counter() - t) * 1000:.0f} ms", "sys")
        # global vocabulary = union of node classes (server_stack.py)
        vocab: dict[str, int] = {}
        for rt in self.nodes:
            for c in rt.classes:
                vocab.setdefault(c, len(vocab))
        self.vocab, self.classes = vocab, list(vocab)
        st = self.cfg.stacker_path
        if st and st.exists():
            import joblib
            obj = joblib.load(st)
            self.meta = obj["stacker"]
            saved_vocab = {_norm_class(k): v for k, v in obj.get("vocab", {}).items()}
            if saved_vocab and saved_vocab != self.vocab:
                self.log("stacker.joblib vocabulary differs from the node checkpoints; re-fit recommended", "warn")
            self.meta_metrics = {"source": "stacker.joblib", **(obj.get("metrics") or {})}
            self.acc_weights = np.ones(len(self.nodes)) / len(self.nodes)
            self.log("meta-learner loaded from stacker.joblib", "sys")
        else:
            self.log("no stacker.joblib — the meta-learner will be fitted on the validation split when you run the federation panel", "warn")

    # ------------------------------------------------------------------ data access
    def _node_idx(self, rt: NodeRuntime, split_code: int) -> np.ndarray:
        c = self.cache
        gidx = np.array([self.vocab[f] for f in rt.classes])
        return np.where((c["split"] == split_code) & np.isin(c["y"], gidx))[0]

    def _global_idx(self, split_code: int) -> np.ndarray:
        return np.where(self.cache["split"] == split_code)[0]

    def _fullres(self, source: str, idx: np.ndarray) -> np.ndarray | None:
        """Backbone-resolution uint8 images (n, 1, S, S) read from the dataset folders, or None when the
        original images are not available (the caller then falls back to up-sampled thumbnails)."""
        files = self.cache.get("files" if source == "known" else "zd_files")
        root = self.cfg.training_dir if source == "known" else self.cfg.zero_day_dir
        if files is None or root is None or not root.exists():
            return None
        from PIL import Image
        size = int(self.cfg.get("model.image_size", 224))
        out = np.zeros((len(idx), 1, size, size), dtype=np.uint8)
        for j, i in enumerate(idx):
            f = root / files[int(i)]
            if not f.exists():
                return None
            out[j, 0] = np.asarray(Image.open(f).convert("L").resize((size, size), Image.BILINEAR), dtype=np.uint8)
        return out

    def _node_split(self, rt: NodeRuntime, split_code: int) -> tuple[np.ndarray, np.ndarray]:
        """Images of the node's own families in a split (0 train, 1 val, 2 test) with LOCAL labels."""
        c = self.cache
        gidx = np.array([self.vocab[f] for f in rt.classes])
        mask = (c["split"] == split_code) & np.isin(c["y"], gidx)
        X = c["X"][mask]
        g2l = {g: i for i, g in enumerate(gidx)}
        y = np.array([g2l[int(v)] for v in c["y"][mask]], dtype=np.int64)
        return X, y

    def _global_split(self, split_code: int) -> tuple[np.ndarray, np.ndarray]:
        c = self.cache
        mask = c["split"] == split_code
        return c["X"][mask], c["y"][mask].astype(np.int64)

    def _node_inputs(self, rt: NodeRuntime, X_thumbs: np.ndarray, key: str | None = None, ref: tuple[str, np.ndarray] | None = None) -> np.ndarray:
        """Surrogate nodes consume the cached thumbnails.  ResNet nodes consume backbone-resolution images read
        from the dataset folders (ref = (source, indices)); only when those are unavailable are the thumbnails
        up-sampled, which lowers the accuracy of trained checkpoints and is logged once."""
        if rt.kind == "surrogate":
            return X_thumbs
        if key and key in self._torch_cache:
            return self._torch_cache[key]
        arr = self._fullres(*ref) if ref is not None else None
        if arr is None:
            if not getattr(self, "_warned_thumbs", False):
                self.log("original dataset images not found; ResNet nodes are fed up-sampled thumbnails (lower accuracy)", "warn")
                self._warned_thumbs = True
            from PIL import Image
            size = int(self.cfg.get("model.image_size", 224))
            arr = np.stack([np.asarray(Image.fromarray(x, "L").resize((size, size), Image.BILINEAR), dtype=np.uint8)[None] for x in X_thumbs])
        if key:
            self._torch_cache[key] = arr
        return arr

    # ------------------------------------------------------------------ stacking
    def align(self, rt: NodeRuntime, probs: np.ndarray) -> np.ndarray:
        out = np.zeros((probs.shape[0], len(self.classes)), dtype=np.float32)
        for c, li in rt.class_to_idx.items():
            out[:, self.vocab[c]] = probs[:, li]
        return out

    def _weights(self) -> np.ndarray:
        if not bool(self.cfg.get("model.accuracy_weighting", True)):
            return np.ones(len(self.nodes)) / len(self.nodes)
        w = np.array([max(rt.val_accuracy, 1e-3) for rt in self.nodes], dtype=np.float64)
        return w / w.sum()

    def stack_features(self, aligned: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
        return np.concatenate([a * w for a, w in zip(aligned, weights)], axis=1)

    def _node_predictions(self, X: np.ndarray, key: str | None = None, batch: int = 64, ref: tuple[str, np.ndarray] | None = None) -> list[np.ndarray]:
        outs = []
        for rt in self.nodes:
            inp = self._node_inputs(rt, X, key, ref)
            probs = np.concatenate([rt.predict_proba(inp[i:i + batch]) for i in range(0, len(inp), batch)]) if len(inp) else np.zeros((0, len(rt.classes)))
            outs.append(self.align(rt, probs))
        return outs

    def fit_meta(self, calibrate: bool = True) -> dict[str, Any]:
        """Server step: run every node on the server's validation pool, weight, fit, test.
        calibrate=True additionally computes out-of-fold confidences on the validation pool
        (5-fold cross_val_predict) to set the open-set threshold without touching test data."""
        Xv, yv = self._global_split(1)
        Xt, yt = self._global_split(2)
        for rt in self.nodes:
            Xn, yn = self._node_split(rt, 1)
            inp = self._node_inputs(rt, Xn, f"{rt.name}:nodeval", ("known", self._node_idx(rt, 1)))
            if rt.kind == "surrogate":
                rt.val_accuracy = rt.model.accuracy(Xn, yn)
            else:
                rt.val_accuracy = float((np.concatenate([rt.predict_proba(inp[i:i + 64]) for i in range(0, len(inp), 64)]).argmax(1) == yn).mean())
        self.acc_weights = self._weights()
        t = time.perf_counter()
        Zv = self.stack_features(self._node_predictions(Xv, "val", ref=("known", self._global_idx(1))), self.acc_weights)
        self.meta = self._make_stacker().fit(Zv, yv)
        fit_ms = (time.perf_counter() - t) * 1000
        if calibrate:
            from sklearn.model_selection import cross_val_predict
            P_oof = cross_val_predict(self._make_stacker(), Zv, yv, cv=5, method="predict_proba")
            self._calibrate_threshold(P_oof)
        elif self.threshold is None:
            self._calibrate_threshold(self.meta.predict_proba(Zv))
        Zt = self.stack_features(self._node_predictions(Xt, "test", ref=("known", self._global_idx(2))), self.acc_weights)
        pred = self.meta.predict(Zt)
        acc = float((pred == yt).mean())
        f1 = float(f1_score(yt, pred, average="macro"))
        per_class = {self.classes[c]: float((pred[yt == c] == c).mean()) for c in np.unique(yt)}
        # node-only baseline: best single node vote (its own top-1 in global space), for the ablation line
        self.meta_metrics = {"accuracy": acc, "f1_macro": f1, "n_val": int(len(yv)), "n_test": int(len(yt)), "fit_ms": round(fit_ms, 1),
                             "stacker": self.stacker_kind, "meta_params": self._meta_size(), "per_class": per_class,
                             "input_dim": int(Zv.shape[1]), "source": "fitted on validation split", "threshold": self.threshold}
        return self.meta_metrics

    @property
    def stacker_kind(self) -> str:
        return str(self.cfg.get("model.stacker", "rf")).lower()

    def _make_stacker(self):
        if self.stacker_kind == "logreg":
            return LogisticRegression(max_iter=1000, C=float(self.cfg.get("model.logreg_C", 10.0)))
        return RandomForestClassifier(n_estimators=int(self.cfg.get("model.rf_trees", 300)), n_jobs=-1, random_state=0)

    def _meta_size(self) -> int:
        m = self.meta
        if hasattr(m, "coef_"):
            return int(m.coef_.size + m.intercept_.size)
        if hasattr(m, "estimators_"):
            return int(sum(t.tree_.node_count for t in m.estimators_))
        return 0

    def _confidence(self, P: np.ndarray) -> np.ndarray:
        if str(self.cfg.get("open_set.method", "max_prob")) == "max_prob":
            return P.max(1)
        Pc = np.clip(P, 1e-12, 1)
        return 1.0 - (-(Pc * np.log(Pc)).sum(1) / np.log(P.shape[1]))

    def _calibrate_threshold(self, P_val: np.ndarray) -> None:
        """Open-set threshold. open_set.threshold = auto -> the confidence below which only
        `known_reject_budget` of the KNOWN validation samples fall (no unseen data is used)."""
        raw = self.cfg.get("open_set.threshold", "auto")
        if isinstance(raw, (int, float)):
            self.threshold = float(raw)
            return
        budget = float(self.cfg.get("open_set.known_reject_budget", 0.05))
        self.threshold = float(np.percentile(self._confidence(P_val), budget * 100))
        self.log(f"open-set threshold auto-calibrated to {self.threshold:.3f} (out-of-fold: rejects {budget * 100:.0f}% of known validation samples)", "fed")

    # ------------------------------------------------------------------ payload accounting
    def payload_round(self) -> dict[str, Any]:
        C = len(self.classes)
        per_node = []
        for rt in self.nodes:
            p = rt.payload()
            n_val = int(np.sum((self.cache["split"] == 1) & np.isin(self.cache["y"], [self.vocab[f] for f in rt.classes]))) if self.cache else 0
            pred_bytes = n_val * C * 4
            w_bytes = p["bytes_int8"] if bool(self.cfg.get("model.quantize_int8_for_transfer", True)) else p["bytes_fp32"]
            per_node.append({"node": rt.name, "params": p["params"], "weights_bytes": w_bytes, "weights_fp32": p["bytes_fp32"],
                             "predictions_bytes": pred_bytes, "n_val": n_val, "upload_bytes": w_bytes + pred_bytes, "download_bytes": 0})
        up = sum(n["upload_bytes"] for n in per_node)
        fedavg = sum(2 * n["weights_fp32"] for n in per_node)              # full FP32 model up AND down
        bb = str(self.cfg.get("model.backbone", "resnet50"))
        ref_p = BACKBONE_PARAMS.get(bb, BACKBONE_PARAMS["resnet50"])
        K, T = len(self.nodes), int(self.cfg.get("federation.rounds", 20))
        ref = {"backbone": bb, "params": ref_p, "int8_bytes": ref_p, "fp32_bytes": 4 * ref_p,
               "proposed_per_node": ref_p + 700 * 28 * 4, "fedavg_per_node": 8 * ref_p,
               "proposed_total": (ref_p + 700 * 28 * 4) * K * T, "fedavg_total": 8 * ref_p * K * T, "gmacs": BACKBONE_GMACS.get(bb, 4.1)}
        return {"per_node": per_node, "proposed_total_bytes": up, "fedavg_total_bytes": fedavg,
                "reduction_x": (fedavg / up) if up else None, "reference_thesis": ref}

    # ------------------------------------------------------------------ federation
    def run_federation(self, rounds: int | None = None, epochs: int | None = None) -> None:
        if self.federation_running:
            return
        threading.Thread(target=self._federation_worker, args=(rounds, epochs), daemon=True).start()

    def _federation_worker(self, rounds: int | None, epochs: int | None) -> None:
        self.federation_running = True
        try:
            T = int(rounds or self.cfg.get("federation.rounds", 20))
            E = int(epochs or self.cfg.get("federation.local_epochs", 5))
            if self.mode == "surrogate":
                self._federation_surrogate(T, E)
            else:
                self._federation_resnet(T)
        except Exception as e:
            self.log(f"federation failed: {e}", "error")
        finally:
            self.federation_running = False
            self.federation_progress = {}

    def _federation_surrogate(self, T: int, E: int) -> None:
        self.log(f"federated training: {T} rounds x {E} local epochs, {len(self.nodes)} class-specialist nodes", "fed")
        # fresh start every run so the learning curve is reproducible
        for rt in self.nodes:
            rt.model = SurrogateNode(rt.name, rt.classes, rt.model.n_components, rt.model.lr, rt.model.seed, rt.model.hidden)
            rt.val_accuracy = 0.0
        self.history = []
        cumulative = 0
        cum_fedavg = 0
        for r in range(1, T + 1):
            t_round = time.perf_counter()
            node_rows = []
            for rt in self.nodes:
                self.federation_progress = {"round": r, "rounds": T, "node": rt.name, "phase": "local training"}
                Xtr, ytr = self._node_split(rt, 0)
                t = time.perf_counter()
                rt.model.train_round(Xtr, ytr, epochs=E)
                Xv, yv = self._node_split(rt, 1)
                rt.val_accuracy = rt.model.accuracy(Xv, yv)
                node_rows.append({"node": rt.name, "train_images": int(len(Xtr)), "local_acc": rt.val_accuracy, "train_ms": round((time.perf_counter() - t) * 1000, 1)})
            self.federation_progress = {"round": r, "rounds": T, "node": "server", "phase": "stacking" + (" + threshold calibration" if r == T else "")}
            m = self.fit_meta(calibrate=(r == T))
            pay = self.payload_round()
            cumulative += pay["proposed_total_bytes"]
            cum_fedavg += pay["fedavg_total_bytes"]
            row = {"round": r, "nodes": node_rows, "weights": [float(w) for w in self.acc_weights], "meta_accuracy": m["accuracy"], "meta_f1": m["f1_macro"],
                   "round_ms": round((time.perf_counter() - t_round) * 1000, 1), "payload": pay, "cumulative_bytes": cumulative, "cumulative_fedavg_bytes": cum_fedavg}
            self.history.append(row)
            self.log(f"round {r}/{T}: meta-learner acc {m['accuracy'] * 100:.1f}% (F1 {m['f1_macro']:.3f}); nodes " +
                     ", ".join(f"{n['node']} {n['local_acc'] * 100:.0f}%" for n in node_rows) +
                     f"; uplink {pay['proposed_total_bytes'] / 1e3:.0f} KB vs FedAvg {pay['fedavg_total_bytes'] / 1e6:.2f} MB", "fed")
        self._save_state()
        self.log(f"federation finished: test accuracy {self.meta_metrics['accuracy'] * 100:.2f}%, macro-F1 {self.meta_metrics['f1_macro']:.3f}", "fed")

    def _federation_resnet(self, T: int) -> None:
        self.log("ResNet mode: node fine-tuning ran offline on GPU (train_node.py); replaying the server-side exchange", "fed")
        self.federation_progress = {"round": 1, "rounds": 1, "node": "server", "phase": "node inference on validation pool"}
        m = self.fit_meta()
        pay = self.payload_round()
        self.history = [{"round": 1, "replay": True, "rounds_configured": T,
                         "nodes": [{"node": rt.name, "local_acc": rt.val_accuracy, "train_images": 0, "train_ms": 0} for rt in self.nodes],
                         "weights": [float(w) for w in self.acc_weights], "meta_accuracy": m["accuracy"], "meta_f1": m["f1_macro"], "payload": pay,
                         "cumulative_bytes": pay["proposed_total_bytes"] * T, "cumulative_fedavg_bytes": pay["fedavg_total_bytes"] * T}]
        self.log(f"meta-learner fitted: test accuracy {m['accuracy'] * 100:.2f}%, macro-F1 {m['f1_macro']:.3f}; per-round uplink {pay['proposed_total_bytes'] / 1e6:.1f} MB "
                 f"({pay['reduction_x']:.1f}x below FedAvg-FP32)", "fed")

    def _save_state(self) -> None:
        if self.mode != "surrogate":
            return
        p = ROOT / "data" / "cache" / "state_surrogate.pkl"
        with open(p, "wb") as f:
            pickle.dump({"nodes": [rt.model.dumps() for rt in self.nodes], "meta": self.meta, "acc_weights": self.acc_weights,
                         "meta_metrics": self.meta_metrics, "history": self.history, "threshold": self.threshold}, f)

    # ------------------------------------------------------------------ scanning
    def ready(self) -> bool:
        return self.meta is not None and all((rt.kind != "surrogate") or (rt.model.clf is not None) for rt in self.nodes)

    def scan_bytes(self, data: bytes, filename: str, source: str = "upload") -> dict[str, Any]:
        t0 = time.perf_counter()
        desc = imaging.describe_bytes(data)
        t1 = time.perf_counter()
        if desc["is_image"]:
            img = imaging.load_image_bytes(data)
            geometry = {"input": "pre-converted image", "raw_width": img.width, "raw_height": img.height}
        else:
            img, geometry = imaging.bytes_to_image(data)
            geometry["input"] = "raw bytes -> grayscale image"
        t2 = time.perf_counter()
        thumb = imaging.to_thumbnail(img, int(self.cache["size"]) if self.cache else 64)[None]
        result: dict[str, Any] = {
            "file": {"name": filename, **desc, "source": source}, "geometry": geometry,
            "image_png_b64": __import__("base64").b64encode(imaging.png_bytes(img)).decode(),
            "hex": imaging.hex_preview(data, 192), "timings_ms": {"describe": round((t1 - t0) * 1000, 2), "convert": round((t2 - t1) * 1000, 2)},
        }
        if not self.ready():
            result["verdict"] = {"status": "not_ready", "message": "Run the federated training first (or place the ResNet checkpoints in ./models)."}
            return result
        full = imaging.to_backbone_array(img, int(self.cfg.get("model.image_size", 224)))[None] if any(rt.kind != "surrogate" for rt in self.nodes) else None
        with self.lock:
            aligned = []
            node_votes = []
            for rt in self.nodes:
                inp = thumb if rt.kind == "surrogate" else full
                p = rt.predict_proba(inp)[0]
                a = self.align(rt, p[None])[0]
                aligned.append(a[None])
                top = int(p.argmax())
                node_votes.append({"node": rt.name, "label": rt.label, "kind": rt.kind, "top_family": rt.classes[top], "top_prob": float(p[top]),
                                   "weight": float(self.acc_weights[len(node_votes)]), "local_acc": rt.val_accuracy, "ms": round(rt.last_ms, 2),
                                   "reply_bytes": int(p.size * 4), "probs": {c: float(p[i]) for c, i in rt.class_to_idx.items()}})
            t3 = time.perf_counter()
            Z = self.stack_features(aligned, self.acc_weights)
            proba = self.meta.predict_proba(Z)[0]
            t4 = time.perf_counter()
        order = np.argsort(-proba)
        top_i = int(order[0])
        family = self.classes[top_i]
        conf = float(proba[top_i])
        method = str(self.cfg.get("open_set.method", "max_prob"))
        thr = float(self.threshold if self.threshold is not None else 0.5)
        ent = float(-(proba[proba > 0] * np.log(proba[proba > 0])).sum() / np.log(len(proba)))
        score = float(self._confidence(proba[None])[0])
        owner = next((v for v in node_votes if family in v["probs"]), None)
        unknown = score < thr
        result["verdict"] = {
            "status": "ok", "family": family, "family_pretty": pretty(family), "confidence": conf, "normalised_entropy": round(ent, 4),
            "open_set": {"method": method, "threshold": thr, "score": round(score, 4), "unknown": bool(unknown)},
            "decision": "UNKNOWN FAMILY — possible zero-day" if unknown else "KNOWN MALWARE FAMILY",
            "top5": [{"family": self.classes[int(i)], "pretty": pretty(self.classes[int(i)]), "prob": float(proba[int(i)])} for i in order[:5]],
            "owner_node": owner["node"] if owner else None, "owner_confidence": float(owner["probs"][family]) if owner else None,
            "node_votes": node_votes, "meta": {"input_dim": int(Z.shape[1]), "weights": [float(w) for w in self.acc_weights], "ms": round((t4 - t3) * 1000, 2)},
        }
        result["timings_ms"].update({"nodes": round((t3 - t2) * 1000, 2), "meta": round((t4 - t3) * 1000, 2), "total": round((t4 - t0) * 1000, 2)})
        result["payload"] = {"image_bytes_sent": len(result["image_png_b64"]) * 3 // 4, "node_reply_bytes": sum(v["reply_bytes"] for v in node_votes)}
        self.scan_history.appendleft({"t": time.time(), "name": filename, "family": family, "confidence": conf, "unknown": bool(unknown), "source": source, "size": len(data)})
        self.log(f"scan {filename}: {'UNKNOWN (possible zero-day)' if unknown else pretty(family)} at {conf * 100:.1f}% in {result['timings_ms']['total']:.0f} ms", "warn" if unknown else "scan")
        return result

    # ------------------------------------------------------------------ zero-day evaluation
    def recalibrate(self) -> float | None:
        """Re-derive the open-set threshold from the current config (out-of-fold on the validation pool)."""
        if not self.ready() or not self.cache:
            return self.threshold
        from sklearn.model_selection import cross_val_predict
        Xv, yv = self._global_split(1)
        Zv = self.stack_features(self._node_predictions(Xv, "val", ref=("known", self._global_idx(1))), self.acc_weights)
        self._calibrate_threshold(cross_val_predict(self._make_stacker(), Zv, yv, cv=5, method="predict_proba"))
        self._save_state()
        return self.threshold

    def zero_day_eval(self) -> dict[str, Any]:
        if not self.ready() or not self.cache or len(self.cache["X_zd"]) == 0:
            return {"status": "not_ready"}
        self.recalibrate()
        thr = float(self.threshold if self.threshold is not None else 0.5)
        method = str(self.cfg.get("open_set.method", "max_prob"))

        def conf_of(X: np.ndarray, key: str, ref: tuple[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            Z = self.stack_features(self._node_predictions(X, key, ref=ref), self.acc_weights)
            P = self.meta.predict_proba(Z)
            return self._confidence(P), P.argmax(1)

        t = time.perf_counter()
        Xt, yt = self._global_split(2)
        s_known, _ = conf_of(Xt, "test", ("known", self._global_idx(2)))
        families = []
        s_unseen_all = []
        for zi, zname in enumerate(self.cache["zd_classes"]):
            zidx = np.where(self.cache["y_zd"] == zi)[0]
            Xz = self.cache["X_zd"][zidx]
            if len(Xz) == 0:
                continue
            s, top = conf_of(Xz, f"zd{zi}", ("zd", zidx))
            s_unseen_all.append(s)
            near = np.bincount(top, minlength=len(self.classes))
            nn_i = np.argsort(-near)[:3]
            families.append({"family": zname, "n": int(len(Xz)), "flagged_unknown": float((s < thr).mean()), "mean_confidence": float(s.mean()),
                             "nearest_known": [{"family": self.classes[int(i)], "pretty": pretty(self.classes[int(i)]), "share": float(near[i] / len(Xz))} for i in nn_i if near[i] > 0]})
        s_unseen = np.concatenate(s_unseen_all) if s_unseen_all else np.zeros(0)
        bins = np.linspace(0, 1, 21)
        hist_known = np.histogram(s_known, bins)[0]
        hist_unseen = np.histogram(s_unseen, bins)[0]
        # threshold sweep: balanced accuracy of the known-vs-unknown decision
        sweep = []
        best = (0.0, thr)
        for cand in np.round(np.arange(0.05, 0.96, 0.05), 2):
            tpr = float((s_unseen < cand).mean()) if len(s_unseen) else 0.0     # unseen correctly flagged
            tnr = float((s_known >= cand).mean())                              # known correctly accepted
            ba = (tpr + tnr) / 2
            sweep.append({"threshold": float(cand), "unseen_flagged": tpr, "known_accepted": tnr, "balanced": ba})
            if ba > best[0]:
                best = (ba, float(cand))
        self.zero_day_result = {
            "status": "ok", "threshold": thr, "method": method, "families": families,
            "known": {"n": int(len(s_known)), "false_unknown_rate": float((s_known < thr).mean()), "mean_confidence": float(s_known.mean())},
            "unseen": {"n": int(len(s_unseen)), "flagged_unknown": float((s_unseen < thr).mean()) if len(s_unseen) else 0.0, "mean_confidence": float(s_unseen.mean()) if len(s_unseen) else 0.0},
            "histogram": {"bins": bins.tolist(), "known": hist_known.tolist(), "unseen": hist_unseen.tolist()},
            "sweep": sweep, "suggested_threshold": best[1], "suggested_balanced_accuracy": best[0], "ms": round((time.perf_counter() - t) * 1000, 0),
        }
        self.log(f"zero-day evaluation: {self.zero_day_result['unseen']['flagged_unknown'] * 100:.1f}% of {len(s_unseen)} unseen-family images flagged unknown at threshold {thr}; "
                 f"{self.zero_day_result['known']['false_unknown_rate'] * 100:.1f}% of known test images false-flagged", "fed")
        return self.zero_day_result

    # ------------------------------------------------------------------ folder watcher (endpoint agent)
    def set_watch(self, enabled: bool) -> None:
        self.watch_enabled = enabled
        d = self.cfg.resolve(self.cfg.get("watch.dir", "watch_inbox"))
        d.mkdir(parents=True, exist_ok=True)
        if enabled:
            self._watch_seen = {p.name for p in d.iterdir() if p.is_file()}
            if self._watch_thread is None or not self._watch_thread.is_alive():
                self._watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
                self._watch_thread.start()
            self.log(f"live watch enabled on {d}", "sys")
        else:
            self.log("live watch disabled", "sys")

    def _watch_loop(self) -> None:
        d = self.cfg.resolve(self.cfg.get("watch.dir", "watch_inbox"))
        exts = tuple(e.lower() for e in (self.cfg.get("watch.extensions") or []))
        poll = float(self.cfg.get("watch.poll_seconds", 1.0))
        quarantine = bool(self.cfg.get("watch.quarantine", False))
        qdir = ROOT / "quarantine"
        while self.watch_enabled:
            try:
                for p in sorted(d.iterdir()):
                    if not p.is_file() or p.name in self._watch_seen or (exts and p.suffix.lower() not in exts):
                        continue
                    # wait until the file stops growing (copy in progress)
                    s1 = p.stat().st_size; time.sleep(0.3)
                    if p.stat().st_size != s1:
                        continue
                    self._watch_seen.add(p.name)
                    data = p.read_bytes()
                    res = self.scan_bytes(data, p.name, source="watch")
                    v = res.get("verdict", {})
                    if quarantine and v.get("status") == "ok":
                        qdir.mkdir(exist_ok=True)
                        shutil.move(str(p), str(qdir / p.name))
                        self.log(f"quarantined {p.name} -> ./quarantine", "warn")
            except Exception as e:
                self.log(f"watch error: {e}", "error")
            time.sleep(poll)

    # ------------------------------------------------------------------ status
    def resources(self) -> dict[str, Any]:
        out: dict[str, Any] = {"uptime_s": round(time.time() - self.started_at)}
        if self._proc:
            mi = self._proc.memory_info()
            out.update({"cpu_percent": self._proc.cpu_percent(None), "rss_mb": round(mi.rss / 1e6, 1), "threads": self._proc.num_threads(),
                        "system_cpu_percent": psutil.cpu_percent(None), "system_ram_percent": psutil.virtual_memory().percent, "cpu_count": psutil.cpu_count()})
        return out

    def status(self) -> dict[str, Any]:
        import sklearn
        versions = {"python": __import__("platform").python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}
        try:
            import torch, torchvision
            versions.update({"torch": torch.__version__, "torchvision": torchvision.__version__, "cuda": torch.cuda.is_available()})
        except Exception:
            versions.update({"torch": None})
        return {
            "mode": self.mode, "ready": self.ready(), "federation_running": self.federation_running, "federation_progress": self.federation_progress,
            "nodes": [{"name": rt.name, "label": rt.label, "kind": rt.kind, "families": rt.classes, "n_families": len(rt.classes), "val_accuracy": rt.val_accuracy,
                       "weight": float(self.acc_weights[i]) if self.acc_weights is not None else None, "info": rt.info()} for i, rt in enumerate(self.nodes)],
            "families": self.classes, "n_families": len(self.classes), "meta": self.meta_metrics, "rounds_done": len(self.history),
            "threshold": self.threshold, "stacker": self.stacker_kind,
            "zero_day_families": list(self.cache["zd_classes"]) if self.cache else [], "cache": {"known": int(len(self.cache["X"])), "unseen": int(len(self.cache["X_zd"])), "px": int(self.cache["size"])} if self.cache else None,
            "watch_enabled": self.watch_enabled, "resources": self.resources(), "versions": versions, "config": self.cfg.summary(),
            "recent_scans": list(self.scan_history)[:12], "payload_now": self.payload_round() if self.nodes and self.cache else None,
        }

    def samples(self) -> dict[str, Any]:
        out: dict[str, list[dict[str, str]]] = {"known": [], "zero_day": []}
        for grp in out:
            d = self.cfg.samples_dir / grp
            if not d.exists():
                continue
            for fam in sorted(p for p in d.iterdir() if p.is_dir()):
                for f in sorted(fam.iterdir())[:6]:
                    out[grp].append({"family": fam.name, "pretty": pretty(fam.name), "file": f.name, "path": f"{grp}/{fam.name}/{f.name}"})
        return out

    def scan_sample(self, rel: str) -> dict[str, Any]:
        p = (self.cfg.samples_dir / rel).resolve()
        if not str(p).startswith(str(self.cfg.samples_dir.resolve())) or not p.exists():
            raise FileNotFoundError(rel)
        return self.scan_bytes(p.read_bytes(), p.name, source="sample")
