"""SigStack engine -- label service, normaliser, encoder slots, stacking, training protocol,
category hold-out (zero-day) study, intake watcher and the verdict store.

Pipeline of one report (Section 4.4.3):
  intake (hash, cache) -> behaviour acquisition -> label service (Eq. 4.4, category map)
  -> normaliser (Eqs. 4.5-4.10, serialise, tokenise, n = 512 policy) -> encoder slots
  -> stacking meta-learner -> unknown-category rule tau_4 (Eq. 4.11) -> verdict store.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict

from . import flatten as fl
from . import stacking as st
from .config import ROOT, Config
from .encoder import RobertaSlot, Slot, SurrogateFeaturizer, SurrogateSlot, analytical_cost_table
from .labels import UNMAPPED, CategoryMap, label_report
from .tokenizer import TokenCounter

try:
    import psutil
except ImportError:  # optional
    psutil = None

MODEL_VERSION_FILE = ROOT / "data" / "cache" / "model_version.txt"


class Engine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.lock = threading.RLock()
        self.events: deque[dict[str, Any]] = deque(maxlen=2000)
        self._event_id = 0
        self.started_at = time.time()
        self.cache: dict[str, Any] | None = None
        self.cmap = CategoryMap(cfg.category_map_path)
        self.categories: list[str] = list(self.cmap.categories)
        self.cat_idx = {c: i for i, c in enumerate(self.categories)}
        self.tokens = TokenCounter(cfg.get("encoder.tokenizer", "roberta-base"))
        self.engines: list[str] = cfg.engine_folders
        # labels
        self.eng_cat: dict[str, list[str | None]] = {}
        self.eng_raw: dict[str, list[str | None]] = {}
        self.y_cat: np.ndarray | None = None          # consensus category index, -1 if outside the labelled corpus
        self.score: np.ndarray | None = None
        self.disputed: np.ndarray | None = None
        self.label_stats: dict[str, Any] = {}
        # models
        self.feat: SurrogateFeaturizer | None = None
        self.slots: dict[str, Slot] = {}            # "category" + one per engine folder
        self.meta = None
        self.threshold: float | None = None
        self.metrics: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.model_version: str = "untrained"
        self.training = False
        self.progress: dict[str, Any] = {}
        self.holdout_running = False
        self.holdout_progress: dict[str, Any] = {}
        self.holdout_result: dict[str, Any] | None = None
        # store
        self.verdicts: deque[dict[str, Any]] = deque(maxlen=200)
        self.verdict_file = ROOT / "data" / "verdicts.jsonl"
        self.watch_enabled = False
        self._watch_seen: set[str] = set()
        self._watch_thread: threading.Thread | None = None
        self._feat_thread: threading.Thread | None = None
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
        self.log("SigStack engine starting", "sys")
        self.log(f"token counter: {self.tokens.kind}" + (f" ({self.tokens.error})" if self.tokens.error else ""), "sys" if not self.tokens.error else "warn")
        self._load_cache()
        if self.cache:
            self._build_labels()
        self._build_slots()
        self.log(f"engine ready: {len(self.slots)} model slots ({', '.join(f'{k}={v.kind}' for k, v in self.slots.items())}), "
                 f"{len(self.categories)} categories, category map {self.cmap.version}", "sys")

    def _load_cache(self) -> None:
        p = self.cfg.cache_path
        if not p.exists():
            self.log(f"report cache not found ({p.name}); run  python scripts/prepare_data.py", "warn")
            return
        with open(p, "rb") as f:
            self.cache = pickle.load(f)
        c = self.cache
        self.log(f"report cache loaded: {len(c['hashes'])} unique hashes, schema {c['schema_version']}, tokens median {int(np.median(c['n_tokens']))} "
                 f"(> 512: {(c['n_tokens'] > 512).mean() * 100:.1f}%)", "sys")

    # ------------------------------------------------------------------ label service over the corpus
    def _build_labels(self) -> None:
        c = self.cache
        n = len(c["hashes"])
        prio = list(self.cfg.get("labels.engine_priority", self.engines))
        min_eng = int(self.cfg.get("labels.min_engines", 1))
        self.eng_raw = {e: c["engine_strings"][e] for e in self.engines}
        self.eng_cat = {}
        gaps: Counter = Counter()
        for e in self.engines:
            out = []
            for s in self.eng_raw[e]:
                if s is None:
                    out.append(None)
                else:
                    cat, _ = self.cmap.map(e, s)
                    out.append(cat)
                    if cat == UNMAPPED:
                        gaps[(e, s)] += 1
            self.eng_cat[e] = out
        y = np.full(n, -1, dtype=np.int64)
        score = np.zeros(n, dtype=np.int8)
        disputed = np.zeros(n, dtype=bool)
        agree_n = agree_ok = 0
        cons_counts: Counter = Counter()
        for i in range(n):
            mapped = {e: self.eng_cat[e][i] for e in self.engines if self.eng_cat[e][i]}
            score[i] = len(mapped)
            cat, disp = self._consensus(mapped, prio)
            disputed[i] = disp
            if len(mapped) > 1:
                agree_n += 1
                agree_ok += 0 if disp else 1
            if cat != UNMAPPED and score[i] >= min_eng:
                y[i] = self.cat_idx[cat]
                cons_counts[cat] += 1
        self.y_cat, self.score, self.disputed = y, score, disputed
        per_engine = {}
        for e in self.engines:
            raw = [s for s in self.eng_raw[e] if s]
            cats = Counter(cc for cc in self.eng_cat[e] if cc)
            per_engine[e] = {"display": self.cfg.display_name(e), "reports": len(raw), "families": len(set(raw)), "categories": dict(cats),
                             "unmapped": cats.get(UNMAPPED, 0), "top_families": Counter(raw).most_common(12)}
        self.label_stats = {
            "unique_hashes": n, "score_distribution": {int(k): int(v) for k, v in Counter(score.tolist()).items()},
            "labelled_corpus": int((y >= 0).sum()), "consensus_distribution": dict(cons_counts), "agreement_rate": (agree_ok / agree_n) if agree_n else None,
            "multi_engine_hashes": agree_n, "disputed": int(disputed.sum()), "per_engine": per_engine,
            "map_gaps": [{"engine": e, "raw": s, "n": int(k)} for (e, s), k in gaps.most_common(30)], "map_version": self.cmap.version, "min_engines": min_eng,
            "token_stats": {"median": int(np.median(c["n_tokens"])), "p90": int(np.percentile(c["n_tokens"], 90)), "share_over_512": float((c["n_tokens"] > 512).mean()),
                            "histogram": self._token_hist()},
        }
        self.log(f"label service ({self.cmap.version}): {self.label_stats['labelled_corpus']} labelled hashes in {len(cons_counts)} categories; "
                 f"engine agreement after mapping {self.label_stats['agreement_rate'] * 100:.1f}% of {agree_n} multi-engine hashes; {len(gaps)} unmapped vendor strings", "sys")

    def _consensus(self, mapped: dict[str, str], prio: list[str]) -> tuple[str, bool]:
        from .labels import consensus
        return consensus(mapped, prio)

    def _token_hist(self) -> dict[str, Any]:
        nt = self.cache["n_tokens"]
        bins = [0, 64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 1024, 1536, 2048, 10 ** 6]
        h = np.histogram(nt, bins)[0]
        return {"bins": bins[:-1], "counts": h.tolist()}

    # ------------------------------------------------------------------ slots
    def _slot_mode(self, slot: str) -> str:
        req = self.cfg.mode
        d = self.cfg.roberta_dir(slot)
        have = bool(d and (d / "config.json").exists())
        if req == "surrogate":
            return "surrogate"
        if req == "roberta" and not have:
            self.log(f"slot {slot}: encoder.mode = roberta but {d} is missing; using the surrogate head", "warn")
        return "roberta" if have else "surrogate"

    def _build_slots(self) -> None:
        sur = self.cfg.get("encoder.surrogate", {}) or {}
        self.feat = SurrogateFeaturizer(int(sur.get("n_features", 2 ** 18)), int(sur.get("ngram_max", 2)), int(sur.get("svd_components", 256)), int(sur.get("seed", 42)))
        rob = self.cfg.get("encoder.roberta", {}) or {}
        texts = self.cache["texts"] if self.cache else None
        for slot in ["category"] + self.engines:
            display = "direct category model" if slot == "category" else f"{self.cfg.display_name(slot)} family model"
            mode = self._slot_mode(slot)
            if mode == "roberta":
                try:
                    rs = RobertaSlot(slot, display, self.cfg.roberta_dir(slot), int(rob.get("max_length", 512)), int(rob.get("batch_size", 8)), bool(rob.get("int8_dynamic", False)), texts)
                    self.slots[slot] = rs
                    self.log(f"slot {slot}: RoBERTa loaded from {rs.dir.name} ({rs.meta['params'] / 1e6:.1f} M params, {rs.meta['layers']} layers, {len(rs.classes)} classes) in {rs.load_ms:.0f} ms", "sys")
                    continue
                except Exception as e:
                    self.log(f"slot {slot}: RoBERTa folder could not be loaded ({e}); using the surrogate head", "warn")
            self.slots[slot] = SurrogateSlot(slot, display, self.feat, str(sur.get("head", "logreg")), float(sur.get("logreg_C", 4.0)), int(sur.get("mlp_hidden", 256)), int(sur.get("seed", 42)))
        if self.cache and any(s.kind == "surrogate" for s in self.slots.values()):
            self._feat_thread = threading.Thread(target=self._build_features, daemon=True)
            self._feat_thread.start()
        self._restore_state()

    def _build_features(self) -> None:
        p = ROOT / "data" / "cache" / "features_surrogate.pkl"
        stamp = float(self.cache.get("built", 0))
        if self.feat.load(p, stamp):
            self.log(f"surrogate feature space restored ({self.feat.F.shape[1]} latent dims for {self.feat.F.shape[0]} reports)", "sys")
            return
        self.log("building the surrogate feature space (hashing -> TF-IDF -> SVD) over the corpus ...", "sys")
        self.feat.build(self.cache["texts"], progress=lambda s: self.log(f"surrogate features: {s}", "sys"))
        self.feat.save(p, stamp)
        self.log(f"surrogate feature space ready in {self.feat.build_ms / 1000:.0f} s ({self.feat.F.shape[1]} latent dims, explained variance {self.feat.info()['explained_variance'] * 100:.0f}%)", "sys")

    def _restore_state(self) -> None:
        p = self.cfg.state_path
        if not p.exists():
            return
        try:
            with open(p, "rb") as f:
                stt = pickle.load(f)
            if stt.get("map_version") != self.cmap.version or stt.get("cache_built") != self.cache.get("built") or stt.get("feat_sig") != self._feat_sig():
                self.log("saved model state belongs to another category-map version, report cache or feature space; retrain", "warn")
                return
            for name, blob in stt["slots"].items():
                s = self.slots.get(name)
                if isinstance(s, SurrogateSlot):
                    s.loads(blob)
            self.meta, self.threshold, self.metrics, self.history = stt["meta"], stt["threshold"], stt["metrics"], stt["history"]
            self.model_version = stt.get("model_version", "restored")
            self.log(f"restored trained state {self.model_version} ({len(self.history)} folds)", "sys")
        except Exception as e:
            self.log(f"could not restore saved state: {e}", "warn")

    def _feat_sig(self) -> tuple:
        return (self.feat.n_features, self.feat.ngram_max, self.feat.n_components) if self.feat else ()

    def _save_state(self) -> None:
        p = self.cfg.state_path
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({"slots": {n: s.dumps() for n, s in self.slots.items() if isinstance(s, SurrogateSlot)}, "meta": self.meta, "threshold": self.threshold,
                         "metrics": self.metrics, "history": self.history, "model_version": self.model_version, "map_version": self.cmap.version,
                         "cache_built": self.cache.get("built"), "feat_sig": self._feat_sig()}, f, protocol=5)

    # ------------------------------------------------------------------ helpers
    def ready(self) -> bool:
        return self.meta is not None and all(s.trained for s in self.slots.values())

    def _family_to_cat(self, engine: str) -> dict[str, str]:
        return {s: self.cmap.map(engine, s)[0] for s in set(x for x in self.eng_raw[engine] if x)}

    def _fold_probs(self, slot: str, P: np.ndarray) -> np.ndarray:
        """Engine family probabilities -> category probabilities (direct slot passes through)."""
        s = self.slots[slot]
        if slot == "category":
            if s.classes == self.categories:
                return P
            out = np.zeros((P.shape[0], len(self.categories)), dtype=np.float32)
            for j, c in enumerate(s.classes):
                if c in self.cat_idx:
                    out[:, self.cat_idx[c]] = P[:, j]
            return out
        return st.fold_to_categories(P, s.classes, self._family_to_cat(slot), self.categories)

    def _selected_rows(self) -> np.ndarray:
        rows = np.where(self.y_cat >= 0)[0]
        cap = int(self.cfg.get("train.max_samples", 0) or 0)
        if cap and cap < len(rows):
            sss = StratifiedShuffleSplit(n_splits=1, train_size=cap, random_state=int(self.cfg.get("train.seed", 42)))
            idx, _ = next(sss.split(rows, self.y_cat[rows]))
            rows = np.sort(rows[idx])
        return rows

    def _engine_family_targets(self, engine: str, rows: np.ndarray, min_support: int) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Classes of the engine's family model and (labelled rows, local labels) inside `rows`."""
        raw = self.eng_raw[engine]
        cnt = Counter(raw[i] for i in rows if raw[i] and self.eng_cat[engine][i] != UNMAPPED)
        classes = sorted(f for f, k in cnt.items() if k >= min_support)
        cidx = {f: j for j, f in enumerate(classes)}
        lab_rows, y = [], []
        for i in rows:
            f = raw[i]
            if f in cidx:
                lab_rows.append(i)
                y.append(cidx[f])
        return classes, np.array(lab_rows, dtype=np.int64), np.array(y, dtype=np.int64)

    # ------------------------------------------------------------------ training protocol (Section 4.3)
    def train(self) -> None:
        if self.training:
            return
        threading.Thread(target=self._train_worker, daemon=True).start()

    def _train_worker(self) -> None:
        self.training = True
        try:
            self._train()
        except Exception as e:
            import traceback
            self.log(f"training failed: {e}", "error")
            traceback.print_exc()
        finally:
            self.training = False
            self.progress = {}

    def _wait_features(self) -> None:
        while any(s.kind == "surrogate" for s in self.slots.values()) and not self.feat.ready:
            self.progress = {"phase": "waiting for the surrogate feature space (hashing -> TF-IDF -> SVD)"}
            time.sleep(0.5)

    def _train(self) -> None:
        cfgt = self.cfg.get("train", {}) or {}
        K = int(cfgt.get("folds", 5))
        seed = int(cfgt.get("seed", 42))
        min_sup = int(cfgt.get("min_family_support", 30))
        rob_rows = int(cfgt.get("roberta_eval_rows", 600))
        self._wait_features()
        rows = self._selected_rows()
        y = self.y_cat[rows]
        n = len(rows)
        self.log(f"training protocol: {K}-fold stratified CV on {n} labelled reports, {len(self.categories)} categories, "
                 f"slots {', '.join(f'{k}:{v.kind}' for k, v in self.slots.items())}", "fed")
        self.history = []
        C = len(self.categories)
        P_oof: dict[str, np.ndarray] = {s: np.zeros((n, C), dtype=np.float32) for s in self.slots}
        fam_pred: dict[str, np.ndarray] = {}      # engine -> predicted family index per selected row (-1 = n/a)
        fam_classes: dict[str, list[str]] = {}
        fam_true: dict[str, np.ndarray] = {}
        pos = {int(r): k for k, r in enumerate(rows)}
        for e in self.engines:
            classes, lab_rows, y_fam = self._engine_family_targets(e, rows, min_sup)
            fam_classes[e] = classes
            fam_true[e] = np.full(n, -1, dtype=np.int64)
            fam_true[e][[pos[int(r)] for r in lab_rows]] = y_fam
            fam_pred[e] = np.full(n, -1, dtype=np.int64)
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
        scored_mask = np.ones(n, dtype=bool)
        t_all = time.perf_counter()
        for k, (tr, te) in enumerate(skf.split(rows, y), start=1):
            t_fold = time.perf_counter()
            fold_row: dict[str, Any] = {"fold": k, "n_train": int(len(tr)), "n_test": int(len(te)), "slots": {}}
            for name, slot in self.slots.items():
                self.progress = {"fold": k, "folds": K, "slot": name, "phase": "local fit" if slot.kind == "surrogate" else "scoring (fine-tuned offline)"}
                t_s = time.perf_counter()
                if name == "category":
                    slot.set_classes(self.categories) if slot.kind == "surrogate" else None
                    tr_rows, tr_y = rows[tr], y[tr]
                else:
                    classes = fam_classes[name]
                    if slot.kind == "surrogate":
                        slot.set_classes(classes)
                    m = fam_true[name][tr] >= 0
                    tr_rows, tr_y = rows[tr][m], fam_true[name][tr][m]
                if slot.kind == "surrogate":
                    if len(tr_rows) == 0 or len(set(tr_y.tolist())) < 2:
                        fold_row["slots"][name] = {"skipped": True}
                        continue
                    slot.fit(tr_rows, tr_y)
                    te_eval = te
                else:
                    rng = np.random.RandomState(seed + k)
                    te_eval = np.sort(rng.choice(te, size=min(rob_rows, len(te)), replace=False))
                    scored_mask[np.setdiff1d(te, te_eval)] = False
                P = slot.predict_proba(rows[te_eval])
                if name != "category":
                    fam_pred[name][te_eval] = P.argmax(1) if P.shape[1] else -1
                P_oof[name][te_eval] = self._fold_probs(name, P)
                acc_rows = te_eval if name == "category" else te_eval[fam_true[name][te_eval] >= 0]
                if name == "category":
                    acc = float((P_oof[name][te_eval].argmax(1) == y[te_eval]).mean()) if len(te_eval) else None
                else:
                    acc = float((fam_pred[name][acc_rows] == fam_true[name][acc_rows]).mean()) if len(acc_rows) else None
                fold_row["slots"][name] = {"kind": slot.kind, "n_train": int(len(tr_rows)), "n_classes": len(slot.classes), "accuracy": acc, "ms": round((time.perf_counter() - t_s) * 1000)}
            fold_row["ms"] = round((time.perf_counter() - t_fold) * 1000)
            self.history.append(fold_row)
            self.log(f"fold {k}/{K}: " + ", ".join(f"{nm} {v['accuracy'] * 100:.1f}%" for nm, v in fold_row["slots"].items() if v.get("accuracy") is not None) + f" · {fold_row['ms'] / 1000:.1f} s", "fed")
        # ---- stacking on the out-of-fold base predictions
        self.progress = {"phase": "stacking meta-learner (out-of-fold) + threshold calibration"}
        keep = np.where(scored_mask)[0]
        Z = np.concatenate([P_oof[s][keep] for s in self.slots], axis=1)
        yk = y[keep]
        meta = st.make_meta(str(cfgt.get("meta_learner", "logreg")), float(cfgt.get("logreg_C", 1.0)), seed)
        P_meta = cross_val_predict(meta, Z, yk, cv=StratifiedKFold(K, shuffle=True, random_state=seed), method="predict_proba")
        meta_pred = P_meta.argmax(1)
        self.meta = st.make_meta(str(cfgt.get("meta_learner", "logreg")), float(cfgt.get("logreg_C", 1.0)), seed).fit(Z, yk)
        os_cfg = self.cfg.get("open_set", {}) or {}
        self.threshold = st.calibrate_threshold(P_meta, str(os_cfg.get("method", "max_prob")), os_cfg.get("threshold", "auto"), float(os_cfg.get("known_reject_budget", 0.1)))
        # ---- metrics
        m: dict[str, Any] = {"n": int(len(keep)), "n_selected": n, "folds": K, "categories": self.categories, "map_version": self.cmap.version,
                             "schema_version": self.cache["schema_version"], "meta_learner": str(cfgt.get("meta_learner", "logreg")), "meta_params": st.meta_size(self.meta),
                             "stack_dim": int(Z.shape[1]), "threshold": self.threshold}
        m["stacking"] = {"summary": st.summary_metrics(yk, meta_pred), "per_class": st.per_class_metrics(yk, meta_pred, self.categories)}
        vote = np.mean([P_oof[s][keep] for s in self.slots], axis=0).argmax(1)
        m["voting"] = {"summary": st.summary_metrics(yk, vote), "per_class": st.per_class_metrics(yk, vote, self.categories)}
        m["direct"] = {"summary": st.summary_metrics(yk, P_oof["category"][keep].argmax(1)), "per_class": st.per_class_metrics(yk, P_oof["category"][keep].argmax(1), self.categories), "kind": self.slots["category"].kind}
        # per-engine model: on ITS labelled rows against ITS mapped category (Table 4.10 columns) and against consensus on all rows (Table 4.9 rows)
        eng_metrics_own: dict[str, dict[str, dict[str, float]]] = {}
        m["engines"] = {}
        for e in self.engines:
            ec = np.array([self.cat_idx.get(self.eng_cat[e][int(r)], -1) if self.eng_cat[e][int(r)] else -1 for r in rows[keep]])
            own = ec >= 0
            pred_e = P_oof[e][keep].argmax(1)
            own_pc = st.per_class_metrics(ec[own], pred_e[own], self.categories) if own.any() else {}
            eng_metrics_own[e] = own_pc
            fam_rows = keep[fam_true[e][keep] >= 0]
            fam_pc = {}
            if len(fam_rows):
                yt, yp = fam_true[e][fam_rows], fam_pred[e][fam_rows]
                pc = st.per_class_metrics(yt, np.where(yp < 0, -1, yp), fam_classes[e])
                fam_pc = dict(sorted(pc.items(), key=lambda kv: -kv[1]["support"]))
            m["engines"][e] = {"display": self.cfg.display_name(e), "kind": self.slots[e].kind, "n_families": len(fam_classes[e]),
                               "own_labels": {"n": int(own.sum()), "summary": st.summary_metrics(ec[own], pred_e[own]) if own.any() else None, "per_class": own_pc},
                               "vs_consensus": {"summary": st.summary_metrics(yk, pred_e), "per_class": st.per_class_metrics(yk, pred_e, self.categories)},
                               "family_accuracy": float((fam_pred[e][fam_rows] == fam_true[e][fam_rows]).mean()) if len(fam_rows) else None,
                               "family_metrics": fam_pc}
        m["stacking_support_weighted"] = st.support_weighted_stacking(eng_metrics_own, self.categories)
        # six-family report view (Table 4.10 grouping)
        groups = self.cfg.get("labels.report_groups", {}) or {}
        gnames = sorted(set(groups.values()))
        if groups:
            gidx = {g: i for i, g in enumerate(gnames)}
            to_g = np.array([gidx.get(groups.get(c, c), -1) for c in self.categories])
            yg, pg = to_g[yk], to_g[meta_pred]
            m["groups"] = {"names": gnames, "stacking": st.per_class_metrics(yg, pg, gnames), "summary": st.summary_metrics(yg, pg),
                           "engines": {e: st.per_class_metrics(to_g[np.array([self.cat_idx.get(self.eng_cat[e][int(r)], 0) if self.eng_cat[e][int(r)] else 0 for r in rows[keep]])][np.array([bool(self.eng_cat[e][int(r)]) and self.eng_cat[e][int(r)] != UNMAPPED for r in rows[keep]])],
                                                                to_g[P_oof[e][keep].argmax(1)][np.array([bool(self.eng_cat[e][int(r)]) and self.eng_cat[e][int(r)] != UNMAPPED for r in rows[keep]])], gnames) for e in self.engines}}
        m["confidence_hist"] = np.histogram(st.confidence(P_meta, str(os_cfg.get("method", "max_prob"))), np.linspace(0, 1, 21))[0].tolist()
        m["ms_total"] = round((time.perf_counter() - t_all) * 1000)
        self.metrics = m
        # ---- final fit on every selected row for serving
        self.progress = {"phase": "final fit of every slot on all selected rows"}
        for name, slot in self.slots.items():
            if slot.kind != "surrogate":
                continue
            if name == "category":
                slot.set_classes(self.categories)
                slot.fit(rows, y)
            else:
                slot.set_classes(fam_classes[name])
                mrows = fam_true[name] >= 0
                if mrows.sum() and len(set(fam_true[name][mrows].tolist())) >= 2:
                    slot.fit(rows[mrows], fam_true[name][mrows])
        self.model_version = time.strftime("m%Y%m%d-%H%M%S")
        self._save_state()
        s = m["stacking"]["summary"]
        self.log(f"training finished: stacking accuracy {s['accuracy'] * 100:.2f}%, macro-F1 {s['f1_macro'] * 100:.2f}% (voting {m['voting']['summary']['f1_macro'] * 100:.2f}%, "
                 f"direct {m['direct']['summary']['f1_macro'] * 100:.2f}%); tau_4 = {self.threshold:.3f}; model version {self.model_version}", "fed")

    # ------------------------------------------------------------------ analyse one report (Section 4.4.3 path)
    def _acquire(self, data: bytes, name: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Return (report, acquisition info).  Accepts a raw JSON report or a <hash>.txt (cache / VirusTotal look-up)."""
        info: dict[str, Any] = {"source": "upload"}
        h = fl.hash_from_name(name)
        cache_dir = self.cfg.resolve(self.cfg.get("acquisition.cache_dir", "data/cache/reports"))
        text = data.decode("utf-8", "replace").strip()
        if name.lower().endswith(".txt") or (len(text) in (32, 64) and all(ch in "0123456789abcdefABCDEF" for ch in text)):
            h = text.lower() if text and all(ch in "0123456789abcdefABCDEF" for ch in text) else h
            info["mode"] = "hash look-up"
            if h and cache_dir and (cache_dir / f"{h}.json").exists():
                info["source"] = "report cache (hash hit)"
                return json.loads((cache_dir / f"{h}.json").read_text(encoding="utf-8")), {**info, "hash": h}
            if h and self.cache and h in self._hash_index():
                info["source"] = "corpus (hash hit)"
                return {"_corpus_row": self._hash_index()[h]}, {**info, "hash": h}
            key = self.cfg.get("acquisition.virustotal_api_key")
            if h and key:
                from .vt import fetch_behaviour
                rep, err = fetch_behaviour(h, key)
                if rep is not None:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    (cache_dir / f"{h}.json").write_text(json.dumps(rep), encoding="utf-8")
                    info["source"] = "VirusTotal API (fetched, now cached)"
                    return rep, {**info, "hash": h}
                info["error"] = err
            else:
                info["error"] = "hash not in the local corpus or report cache; set acquisition.virustotal_api_key to fetch it"
            return None, {**info, "hash": h}
        try:
            rep = json.loads(text)
        except Exception as e:
            return None, {**info, "error": f"not a JSON behaviour report ({e})", "hash": h}
        if isinstance(rep, dict) and "data" in rep and "attributes" in (rep.get("data") or {}):
            from .vt import v3_to_v1
            rep = v3_to_v1(rep)
            info["converted"] = "VirusTotal v3 behaviour_summary -> schema v1"
        return rep, {**info, "hash": h}

    def _hash_index(self) -> dict[str, int]:
        if not hasattr(self, "_hidx"):
            self._hidx = {h: i for i, h in enumerate(self.cache["hashes"])} if self.cache else {}
        return self._hidx

    def analyze(self, data: bytes, name: str, source: str = "upload", engine_strings: dict[str, str] | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        sha256 = hashlib.sha256(data).hexdigest()
        md5 = hashlib.md5(data).hexdigest()
        rep, acq = self._acquire(data, name)
        t1 = time.perf_counter()
        result: dict[str, Any] = {"file": {"name": name, "size_bytes": len(data), "sha256": sha256, "md5": md5, "source": source, "hash": acq.get("hash")},
                                  "acquisition": acq, "timings_ms": {"acquire": round((t1 - t0) * 1000, 2)}}
        if rep is None:
            result["verdict"] = {"status": "error", "message": acq.get("error", "no report")}
            return result
        # corpus row look-up (hash hit on a known report)
        corpus_row = rep.get("_corpus_row") if isinstance(rep, dict) else None
        ncfg = self.cfg.get("normalizer", {}) or {}
        schema = fl.SCHEMAS[str(ncfg.get("schema_version", "v1"))]
        if corpus_row is not None:
            text = self.cache["texts"][corpus_row]
            norm = {"flat_keys": None, "dropped_keys": [], "row": {}, "stats": {"schema_keys": len(schema), "present": int(self.cache["present"][corpus_row]), "padded": len(schema) - int(self.cache["present"][corpus_row]), "outside_schema": [], "n_values": int(self.cache["n_values"][corpus_row])},
                    "category_counts": dict(zip(self.cache["cat_order"], self.cache["cat_counts"][corpus_row].tolist())), "text": text, "segments": fl.serialize({}, list(ncfg.get("category_order", fl.DEFAULT_ORDER)))[1], "from_cache": True}
        else:
            norm = fl.normalise_report(rep, schema, list(ncfg.get("drop_keys", ["success"])), list(ncfg.get("category_order", fl.DEFAULT_ORDER)),
                                       max_items=int(ncfg.get("max_items_per_key", 20)), mask_hash=bool(ncfg.get("mask_sample_hash", True)), max_chars=int(self.cfg.get("data.text_max_chars", 3000)))
        t2 = time.perf_counter()
        tok = self.tokens.count(norm["text"], int(self.cfg.get("encoder.roberta.max_length", 512)))
        # which segments survive the n-token cut
        if tok["truncated"] and tok["cut_chars"] is not None:
            for s in norm["segments"]:
                s["kept"] = "full" if s["start"] + s["chars"] <= tok["cut_chars"] else ("partial" if s["start"] < tok["cut_chars"] else "dropped")
        else:
            for s in norm["segments"]:
                s["kept"] = "full"
        t3 = time.perf_counter()
        # label service
        h = acq.get("hash")
        strings = engine_strings
        if strings is None and h and self.cache and h in self._hash_index():
            i = self._hash_index()[h]
            strings = {e: self.eng_raw[e][i] for e in self.engines if self.eng_raw[e][i]}
        labels = label_report(strings or {}, self.cmap, list(self.cfg.get("labels.engine_priority", self.engines))) if strings else None
        t4 = time.perf_counter()
        result.update({"normalizer": {k: v for k, v in norm.items() if k != "row"}, "row_preview": {k: (v[:160] + "…" if len(v) > 160 else v) for k, v in norm.get("row", {}).items()},
                       "text_display": fl.display_text(norm["text"])[:2200], "tokens": tok, "labels": labels,
                       "schema_version": str(ncfg.get("schema_version", "v1"))})
        result["timings_ms"].update({"normalise": round((t2 - t1) * 1000, 2), "tokenise": round((t3 - t2) * 1000, 2), "label_service": round((t4 - t3) * 1000, 2)})
        if not self.ready():
            result["verdict"] = {"status": "not_ready", "message": "Run the training protocol first (or place the fine-tuned RoBERTa folders in ./models)."}
            return result
        # encoder slots + stacking
        with self.lock:
            slot_out = []
            blocks = []
            t5 = time.perf_counter()
            F_shared = self.feat.transform([norm["text"]]) if any(s.kind == "surrogate" for s in self.slots.values()) else None
            feat_ms = (time.perf_counter() - t5) * 1000
            for name, slot in self.slots.items():
                P = slot.predict_proba_features(F_shared) if slot.kind == "surrogate" else slot.predict_proba_texts([norm["text"]])
                Pc = self._fold_probs(name, P)[0]
                blocks.append(Pc[None])
                top = int(P[0].argmax()) if P.shape[1] else 0
                slot_out.append({"slot": name, "display": slot.display, "kind": slot.kind, "engine": None if name == "category" else self.cfg.display_name(name),
                                 "top_label": slot.classes[top] if slot.classes else None, "top_prob": float(P[0][top]) if P.shape[1] else 0.0,
                                 "top_category": self.categories[int(Pc.argmax())], "category_prob": float(Pc.max()), "ms": round(slot.last_ms, 2),
                                 "category_probs": {c: float(Pc[i]) for i, c in enumerate(self.categories)}, "reply_bytes": int(Pc.size * 4)})
            t6 = time.perf_counter()
            Z = np.concatenate(blocks, axis=1)
            proba = self.meta.predict_proba(Z)[0]
            t7 = time.perf_counter()
        order = np.argsort(-proba)
        top_i = int(order[0])
        os_cfg = self.cfg.get("open_set", {}) or {}
        method = str(os_cfg.get("method", "max_prob"))
        thr = float(self.threshold if self.threshold is not None else 0.5)
        score = float(st.confidence(proba[None], method)[0])
        unknown = score < thr
        short = tok["n_tokens"] < 24
        result["verdict"] = {
            "status": "ok", "category": self.categories[top_i], "confidence": float(proba[top_i]),
            "open_set": {"method": method, "threshold": thr, "score": round(score, 4), "unknown": bool(unknown)},
            "decision": "UNKNOWN CATEGORY — possible zero-day" if unknown else "KNOWN CATEGORY",
            "short_report": short, "top5": [{"category": self.categories[int(i)], "prob": float(proba[int(i)])} for i in order[:5]],
            "slots": slot_out, "meta": {"input_dim": int(Z.shape[1]), "ms": round((t7 - t6) * 1000, 2), "kind": str(self.cfg.get("train.meta_learner", "logreg"))},
            "agrees_with_labels": (labels["consensus"] == self.categories[top_i]) if labels and labels["consensus"] != UNMAPPED else None,
            "versions": {"model": self.model_version, "category_map": self.cmap.version, "schema": str(ncfg.get("schema_version", "v1"))},
        }
        result["timings_ms"].update({"encoder": round((t6 - t5) * 1000, 2), "shared_features": round(feat_ms, 2), "stacking": round((t7 - t6) * 1000, 2), "total": round((t7 - t0) * 1000, 2)})
        result["payload"] = {"report_bytes": len(data), "text_bytes": len(norm["text"].encode("utf-8")), "tokens_sent": min(tok["n_tokens"], tok["max_length"]), "slot_reply_bytes": sum(s["reply_bytes"] for s in slot_out)}
        self._store_verdict(result)
        v = result["verdict"]
        self.log(f"analyse {name}: {'UNKNOWN CATEGORY (possible zero-day)' if unknown else v['category']} at {v['confidence'] * 100:.1f}% "
                 f"({tok['n_tokens']} tokens{', truncated' if tok['truncated'] else ''}) in {result['timings_ms']['total']:.0f} ms", "warn" if unknown else "scan")
        return result

    def _store_verdict(self, res: dict[str, Any]) -> None:
        v = res["verdict"]
        rec = {"t": time.time(), "name": res["file"]["name"], "hash": res["file"].get("hash"), "source": res["file"]["source"], "category": v["category"], "confidence": v["confidence"],
               "unknown": v["open_set"]["unknown"], "n_tokens": res["tokens"]["n_tokens"], "truncated": res["tokens"]["truncated"], "score": (res["labels"] or {}).get("score"),
               "consensus": (res["labels"] or {}).get("consensus"), "raw": (res["labels"] or {}).get("raw"), "versions": v["versions"], "ms": res["timings_ms"]["total"]}
        self.verdicts.appendleft(rec)
        try:
            self.verdict_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.verdict_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------ category hold-out study (Table 4.7 protocol, Section 4.4.4 release gate)
    def run_holdout(self, categories: list[str] | None = None) -> None:
        if self.holdout_running:
            return
        threading.Thread(target=self._holdout_worker, args=(categories,), daemon=True).start()

    def _holdout_worker(self, categories: list[str] | None) -> None:
        self.holdout_running = True
        try:
            self._holdout(categories or list(self.categories))
        except Exception as e:
            import traceback
            self.log(f"hold-out study failed: {e}", "error")
            traceback.print_exc()
        finally:
            self.holdout_running = False
            self.holdout_progress = {}

    def _holdout(self, cats: list[str]) -> None:
        self._wait_features()
        cfgt = self.cfg.get("train", {}) or {}
        seed = int(cfgt.get("seed", 42))
        min_sup = int(cfgt.get("min_family_support", 30))
        os_cfg = self.cfg.get("open_set", {}) or {}
        method = str(os_cfg.get("method", "max_prob"))
        budget = float(os_cfg.get("known_reject_budget", 0.1))
        rows_all = self._selected_rows()
        cap = int(cfgt.get("holdout_rows", 12000))
        if cap and cap < len(rows_all):
            sss = StratifiedShuffleSplit(n_splits=1, train_size=cap, random_state=seed)
            idx, _ = next(sss.split(rows_all, self.y_cat[rows_all]))
            rows_all = np.sort(rows_all[idx])
        sur = self.cfg.get("encoder.surrogate", {}) or {}
        results = []
        t_all = time.perf_counter()
        self.log(f"category hold-out study on {len(rows_all)} reports: each category is removed from training in turn (surrogate heads, shared unsupervised feature space)", "fed")
        for ci, cat in enumerate(cats):
            t_c = time.perf_counter()
            self.holdout_progress = {"category": cat, "i": ci + 1, "n": len(cats), "phase": "fit on the other categories"}
            c = self.cat_idx[cat]
            y = self.y_cat[rows_all]
            held = rows_all[y == c]
            known = rows_all[y != c]
            if len(held) < 5 or len(known) < 50:
                results.append({"category": cat, "n_unseen": int(len(held)), "skipped": True})
                continue
            # base models on 80 % of the known rows; meta on the remaining 20 % (hold-out stacking); tau_4 out-of-fold on that 20 %
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(known))
            cut = int(len(known) * 0.8)
            base_rows, meta_rows = np.sort(known[perm[:cut]]), np.sort(known[perm[cut:]])
            slots: dict[str, SurrogateSlot] = {}
            known_cats = [k for k in self.categories if k != cat]
            for name in self.slots:
                s = SurrogateSlot(name, name, self.feat, str(sur.get("head", "logreg")), float(sur.get("logreg_C", 4.0)), int(sur.get("mlp_hidden", 256)), seed)
                if name == "category":
                    s.set_classes(self.categories)
                    s.fit(base_rows, self.y_cat[base_rows])
                else:
                    classes, lab_rows, y_fam = self._engine_family_targets(name, base_rows, min_sup)
                    # remove families that map to the held-out category (they are "unseen" too)
                    f2c = self._family_to_cat(name)
                    keep_cls = [f for f in classes if f2c.get(f) != cat]
                    if len(keep_cls) < 2:
                        continue
                    remap = {classes.index(f): j for j, f in enumerate(keep_cls)}
                    m = np.array([int(v) in remap for v in y_fam])
                    s.set_classes(keep_cls)
                    s.fit(lab_rows[m], np.array([remap[int(v)] for v in y_fam[m]]))
                slots[name] = s

            def stack(rws: np.ndarray) -> np.ndarray:
                blocks = []
                for name, s in slots.items():
                    P = s.predict_proba(rws)
                    blocks.append(self._fold_probs_with(name, s, P))
                return np.concatenate(blocks, axis=1)

            Zm = stack(meta_rows)
            ym = self.y_cat[meta_rows]
            meta = st.make_meta("logreg", float(cfgt.get("logreg_C", 1.0)), seed)
            P_oof = cross_val_predict(meta, Zm, ym, cv=StratifiedKFold(3, shuffle=True, random_state=seed), method="predict_proba")
            # map meta class order (only known categories appear) to full category space
            meta.fit(Zm, ym)
            thr = st.calibrate_threshold(P_oof, method, os_cfg.get("threshold", "auto"), budget)
            s_known = st.confidence(P_oof, method)
            self.holdout_progress["phase"] = "scoring the unseen category"
            Zh = stack(held)
            Ph = meta.predict_proba(Zh)
            s_unseen = st.confidence(Ph, method)
            nearest = Counter(self.categories[int(meta.classes_[j])] for j in Ph.argmax(1))
            known_f1 = st.macro_f1(ym, meta.classes_[P_oof.argmax(1)])
            bins = np.linspace(0, 1, 21)
            results.append({"category": cat, "n_unseen": int(len(held)), "n_known_val": int(len(meta_rows)), "threshold": thr,
                            "flagged_unknown": float((s_unseen < thr).mean()), "known_false_unknown": float((s_known < thr).mean()),
                            "mean_conf_unseen": float(s_unseen.mean()), "mean_conf_known": float(s_known.mean()), "known_macro_f1": known_f1,
                            "nearest": [{"category": k, "share": v / len(held)} for k, v in nearest.most_common(3)],
                            "hist_known": np.histogram(s_known, bins)[0].tolist(), "hist_unseen": np.histogram(s_unseen, bins)[0].tolist(),
                            "sweep": [{"threshold": float(t), "unseen_flagged": float((s_unseen < t).mean()), "known_accepted": float((s_known >= t).mean()),
                                       "balanced": (float((s_unseen < t).mean()) + float((s_known >= t).mean())) / 2} for t in np.round(np.arange(0.05, 0.96, 0.05), 2)],
                            "ms": round((time.perf_counter() - t_c) * 1000)})
            self.log(f"hold-out {cat}: {results[-1]['flagged_unknown'] * 100:.0f}% of {len(held)} unseen reports flagged unknown at tau_4 = {thr:.3f}; "
                     f"nearest known {results[-1]['nearest'][0]['category']}; known false-unknown {results[-1]['known_false_unknown'] * 100:.1f}% · {results[-1]['ms'] / 1000:.0f} s", "fed")
            self.holdout_result = {"status": "running", "results": results, "method": method, "budget": budget}
        self.holdout_result = {"status": "ok", "results": results, "method": method, "budget": budget, "bins": np.linspace(0, 1, 21).tolist(), "ms": round((time.perf_counter() - t_all) * 1000),
                               "rows": int(len(rows_all)), "encoder": "surrogate heads on the shared feature space"}

    def _fold_probs_with(self, name: str, s: Slot, P: np.ndarray) -> np.ndarray:
        if name == "category":
            return P
        return st.fold_to_categories(P, s.classes, self._family_to_cat(name), self.categories)

    # ------------------------------------------------------------------ intake watcher
    def set_watch(self, enabled: bool) -> None:
        self.watch_enabled = enabled
        d = self.cfg.resolve(self.cfg.get("watch.dir", "watch_inbox"))
        d.mkdir(parents=True, exist_ok=True)
        if enabled:
            self._watch_seen = {p.name for p in d.iterdir() if p.is_file()}
            if self._watch_thread is None or not self._watch_thread.is_alive():
                self._watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
                self._watch_thread.start()
            self.log(f"intake watch enabled on {d}", "sys")
        else:
            self.log("intake watch disabled", "sys")

    def _watch_loop(self) -> None:
        d = self.cfg.resolve(self.cfg.get("watch.dir", "watch_inbox"))
        exts = tuple(e.lower() for e in (self.cfg.get("watch.extensions") or []))
        poll = float(self.cfg.get("watch.poll_seconds", 1.0))
        while self.watch_enabled:
            try:
                for p in sorted(d.iterdir()):
                    if not p.is_file() or p.name in self._watch_seen or (exts and p.suffix.lower() not in exts):
                        continue
                    s1 = p.stat().st_size
                    time.sleep(0.3)
                    if p.stat().st_size != s1:
                        continue
                    self._watch_seen.add(p.name)
                    self.analyze(p.read_bytes(), p.name, source="watch")
            except Exception as e:
                self.log(f"watch error: {e}", "error")
            time.sleep(poll)

    # ------------------------------------------------------------------ samples
    def samples(self) -> dict[str, Any]:
        idx = self.cfg.samples_dir / "index.json"
        if not idx.exists():
            return {"samples": []}
        items = json.loads(idx.read_text(encoding="utf-8"))
        for it in items:
            it["engines_display"] = {self.cfg.display_name(k): v for k, v in it.get("engines", {}).items()}
        return {"samples": items}

    def analyze_sample(self, rel: str) -> dict[str, Any]:
        p = (self.cfg.samples_dir / rel).resolve()
        if not str(p).startswith(str(self.cfg.samples_dir.resolve())) or not p.exists():
            raise FileNotFoundError(rel)
        idx = json.loads((self.cfg.samples_dir / "index.json").read_text(encoding="utf-8"))
        strings = next((it.get("engines") for it in idx if it["path"].replace("\\", "/") == rel.replace("\\", "/")), None)
        return self.analyze(p.read_bytes(), p.name, source="sample", engine_strings=strings)

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
            import torch
            versions["torch"] = torch.__version__
            versions["cuda"] = torch.cuda.is_available()
        except Exception:
            versions["torch"] = None
        try:
            import transformers
            versions["transformers"] = transformers.__version__
        except Exception:
            versions["transformers"] = None
        modes = {k: v.kind for k, v in self.slots.items()}
        return {
            "mode": "roberta" if all(m == "roberta" for m in modes.values()) else ("mixed" if any(m == "roberta" for m in modes.values()) else "surrogate"),
            "ready": self.ready(), "training": self.training, "progress": self.progress, "holdout_running": self.holdout_running, "holdout_progress": self.holdout_progress,
            "features_ready": bool(self.feat and self.feat.ready) or all(m == "roberta" for m in modes.values()),
            "slots": [{"slot": k, "display": v.display, "kind": v.kind, "engine": None if k == "category" else self.cfg.display_name(k), "n_classes": len(v.classes), "trained": v.trained, "info": v.info()} for k, v in self.slots.items()],
            "categories": self.categories, "metrics_summary": {k: self.metrics.get(k) for k in ("n", "folds", "threshold", "stack_dim", "meta_params")} | ({"stacking": self.metrics["stacking"]["summary"]} if self.metrics else {}),
            "threshold": self.threshold, "model_version": self.model_version, "map_version": self.cmap.version, "schema_version": self.cache["schema_version"] if self.cache else None,
            "label_stats": self.label_stats, "tokenizer": self.tokens.info(), "surrogate_features": self.feat.info() if self.feat and self.feat.ready else None,
            "cost_table": analytical_cost_table(int(self.cfg.get("encoder.roberta.max_length", 512))),
            "watch_enabled": self.watch_enabled, "resources": self.resources(), "versions": versions, "config": self.cfg.summary(),
            "recent": list(self.verdicts)[:15], "cache": {"hashes": len(self.cache["hashes"]), "built": self.cache.get("built")} if self.cache else None,
        }
