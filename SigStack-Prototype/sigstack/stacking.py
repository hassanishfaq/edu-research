"""Stacking service -- Section 4.3 (Tables 4.9-4.12) and the unknown-category rule of Eq. (4.11).

  stack input   [ p_engine1(c) | p_engine2(c) | p_engine3(c) | p_direct(c) ]   (per-engine family models
                folded to categories through the map, plus the direct category model)
  meta-learner  logistic regression (Table 4.11)
  Stacking Precision / F1 (Table 4.10):  sum_e P_e(c) * S_e(c) / sum_e S_e(c)   support-weighted over engines
  tau_4 (Eq. 4.11):  Q_beta { max_c p(c|x) : x in known validation }   calibrated out-of-fold
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support


def fold_to_categories(p_family: np.ndarray, family_classes: list[str], family_to_cat: dict[str, str], categories: list[str]) -> np.ndarray:
    """Sum the family probabilities of an engine model into the standardised categories."""
    out = np.zeros((p_family.shape[0], len(categories)), dtype=np.float32)
    cidx = {c: i for i, c in enumerate(categories)}
    for j, fam in enumerate(family_classes):
        c = family_to_cat.get(fam)
        if c in cidx:
            out[:, cidx[c]] += p_family[:, j]
    return out


def make_meta(kind: str = "logreg", C: float = 1.0, seed: int = 42):
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    return LogisticRegression(max_iter=2000, C=C)


def meta_size(m) -> int:
    if hasattr(m, "coef_"):
        return int(m.coef_.size + m.intercept_.size)
    if hasattr(m, "estimators_"):
        return int(sum(t.tree_.node_count for t in m.estimators_))
    return 0


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict[str, dict[str, float]]:
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i, c in enumerate(classes)}


def summary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float((y_true == y_pred).mean()), "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f), "n": int(len(y_true))}


def support_weighted_stacking(engine_metrics: dict[str, dict[str, dict[str, float]]], categories: list[str]) -> dict[str, dict[str, float]]:
    """Table 4.10 'Stacking Precision' and 'Stacking F1': support-weighted average over the engines."""
    out = {}
    for c in categories:
        num_p = num_f = den = 0.0
        for eng, m in engine_metrics.items():
            mc = m.get(c)
            if not mc or mc["support"] == 0:
                continue
            num_p += mc["precision"] * mc["support"]
            num_f += mc["f1"] * mc["support"]
            den += mc["support"]
        out[c] = {"precision": num_p / den if den else 0.0, "f1": num_f / den if den else 0.0, "support": int(den)}
    return out


def confidence(P: np.ndarray, method: str = "max_prob") -> np.ndarray:
    if method == "entropy":
        Pc = np.clip(P, 1e-12, 1)
        return 1.0 - (-(Pc * np.log(Pc)).sum(1) / np.log(P.shape[1]))
    return P.max(1)


def calibrate_threshold(P_oof_known: np.ndarray, method: str, raw_threshold: Any, budget: float) -> float:
    """Eq. (4.11): the beta-quantile of the out-of-fold confidences of KNOWN validation samples."""
    if isinstance(raw_threshold, (int, float)):
        return float(raw_threshold)
    return float(np.percentile(confidence(P_oof_known, method), budget * 100))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
