"""LinearSVC construction, temperature-scaled pseudo-probabilities and metrics."""

from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss, recall_score
from sklearn.svm import LinearSVC

from ntl.labels import LABELS, N_CLASSES, family


def make_svc(cfg: dict) -> LinearSVC:
    m = cfg["model"]
    return LinearSVC(
        C=float(m["C"]),
        class_weight=m.get("class_weight", "balanced"),
        max_iter=int(m.get("max_iter", 5000)),
        random_state=int(cfg.get("seed", 42)),
    )


def scores_in_label_order(clf: LinearSVC, scores: np.ndarray) -> np.ndarray:
    """decision_function columns follow clf.classes_; reorder into labels.py order (ids 0..17)."""
    out = np.full((scores.shape[0], N_CLASSES), -np.inf if scores.size else 0.0, dtype=np.float32)
    out[:, np.asarray(clf.classes_, dtype=int)] = scores
    return out


def pseudo_proba(scores: np.ndarray, T: float) -> np.ndarray:
    return softmax(np.asarray(scores, dtype=np.float64) / float(T), axis=1)


def select_temperature(scores: np.ndarray, y: np.ndarray, grid) -> tuple[float, dict[float, float]]:
    """Pick T minimising OOF log-loss of softmax(scores / T)."""
    losses = {float(T): float(log_loss(y, pseudo_proba(scores, T), labels=list(range(N_CLASSES)))) for T in grid}
    best_T = min(losses, key=losses.get)
    return best_T, losses


def per_class_recall(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rec = recall_score(y, pred, labels=list(range(N_CLASSES)), average=None, zero_division=0)
    return {LABELS[i]: float(r) for i, r in enumerate(rec)}


def worst_confusions(y: np.ndarray, pred: np.ndarray, top: int = 20) -> list[dict]:
    """Largest off-diagonal (true, pred) counts; `same_family` flags sibling confusions."""
    cm = confusion_matrix(y, pred, labels=list(range(N_CLASSES)))
    np.fill_diagonal(cm, 0)
    flat = np.argsort(cm, axis=None)[::-1][:top]
    rows = []
    for idx in flat:
        t, p = np.unravel_index(idx, cm.shape)
        if cm[t, p] == 0:
            break
        rows.append({
            "true": LABELS[t],
            "pred": LABELS[p],
            "count": int(cm[t, p]),
            "same_family": family(LABELS[t]) == family(LABELS[p]),
        })
    return rows


def balanced_acc(y, pred) -> float:
    return float(balanced_accuracy_score(y, pred))
