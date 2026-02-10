from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return float("nan")
    yt = y_true[m]
    yp = y_pred[m]
    denom = np.sum((yt - np.mean(yt)) ** 2)
    if denom == 0:
        return float("nan")
    return float(1 - np.sum((yt - yp) ** 2) / denom)


def kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Kling-Gupta Efficiency (KGE).
    Robust to constant series (returns NaN if undefined).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return float("nan")

    yt = y_true[m]
    yp = y_pred[m]

    if yt.size < 2:
        return float("nan")

    std_t = np.std(yt)
    std_p = np.std(yp)
    if std_t == 0 or std_p == 0:
        return float("nan")

    r = float(np.corrcoef(yt, yp)[0, 1])
    alpha = float(std_p / std_t)
    mean_t = float(np.mean(yt))
    mean_p = float(np.mean(yp))
    if mean_t == 0:
        return float("nan")
    beta = float(mean_p / mean_t)

    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))