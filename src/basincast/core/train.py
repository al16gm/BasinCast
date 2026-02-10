from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge

try:
    from xgboost import XGBRegressor

    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

from basincast.core.metrics import kge, rmse


@dataclass(frozen=True)
class TrainConfig:
    random_state: int = 42


def model_zoo(cfg: TrainConfig) -> Dict[str, object]:
    models: Dict[str, object] = {
        "rf": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=cfg.random_state,
            n_jobs=-1,
        ),
        "gbr": GradientBoostingRegressor(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=4,
            random_state=cfg.random_state,
        ),
        "bayes_ridge": BayesianRidge(),
    }
    if _HAS_XGB:
        models["xgb"] = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
    return models


def make_model(model_type: str, cfg: TrainConfig) -> object:
    zoo = model_zoo(cfg)
    if model_type not in zoo:
        raise ValueError(f"Unknown model_type: {model_type}")
    return zoo[model_type]


def train_and_select(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: TrainConfig = TrainConfig(),
) -> tuple[str, Dict[str, float | str]]:
    """
    Train all candidate models; select best by validation KGE on delta_y.
    Returns best model_type and its validation metrics.
    """
    zoo = model_zoo(cfg)
    results = []

    for name, model in zoo.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_val) if len(X_val) else np.array([])
            met = {
                "model_type": name,
                "kge_val": float(kge(y_val.to_numpy(), pred)) if len(X_val) else float("nan"),
                "rmse_val": float(rmse(y_val.to_numpy(), pred)) if len(X_val) else float("nan"),
            }
            results.append(met)
        except Exception:
            continue

    if not results:
        raise RuntimeError("No model could be trained.")

    kges = [r["kge_val"] for r in results]
    if all(np.isnan(kges)):
        best = min(results, key=lambda x: x["rmse_val"])
    else:
        best = max(results, key=lambda x: (-1e9 if np.isnan(x["kge_val"]) else x["kge_val"]))

    return str(best["model_type"]), best