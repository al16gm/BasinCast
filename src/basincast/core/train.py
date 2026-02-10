from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error

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
            n_estimators=400,
            min_samples_leaf=2,
            random_state=cfg.random_state,
            n_jobs=-1,
        ),
        "gbr": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            random_state=cfg.random_state,
        ),
        "bayes_ridge": BayesianRidge(),
    }
    if _HAS_XGB:
        models["xgb"] = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
    return models


def train_and_select(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: TrainConfig = TrainConfig(),
) -> tuple[object, Dict[str, float | str]]:
    """
    Train all models and select best by KGE (validation).
    Returns: (best_model, metrics_dict)
    """
    zoo = model_zoo(cfg)
    results = []

    for name, model in zoo.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_val) if len(X_val) > 0 else np.array([])
            met = {
                "model_type": name,
                "kge_val": float(kge(y_val.to_numpy(), pred)) if len(X_val) > 0 else float("nan"),
                "rmse_val": float(rmse(y_val.to_numpy(), pred)) if len(X_val) > 0 else float("nan"),
            }
            results.append((model, met))
        except Exception as e:
            # Skip failing models in MVP
            continue

    if len(results) == 0:
        raise RuntimeError("No model could be trained (check input data).")

    # Prefer max KGE; if KGE is NaN for all, fallback to min RMSE
    kges = [r[1]["kge_val"] for r in results]
    if all(np.isnan(kges)):
        best = min(results, key=lambda x: x[1]["rmse_val"])
    else:
        best = max(results, key=lambda x: (-1e9 if np.isnan(x[1]["kge_val"]) else x[1]["kge_val"]))

    return best[0], best[1]