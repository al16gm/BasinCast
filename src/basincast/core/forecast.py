from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd

from basincast.core.features import FeatureConfig, get_feature_columns


@dataclass(frozen=True)
class ForecastConfig:
    horizons: List[int]
    non_negative: bool = True


def forecast_recursive_endo(
    model: object,
    history: pd.DataFrame,
    start_date: pd.Timestamp,
    fc_cfg: ForecastConfig,
    feat_cfg: FeatureConfig = FeatureConfig(),
) -> pd.DataFrame:
    """
    Recursive multi-step forecast in delta space.

    history must contain at least: date, value
    start_date is last observed month-start date.

    Output columns:
      date, horizon, y_forecast, delta_y_pred
    """
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.sort_values("date")

    max_h = int(max(fc_cfg.horizons))
    feat_cols = get_feature_columns(feat_cfg)

    # Build buffer with last 12 observed levels (for lag12)
    last_values = history["value"].dropna().to_numpy(dtype=float)
    if len(last_values) == 0:
        raise ValueError("Empty history values.")

    y_prev = float(last_values[-1])

    # If not enough for lag12, pad using earliest value
    if len(last_values) >= 12:
        lag12_buffer = deque(last_values[-12:].tolist(), maxlen=12)
    else:
        pad = [float(last_values[0])] * (12 - len(last_values))
        lag12_buffer = deque(pad + last_values.tolist(), maxlen=12)

    delta_prev = 0.0  # first step delta lag1 (policy: repeat_first / 0.0)

    results: List[Dict[str, object]] = []
    month = int(pd.Timestamp(start_date).month)

    for h in range(1, max_h + 1):
        # Features for step h
        value_lag1 = y_prev
        value_lag12 = float(lag12_buffer[0])

        month_sin = np.sin(2 * np.pi * month / 12.0)
        month_cos = np.cos(2 * np.pi * month / 12.0)

        row = {
            "value_lag1": value_lag1,
            "value_lag12": value_lag12,
        }
        if feat_cfg.add_delta_lag1:
            row["delta_y_lag1"] = delta_prev
        if feat_cfg.add_seasonality:
            row["month_sin"] = month_sin
            row["month_cos"] = month_cos

        X = pd.DataFrame([row], columns=feat_cols)
        delta_pred = float(model.predict(X)[0])

        y_fc = y_prev + delta_pred
        if fc_cfg.non_negative:
            y_fc = max(0.0, y_fc)

        fc_date = (pd.Timestamp(start_date) + pd.DateOffset(months=h)).to_period("M").to_timestamp()

        if h in fc_cfg.horizons:
            results.append(
                {
                    "date": fc_date,
                    "horizon": h,
                    "y_forecast": y_fc,
                    "delta_y_pred": delta_pred,
                }
            )

        # Update buffers
        lag12_buffer.append(y_fc)
        delta_prev = delta_pred
        y_prev = y_fc
        month = (month % 12) + 1

    return pd.DataFrame(results)