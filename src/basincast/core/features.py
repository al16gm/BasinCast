from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    """
    ENDO feature family for monthly level series.
    We model delta_y (monthly increment) to stabilize learning.
    """
    use_delta: bool = True
    lags: Tuple[int, int] = (1, 12)  # lag1 and lag12
    add_delta_lag1: bool = True
    add_seasonality: bool = True


def ensure_month_start(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[date_col] = out[date_col].dt.to_period("M").dt.to_timestamp()
    return out


def build_endo_features(
    g: pd.DataFrame,
    cfg: FeatureConfig = FeatureConfig(),
) -> pd.DataFrame:
    """
    Build ENDO features for a single point_id.

    Input g columns must include: date, value
    Output includes target 'delta_y' and feature columns.
    """
    df = g.copy()
    df = ensure_month_start(df, "date")
    df = df.sort_values("date")

    # Target: delta_y
    df["delta_y"] = df["value"].diff()

    # Lags in level space
    lag1, lag12 = cfg.lags
    df["value_lag1"] = df["value"].shift(lag1)
    df["value_lag12"] = df["value"].shift(lag12)

    if cfg.add_delta_lag1:
        df["delta_y_lag1"] = df["delta_y"].shift(1)

    if cfg.add_seasonality:
        m = df["date"].dt.month.astype(int)
        df["month_sin"] = np.sin(2 * np.pi * m / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * m / 12.0)

    # Keep only needed columns (but don't drop metadata columns here)
    return df


def get_feature_columns(cfg: FeatureConfig = FeatureConfig()) -> List[str]:
    cols = ["value_lag1", "value_lag12"]
    if cfg.add_delta_lag1:
        cols.append("delta_y_lag1")
    if cfg.add_seasonality:
        cols += ["month_sin", "month_cos"]
    return cols


def split_train_val_by_months(
    df_feat: pd.DataFrame,
    val_months: int = 36,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-safe split: last `val_months` rows (by date) are validation.
    """
    df_feat = df_feat.sort_values("date").reset_index(drop=True)
    if len(df_feat) <= val_months + 24:
        # keep at least some training; fallback
        val_months = max(12, min(val_months, max(0, len(df_feat) // 4)))

    if val_months <= 0:
        return df_feat, df_feat.iloc[0:0].copy()

    cut = len(df_feat) - val_months
    train = df_feat.iloc[:cut].copy()
    val = df_feat.iloc[cut:].copy()
    return train, val