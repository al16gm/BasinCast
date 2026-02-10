from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from basincast.core.metrics import kge, rmse
from basincast.core.features import FeatureConfig, get_feature_columns
from basincast.core.forecast import ForecastConfig, forecast_recursive_endo


@dataclass(frozen=True)
class BacktestConfig:
    test_months: int = 96  # default: 8 years
    kge_threshold: float = 0.6
    min_pairs_per_horizon: int = 24  # minimum evaluation samples per horizon
    horizons: Tuple[int, ...] = (12, 24, 36, 48)


def _seasonal_baseline_pred(
    history: pd.DataFrame, start_date: pd.Timestamp, horizon: int
) -> float:
    """
    Seasonal persistence baseline:
      y(t+h) = y(t + h - 12) if h<=12 (same month previous year)
      for h>12: recursively repeat 12-month cycle
    """
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    hist_map = dict(zip(history["date"].tolist(), history["value"].astype(float).tolist()))

    def rec(h: int) -> float:
        if h <= 12:
            d = (pd.Timestamp(start_date) + pd.DateOffset(months=h - 12)).to_period("M").to_timestamp()
            if d in hist_map and np.isfinite(hist_map[d]):
                return float(hist_map[d])
            # fallback: last observed
            return float(hist_map.get(pd.Timestamp(start_date), list(hist_map.values())[-1]))
        return rec(h - 12)

    return rec(int(horizon))


def backtest_holdout_by_horizon(
    model: object,
    full_history: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    point_id: str,
    resource_type: str,
    unit: str,
    bt_cfg: BacktestConfig,
    feat_cfg: FeatureConfig = FeatureConfig(),
) -> Tuple[pd.DataFrame, Dict[str, float | int | str]]:
    """
    Paper-2 style protocol:
      - Train on data <= cutoff_date (done outside)
      - Evaluate on holdout (data > cutoff_date) using rolling origins:
          for each origin in holdout where origin+h exists -> predict y(origin+h)
    Returns:
      skill_df rows: family in {"ENDO_ML","BASELINE_SEASONAL"} per horizon
      summary dict including reliability_horizon
    """
    df = full_history.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=["date", "value"]).sort_values("date")

    # Holdout dates
    hold = df[df["date"] > cutoff_date].copy()
    if hold.empty:
        skill_df = pd.DataFrame()
        summary = {
            "point_id": point_id,
            "status": "SKIP (empty holdout)",
            "reliability_horizon": 0,
        }
        return skill_df, summary

    horizons = list(bt_cfg.horizons)
    max_h = max(horizons)

    # map for true values lookup
    true_map = dict(zip(df["date"].tolist(), df["value"].astype(float).tolist()))

    rows = []
    # evaluate each horizon
    for h in horizons:
        preds_ml: List[float] = []
        preds_bl: List[float] = []
        trues: List[float] = []

        # rolling origins inside holdout
        for origin in hold["date"].tolist():
            target_date = (pd.Timestamp(origin) + pd.DateOffset(months=int(h))).to_period("M").to_timestamp()
            if target_date not in true_map:
                continue  # cannot score
            # ensure we have history available up to origin (operational setting)
            hist_until = df[df["date"] <= origin][["date", "value"]].copy()
            if len(hist_until) < 24:
                continue

            # ML forecast for this horizon only
            fc_cfg = ForecastConfig(horizons=[int(h)], non_negative=True)
            fc = forecast_recursive_endo(
                model=model,
                history=hist_until,
                start_date=pd.Timestamp(origin),
                fc_cfg=fc_cfg,
                feat_cfg=feat_cfg,
            )
            if fc.empty:
                continue
            y_pred = float(fc["y_forecast"].iloc[-1])
            y_true = float(true_map[target_date])

            # baseline
            y_bl = float(_seasonal_baseline_pred(hist_until, pd.Timestamp(origin), int(h)))

            preds_ml.append(y_pred)
            preds_bl.append(y_bl)
            trues.append(y_true)

        n_pairs = int(len(trues))
        if n_pairs == 0:
            kge_ml = float("nan")
            rmse_ml = float("nan")
            kge_bl = float("nan")
            rmse_bl = float("nan")
        else:
            tr = np.asarray(trues, dtype=float)
            pm = np.asarray(preds_ml, dtype=float)
            pb = np.asarray(preds_bl, dtype=float)

            kge_ml = float(kge(tr, pm))
            rmse_ml = float(rmse(tr, pm))
            kge_bl = float(kge(tr, pb))
            rmse_bl = float(rmse(tr, pb))

        # rows ML + baseline
        rows.append(
            {
                "point_id": point_id,
                "resource_type": resource_type,
                "unit": unit,
                "family": "ENDO_ML",
                "horizon": int(h),
                "kge": kge_ml,
                "rmse": rmse_ml,
                "n_pairs": n_pairs,
            }
        )
        rows.append(
            {
                "point_id": point_id,
                "resource_type": resource_type,
                "unit": unit,
                "family": "BASELINE_SEASONAL",
                "horizon": int(h),
                "kge": kge_bl,
                "rmse": rmse_bl,
                "n_pairs": n_pairs,
            }
        )

    skill_df = pd.DataFrame(rows)

    # reliability horizon from ML only
    ml = skill_df[skill_df["family"] == "ENDO_ML"].copy()
    rel = 0
    for h in sorted(horizons):
        row = ml[ml["horizon"] == h]
        if row.empty:
            continue
        n_pairs = int(row["n_pairs"].iloc[0])
        kgeh = float(row["kge"].iloc[0]) if np.isfinite(row["kge"].iloc[0]) else float("nan")
        if n_pairs >= bt_cfg.min_pairs_per_horizon and np.isfinite(kgeh) and kgeh >= bt_cfg.kge_threshold:
            rel = int(h)

    summary = {
        "point_id": point_id,
        "status": "OK",
        "reliability_horizon": int(rel),
        "kge_threshold": float(bt_cfg.kge_threshold),
        "test_months": int(bt_cfg.test_months),
        "min_pairs_per_horizon": int(bt_cfg.min_pairs_per_horizon),
    }
    return skill_df, summary