from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from basincast.core.metrics import kge, rmse
from basincast.core.features import FeatureConfig
from basincast.core.forecast import ForecastConfig, forecast_recursive_endo
from basincast.core.quality import seasonal_baseline_value, to_month_start


@dataclass(frozen=True)
class BacktestConfig:
    test_months: int = 96
    min_pairs_per_horizon: int = 24
    horizons: Tuple[int, ...] = (12, 24, 36, 48)


def backtest_holdout_by_horizon(
    model: object,
    full_history: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    point_id: str,
    resource_type: str,
    unit: str,
    bt_cfg: BacktestConfig,
    feat_cfg: FeatureConfig = FeatureConfig(),
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Paper-2 holdout protocol (rolling origins within the holdout):
      - for each origin in holdout where origin+h exists -> predict y(origin+h)
    Produces skill for ENDO_ML and BASELINE_SEASONAL.
    """
    df = full_history.copy()
    df["date"] = to_month_start(df["date"])
    df = df.dropna(subset=["date", "value"]).sort_values("date")

    hold = df[df["date"] > cutoff_date].copy()
    if hold.empty:
        return pd.DataFrame(), {"status": "SKIP_EMPTY_HOLDOUT", "reliability_horizon": 0}

    horizons = list(bt_cfg.horizons)

    true_map = dict(zip(df["date"].tolist(), df["value"].astype(float).tolist()))
    rows = []

    for h in horizons:
        preds_ml: List[float] = []
        preds_bl: List[float] = []
        trues: List[float] = []

        for origin in hold["date"].tolist():
            target_date = (pd.Timestamp(origin) + pd.DateOffset(months=int(h))).to_period("M").to_timestamp()
            if target_date not in true_map:
                continue

            hist_until = df[df["date"] <= origin][["date", "value"]].copy()
            if len(hist_until) < 24:
                continue

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
            y_bl = float(seasonal_baseline_value(hist_until, pd.Timestamp(origin), int(h)))

            preds_ml.append(y_pred)
            preds_bl.append(y_bl)
            trues.append(y_true)

        n_pairs = int(len(trues))
        if n_pairs == 0:
            kge_ml = rmse_ml = kge_bl = rmse_bl = float("nan")
        else:
            tr = np.asarray(trues, dtype=float)
            pm = np.asarray(preds_ml, dtype=float)
            pb = np.asarray(preds_bl, dtype=float)

            kge_ml = float(kge(tr, pm))
            rmse_ml = float(rmse(tr, pm))
            kge_bl = float(kge(tr, pb))
            rmse_bl = float(rmse(tr, pb))

        rows.append(
            {"point_id": point_id, "resource_type": resource_type, "unit": unit,
             "family": "ENDO_ML", "horizon": int(h), "kge": kge_ml, "rmse": rmse_ml, "n_pairs": n_pairs}
        )
        rows.append(
            {"point_id": point_id, "resource_type": resource_type, "unit": unit,
             "family": "BASELINE_SEASONAL", "horizon": int(h), "kge": kge_bl, "rmse": rmse_bl, "n_pairs": n_pairs}
        )

    skill_df = pd.DataFrame(rows)
    summary = {"status": "OK"}
    return skill_df, summary