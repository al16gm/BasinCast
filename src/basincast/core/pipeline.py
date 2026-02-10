from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from basincast.core.features import FeatureConfig, build_endo_features, get_feature_columns, split_train_val_by_months
from basincast.core.train import TrainConfig, train_and_select
from basincast.core.forecast import ForecastConfig, forecast_recursive_endo


@dataclass(frozen=True)
class PipelineConfig:
    horizons: List[int] = (1, 3, 6, 12, 24, 36, 48)
    val_months: int = 36
    min_history_months: int = 120
    random_state: int = 42


def run_pipeline(
    canonical_csv: str | Path,
    outdir: str | Path = "outputs",
    cfg: PipelineConfig = PipelineConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run ENDO-only forecasting pipeline for all points in canonical_timeseries.csv.

    Returns:
      forecasts_df, metrics_df
    """
    canonical_csv = Path(canonical_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(canonical_csv)
    required = {"date", "point_id", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in canonical CSV: {sorted(list(missing))}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=["date", "point_id", "value"]).copy()
    df["point_id"] = df["point_id"].astype(str).map(str.strip)

    # Optional metadata columns
    if "resource_type" not in df.columns:
        df["resource_type"] = ""
    if "unit" not in df.columns:
        df["unit"] = ""

    feat_cfg = FeatureConfig()
    feat_cols = get_feature_columns(feat_cfg)
    train_cfg = TrainConfig(random_state=cfg.random_state)
    fc_cfg = ForecastConfig(horizons=list(cfg.horizons), non_negative=True)

    all_forecasts = []
    all_metrics = []

    for pid, g in df.groupby("point_id"):
        g = g.sort_values("date").copy()
        resource_type = str(g["resource_type"].dropna().iloc[0]) if g["resource_type"].dropna().shape[0] > 0 else ""
        unit = str(g["unit"].dropna().iloc[0]) if g["unit"].dropna().shape[0] > 0 else ""

        # Minimum length check
        n_months = g["date"].nunique()
        if n_months < cfg.min_history_months:
            all_metrics.append(
                {
                    "point_id": pid,
                    "resource_type": resource_type,
                    "unit": unit,
                    "status": f"SKIP (history {n_months} < {cfg.min_history_months})",
                }
            )
            continue

        df_feat = build_endo_features(g[["date", "value"]], feat_cfg)

        # Drop rows with NaNs in features/target due to lagging
        df_feat = df_feat.dropna(subset=feat_cols + ["delta_y"]).copy()
        if len(df_feat) < 60:
            all_metrics.append(
                {
                    "point_id": pid,
                    "resource_type": resource_type,
                    "unit": unit,
                    "status": "SKIP (too few feature rows after lagging)",
                }
            )
            continue

        train_df, val_df = split_train_val_by_months(df_feat, val_months=cfg.val_months)

        X_train = train_df[feat_cols]
        y_train = train_df["delta_y"]
        X_val = val_df[feat_cols]
        y_val = val_df["delta_y"]

        model, met = train_and_select(X_train, y_train, X_val, y_val, cfg=train_cfg)

        # Forecast from last observed date
        last_date = pd.to_datetime(g["date"].max())
        fc = forecast_recursive_endo(
            model=model,
            history=g[["date", "value"]],
            start_date=last_date,
            fc_cfg=fc_cfg,
            feat_cfg=feat_cfg,
        )
        fc["point_id"] = pid
        fc["resource_type"] = resource_type
        fc["unit"] = unit
        fc["family"] = "ENDO"
        fc["model_type"] = met["model_type"]
        all_forecasts.append(fc)

        all_metrics.append(
            {
                "point_id": pid,
                "resource_type": resource_type,
                "unit": unit,
                "status": "OK",
                "model_type": met["model_type"],
                "kge_val": met["kge_val"],
                "rmse_val": met["rmse_val"],
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "train_start": str(pd.to_datetime(train_df["date"].min()).date()),
                "train_end": str(pd.to_datetime(train_df["date"].max()).date()),
                "val_start": str(pd.to_datetime(val_df["date"].min()).date()) if len(val_df) else "",
                "val_end": str(pd.to_datetime(val_df["date"].max()).date()) if len(val_df) else "",
                "last_observed_date": str(pd.to_datetime(last_date).date()),
            }
        )

    forecasts_df = pd.concat(all_forecasts, ignore_index=True) if len(all_forecasts) else pd.DataFrame()
    metrics_df = pd.DataFrame(all_metrics)

    forecasts_path = outdir / "forecasts_v0_5.csv"
    metrics_path = outdir / "metrics_v0_5.csv"

    forecasts_df.to_csv(forecasts_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    return forecasts_df, metrics_df