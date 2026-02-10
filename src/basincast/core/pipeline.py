from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from basincast.core.features import FeatureConfig, build_endo_features, get_feature_columns, split_train_val_by_months
from basincast.core.train import TrainConfig, make_model, train_and_select
from basincast.core.forecast import ForecastConfig, forecast_recursive_endo
from basincast.core.backtest import BacktestConfig, backtest_holdout_by_horizon


@dataclass(frozen=True)
class PipelineConfig:
    # Operational horizons (future output)
    horizons: Tuple[int, ...] = (12, 24, 36, 48)

    # Internal protocol (paper-2 holdout)
    test_months: int = 96
    kge_threshold: float = 0.6
    min_pairs_per_horizon: int = 24

    # Inner validation for model selection (inside training only)
    val_months: int = 36

    # Input requirements
    min_history_months: int = 120

    random_state: int = 42


def _month_start(ts: pd.Series) -> pd.Series:
    d = pd.to_datetime(ts, errors="coerce")
    return d.dt.to_period("M").dt.to_timestamp()


def run_pipeline(
    canonical_csv: str | Path,
    outdir: str | Path = "outputs",
    cfg: PipelineConfig = PipelineConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Core v0.5.1:
      - Paper-2 style holdout evaluation on last cfg.test_months months (internal)
      - Reliability horizon: max horizon with KGE >= cfg.kge_threshold (and enough pairs)
      - Final model fit on full history
      - Operational forecast from last observed date for cfg.horizons
    Outputs:
      forecasts_v0_5_1.csv, metrics_v0_5_1.csv, skill_by_horizon_v0_5_1.csv
    """
    canonical_csv = Path(canonical_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(canonical_csv)
    required = {"date", "point_id", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in canonical CSV: {sorted(list(missing))}")

    df["date"] = _month_start(df["date"])
    df = df.dropna(subset=["date", "point_id", "value"]).copy()
    df["point_id"] = df["point_id"].astype(str).map(str.strip)

    if "resource_type" not in df.columns:
        df["resource_type"] = ""
    if "unit" not in df.columns:
        df["unit"] = ""

    feat_cfg = FeatureConfig()
    feat_cols = get_feature_columns(feat_cfg)
    train_cfg = TrainConfig(random_state=cfg.random_state)
    fc_cfg = ForecastConfig(horizons=list(cfg.horizons), non_negative=True)

    bt_cfg = BacktestConfig(
        test_months=cfg.test_months,
        kge_threshold=cfg.kge_threshold,
        min_pairs_per_horizon=cfg.min_pairs_per_horizon,
        horizons=tuple(cfg.horizons),
    )

    all_forecasts = []
    all_metrics = []
    all_skill = []

    for pid, g in df.groupby("point_id"):
        g = g.sort_values("date").copy()
        resource_type = str(g["resource_type"].dropna().iloc[0]) if g["resource_type"].dropna().shape[0] > 0 else ""
        unit = str(g["unit"].dropna().iloc[0]) if g["unit"].dropna().shape[0] > 0 else ""

        n_months = int(g["date"].nunique())
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

        max_date = pd.to_datetime(g["date"].max())
        cutoff = (max_date - pd.DateOffset(months=int(cfg.test_months))).to_period("M").to_timestamp()

        train_hist = g[g["date"] <= cutoff][["date", "value"]].copy()
        hold_hist = g[g["date"] > cutoff][["date", "value"]].copy()

        if len(hold_hist) < max(cfg.horizons) + 12:
            all_metrics.append(
                {
                    "point_id": pid,
                    "resource_type": resource_type,
                    "unit": unit,
                    "status": "SKIP (holdout too short for horizons)",
                }
            )
            continue

        # --------- 1) Model selection inside TRAIN only ---------
        df_feat_train = build_endo_features(train_hist, feat_cfg)
        df_feat_train = df_feat_train.dropna(subset=feat_cols + ["delta_y"]).copy()
        if len(df_feat_train) < 60:
            all_metrics.append(
                {
                    "point_id": pid,
                    "resource_type": resource_type,
                    "unit": unit,
                    "status": "SKIP (too few train feature rows after lagging)",
                }
            )
            continue

        inner_tr, inner_val = split_train_val_by_months(df_feat_train, val_months=cfg.val_months)
        X_tr = inner_tr[feat_cols]
        y_tr = inner_tr["delta_y"]
        X_val = inner_val[feat_cols]
        y_val = inner_val["delta_y"]

        best_type, sel_met = train_and_select(X_tr, y_tr, X_val, y_val, cfg=train_cfg)

        # Refit chosen model on ALL train_feat (still no holdout leakage)
        model_bt = make_model(best_type, train_cfg)
        X_train_all = df_feat_train[feat_cols]
        y_train_all = df_feat_train["delta_y"]
        model_bt.fit(X_train_all, y_train_all)

        # --------- 2) Holdout evaluation (paper-2) ---------
        full_hist = g[["date", "value"]].copy()
        skill_df, bt_summary = backtest_holdout_by_horizon(
            model=model_bt,
            full_history=full_hist,
            cutoff_date=cutoff,
            point_id=str(pid),
            resource_type=resource_type,
            unit=unit,
            bt_cfg=bt_cfg,
            feat_cfg=feat_cfg,
        )
        if not skill_df.empty:
            skill_df["model_type"] = best_type
            all_skill.append(skill_df)

        # --------- 3) Final fit on FULL history (operational) ---------
        df_feat_full = build_endo_features(full_hist, feat_cfg)
        df_feat_full = df_feat_full.dropna(subset=feat_cols + ["delta_y"]).copy()

        model_final = make_model(best_type, train_cfg)
        model_final.fit(df_feat_full[feat_cols], df_feat_full["delta_y"])

        # Operational forecast from last observed date
        last_date = pd.to_datetime(full_hist["date"].max())
        fc = forecast_recursive_endo(
            model=model_final,
            history=full_hist,
            start_date=last_date,
            fc_cfg=fc_cfg,
            feat_cfg=feat_cfg,
        )
        fc["point_id"] = str(pid)
        fc["resource_type"] = resource_type
        fc["unit"] = unit
        fc["family"] = "ENDO"
        fc["model_type"] = best_type
        fc["reliability_horizon"] = int(bt_summary.get("reliability_horizon", 0))
        all_forecasts.append(fc)

        # Summarize skill (ML only)
        ml = skill_df[skill_df["family"] == "ENDO_ML"].copy() if not skill_df.empty else pd.DataFrame()
        kge12 = float(ml.loc[ml["horizon"] == 12, "kge"].iloc[0]) if (not ml.empty and (ml["horizon"] == 12).any()) else float("nan")
        kge24 = float(ml.loc[ml["horizon"] == 24, "kge"].iloc[0]) if (not ml.empty and (ml["horizon"] == 24).any()) else float("nan")
        kge36 = float(ml.loc[ml["horizon"] == 36, "kge"].iloc[0]) if (not ml.empty and (ml["horizon"] == 36).any()) else float("nan")
        kge48 = float(ml.loc[ml["horizon"] == 48, "kge"].iloc[0]) if (not ml.empty and (ml["horizon"] == 48).any()) else float("nan")

        all_metrics.append(
            {
                "point_id": str(pid),
                "resource_type": resource_type,
                "unit": unit,
                "status": "OK",
                "model_type": best_type,
                "kge_inner_val": float(sel_met.get("kge_val", float("nan"))),
                "rmse_inner_val": float(sel_met.get("rmse_val", float("nan"))),
                "cutoff_date": str(pd.to_datetime(cutoff).date()),
                "last_observed_date": str(pd.to_datetime(last_date).date()),
                "reliability_horizon": int(bt_summary.get("reliability_horizon", 0)),
                "kge_12": kge12,
                "kge_24": kge24,
                "kge_36": kge36,
                "kge_48": kge48,
                "kge_threshold": float(cfg.kge_threshold),
                "test_months": int(cfg.test_months),
            }
        )

    forecasts_df = pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()
    metrics_df = pd.DataFrame(all_metrics)
    skill_all = pd.concat(all_skill, ignore_index=True) if all_skill else pd.DataFrame()

    forecasts_df.to_csv(outdir / "forecasts_v0_5_1.csv", index=False)
    metrics_df.to_csv(outdir / "metrics_v0_5_1.csv", index=False)
    skill_all.to_csv(outdir / "skill_by_horizon_v0_5_1.csv", index=False)

    return forecasts_df, metrics_df, skill_all