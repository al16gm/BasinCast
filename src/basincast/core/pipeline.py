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
from basincast.core.quality import (
    QualityConfig,
    clean_monthly_series,
    horizons_supported_by_holdout,
    compute_grade_horizons,
    choose_decision,
    baseline_forecast,
    to_month_start,
)


@dataclass(frozen=True)
class PipelineConfig:
    horizons: Tuple[int, ...] = (12, 24, 36, 48)

    test_months: int = 96
    val_months: int = 36

    min_history_months: int = 120
    min_train_months: int = 120

    planning_kge_threshold: float = 0.6
    advisory_kge_threshold: float = 0.3
    min_pairs_per_horizon: int = 24

    random_state: int = 42


def run_pipeline(
    canonical_csv: str | Path,
    outdir: str | Path = "outputs",
    cfg: PipelineConfig = PipelineConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Core v0.5.2:
      - Graceful handling: OK/WARN/SKIP + reasons
      - Paper-2 holdout protocol on last `test_months` (auto-adjust if needed)
      - Reliability horizons: planning (KGE>=0.6) and advisory (KGE>=0.3)
      - Decision fallback: MODEL vs BASELINE vs NOT_RELIABLE
      - Forecast export includes model, baseline, and selected
    """
    canonical_csv = Path(canonical_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(canonical_csv)
    required = {"date", "point_id", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in canonical CSV: {sorted(list(missing))}")

    df["date"] = to_month_start(df["date"])
    df = df.dropna(subset=["date", "point_id", "value"]).copy()
    df["point_id"] = df["point_id"].astype(str).map(str.strip)

    if "resource_type" not in df.columns:
        df["resource_type"] = ""
    if "unit" not in df.columns:
        df["unit"] = ""

    feat_cfg = FeatureConfig()
    feat_cols = get_feature_columns(feat_cfg)
    train_cfg = TrainConfig(random_state=cfg.random_state)

    qc = QualityConfig(
        planning_kge_threshold=cfg.planning_kge_threshold,
        advisory_kge_threshold=cfg.advisory_kge_threshold,
        min_pairs_per_horizon=cfg.min_pairs_per_horizon,
        non_negative=True,
    )

    all_forecasts = []
    all_metrics = []
    all_skill = []

    horizons_req = list(cfg.horizons)

    for pid, g0 in df.groupby("point_id"):
        reasons: List[str] = []
        warnings: List[str] = []

        g0 = g0.sort_values("date").copy()
        resource_type = str(g0["resource_type"].dropna().iloc[0]) if g0["resource_type"].dropna().shape[0] > 0 else ""
        unit = str(g0["unit"].dropna().iloc[0]) if g0["unit"].dropna().shape[0] > 0 else ""

        # Clean series per point
        g, w = clean_monthly_series(g0[["date", "value"]])
        warnings.extend(w)

        n_months = int(g["date"].nunique())
        if n_months < cfg.min_history_months:
            all_metrics.append({
                "point_id": str(pid), "resource_type": resource_type, "unit": unit,
                "status": "SKIP", "reason": f"HISTORY_TOO_SHORT({n_months}<{cfg.min_history_months})",
            })
            continue

        if resource_type == "":
            warnings.append("MISSING_RESOURCE_TYPE")
        if unit == "":
            warnings.append("MISSING_UNIT")

        max_date = pd.to_datetime(g["date"].max()).to_period("M").to_timestamp()

        # Auto-adjust holdout if series is not long enough for the requested test_months + min_train_months
        test_months = int(cfg.test_months)
        if n_months < cfg.min_train_months + test_months:
            test_months = max(24, n_months - cfg.min_train_months)
            warnings.append(f"TEST_MONTHS_ADJUSTED={test_months}")
        if test_months < 24:
            all_metrics.append({
                "point_id": str(pid), "resource_type": resource_type, "unit": unit,
                "status": "SKIP", "reason": "INSUFFICIENT_LENGTH_FOR_HOLDOUT",
            })
            continue

        cutoff = (max_date - pd.DateOffset(months=int(test_months))).to_period("M").to_timestamp()
        train_hist = g[g["date"] <= cutoff].copy()
        hold_hist = g[g["date"] > cutoff].copy()
        holdout_months = int(hold_hist["date"].nunique())

        supported_h, unsupported_h = horizons_supported_by_holdout(holdout_months, horizons_req, qc.min_pairs_per_horizon)
        if unsupported_h:
            warnings.append(f"UNSUPPORTED_HORIZONS_DUE_TO_HOLDOUT={unsupported_h}")

        # If no horizons can be scored with enough pairs, we still can forecast, but reliability must be 0
        score_horizons = supported_h if supported_h else []

        # --- Train features on TRAIN only (no leakage) ---
        df_feat_train = build_endo_features(train_hist, feat_cfg).dropna(subset=feat_cols + ["delta_y"]).copy()
        if len(df_feat_train) < 60:
            all_metrics.append({
                "point_id": str(pid), "resource_type": resource_type, "unit": unit,
                "status": "SKIP", "reason": "TOO_FEW_TRAIN_FEATURE_ROWS_AFTER_LAGGING",
            })
            continue

        inner_tr, inner_val = split_train_val_by_months(df_feat_train, val_months=int(cfg.val_months))
        X_tr, y_tr = inner_tr[feat_cols], inner_tr["delta_y"]
        X_val, y_val = inner_val[feat_cols], inner_val["delta_y"]

        best_type, sel_met = train_and_select(X_tr, y_tr, X_val, y_val, cfg=train_cfg)

        # Fit model for backtest on ALL train_hist
        model_bt = make_model(best_type, train_cfg)
        model_bt.fit(df_feat_train[feat_cols], df_feat_train["delta_y"])

        # --- Backtest on holdout (paper-2 protocol) ---
        bt_cfg = BacktestConfig(
            test_months=int(test_months),
            min_pairs_per_horizon=int(qc.min_pairs_per_horizon),
            horizons=tuple(horizons_req),
        )
        skill_df, bt_summary = backtest_holdout_by_horizon(
            model=model_bt,
            full_history=g[["date", "value"]],
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

        # Compute planning/advisory horizons (ML and baseline)
        ml_plan, ml_adv = compute_grade_horizons(skill_df, "ENDO_ML", qc, horizons_req)
        bl_plan, bl_adv = compute_grade_horizons(skill_df, "BASELINE_SEASONAL", qc, horizons_req)

        decision, selected_family = choose_decision(ml_plan, ml_adv, bl_plan, bl_adv)

        if not score_horizons:
            # Can't score reliably -> force not reliable horizons
            warnings.append("INSUFFICIENT_BACKTEST_PAIRS_FOR_RELIABILITY")
            ml_plan = 0
            ml_adv = 0
            bl_plan = 0
            bl_adv = 0
            if decision != "NOT_RELIABLE":
                warnings.append("DECISION_DOWNGRADED_TO_NOT_RELIABLE")
            decision, selected_family = "NOT_RELIABLE", "NONE"

        # Status: OK only if we have planning-grade usable output; otherwise WARN
        if decision in ("MODEL_PLANNING", "BASELINE_PLANNING") and not warnings:
            status = "OK"
        else:
            status = "WARN"
        
        reason = ""
        if decision == "NOT_RELIABLE":
            reason = "SKILL_BELOW_ADVISORY_THRESHOLD"
        elif decision in ("MODEL_ADVISORY", "BASELINE_ADVISORY"):
            reason = "ONLY_ADVISORY_GRADE"

        # --- Final fit on FULL history for operational forecast ---
        full_hist = g[["date", "value"]].copy()
        df_feat_full = build_endo_features(full_hist, feat_cfg).dropna(subset=feat_cols + ["delta_y"]).copy()
        model_final = make_model(best_type, train_cfg)
        model_final.fit(df_feat_full[feat_cols], df_feat_full["delta_y"])

        last_date = pd.to_datetime(full_hist["date"].max()).to_period("M").to_timestamp()

        fc_cfg = ForecastConfig(horizons=horizons_req, non_negative=True)
        fc_model = forecast_recursive_endo(
            model=model_final,
            history=full_hist,
            start_date=last_date,
            fc_cfg=fc_cfg,
            feat_cfg=feat_cfg,
        ).rename(columns={"y_forecast": "y_forecast_model"})

        fc_base = baseline_forecast(full_hist, last_date, horizons_req, non_negative=True).rename(columns={"y_forecast": "y_forecast_baseline"})

        fc = fc_model.merge(fc_base, on=["date", "horizon"], how="outer")
        fc["point_id"] = str(pid)
        fc["resource_type"] = resource_type
        fc["unit"] = unit
        fc["model_type"] = best_type
        fc["status"] = status
        fc["decision"] = decision
        fc["selected_family"] = selected_family

        fc["planning_horizon_ml"] = int(ml_plan)
        fc["advisory_horizon_ml"] = int(ml_adv)
        fc["planning_horizon_baseline"] = int(bl_plan)
        fc["advisory_horizon_baseline"] = int(bl_adv)

        # Selected forecast
        if selected_family == "BASELINE_SEASONAL":
            fc["y_forecast_selected"] = fc["y_forecast_baseline"]
        else:
            fc["y_forecast_selected"] = fc["y_forecast_model"]

        # Reliability flags (selected)
        planning_sel = bl_plan if selected_family == "BASELINE_SEASONAL" else ml_plan
        advisory_sel = bl_adv if selected_family == "BASELINE_SEASONAL" else ml_adv

        fc["is_planning_reliable"] = fc["horizon"].astype(int) <= int(planning_sel) if planning_sel > 0 else False
        fc["is_advisory_reliable"] = fc["horizon"].astype(int) <= int(advisory_sel) if advisory_sel > 0 else False

        # Attach warnings list
        fc["warnings"] = ";".join(warnings) if warnings else ""
        all_forecasts.append(fc)

        # Put KGE by horizon into metrics row (ML and baseline)
        def _get_kge(fam: str, h: int) -> float:
            r = skill_df[(skill_df["family"] == fam) & (skill_df["horizon"] == int(h))]
            if r.empty:
                return float("nan")
            return float(r["kge"].iloc[0]) if np.isfinite(r["kge"].iloc[0]) else float("nan")

        met_row = {
            "point_id": str(pid),
            "resource_type": resource_type,
            "unit": unit,
            "status": status,
            "reason": reason,
            "warnings": ";".join(warnings) if warnings else "",
            "model_type": best_type,
            "kge_inner_val": float(sel_met.get("kge_val", float("nan"))),
            "rmse_inner_val": float(sel_met.get("rmse_val", float("nan"))),
            "cutoff_date": str(pd.to_datetime(cutoff).date()),
            "last_observed_date": str(pd.to_datetime(last_date).date()),
            "decision": decision,
            "selected_family": selected_family,
            "planning_horizon_ml": int(ml_plan),
            "advisory_horizon_ml": int(ml_adv),
            "planning_horizon_baseline": int(bl_plan),
            "advisory_horizon_baseline": int(bl_adv),
            "planning_kge_threshold": float(qc.planning_kge_threshold),
            "advisory_kge_threshold": float(qc.advisory_kge_threshold),
            "test_months_used": int(test_months),
            "holdout_months": int(holdout_months),
        }

        for h in horizons_req:
            met_row[f"kge_ml_{h}"] = _get_kge("ENDO_ML", int(h))
            met_row[f"kge_bl_{h}"] = _get_kge("BASELINE_SEASONAL", int(h))

        all_metrics.append(met_row)

    forecasts_df = pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()
    metrics_df = pd.DataFrame(all_metrics)
    skill_all = pd.concat(all_skill, ignore_index=True) if all_skill else pd.DataFrame()

    forecasts_df.to_csv(outdir / "forecasts_v0_5_2.csv", index=False)
    metrics_df.to_csv(outdir / "metrics_v0_5_2.csv", index=False)
    skill_all.to_csv(outdir / "skill_by_horizon_v0_5_2.csv", index=False)

    return forecasts_df, metrics_df, skill_all