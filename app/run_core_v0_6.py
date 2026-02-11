from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error


# -----------------------------
# Metrics
# -----------------------------
def kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2:
        return np.nan

    r = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(r):
        return np.nan

    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    if std_true == 0 or std_pred == 0:
        return np.nan

    alpha = std_pred / std_true
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    if mean_true == 0:
        return np.nan
    beta = mean_pred / mean_true

    return 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# -----------------------------
# Feature engineering
# -----------------------------
EXOG_COLS = ["precip_mm_month_est", "t2m_c", "tmax_c", "tmin_c"]


def _month_sin_cos(dates: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    m = pd.to_datetime(dates).dt.month.to_numpy()
    return np.sin(2 * np.pi * m / 12.0), np.cos(2 * np.pi * m / 12.0)


def make_features(df: pd.DataFrame, family: str) -> pd.DataFrame:
    """
    Build a modeling table in incremental space (delta_y) with ENDO or EXOG family.

    ENDO features:
      - value_lag1, value_lag12, delta_lag1, month_sin, month_cos

    EXOG features:
      - ENDO + exog_lag1 for precip/temp

    Returns a DataFrame with feature columns + delta_y (target), indexed by date.
    """
    g = df.copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    g = g.sort_values("date").reset_index(drop=True)

    # Basic target
    g["delta_y"] = g["value"].diff()

    # Lags
    g["value_lag1"] = g["value"].shift(1)
    g["value_lag12"] = g["value"].shift(12)
    g["delta_lag1"] = g["delta_y"].shift(1)

    # Seasonality
    msin, mcos = _month_sin_cos(g["date"])
    g["month_sin"] = msin
    g["month_cos"] = mcos

    feat_cols = ["value_lag1", "value_lag12", "delta_lag1", "month_sin", "month_cos"]

    if family == "EXOG_ML":
        # require exog columns
        for c in EXOG_COLS:
            if c not in g.columns:
                raise ValueError(f"Missing EXOG column '{c}' for family=EXOG_ML")

        # Use lag1 exog (safe + consistent with recursive usage)
        for c in EXOG_COLS:
            g[f"{c}_lag1"] = g[c].shift(1)
            feat_cols.append(f"{c}_lag1")

    out = g[["date", "delta_y"] + feat_cols].dropna().reset_index(drop=True)
    return out


# -----------------------------
# Models
# -----------------------------
def model_zoo() -> Dict[str, object]:
    return {
        "bayes_ridge": BayesianRidge(),
        "gbr": GradientBoostingRegressor(random_state=42),
        "rf": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
    }


def train_select_model(X_tr, y_tr, X_val, y_val) -> Tuple[object, str, float, float]:
    best = None
    best_name = ""
    best_kge = -np.inf
    best_rmse = np.inf

    for name, model in model_zoo().items():
        try:
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            k = kge(y_val, pred)
            r = rmse(y_val, pred)
            if np.isnan(k):
                continue
            if k > best_kge:
                best = model
                best_name = name
                best_kge = float(k)
                best_rmse = float(r)
        except Exception:
            continue

    if best is None:
        raise RuntimeError("No model could be trained successfully.")
    return best, best_name, best_kge, best_rmse


# -----------------------------
# Recursive forecasting for backtest
# -----------------------------
def recursive_forecast_delta(
    model,
    history: pd.DataFrame,
    exog_df: pd.DataFrame | None,
    start_date: pd.Timestamp,
    horizon: int,
    family: str,
) -> float:
    """
    Forecast y(start_date + horizon months) using recursive delta_y predictions.
    history: full df with columns date,value (+ exog if needed), sorted.
    exog_df: full df with exog columns by date (same length/range as history), sorted.
    """
    hist = history.copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date").reset_index(drop=True)

    # Build a buffer of past y values up to start_date
    start_date = pd.to_datetime(start_date)
    idx0 = hist.index[hist["date"] == start_date]
    if len(idx0) == 0:
        raise ValueError(f"start_date {start_date} not found in history.")
    i0 = int(idx0[0])

    # buffer contains actual values up to start_date
    y_buffer = hist.loc[:i0, "value"].to_list()
    # last delta (approx)
    if len(y_buffer) >= 2:
        last_delta = y_buffer[-1] - y_buffer[-2]
    else:
        last_delta = 0.0

    cur_date = start_date

    for step in range(1, horizon + 1):
        next_date = (cur_date + pd.DateOffset(months=1)).to_period("M").to_timestamp()

        # lags
        y_lag1 = y_buffer[-1]
        if len(y_buffer) >= 12:
            y_lag12 = y_buffer[-12]
        else:
            # not enough memory; repeat last observed
            y_lag12 = y_lag1

        delta_lag1 = last_delta

        # seasonality
        m = int(next_date.month)
        month_sin = float(np.sin(2 * np.pi * m / 12.0))
        month_cos = float(np.cos(2 * np.pi * m / 12.0))

        feats = {
            "value_lag1": y_lag1,
            "value_lag12": y_lag12,
            "delta_lag1": delta_lag1,
            "month_sin": month_sin,
            "month_cos": month_cos,
        }

        if family == "EXOG_ML":
            if exog_df is None:
                raise ValueError("EXOG_ML requires exog_df.")
            # lag1 exog for next_date => use exog at (next_date - 1 month)
            exog_lag_date = (next_date - pd.DateOffset(months=1)).to_period("M").to_timestamp()
            row = exog_df.loc[exog_df["date"] == exog_lag_date]
            if len(row) == 0:
                # fallback: use monthly climatology from available exog_df
                mm = int(exog_lag_date.month)
                clim = exog_df.groupby(exog_df["date"].dt.month)[EXOG_COLS].mean(numeric_only=True)
                vals = clim.loc[mm].to_dict()
            else:
                vals = row.iloc[0][EXOG_COLS].to_dict()

            for c in EXOG_COLS:
                feats[f"{c}_lag1"] = float(vals[c])

        X = pd.DataFrame([feats])
        delta_pred = float(model.predict(X.to_numpy())[0])

        y_next = max(0.0, y_lag1 + delta_pred)

        # update buffer
        last_delta = delta_pred
        y_buffer.append(y_next)
        cur_date = next_date

    return float(y_buffer[-1])


def seasonal_baseline(history: pd.DataFrame, start_date: pd.Timestamp, horizon: int) -> float:
    """
    Seasonal naive: y_hat(target) = y(target - 12 months) if available, else y(start_date).
    """
    hist = history.copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date").reset_index(drop=True)

    start_date = pd.to_datetime(start_date)
    target = (start_date + pd.DateOffset(months=horizon)).to_period("M").to_timestamp()
    ref = (target - pd.DateOffset(months=12)).to_period("M").to_timestamp()

    ref_row = hist.loc[hist["date"] == ref]
    if len(ref_row) > 0:
        return float(ref_row.iloc[0]["value"])

    start_row = hist.loc[hist["date"] == start_date]
    if len(start_row) > 0:
        return float(start_row.iloc[0]["value"])

    return float(hist.iloc[-1]["value"])


# -----------------------------
# Backtest + decision
# -----------------------------
@dataclass
class RunConfig:
    horizons: List[int] = None
    holdout_months: int = 96
    inner_val_months: int = 36
    planning_kge: float = 0.6
    advisory_kge: float = 0.3

    def __post_init__(self):
        if self.horizons is None:
            self.horizons = [12, 24, 36, 48]


def planning_horizon_from_kges(kges: Dict[int, float], threshold: float) -> int:
    ok = [h for h, v in kges.items() if (v is not None and not np.isnan(v) and v >= threshold)]
    return max(ok) if ok else 0


def run_point(df_point: pd.DataFrame, cfg: RunConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dfp = df_point.copy()
    dfp["date"] = pd.to_datetime(dfp["date"], errors="coerce")
    dfp = dfp.sort_values("date").reset_index(drop=True)

    point_id = dfp["point_id"].iloc[0]
    resource_type = dfp["resource_type"].iloc[0]
    unit = dfp["unit"].iloc[0]

    last_date = pd.to_datetime(dfp["date"].max()).to_period("M").to_timestamp()
    cutoff = (last_date - pd.DateOffset(months=cfg.holdout_months)).to_period("M").to_timestamp()

    # families to test
    families = ["ENDO_ML"]
    has_exog = all(c in dfp.columns for c in EXOG_COLS)
    if has_exog:
        families.append("EXOG_ML")
    families.append("BASELINE_SEASONAL")

    warnings = []

    if len(dfp) < (cfg.holdout_months + 120):
        warnings.append("SHORT_SERIES")

    # inner train/val split inside TRAIN portion
    train_mask = dfp["date"] <= cutoff
    train_df = dfp.loc[train_mask].copy()
    if len(train_df) < (cfg.inner_val_months + 120):
        warnings.append("SHORT_TRAIN_WINDOW")

    inner_cut = (cutoff - pd.DateOffset(months=cfg.inner_val_months)).to_period("M").to_timestamp()
    inner_tr = train_df.loc[train_df["date"] <= inner_cut].copy()
    inner_val = train_df.loc[train_df["date"] > inner_cut].copy()

    model_info = {}  # family -> (model, model_type, kge_inner, rmse_inner)
    skill_rows = []

    # Prepare exog_df for recursive function
    exog_df = None
    if has_exog:
        exog_df = dfp[["date"] + EXOG_COLS].copy()
        exog_df["date"] = pd.to_datetime(exog_df["date"]).dt.to_period("M").dt.to_timestamp()

    # Train ML families
    for fam in families:
        if fam == "BASELINE_SEASONAL":
            continue

        feats_all = make_features(train_df, fam)
        feats_tr = feats_all.loc[feats_all["date"] <= inner_cut].copy()
        feats_val = feats_all.loc[feats_all["date"] > inner_cut].copy()

        X_tr = feats_tr.drop(columns=["date", "delta_y"]).to_numpy()
        y_tr = feats_tr["delta_y"].to_numpy()
        X_val = feats_val.drop(columns=["date", "delta_y"]).to_numpy()
        y_val = feats_val["delta_y"].to_numpy()

        model, model_type, k_in, r_in = train_select_model(X_tr, y_tr, X_val, y_val)
        model_info[fam] = (model, model_type, k_in, r_in)

    # Evaluate horizons on holdout origins
    holdout_origins = dfp.loc[(dfp["date"] >= cutoff) & (dfp["date"] <= last_date), "date"].tolist()

    # For each family and horizon, compute pairs
    kge_by_family = {fam: {} for fam in families}
    rmse_by_family = {fam: {} for fam in families}
    npairs_by_family = {fam: {} for fam in families}

    dfp_dates = pd.to_datetime(dfp["date"]).dt.to_period("M").dt.to_timestamp()
    dfp = dfp.copy()
    dfp["date"] = dfp_dates

    for fam in families:
        for h in cfg.horizons:
            y_true = []
            y_pred = []

            for origin in holdout_origins:
                origin = pd.to_datetime(origin).to_period("M").to_timestamp()
                target = (origin + pd.DateOffset(months=h)).to_period("M").to_timestamp()

                # only evaluate where target exists in observed df
                row_t = dfp.loc[dfp["date"] == target]
                if len(row_t) == 0:
                    continue

                y_t = float(row_t.iloc[0]["value"])

                if fam == "BASELINE_SEASONAL":
                    y_hat = seasonal_baseline(dfp[["date", "value"]], origin, h)
                else:
                    model = model_info[fam][0]
                    y_hat = recursive_forecast_delta(
                        model=model,
                        history=dfp[["date", "value"]],
                        exog_df=exog_df,
                        start_date=origin,
                        horizon=h,
                        family=fam,
                    )

                y_true.append(y_t)
                y_pred.append(y_hat)

            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)

            n_pairs = int(len(y_true))
            npairs_by_family[fam][h] = n_pairs

            if n_pairs < 24:
                k = np.nan
                r = np.nan
            else:
                k = kge(y_true, y_pred)
                r = rmse(y_true, y_pred)

            kge_by_family[fam][h] = float(k) if k is not None else np.nan
            rmse_by_family[fam][h] = float(r) if r is not None else np.nan

            skill_rows.append({
                "point_id": point_id,
                "resource_type": resource_type,
                "unit": unit,
                "family": fam,
                "horizon": h,
                "kge": kge_by_family[fam][h],
                "rmse": rmse_by_family[fam][h],
                "n_pairs": n_pairs,
                "model_type": model_info[fam][1] if fam in model_info else "",
            })

    skill_df = pd.DataFrame(skill_rows)

    # Decide best family using planning/advisory horizons
    planning_h = {}
    advisory_h = {}
    for fam in families:
        planning_h[fam] = planning_horizon_from_kges(kge_by_family[fam], cfg.planning_kge)
        advisory_h[fam] = planning_horizon_from_kges(kge_by_family[fam], cfg.advisory_kge)

    # Primary: max planning horizon; Secondary: max advisory; Tertiary: max kge at 12
    def rank_key(fam: str):
        k12 = kge_by_family[fam].get(12, np.nan)
        k12 = -1e9 if np.isnan(k12) else float(k12)
        return (planning_h[fam], advisory_h[fam], k12)

    selected = sorted(families, key=rank_key, reverse=True)[0]

    # Decision label
    if planning_h[selected] > 0:
        decision = "MODEL_PLANNING" if selected in ("ENDO_ML", "EXOG_ML") else "BASELINE_PLANNING"
        status = "OK"
        reason = ""
    elif advisory_h[selected] > 0:
        decision = "MODEL_ADVISORY" if selected in ("ENDO_ML", "EXOG_ML") else "BASELINE_ADVISORY"
        status = "WARN"
        reason = "ONLY_ADVISORY_GRADE"
    else:
        decision = "NOT_RELIABLE"
        status = "WARN"
        reason = "SKILL_BELOW_ADVISORY_THRESHOLD"

    # Location confidence
    loc_conf = "HIGH"
    if dfp["lat"].isna().any() or dfp["lon"].isna().any():
        loc_conf = "LOW"

    meteo_source = ""
    if "source" in dfp.columns:
        try:
            meteo_source = str(dfp["source"].mode(dropna=True).iloc[0])
        except Exception:
            meteo_source = ""

    # Metrics row
    met = {
        "point_id": point_id,
        "resource_type": resource_type,
        "unit": unit,
        "status": status,
        "reason": reason,
        "warnings": ";".join(warnings) if warnings else "",
        "decision": decision,
        "selected_family": selected,
        "planning_horizon": int(planning_h[selected]),
        "advisory_horizon": int(advisory_h[selected]),
        "planning_kge_threshold": cfg.planning_kge,
        "advisory_kge_threshold": cfg.advisory_kge,
        "cutoff_date": str(cutoff.date()),
        "last_observed_date": str(last_date.date()),
        "location_confidence": loc_conf,
        "meteo_source": meteo_source,
    }

    # Attach horizon-wise KGE columns
    for fam in families:
        tag = "ml" if fam in ("ENDO_ML", "EXOG_ML") else "bl"
        for h in cfg.horizons:
            met[f"kge_{fam.lower()}_{h}"] = kge_by_family[fam].get(h, np.nan)

    metrics_df = pd.DataFrame([met])

    # Produce final forecasts from last_date (future exog uses climatology if EXOG_ML)
    forecasts = []
    if selected == "BASELINE_SEASONAL":
        for h in cfg.horizons:
            y_hat = seasonal_baseline(dfp[["date", "value"]], last_date, h)
            forecasts.append({
                "date": str((last_date + pd.DateOffset(months=h)).date()),
                "horizon": h,
                "y_forecast": y_hat,
                "point_id": point_id,
                "resource_type": resource_type,
                "unit": unit,
                "family": selected,
                "model_type": "",
                "exog_future_mode": "",
            })
    else:
        model = model_info[selected][0]
        model_type = model_info[selected][1]
        for h in cfg.horizons:
            y_hat = recursive_forecast_delta(
                model=model,
                history=dfp[["date", "value"]],
                exog_df=exog_df,
                start_date=last_date,
                horizon=h,
                family=selected,
            )
            forecasts.append({
                "date": str((last_date + pd.DateOffset(months=h)).date()),
                "horizon": h,
                "y_forecast": y_hat,
                "point_id": point_id,
                "resource_type": resource_type,
                "unit": unit,
                "family": selected,
                "model_type": model_type,
                "exog_future_mode": "CLIMATOLOGY" if selected == "EXOG_ML" else "",
            })

    forecasts_df = pd.DataFrame(forecasts)
    return metrics_df, skill_df, forecasts_df


def main() -> None:
    ap = argparse.ArgumentParser(description="BasinCast Core v0.6 (ENDO vs EXOG vs Baseline).")
    ap.add_argument("--input", required=True, help="Path to canonical_with_meteo.csv (or canonical without meteo)")
    ap.add_argument("--outdir", default="outputs", help="Output folder")
    ap.add_argument("--holdout_months", type=int, default=96)
    ap.add_argument("--inner_val_months", type=int, default=36)
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(inp)

    df = pd.read_csv(inp)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.sort_values(["point_id", "date"]).reset_index(drop=True)

    required = {"point_id", "date", "value", "resource_type", "unit"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required canonical columns: {sorted(missing)}")

    cfg = RunConfig(
        horizons=[12, 24, 36, 48],
        holdout_months=args.holdout_months,
        inner_val_months=args.inner_val_months,
    )

    all_metrics = []
    all_skill = []
    all_fc = []

    for pid, g in df.groupby("point_id", sort=True):
        m, s, f = run_point(g, cfg)
        all_metrics.append(m)
        all_skill.append(s)
        all_fc.append(f)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    skill_df = pd.concat(all_skill, ignore_index=True)
    fc_df = pd.concat(all_fc, ignore_index=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_path = outdir / "metrics_v0_6.csv"
    skill_path = outdir / "skill_v0_6.csv"
    fc_path = outdir / "forecasts_v0_6.csv"

    metrics_df.to_csv(metrics_path, index=False)
    skill_df.to_csv(skill_path, index=False)
    fc_df.to_csv(fc_path, index=False)

    print("BasinCast Core v0.6 ✅")
    print(f"Metrics rows:   {len(metrics_df)} | Saved: {metrics_path}")
    print(f"Skill rows:     {len(skill_df)} | Saved: {skill_path}")
    print(f"Forecasts rows: {len(fc_df)} | Saved: {fc_path}")
    print(metrics_df.head(1).to_csv(index=False).strip())


if __name__ == "__main__":
    main()