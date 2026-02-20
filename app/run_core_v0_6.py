from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Ensure "src/" is importable when running from Streamlit/subprocess
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from basincast.scenarios import CMIP6IntakeDeltaProvider, apply_year_month_deltas_to_exog

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
EXOG_METEO_COLS = ["precip_mm_month_est", "t2m_c", "tmax_c", "tmin_c"]
DEMAND_COL = "demand"  # canonical column name (both user upload and open-demand write this)


def _to_month_start(x) -> pd.Timestamp:
    return pd.to_datetime(x, errors="coerce").to_period("M").to_timestamp()


def _month_sin_cos(dates: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    m = pd.to_datetime(dates).dt.month.to_numpy()
    return np.sin(2 * np.pi * m / 12.0), np.cos(2 * np.pi * m / 12.0)


def make_features(df: pd.DataFrame, family: str) -> pd.DataFrame:
    """
    Build a modeling table in incremental space (delta_y) with ENDO / EXOG_METEO / DEMAND.

    Families:
      - ENDO_ML: only endogenous lags + month sin/cos
      - EXOG_ML: ENDO + meteo_lag1
      - DEMAND_ML: ENDO + demand_lag1
      - EXOG_ML_DEMAND: ENDO + meteo_lag1 + demand_lag1

    Returns: DataFrame with ['date', 'delta_y'] + ordered feature columns, dropping NaNs.
    """
    g = df.copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    g = g.sort_values("date").reset_index(drop=True)

    g["delta_y"] = g["value"].diff()

    g["value_lag1"] = g["value"].shift(1)
    g["value_lag12"] = g["value"].shift(12)
    g["delta_lag1"] = g["delta_y"].shift(1)

    msin, mcos = _month_sin_cos(g["date"])
    g["month_sin"] = msin
    g["month_cos"] = mcos

    feat_cols = ["value_lag1", "value_lag12", "delta_lag1", "month_sin", "month_cos"]

    # --- Meteo part ---
    if family in ("EXOG_ML", "EXOG_ML_DEMAND"):
        for c in EXOG_METEO_COLS:
            if c not in g.columns:
                raise ValueError(f"Missing EXOG meteo column '{c}' for family={family}")
        for c in EXOG_METEO_COLS:
            g[f"{c}_lag1"] = pd.to_numeric(g[c], errors="coerce").shift(1)
            feat_cols.append(f"{c}_lag1")

    # --- Demand part ---
    if family in ("DEMAND_ML", "EXOG_ML_DEMAND"):
        if DEMAND_COL not in g.columns:
            raise ValueError(f"Missing demand column '{DEMAND_COL}' for family={family}")
        g["demand_lag1"] = pd.to_numeric(g[DEMAND_COL], errors="coerce").shift(1)
        feat_cols.append("demand_lag1")

    out = g[["date", "delta_y"] + feat_cols].dropna().reset_index(drop=True)
    return out


# -----------------------------
# Scenario deltas (monthly) - DO NOT INVENT VALUES
# Fill these with your paper Table 2 values when ready.
# Convention:
# - precip_mult: multiplicative change on precipitation (e.g., -0.10 => -10%)
# - temp_add: additive delta in °C applied to temperature-like columns
# -----------------------------
SCENARIO_DELTAS = {
    "Base": {
        "precip_mult": [0.0] * 12,
        "temp_add": [0.0] * 12,
    },
    "Favorable": {
        "precip_mult": [0.0] * 12,
        "temp_add": [0.0] * 12,
    },
    "Unfavorable": {
        "precip_mult": [0.0] * 12,
        "temp_add": [0.0] * 12,
    },
}


def apply_meteo_deltas(
    exog_df: pd.DataFrame,
    scenario: str,
    start_apply_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Apply monthly deltas to meteo columns in exog_df starting from start_apply_date (inclusive).
    This is a deterministic delta-change scenario constructor.

    - Precip column: multiplicative change
    - Temperature-like columns: additive change (°C)

    Does NOT touch historical months prior to start_apply_date.
    """
    if exog_df is None or exog_df.empty:
        return exog_df

    if scenario not in SCENARIO_DELTAS:
        raise ValueError(f"Unknown scenario: {scenario}")

    df = exog_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=["date"]).sort_values("date")

    mask = df["date"] >= _to_month_start(start_apply_date)
    if not mask.any():
        return df

    deltas = SCENARIO_DELTAS[scenario]
    precip_mult = np.asarray(deltas["precip_mult"], dtype=float)
    temp_add = np.asarray(deltas["temp_add"], dtype=float)

    months = df.loc[mask, "date"].dt.month.values - 1  # 0..11

    # Apply precipitation multiplier ONLY to precip column if present
    if "precip_mm_month_est" in df.columns:
        df.loc[mask, "precip_mm_month_est"] = (
            pd.to_numeric(df.loc[mask, "precip_mm_month_est"], errors="coerce").astype(float)
            * (1.0 + precip_mult[months])
        )

    # Apply temperature additive delta to temp-like columns if present
    for c in ["t2m_c", "tmax_c", "tmin_c"]:
        if c in df.columns:
            df.loc[mask, c] = (
                pd.to_numeric(df.loc[mask, c], errors="coerce").astype(float)
                + temp_add[months]
            )

    return df

# -----------------------------
# Models
# -----------------------------
def model_zoo() -> Dict[str, object]:
    return {
        "bayes_ridge": BayesianRidge(),
        "gbr": GradientBoostingRegressor(random_state=42),
        "rf": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
    }


def train_select_model(X_tr, y_tr, X_val=None, y_val=None) -> Tuple[object, str, float, float]:
    """
    Robust selection:
      - Prefer max KGE on validation when defined
      - If all KGE are NaN, fallback to min RMSE
      - If X_val is empty, fallback to min train RMSE
    """
    candidates = []  # (name, model, kge_val, rmse_val)

    for name, model in model_zoo().items():
        try:
            if X_tr is None or len(X_tr) == 0:
                continue

            model.fit(X_tr, y_tr)

            if X_val is None or len(X_val) == 0:
                pred_tr = model.predict(X_tr)
                k_val = np.nan
                r_val = rmse(y_tr, pred_tr)
            else:
                pred = model.predict(X_val)
                k_val = kge(y_val, pred)
                r_val = rmse(y_val, pred)

            candidates.append((name, model, float(k_val) if k_val is not None else np.nan, float(r_val)))
        except Exception:
            continue

    if not candidates:
        raise RuntimeError("No model could be trained successfully (empty/invalid train window).")

    finite = [c for c in candidates if not np.isnan(c[2])]
    if finite:
        best = max(finite, key=lambda c: c[2])
    else:
        best = min(candidates, key=lambda c: c[3])

    best_name, best_model, best_kge, best_rmse = best
    return best_model, str(best_name), float(best_kge), float(best_rmse)


# -----------------------------
# Recursive forecasting (MONTHLY PATH)
# -----------------------------
def recursive_forecast_path_delta(
    model,
    history: pd.DataFrame,
    exog_df: pd.DataFrame | None,
    start_date: pd.Timestamp,
    horizon_max: int,
    family: str,
) -> List[Tuple[pd.Timestamp, float]]:
    """
    Returns list of (date, y_forecast) for steps 1..horizon_max using recursive delta_y predictions.

    Exogenous handling:
      - EXOG_ML: uses meteo (lag1), fallback to monthly climatology if missing for a month
      - DEMAND_ML: uses demand (lag1), fallback to monthly climatology if missing for a month
      - EXOG_ML_DEMAND: uses both
    """
    hist = history.copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date").reset_index(drop=True)

    start_date = _to_month_start(start_date)
    idx0 = hist.index[hist["date"] == start_date]
    if len(idx0) == 0:
        raise ValueError(f"start_date {start_date} not found in history.")
    i0 = int(idx0[0])

    y_buffer = hist.loc[:i0, "value"].astype(float).to_list()
    last_delta = (y_buffer[-1] - y_buffer[-2]) if len(y_buffer) >= 2 else 0.0
    cur_date = start_date

    # Determine which exog components are required
    need_meteo = family in ("EXOG_ML", "EXOG_ML_DEMAND")
    need_demand = family in ("DEMAND_ML", "EXOG_ML_DEMAND")

    ed = None
    clim_meteo = None
    clim_demand = None

    if need_meteo or need_demand:
        if exog_df is None:
            raise ValueError(f"{family} requires exog_df (meteo and/or demand).")

        ed = exog_df.copy()
        ed["date"] = pd.to_datetime(ed["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()

        if need_meteo:
            for c in EXOG_METEO_COLS:
                if c not in ed.columns:
                    raise ValueError(f"{family} requires meteo column '{c}' in exog_df.")
            clim_meteo = ed.groupby(ed["date"].dt.month)[EXOG_METEO_COLS].mean(numeric_only=True)

        if need_demand:
            if DEMAND_COL not in ed.columns:
                raise ValueError(f"{family} requires demand column '{DEMAND_COL}' in exog_df.")
            tmp = pd.to_numeric(ed[DEMAND_COL], errors="coerce")
            clim_demand = tmp.groupby(ed["date"].dt.month).mean()

    preds: List[Tuple[pd.Timestamp, float]] = []

    # Feature column order MUST match training
    feat_cols = ["value_lag1", "value_lag12", "delta_lag1", "month_sin", "month_cos"]
    if need_meteo:
        for c in EXOG_METEO_COLS:
            feat_cols.append(f"{c}_lag1")
    if need_demand:
        feat_cols.append("demand_lag1")

    for _ in range(horizon_max):
        next_date = (cur_date + pd.DateOffset(months=1)).to_period("M").to_timestamp()

        y_lag1 = float(y_buffer[-1])
        y_lag12 = float(y_buffer[-12]) if len(y_buffer) >= 12 else y_lag1
        delta_lag1 = float(last_delta)

        m = int(next_date.month)
        feats = {
            "value_lag1": y_lag1,
            "value_lag12": y_lag12,
            "delta_lag1": delta_lag1,
            "month_sin": float(np.sin(2 * np.pi * m / 12.0)),
            "month_cos": float(np.cos(2 * np.pi * m / 12.0)),
        }

        # Use lag1 exog at previous month
        exog_lag_date = (next_date - pd.DateOffset(months=1)).to_period("M").to_timestamp()

        if need_meteo:
            row = ed.loc[ed["date"] == exog_lag_date] if ed is not None else pd.DataFrame()
            if len(row) == 0 and clim_meteo is not None:
                vals = clim_meteo.loc[int(exog_lag_date.month)].to_dict()
            else:
                vals = row.iloc[0][EXOG_METEO_COLS].to_dict()

            for c in EXOG_METEO_COLS:
                feats[f"{c}_lag1"] = float(vals[c])

        if need_demand:
            row = ed.loc[ed["date"] == exog_lag_date] if ed is not None else pd.DataFrame()
            if len(row) == 0 and clim_demand is not None:
                dval = float(clim_demand.loc[int(exog_lag_date.month)])
            else:
                dval = float(pd.to_numeric(row.iloc[0][DEMAND_COL], errors="coerce"))

            feats["demand_lag1"] = dval

        X = pd.DataFrame([[feats[c] for c in feat_cols]], columns=feat_cols).to_numpy()
        delta_pred = float(model.predict(X)[0])

        y_next = max(0.0, y_lag1 + delta_pred)

        preds.append((next_date, float(y_next)))

        last_delta = delta_pred
        y_buffer.append(y_next)
        cur_date = next_date

    return preds


def seasonal_naive_path(history: pd.DataFrame, start_date: pd.Timestamp, horizon_max: int) -> List[Tuple[pd.Timestamp, float]]:

    def extend_exog_with_monthly_climatology(
        exog_df: pd.DataFrame,
        origin: pd.Timestamp,
        horizon_max: int,
        include_meteo: bool,
        include_demand: bool,
    ) -> pd.DataFrame:
        """
        Extend exog_df (historical) with future rows up to origin + horizon_max months.
        Future values are filled using monthly climatology computed from historical data.

        This keeps existing fallback logic but makes scenario exog effective beyond h=1.
        """
        if exog_df is None or exog_df.empty:
            return exog_df

        ed = exog_df.copy()
        ed["date"] = pd.to_datetime(ed["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        ed = ed.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        # Compute monthly climatologies from historical data
        clim_meteo = None
        clim_demand = None

        if include_meteo:
            # EXOG_METEO_COLS must exist in your file (it does in v0.6)
            clim_meteo = ed.groupby(ed["date"].dt.month)[EXOG_METEO_COLS].mean(numeric_only=True)

        if include_demand and (DEMAND_COL in ed.columns):
            tmp = pd.to_numeric(ed[DEMAND_COL], errors="coerce")
            clim_demand = tmp.groupby(ed["date"].dt.month).mean()

        origin = _to_month_start(origin)
        last_needed = _to_month_start(origin + pd.DateOffset(months=int(horizon_max)))

        future_dates = pd.date_range(
            start=_to_month_start(origin + pd.DateOffset(months=1)),
            end=last_needed,
            freq="MS",
        )

        existing = set(ed["date"].tolist())
        rows: List[dict] = []

        for d in future_dates:
            d = _to_month_start(d)
            if d in existing:
                continue

            m = int(d.month)
            row: dict = {"date": d}

            if include_meteo and (clim_meteo is not None):
                vals = clim_meteo.loc[m].to_dict()
                for c in EXOG_METEO_COLS:
                    row[c] = float(vals[c])

            if include_demand and (clim_demand is not None):
                row[DEMAND_COL] = float(clim_demand.loc[m])

            rows.append(row)

        if rows:
            ed = pd.concat([ed, pd.DataFrame(rows)], ignore_index=True)
            ed = ed.sort_values("date").reset_index(drop=True)

        return ed

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


def run_point(
    df_point: pd.DataFrame,
    cfg: RunConfig,
    baseline_start_year: int,
    baseline_end_year: int,
    future_end_year: int = 2050,
    cmip6_cache_dir: str = "outputs/cache/cmip6_deltas",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    dfp = df_point.copy()
    dfp["date"] = pd.to_datetime(dfp["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    dfp = dfp.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)

    point_id = str(dfp["point_id"].iloc[0])
    resource_type = str(dfp["resource_type"].iloc[0]) if "resource_type" in dfp.columns else ""
    unit = str(dfp["unit"].iloc[0]) if "unit" in dfp.columns else ""

    last_date = _to_month_start(dfp["date"].max())
    cutoff = _to_month_start(last_date - pd.DateOffset(months=int(cfg.holdout_months)))

    # Candidate families (data-driven)
    has_meteo = (
        all(c in dfp.columns for c in EXOG_METEO_COLS)
        and dfp[EXOG_METEO_COLS].notna().any().all()
    )
    has_demand = (DEMAND_COL in dfp.columns) and pd.to_numeric(dfp[DEMAND_COL], errors="coerce").notna().any()

    families = ["ENDO_ML"]
    if has_meteo:
        families.append("EXOG_ML")
    if has_demand:
        families.append("DEMAND_ML")           # <-- demand without meteo supported
    if has_meteo and has_demand:
        families.append("EXOG_ML_DEMAND")      # <-- both
    families.append("BASELINE_SEASONAL")

    warnings: List[str] = []
    if len(dfp) < (cfg.holdout_months + 48):
        warnings.append("SHORT_SERIES")

    train_df = dfp.loc[dfp["date"] <= cutoff].copy()
    if len(train_df) < (cfg.inner_val_months + 24):
        warnings.append("SHORT_TRAIN_WINDOW")

    inner_cut = _to_month_start(cutoff - pd.DateOffset(months=int(cfg.inner_val_months)))
    inner_tr = train_df.loc[train_df["date"] <= inner_cut].copy()
    inner_val = train_df.loc[train_df["date"] > inner_cut].copy()

    model_info: Dict[str, Tuple[object, str, float, float]] = {}  # fam -> (model, model_type, kge_inner, rmse_inner)

    exog_df = None
    exog_cols: List[str] = []
    if has_meteo:
        exog_cols += EXOG_METEO_COLS
    if has_demand:
        exog_cols += [DEMAND_COL]

    if exog_cols:
        exog_df = dfp[["date"] + exog_cols].copy()
        exog_df["date"] = pd.to_datetime(exog_df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()

    # Train ML families (DO NOT CRASH if a family fails)
    for fam in families:
        if fam == "BASELINE_SEASONAL":
            continue
        try:
            feats_all = make_features(train_df, fam)
            feats_tr = feats_all.loc[feats_all["date"] <= inner_cut].copy()
            feats_val = feats_all.loc[feats_all["date"] > inner_cut].copy()

            if len(feats_tr) < 24:
                raise RuntimeError("Not enough rows to train")

            X_tr = feats_tr.drop(columns=["date", "delta_y"]).to_numpy()
            y_tr = feats_tr["delta_y"].to_numpy()
            X_val = feats_val.drop(columns=["date", "delta_y"]).to_numpy()
            y_val = feats_val["delta_y"].to_numpy()

            model, model_type, k_in, r_in = train_select_model(X_tr, y_tr, X_val, y_val)
            model_info[fam] = (model, model_type, k_in, r_in)
        except Exception:
            warnings.append(f"SKIP_{fam}_NO_MODEL")

    # Evaluate horizons on holdout origins (months inside holdout window)
    holdout_origins = dfp.loc[(dfp["date"] >= cutoff) & (dfp["date"] <= last_date), "date"].tolist()

    kge_by_family = {fam: {} for fam in families}
    rmse_by_family = {fam: {} for fam in families}
    npairs_by_family = {fam: {} for fam in families}

    skill_rows = []

    for fam in families:
        for h in cfg.horizons:
            y_true = []
            y_pred = []

            for origin in holdout_origins:
                origin = _to_month_start(origin)
                target = _to_month_start(origin + pd.DateOffset(months=int(h)))

                row_t = dfp.loc[dfp["date"] == target]
                if len(row_t) == 0:
                    continue

                y_t = float(row_t.iloc[0]["value"])

                if fam == "BASELINE_SEASONAL":
                    # recursive seasonal naive (multi-step)
                    y_hat = seasonal_naive_path(dfp[["date", "value"]], origin, int(h))[-1][1]
                else:
                    if fam not in model_info:
                        continue
                    model = model_info[fam][0]
                    y_hat = recursive_forecast_path_delta(
                        model=model,
                        history=dfp[["date", "value"]],
                        exog_df=exog_df,
                        start_date=origin,
                        horizon_max=int(h),
                        family=fam,
                    )[-1][1]

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

            skill_rows.append(
                {
                    "point_id": point_id,
                    "resource_type": resource_type,
                    "unit": unit,
                    "family": fam,
                    "horizon": int(h),
                    "kge": kge_by_family[fam][h],
                    "rmse": rmse_by_family[fam][h],
                    "n_pairs": n_pairs,
                    "model_type": (model_info[fam][1] if fam in model_info else ("seasonal_naive" if fam == "BASELINE_SEASONAL" else "")),
                }
            )

    skill_df = pd.DataFrame(skill_rows)

    # Decide best family: planning -> advisory -> KGE@12
    planning_h = {fam: planning_horizon_from_kges(kge_by_family[fam], cfg.planning_kge) for fam in families}
    advisory_h = {fam: planning_horizon_from_kges(kge_by_family[fam], cfg.advisory_kge) for fam in families}

    def rank_key(fam: str):
        k12 = kge_by_family[fam].get(12, np.nan)
        k12 = -1e9 if np.isnan(k12) else float(k12)
        return (int(planning_h[fam]), int(advisory_h[fam]), k12)

    selected = sorted(families, key=rank_key, reverse=True)[0]

    # Decision label
    is_baseline = (selected == "BASELINE_SEASONAL")

    if planning_h[selected] > 0:
        decision = "MODEL_PLANNING" if not is_baseline else "BASELINE_PLANNING"
        status = "OK"
        reason = ""
    elif advisory_h[selected] > 0:
        decision = "MODEL_ADVISORY" if not is_baseline else "BASELINE_ADVISORY"
        status = "WARN"
        reason = "ONLY_ADVISORY_GRADE"
    else:
        decision = "NOT_RELIABLE"
        status = "WARN"
        reason = "SKILL_BELOW_ADVISORY_THRESHOLD"

    # Location confidence
    loc_conf = "UNKNOWN"
    if {"lat", "lon"}.issubset(set(dfp.columns)):
        loc_conf = "HIGH"
        if dfp["lat"].isna().any() or dfp["lon"].isna().any():
            loc_conf = "LOW"

    meteo_source = ""
    if "source" in dfp.columns:
        try:
            meteo_source = str(dfp["source"].mode(dropna=True).iloc[0])
        except Exception:
            meteo_source = ""

    selected_model_type = ""
    if selected == "BASELINE_SEASONAL":
        selected_model_type = "seasonal_naive"
    else:
        selected_model_type = model_info[selected][1] if selected in model_info else ""

    metrics_row = {
        "point_id": point_id,
        "resource_type": resource_type,
        "unit": unit,
        "status": status,
        "reason": reason,
        "warnings": ";".join(sorted(set(warnings))) if warnings else "",
        "decision": decision,
        "selected_family": selected,
        "selected_model_type": selected_model_type,
        "planning_horizon": int(planning_h[selected]),
        "advisory_horizon": int(advisory_h[selected]),
        "planning_kge_threshold": float(cfg.planning_kge),
        "advisory_kge_threshold": float(cfg.advisory_kge),
        "cutoff_date": str(cutoff.date()),
        "last_observed_date": str(last_date.date()),
        "location_confidence": loc_conf,
        "meteo_source": meteo_source,
    }
    metrics_df = pd.DataFrame([metrics_row])

    # -----------------------------
    # Forecasts (FULL MONTHLY PATH 1..Hmax)
    # -----------------------------
    Hmax = int(max(cfg.horizons)) if cfg.horizons else 48

    forecasts: List[dict] = []
    origin = last_date

    if selected == "BASELINE_SEASONAL":
        path = seasonal_naive_path(dfp[["date", "value"]], origin, Hmax)
        for i, (dtt, yhat) in enumerate(path, start=1):
            forecasts.append(
                {
                    "point_id": point_id,
                    "resource_type": resource_type,
                    "unit": unit,
                    "family": selected,
                    "model_type": "seasonal_naive",
                    "date": str(_to_month_start(dtt).date()),
                    "horizon": int(i),
                    "y_forecast": float(yhat),
                    "is_key_horizon": bool(i in cfg.horizons),
                }
            )
    else:
        # retrain selected ML on ALL available history (up to last_date)
        final_model = None
        final_model_type = ""

        try:
            feats_all = make_features(dfp, selected)
            if len(feats_all) < 24:
                raise RuntimeError("Not enough rows to train final model")
            X_all = feats_all.drop(columns=["date", "delta_y"]).to_numpy()
            y_all = feats_all["delta_y"].to_numpy()
            final_model, final_model_type, _, _ = train_select_model(X_all, y_all, None, None)

            path = recursive_forecast_path_delta(
                model=final_model,
                history=dfp[["date", "value"]],
                exog_df=exog_df,
                start_date=origin,
                horizon_max=Hmax,
                family=selected,
            )
            for i, (dtt, yhat) in enumerate(path, start=1):
                forecasts.append(
                    {
                        "point_id": point_id,
                        "resource_type": resource_type,
                        "unit": unit,
                        "family": selected,
                        "model_type": str(final_model_type),
                        "date": str(_to_month_start(dtt).date()),
                        "horizon": int(i),
                        "y_forecast": float(yhat),
                        "is_key_horizon": bool(i in cfg.horizons),
                    }
                )
        except Exception:
            # fallback to baseline if final model fails
            path = seasonal_naive_path(dfp[["date", "value"]], origin, Hmax)
            for i, (dtt, yhat) in enumerate(path, start=1):
                forecasts.append(
                    {
                        "point_id": point_id,
                        "resource_type": resource_type,
                        "unit": unit,
                        "family": "BASELINE_SEASONAL_FALLBACK",
                        "model_type": "seasonal_naive",
                        "date": str(_to_month_start(dtt).date()),
                        "horizon": int(i),
                        "y_forecast": float(yhat),
                        "is_key_horizon": bool(i in cfg.horizons),
                    }
                )

    forecasts_df = pd.DataFrame(forecasts)

     # -----------------------------
    # NEW: Scenario forecasts (CMIP6 delta-change, year-month dependent)
    # Families map to SSPs:
    #   Favorable -> ssp126
    #   Base -> ssp245
    #   Unfavorable -> ssp585
    # -----------------------------
    scenario_rows: List[dict] = []

    try:
        needs_meteo_for_selected = selected in ("EXOG_ML", "EXOG_ML_DEMAND")
        if needs_meteo_for_selected and final_model is not None and (exog_df is not None) and (not exog_df.empty):
            # Extend exog to cover future months (so scenarios affect the whole horizon)
            exog_ext = extend_exog_with_monthly_climatology(
                exog_df=exog_df,
                origin=origin,
                horizon_max=Hmax,
                include_meteo=True,
                include_demand=(selected == "EXOG_ML_DEMAND"),
            )

            # Baseline is global from the input file; future starts after baseline end
            future_start_year = int(baseline_end_year) + 1
            if future_start_year <= int(future_end_year):
                provider = CMIP6IntakeDeltaProvider(
                    cache_dir=Path(cmip6_cache_dir),
                    models=None,
                    statistic="mean",
                    grid_method="nearest",
                    cat_url=None,
                )

                # point lat/lon (required for CMIP6 extraction)
                lat = float(dfp["lat"].iloc[0]) if "lat" in dfp.columns else np.nan
                lon = float(dfp["lon"].iloc[0]) if "lon" in dfp.columns else np.nan

                if not np.isnan(lat) and not np.isnan(lon):
                    for scen_name, ssp in [("Favorable", "ssp126"), ("Base", "ssp245"), ("Unfavorable", "ssp585")]:
                        deltas_df = provider.get_deltas(
                            lat=lat,
                            lon=lon,
                            family=scen_name,  # provider maps family -> ssp internally
                            baseline_start=baseline_start_year,
                            baseline_end=baseline_end_year,
                            future_start=future_start_year,
                            future_end=future_end_year,
                        )

                        exog_scen = apply_year_month_deltas_to_exog(
                            exog_future=exog_ext,
                            deltas=deltas_df,
                            date_col="date",
                            precip_col="precip_mm_month_est",
                            temp_col="t2m_c",
                            tmax_col="tmax_c",
                            tmin_col="tmin_c",
                        )

                        path_scen = recursive_forecast_path_delta(
                            model=final_model,
                            history=dfp[["date", "value"]],
                            exog_df=exog_scen,
                            start_date=origin,
                            horizon_max=Hmax,
                            family=selected,
                        )

                        for i, (dtt, yhat) in enumerate(path_scen, start=1):
                            scenario_rows.append(
                                {
                                    "point_id": point_id,
                                    "resource_type": resource_type,
                                    "unit": unit,
                                    "family": selected,
                                    "model_type": str(final_model_type),
                                    "scenario": scen_name,
                                    "ssp": ssp,
                                    "date": str(_to_month_start(dtt).date()),
                                    "horizon": int(i),
                                    "y_forecast": float(yhat),
                                }
                            )
    except Exception:
        # If scenario generation fails, keep it empty (do not break production run)
        pass

    forecasts_scenarios_df = pd.DataFrame(scenario_rows)

    return metrics_df, skill_df, forecasts_df, forecasts_scenarios_df


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
    # Global baseline from uploaded canonical file (dynamic)
    baseline_start_year = int(df["date"].min().year)
    baseline_end_year = int(df["date"].max().year)

    required = {"point_id", "date", "value", "resource_type", "unit"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required canonical columns: {sorted(missing)}")

    cfg = RunConfig(
        horizons=[12, 24, 36, 48],
        holdout_months=args.holdout_months,
        inner_val_months=args.inner_val_months,
    )

    all_metrics: List[pd.DataFrame] = []
    all_skill: List[pd.DataFrame] = []
    all_fc: List[pd.DataFrame] = []
    all_fc_scen: List[pd.DataFrame] = []

    for point_id, g in df.groupby("point_id", sort=True):
        m, s, f, fs = run_point(
            g,
            cfg,
            baseline_start_year=baseline_start_year,
            baseline_end_year=baseline_end_year,
            future_end_year=2050,
            cmip6_cache_dir=str(Path(args.outdir) / "cache" / "cmip6_deltas"),
        )

    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    skill_df = pd.concat(all_skill, ignore_index=True) if all_skill else pd.DataFrame()
    fc_df = pd.concat(all_fc, ignore_index=True) if all_fc else pd.DataFrame()

    # If no scenarios were generated, still create an empty DF with stable columns
    if all_fc_scen:
        forecasts_scenarios_all = pd.concat(all_fc_scen, ignore_index=True)
    else:
        forecasts_scenarios_all = pd.DataFrame(
            columns=[
                "point_id", "resource_type", "unit", "family", "model_type",
                "scenario", "ssp", "date", "horizon", "y_forecast"
            ]
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_path = outdir / "metrics_v0_6.csv"
    skill_path = outdir / "skill_v0_6.csv"
    fc_path = outdir / "forecasts_v0_6.csv"
    fc_scen_path = outdir / "forecasts_scenarios_v0_6.csv"

    metrics_df.to_csv(metrics_path, index=False)
    skill_df.to_csv(skill_path, index=False)
    fc_df.to_csv(fc_path, index=False)
    forecasts_scenarios_all.to_csv(fc_scen_path, index=False)

    print("BasinCast Core v0.6 OK")
    print(f"Metrics rows:            {len(metrics_df)} | Saved: {metrics_path}")
    print(f"Skill rows:              {len(skill_df)} | Saved: {skill_path}")
    print(f"Forecast rows:           {len(fc_df)} | Saved: {fc_path}")
    print(f"Forecast scenarios rows: {len(forecasts_scenarios_all)} | Saved: {fc_scen_path}")
    if len(metrics_df) > 0:
        print(metrics_df.head(1).to_csv(index=False).strip())