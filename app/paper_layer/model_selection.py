# app/paper_layer/model_selection.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ---------- Metrics ----------

def kge(sim: np.ndarray, obs: np.ndarray) -> float:
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    m = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[m], obs[m]
    if len(sim) < 3:
        return np.nan
    mean_sim, mean_obs = np.mean(sim), np.mean(obs)
    std_sim, std_obs = np.std(sim, ddof=0), np.std(obs, ddof=0)
    if mean_obs == 0 or std_obs == 0:
        return np.nan
    r = np.corrcoef(sim, obs)[0, 1] if len(sim) > 1 else np.nan
    alpha = std_sim / std_obs if std_obs != 0 else np.nan
    beta = mean_sim / mean_obs if mean_obs != 0 else np.nan
    if not np.isfinite(r) or not np.isfinite(alpha) or not np.isfinite(beta):
        return np.nan
    return 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)

# ---------- Dates ----------

def to_month_start_series(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp()

def seasonal_naive_path(hist: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> np.ndarray:
    # hist: columns ["date","value"] monthly
    obs_map = dict(zip(hist["date"].tolist(), hist["value"].astype(float).tolist()))
    y0 = obs_map.get(origin, float(hist["value"].iloc[-1]))
    out = []
    last_y = y0
    for step in range(1, horizon + 1):
        target = (origin + pd.DateOffset(months=step)).to_period("M").to_timestamp()
        ref = (target - pd.DateOffset(months=12)).to_period("M").to_timestamp()
        if ref in obs_map and np.isfinite(obs_map[ref]):
            last_y = float(obs_map[ref])
        out.append(last_y)
    return np.array(out, dtype=float)

# ---------- Features (ENDO only for v0.13) ----------

def make_endo_features(df: pd.DataFrame, lags: List[int]) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["date"] = to_month_start_series(df["date"])
    df = df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)

    for L in lags:
        df[f"y_lag_{L}"] = df["value"].shift(L)

    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    X = df[[c for c in df.columns if c.startswith("y_lag_")] + ["month_sin", "month_cos"]]
    y = df["value"]
    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)

# ---------- Backtesting ----------

@dataclass
class SelectionConfig:
    horizons: List[int]
    lags: List[int] = (1, 2, 3, 6, 12)
    min_train: int = 36
    n_origins: int = 24  # last 24 months as rolling origins (if possible)

def _rolling_origins(dates: pd.Series, n_origins: int, min_train: int) -> List[pd.Timestamp]:
    unique_dates = pd.Series(dates.unique()).sort_values()
    if len(unique_dates) < (min_train + 1):
        return []
    # take last n_origins possible origins, leaving at least min_train before origin
    origins = []
    for i in range(len(unique_dates) - 1, -1, -1):
        origin = unique_dates.iloc[i]
        # number of points up to origin (inclusive)
        n_train = (unique_dates <= origin).sum()
        if n_train >= min_train:
            origins.append(origin)
        if len(origins) >= n_origins:
            break
    return sorted(origins)

def backtest_models_endo(df: pd.DataFrame, cfg: SelectionConfig) -> pd.DataFrame:
    """
    Returns skill table with columns:
    model, family, h, kge_mean, kge_std, fail_rate
    """
    df = df.copy()
    df["date"] = to_month_start_series(df["date"])
    df = df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)

    origins = _rolling_origins(df["date"], cfg.n_origins, cfg.min_train)
    if not origins:
        return pd.DataFrame()

    models = {
        "BayesianRidge": BayesianRidge(),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    # Precompute features aligned to df
    X_all, y_all = make_endo_features(df, list(cfg.lags))
    # We need mapping to dates; rebuild aligned frame
    df_feat = df.copy()
    for L in cfg.lags:
        df_feat[f"y_lag_{L}"] = df_feat["value"].shift(L)
    df_feat["month"] = df_feat["date"].dt.month
    df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month"] / 12.0)
    df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12.0)
    feat_cols = [f"y_lag_{L}" for L in cfg.lags] + ["month_sin", "month_cos"]
    df_feat = df_feat.dropna(subset=feat_cols + ["value"]).reset_index(drop=True)

    # For seasonal naive, we evaluate directly on original df (monthly)
    hist = df[["date", "value"]].copy()

    rows = []
    for model_name in ["seasonal_naive"] + list(models.keys()):
        family = "BASELINE" if model_name == "seasonal_naive" else "ENDO"
        per_h_scores: Dict[int, List[float]] = {h: [] for h in cfg.horizons}
        fails = 0
        total = 0

        for origin in origins:
            total += 1
            try:
                # training subset ends at origin (inclusive)
                train_mask = df_feat["date"] <= origin
                train = df_feat.loc[train_mask].copy()
                if len(train) < cfg.min_train:
                    raise ValueError("Not enough training rows after feature filtering.")

                if model_name == "seasonal_naive":
                    # produce predictions for each h using path and pick last step
                    Hmax = int(max(cfg.horizons))
                    path = seasonal_naive_path(hist, origin, Hmax)
                    # true values at each horizon
                    for h in cfg.horizons:
                        target_date = (origin + pd.DateOffset(months=int(h))).to_period("M").to_timestamp()
                        obs_row = hist.loc[hist["date"] == target_date, "value"]
                        if len(obs_row) == 0:
                            per_h_scores[h].append(np.nan)
                        else:
                            sim = np.array([path[int(h)-1]], dtype=float)  # step h -> index h-1
                            obs = np.array([float(obs_row.iloc[0])], dtype=float)
                            # KGE on single point is undefined -> store NaN
                            per_h_scores[h].append(np.nan)
                    continue

                # fit model
                m = models[model_name]
                X_train = train[feat_cols].values
                y_train = train["value"].values
                m.fit(X_train, y_train)

                # recursive forecast: step-by-step, updating lags with predictions
                # Start from last known row at origin within df_feat
                base_row = df_feat.loc[df_feat["date"] == origin]
                if len(base_row) == 0:
                    raise ValueError("Origin not found in feature-aligned table.")
                state = base_row.iloc[0].copy()

                preds = {}
                for step in range(1, int(max(cfg.horizons)) + 1):
                    # build feature vector from current state
                    x = np.array([state[c] for c in feat_cols], dtype=float).reshape(1, -1)
                    yhat = float(m.predict(x)[0])
                    preds[step] = yhat

                    # advance state to next month: update lags
                    next_date = (origin + pd.DateOffset(months=step)).to_period("M").to_timestamp()
                    # shift lag features: y_lag_1 becomes previous y, etc.
                    # easiest: rebuild lags from a buffer
                    # We'll keep a buffer of last max(lags) values
                    # For simplicity: store predicted values in a dict and recompute lags from history+preds
                    # (robust and leakage-free)
                    # Compute y at next_date- L from observed if <= origin else from preds
                    for L in cfg.lags:
                        ref_date = (next_date - pd.DateOffset(months=int(L))).to_period("M").to_timestamp()
                        if ref_date <= origin:
                            ref_val = hist.loc[hist["date"] == ref_date, "value"]
                            state[f"y_lag_{L}"] = float(ref_val.iloc[0]) if len(ref_val) else np.nan
                        else:
                            ref_step = int((ref_date - origin).days / 30.0)  # approximate; corrected below
                            # safer: compute month difference using periods
                            ref_step = int((ref_date.to_period("M") - origin.to_period("M")))
                            state[f"y_lag_{L}"] = float(preds.get(ref_step, np.nan))

                    state["date"] = next_date
                    month = int(pd.Timestamp(next_date).month)
                    state["month_sin"] = float(np.sin(2 * np.pi * month / 12.0))
                    state["month_cos"] = float(np.cos(2 * np.pi * month / 12.0))

                # score for each horizon against observed (vector KGE needs >2 points -> we score across origins later)
                for h in cfg.horizons:
                    target_date = (origin + pd.DateOffset(months=int(h))).to_period("M").to_timestamp()
                    obs_row = hist.loc[hist["date"] == target_date, "value"]
                    if len(obs_row) == 0:
                        per_h_scores[h].append(np.nan)
                    else:
                        per_h_scores[h].append(float(preds[int(h)]))

            except Exception:
                fails += 1
                for h in cfg.horizons:
                    per_h_scores[h].append(np.nan)

        # Convert per-h predictions into KGE across origins (now we have a vector per horizon)
        for h in cfg.horizons:
            # obs for each origin at horizon h
            obs_vec = []
            sim_vec = []
            for i, origin in enumerate(origins):
                target_date = (origin + pd.DateOffset(months=int(h))).to_period("M").to_timestamp()
                obs_row = hist.loc[hist["date"] == target_date, "value"]
                obs = float(obs_row.iloc[0]) if len(obs_row) else np.nan
                sim = per_h_scores[h][i]
                obs_vec.append(obs)
                sim_vec.append(sim)
            score = kge(np.array(sim_vec), np.array(obs_vec))
            rows.append({
                "model": model_name,
                "family": family,
                "h": int(h),
                "kge": float(score) if np.isfinite(score) else np.nan,
                "fail_rate": fails / max(total, 1),
                "n_origins": len(origins),
            })

    return pd.DataFrame(rows)

def select_winner(skill_df: pd.DataFrame, horizons_focus: Tuple[int,int]=(1,12)) -> pd.DataFrame:
    """
    Produces leaderboard with aggregate metrics:
    - mean KGE over horizons in focus
    - reliability horizons based on thresholds (computed elsewhere, can be added later)
    """
    if skill_df is None or skill_df.empty:
        return pd.DataFrame()

    hmin, hmax = horizons_focus
    df = skill_df.copy()
    df_focus = df[(df["h"] >= hmin) & (df["h"] <= hmax)].copy()

    agg = (
        df_focus.groupby(["model","family"], as_index=False)
        .agg(kge_mean=("kge","mean"), kge_std=("kge","std"), fail_rate=("fail_rate","max"), n_origins=("n_origins","max"))
        .sort_values(["kge_mean","fail_rate"], ascending=[False, True])
        .reset_index(drop=True)
    )
    agg["rank"] = np.arange(1, len(agg)+1)
    return agg