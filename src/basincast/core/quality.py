from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityConfig:
    planning_kge_threshold: float = 0.6
    advisory_kge_threshold: float = 0.3
    min_pairs_per_horizon: int = 24
    non_negative: bool = True


def to_month_start(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, errors="coerce")
    return d.dt.to_period("M").dt.to_timestamp()


def clean_monthly_series(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """
    Cleans a single point series:
      - forces month-start dates
      - resolves duplicate dates by mean(value)
      - sorts chronologically
      - reports missing months count (does NOT fill)
    Expected columns: date, value
    """
    warnings: List[str] = []
    out = df.copy()

    out["date"] = to_month_start(out["date"])
    out = out.dropna(subset=["date"]).copy()

    # Coerce value numeric (comma/dot)
    if "value" in out.columns:
        out["value"] = (
            out["value"]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )
        out["value"] = pd.to_numeric(out["value"], errors="coerce")

    out = out.dropna(subset=["value"]).copy()
    out = out.sort_values("date")

    # Handle duplicates by averaging
    if out["date"].duplicated().any():
        warnings.append("DUPLICATE_DATES_AVERAGED")
        out = out.groupby("date", as_index=False)["value"].mean()
        out = out.sort_values("date")

    # Missing months diagnostic
    if len(out) >= 2:
        full_idx = pd.date_range(out["date"].min(), out["date"].max(), freq="MS")
        missing = len(full_idx) - out["date"].nunique()
        if missing > 0:
            warnings.append(f"MISSING_MONTHS={missing}")

    return out, warnings


def horizons_supported_by_holdout(holdout_months: int, horizons: List[int], min_pairs: int) -> tuple[List[int], List[int]]:
    """
    For each horizon h, n_pairs = holdout_months - h.
    Supported if n_pairs >= min_pairs.
    Returns (supported, unsupported)
    """
    supported, unsupported = [], []
    for h in horizons:
        n_pairs = holdout_months - int(h)
        if n_pairs >= min_pairs:
            supported.append(int(h))
        else:
            unsupported.append(int(h))
    return supported, unsupported


def compute_grade_horizons(
    skill_df: pd.DataFrame,
    family: str,
    qc: QualityConfig,
    horizons: List[int],
) -> tuple[int, int]:
    """
    Returns:
      planning_horizon: max h with KGE>=planning threshold and enough pairs
      advisory_horizon: max h with KGE>=advisory threshold and enough pairs
    """
    if skill_df.empty:
        return 0, 0

    fam = skill_df[skill_df["family"] == family].copy()
    if fam.empty:
        return 0, 0

    planning = 0
    advisory = 0

    for h in sorted(horizons):
        row = fam[fam["horizon"] == int(h)]
        if row.empty:
            continue
        n_pairs = int(row["n_pairs"].iloc[0])
        kgeh = row["kge"].iloc[0]
        if n_pairs < qc.min_pairs_per_horizon or not np.isfinite(kgeh):
            continue
        if float(kgeh) >= qc.advisory_kge_threshold:
            advisory = int(h)
        if float(kgeh) >= qc.planning_kge_threshold:
            planning = int(h)

    return planning, advisory


def choose_decision(
    ml_planning: int,
    ml_advisory: int,
    bl_planning: int,
    bl_advisory: int,
) -> tuple[str, str]:
    """
    Returns (decision, selected_family)
    decision in:
      MODEL_PLANNING, BASELINE_PLANNING, MODEL_ADVISORY, BASELINE_ADVISORY, NOT_RELIABLE
    """
    if ml_planning > 0:
        return "MODEL_PLANNING", "ENDO_ML"
    if bl_planning > 0:
        return "BASELINE_PLANNING", "BASELINE_SEASONAL"
    if ml_advisory > 0:
        return "MODEL_ADVISORY", "ENDO_ML"
    if bl_advisory > 0:
        return "BASELINE_ADVISORY", "BASELINE_SEASONAL"
    return "NOT_RELIABLE", "NONE"


def seasonal_baseline_value(history: pd.DataFrame, start_date: pd.Timestamp, horizon: int) -> float:
    """
    Seasonal persistence baseline:
      y(t+h) = y(t + h - 12) for h<=12, else repeat 12-month cycle recursively.
    If missing, fallback to last observed.
    """
    h = int(horizon)
    hist = history.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    hist = hist.dropna(subset=["date", "value"]).sort_values("date")
    if hist.empty:
        return float("nan")

    hist_map = dict(zip(hist["date"].tolist(), hist["value"].astype(float).tolist()))
    last_obs = float(hist["value"].iloc[-1])

    def rec(k: int) -> float:
        if k <= 12:
            d = (pd.Timestamp(start_date) + pd.DateOffset(months=k - 12)).to_period("M").to_timestamp()
            v = hist_map.get(d, last_obs)
            return float(v)
        return rec(k - 12)

    return rec(h)


def baseline_forecast(history: pd.DataFrame, start_date: pd.Timestamp, horizons: List[int], non_negative: bool = True) -> pd.DataFrame:
    rows = []
    for h in horizons:
        d = (pd.Timestamp(start_date) + pd.DateOffset(months=int(h))).to_period("M").to_timestamp()
        y = float(seasonal_baseline_value(history, pd.Timestamp(start_date), int(h)))
        if non_negative and np.isfinite(y):
            y = max(0.0, y)
        rows.append({"date": d, "horizon": int(h), "y_forecast": y})
    return pd.DataFrame(rows)