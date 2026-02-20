from __future__ import annotations

import numpy as np
import pandas as pd


def detect_meteo_columns(df: pd.DataFrame) -> dict:
    """
    Robust autodetection of precipitation and temperature-like columns.
    Prefers canonical BasinCast names when available.
    """
    cols = set(df.columns)

    precip_candidates = [
        "precip_mm_month_est", "precip_mm", "pr", "precip", "precipitation"
    ]
    temp_candidates = [
        "t2m_c", "tas", "temp_c", "temperature_c"
    ]
    tmax_candidates = ["tmax_c", "tasmax", "tmax"]
    tmin_candidates = ["tmin_c", "tasmin", "tmin"]

    def pick(cands):
        for c in cands:
            if c in cols:
                return c
        return None

    return {
        "precip": pick(precip_candidates),
        "temp": pick(temp_candidates),
        "tmax": pick(tmax_candidates),
        "tmin": pick(tmin_candidates),
    }


def apply_year_month_deltas_to_exog(
    exog_future: pd.DataFrame,
    deltas: pd.DataFrame,
    date_col: str = "date",
    precip_col: str | None = None,
    temp_col: str | None = None,
    tmax_col: str | None = None,
    tmin_col: str | None = None,
) -> pd.DataFrame:
    """
    Apply year-month deltas to exogenous future dataframe.

    Parameters
    ----------
    exog_future:
        DataFrame with a monthly date column and meteo columns.
    deltas:
        DataFrame with columns: year, month, delta_temp_add, delta_precip_mult
    """
    if exog_future is None or exog_future.empty:
        return exog_future
    if deltas is None or deltas.empty:
        return exog_future

    df = exog_future.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    mapping = detect_meteo_columns(df)
    precip_col = precip_col or mapping["precip"]
    temp_col = temp_col or mapping["temp"]
    tmax_col = tmax_col or mapping["tmax"]
    tmin_col = tmin_col or mapping["tmin"]

    deltas2 = deltas.copy()
    deltas2["year"] = deltas2["year"].astype(int)
    deltas2["month"] = deltas2["month"].astype(int)

    df["year"] = df[date_col].dt.year.astype(int)
    df["month"] = df[date_col].dt.month.astype(int)

    df = df.merge(deltas2, on=["year", "month"], how="left")

    # Temperature additive deltas
    for c in [temp_col, tmax_col, tmin_col]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float) + df["delta_temp_add"].fillna(0.0).astype(float)

    # Precip multiplicative deltas: x * (1 + mult)
    if precip_col and precip_col in df.columns:
        mult = df["delta_precip_mult"].fillna(0.0).astype(float)
        df[precip_col] = pd.to_numeric(df[precip_col], errors="coerce").astype(float) * (1.0 + mult)

    df = df.drop(columns=["year", "month", "delta_temp_add", "delta_precip_mult"], errors="ignore")
    return df