from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from basincast.meteo.power import PowerMonthlyConfig, fetch_power_monthly

METEO_COLS = ["precip_mm_month_est", "t2m_c", "tmax_c", "tmin_c"]


def to_month_start(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp()


def coerce_float_series(s: pd.Series) -> pd.Series:
    """
    Robust numeric parsing for European formats:
      - "1,23" -> 1.23
      - "1.234,56" -> 1234.56
      - "1 234,56" -> 1234.56
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    x = s.astype(str).str.strip()

    # remove spaces used as thousand separators
    x = x.str.replace(" ", "", regex=False)

    # if both '.' and ',' exist -> assume '.' thousands and ',' decimal
    both = x.str.contains(r"\.") & x.str.contains(r",")
    x.loc[both] = x.loc[both].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)

    # if only ',' exists -> decimal comma
    only_comma = x.str.contains(",") & ~x.str.contains(r"\.")
    x.loc[only_comma] = x.loc[only_comma].str.replace(",", ".", regex=False)

    return pd.to_numeric(x, errors="coerce")


def canonical_has_meteo(df: pd.DataFrame) -> bool:
    return all((c in df.columns) for c in METEO_COLS) and df[METEO_COLS].notna().any().any()


def _guess_col(cols: List[str], candidates: List[str]) -> str:
    lc = {c.lower(): c for c in cols}
    for cand in candidates:
        for c in cols:
            if cand in c.lower():
                return c
        if cand.lower() in lc:
            return lc[cand.lower()]
    return cols[0] if cols else ""


def read_table_from_upload(upload) -> pd.DataFrame:
    name = (upload.name or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(upload)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(upload)
    # fallback
    return pd.read_csv(upload)


@dataclass(frozen=True)
class MeteoMapping:
    date_col: str
    precip_col: str
    t2m_col: str
    tmax_col: str
    tmin_col: str
    point_id_col: str = ""  # optional


def build_user_meteo_table(
    raw: pd.DataFrame,
    canonical_point_ids: List[str],
    mapping: MeteoMapping,
    mode: str,
    single_point_id: str = "",
) -> pd.DataFrame:
    """
    mode:
      - "HAS_POINT_ID": raw contains point_id column
      - "SINGLE_POINT": apply all rows to one selected point
      - "COMMON_ALL": apply same meteo to every point_id (replicate)
    """
    df = raw.copy()

    df["date"] = to_month_start(df[mapping.date_col])

    df["precip_mm_month_est"] = coerce_float_series(df[mapping.precip_col])
    df["t2m_c"] = coerce_float_series(df[mapping.t2m_col])
    df["tmax_c"] = coerce_float_series(df[mapping.tmax_col])
    df["tmin_c"] = coerce_float_series(df[mapping.tmin_col])

    out = df[["date"] + METEO_COLS].dropna(subset=["date"]).copy()
    out["source"] = "USER_UPLOAD"

    if mode == "HAS_POINT_ID":
        if not mapping.point_id_col:
            raise ValueError("point_id_col is required for HAS_POINT_ID")
        out["point_id"] = df[mapping.point_id_col].astype(str)
        out = out.dropna(subset=["point_id"])
        return out[["point_id", "date"] + METEO_COLS + ["source"]]

    if mode == "SINGLE_POINT":
        if not single_point_id:
            raise ValueError("single_point_id is required for SINGLE_POINT")
        out["point_id"] = str(single_point_id)
        return out[["point_id", "date"] + METEO_COLS + ["source"]]

    # COMMON_ALL: replicate to all points
    reps = []
    for pid in canonical_point_ids:
        tmp = out.copy()
        tmp["point_id"] = str(pid)
        reps.append(tmp)
    return pd.concat(reps, ignore_index=True)[["point_id", "date"] + METEO_COLS + ["source"]]


def ensure_latlon_per_point(canonical: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per point_id with lat/lon (mode), used for NASA fetch.
    """
    need = {"point_id", "lat", "lon"}
    miss = need - set(canonical.columns)
    if miss:
        raise ValueError(f"Canonical missing required columns for NASA fetch: {sorted(miss)}")

    g = canonical.copy()
    g["point_id"] = g["point_id"].astype(str)
    latlon = (
        g.dropna(subset=["lat", "lon"])
         .groupby("point_id", as_index=False)[["lat", "lon"]]
         .agg(lambda s: s.mode(dropna=True).iloc[0] if len(s.mode(dropna=True)) else s.dropna().iloc[0])
    )
    return latlon


def fetch_nasa_power_for_canonical(canonical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch monthly meteo from NASA POWER for each point_id (using lat/lon),
    for the date-span of the canonical.
    Returns:
      - meteo_df (point_id,date,meteo...,source)
      - canonical_with_meteo (merged)
    """
    df = canonical.copy()
    df["point_id"] = df["point_id"].astype(str)
    df["date"] = to_month_start(df["date"])

    start_year = int(pd.to_datetime(df["date"].min()).year)
    end_year = int(pd.to_datetime(df["date"].max()).year)

    latlon = ensure_latlon_per_point(df)
    if latlon.empty:
        raise ValueError("No valid lat/lon found for any point_id.")

    meteo_all = []
    for _, row in latlon.iterrows():
        pid = str(row["point_id"])
        lat = float(row["lat"])
        lon = float(row["lon"])
        met = fetch_power_monthly(
            lat=lat,
            lon=lon,
            start_year=start_year,
            end_year=end_year,
            cfg=PowerMonthlyConfig(),
        )
        met["point_id"] = pid
        meteo_all.append(met)

    meteo_df = pd.concat(meteo_all, ignore_index=True)
    meteo_df["date"] = to_month_start(meteo_df["date"])

    merged = df.merge(
        meteo_df[["point_id", "date"] + METEO_COLS + ["source"]],
        on=["point_id", "date"],
        how="left",
    )
    return meteo_df, merged