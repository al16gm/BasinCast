from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd

from basincast.translator.parsing import parse_date_series, parse_numeric_series, coerce_month_start

METEO_COLS = ["precip_mm_month_est", "t2m_c", "tmax_c", "tmin_c"]


@dataclass(frozen=True)
class MeteoMapping:
    date_col: str
    precip_col: str
    t2m_col: str
    tmax_col: str
    tmin_col: str
    point_id_col: str = ""  # optional


def to_month_start(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp()


def canonical_has_meteo(canonical: pd.DataFrame) -> bool:
    return all(c in canonical.columns for c in METEO_COLS)


def read_table_from_upload(upload) -> pd.DataFrame:
    name = getattr(upload, "name", "")
    if str(name).lower().endswith(".csv"):
        return pd.read_csv(upload)
    return pd.read_excel(upload)


def _aggregate_meteo_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregation rules:
      precip -> SUM
      t2m    -> MEAN
      tmax   -> MAX
      tmin   -> MIN
    """
    g = df.copy()
    g["date"] = to_month_start(g["date"])

    def first_valid(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else np.nan

    agg = {
        "precip_mm_month_est": "sum",
        "t2m_c": "mean",
        "tmax_c": "max",
        "tmin_c": "min",
        "source": first_valid,
    }

    out = g.groupby(["point_id", "date"], as_index=False).agg(agg)
    out = out.sort_values(["point_id", "date"]).reset_index(drop=True)
    return out


def build_user_meteo_table(
    raw: pd.DataFrame,
    canonical_point_ids: List[str],
    mapping: MeteoMapping,
    mode: str,
    single_point_id: str = "",
) -> pd.DataFrame:
    """
    Builds a canonical meteo table with monthly rows:
      point_id, date, precip_mm_month_est, t2m_c, tmax_c, tmin_c, source

    mode:
      - HAS_POINT_ID: raw has point_id column
      - SINGLE_POINT: raw applies to a single point_id (selected by user)
      - COMMON_ALL: raw applies to all point_ids equally
    """
    df = raw.copy()

    # Parse dates robustly
    date_parsed, _ = parse_date_series(df[mapping.date_col], date_hint="auto")
    df["_date_raw"] = pd.to_datetime(date_parsed, errors="coerce")
    df["date"] = coerce_month_start(df["_date_raw"])

    # Parse numeric columns robustly (accept comma/dot)
    p, _ = parse_numeric_series(df[mapping.precip_col], locale_hint="auto")
    t2m, _ = parse_numeric_series(df[mapping.t2m_col], locale_hint="auto")
    tmax, _ = parse_numeric_series(df[mapping.tmax_col], locale_hint="auto")
    tmin, _ = parse_numeric_series(df[mapping.tmin_col], locale_hint="auto")

    df["precip_mm_month_est"] = p
    df["t2m_c"] = t2m
    df["tmax_c"] = tmax
    df["tmin_c"] = tmin
    df["source"] = "USER_UPLOAD"

    if mode == "HAS_POINT_ID":
        if not mapping.point_id_col:
            raise ValueError("mode=HAS_POINT_ID requires mapping.point_id_col")
        df["point_id"] = df[mapping.point_id_col].astype(str).str.strip()

    elif mode == "SINGLE_POINT":
        if not single_point_id:
            raise ValueError("mode=SINGLE_POINT requires single_point_id")
        df["point_id"] = str(single_point_id)

    elif mode == "COMMON_ALL":
        # We'll aggregate first as a single series, then replicate to all points
        df["point_id"] = "COMMON_ALL"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Keep only rows where we have a month
    df = df.dropna(subset=["date"]).copy()

    # Aggregate to monthly (handles daily/irregular automatically)
    met = _aggregate_meteo_monthly(df[["point_id", "date"] + METEO_COLS + ["source"]])

    if mode == "COMMON_ALL":
        # replicate monthly meteo to all canonical points
        met_base = met.drop(columns=["point_id"]).copy()
        out_all = []
        for pid in canonical_point_ids:
            tmp = met_base.copy()
            tmp["point_id"] = str(pid)
            out_all.append(tmp)
        met = pd.concat(out_all, ignore_index=True).sort_values(["point_id", "date"]).reset_index(drop=True)

    # Filter to canonical point ids (safety)
    met = met[met["point_id"].isin([str(x) for x in canonical_point_ids])].copy()

    return met


def fetch_nasa_power_for_canonical(canonical_df: pd.DataFrame):
    """
    This function should already exist in your project and call NASA POWER.
    We keep the signature used by translator_app.
    """
    from basincast.meteo.power import PowerMonthlyConfig, fetch_power_monthly  # existing module

    c = canonical_df.copy()
    c["date"] = to_month_start(c["date"])
    c["point_id"] = c["point_id"].astype(str)

    # One lat/lon per point
    pts = (
        c.dropna(subset=["lat", "lon"])
        .groupby("point_id", as_index=False)[["lat", "lon"]]
        .first()
    )

    if pts.empty:
        raise ValueError("No valid lat/lon found in canonical_df.")

    # Determine year span from canonical
    years = pd.to_datetime(c["date"], errors="coerce").dt.year.dropna().astype(int)
    start_year = int(years.min())
    end_year = int(years.max())

    all_rows = []
    for _, row in pts.iterrows():
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
        all_rows.append(met)

    meteo_df = pd.concat(all_rows, ignore_index=True)
    meteo_df["date"] = to_month_start(meteo_df["date"])
    meteo_df["source"] = "NASA_POWER_MONTHLY"

    canonical_with_meteo = c.merge(
        meteo_df[["point_id", "date"] + METEO_COLS + ["source"]],
        on=["point_id", "date"],
        how="left",
    )

    return meteo_df, canonical_with_meteo