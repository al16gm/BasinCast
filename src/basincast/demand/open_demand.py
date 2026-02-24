# src/basincast/demand/open_demand.py
from __future__ import annotations

import hashlib
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------
# Dataset notes (paper-friendly)
# ---------------------------------------------------------------------
# Source: Harvard Dataverse (Tethys SSP/RCP demand / withdrawals)
# Access pattern: https://dataverse.harvard.edu/api/access/datafile/{FILE_ID}?format=original
# The ZIPs used here contain sectoral monthly withdrawals in km3/month (per grid cell).
# We build TOTAL demand as sum of the 6 sector CSVs.
#
# IMPORTANT: Units
# - input CSV values: km3/month (as suggested by filenames "*_km3permonth.csv")
# - output: hm3/month (1 km3 = 1000 hm3) to match typical Spanish basin accounting


# -----------------------------
# Defaults for your current case
# -----------------------------
DEFAULT_SCENARIO = "ssp1_rcp26"
DEFAULT_GCM = "gfdl"

# Your confirmed best candidates for withdrawals_sectors_monthly:
DEFAULT_FILE_ID_MONTHLY_1 = 6062173  # withdrawals_sectors_monthly_1.zip (gfdl, ssp1_rcp26)
DEFAULT_FILE_ID_MONTHLY_2 = 6062170  # withdrawals_sectors_monthly_2.zip (gfdl, ssp1_rcp26)

DEFAULT_ZIP_NAME_1 = f"{DEFAULT_SCENARIO}_{DEFAULT_GCM}_withdrawals_sectors_monthly_1.zip"
DEFAULT_ZIP_NAME_2 = f"{DEFAULT_SCENARIO}_{DEFAULT_GCM}_withdrawals_sectors_monthly_2.zip"

DEFAULT_CACHE_DIR = Path("outputs") / "cache" / "demand"

DATAVERSE_ACCESS_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"


@dataclass(frozen=True)
class OpenDemandConfig:
    scenario: str = DEFAULT_SCENARIO
    gcm: str = DEFAULT_GCM
    file_id_1: int = DEFAULT_FILE_ID_MONTHLY_1
    file_id_2: int = DEFAULT_FILE_ID_MONTHLY_2
    zip_name_1: str = DEFAULT_ZIP_NAME_1
    zip_name_2: str = DEFAULT_ZIP_NAME_2
    cache_dir: Path = DEFAULT_CACHE_DIR
    ssl_verify: bool = True  # keep True; corporate env should install certs if needed
    timeout_s: int = 120
    chunk_bytes: int = 1024 * 1024  # 1 MB


def _month_start(ts: Union[str, pd.Timestamp]) -> pd.Timestamp:
    t = pd.to_datetime(ts, errors="coerce")
    if pd.isna(t):
        raise ValueError(f"Cannot parse date: {ts}")
    return t.to_period("M").to_timestamp()


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    start = _month_start(start)
    end = _month_start(end)
    if end < start:
        return []
    return list(pd.date_range(start=start, end=end, freq="MS"))


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _snap_to_half_degree(x: float) -> float:
    return round(float(x) * 2.0) / 2.0


def _normalize_bbox_for_0p5deg_grid(
    bbox: Tuple[float, float, float, float],
    min_halfwidth: float = 0.26,
) -> Tuple[Tuple[float, float, float, float], Dict[str, str]]:
    """
    If bbox is very narrow (point-like), snap its center to the 0.5° grid and
    expand to a minimal window so at least one grid cell is included.
    Returns (bbox_effective, meta).
    """
    lat0, lon0, lat1, lon1 = bbox
    lat_min, lat_max = sorted([float(lat0), float(lat1)])
    lon_min, lon_max = sorted([float(lon0), float(lon1)])

    width_lat = lat_max - lat_min
    width_lon = lon_max - lon_min

    c_lat = 0.5 * (lat_min + lat_max)
    c_lon = 0.5 * (lon_min + lon_max)

    meta: Dict[str, str] = {
        "bbox_input": f"({lat_min},{lon_min},{lat_max},{lon_max})",
        "bbox_snapped": "false",
    }

    # If bbox is narrower than ~one grid step, we snap+expand
    if (width_lat < 0.75) and (width_lon < 0.75):
        s_lat = _snap_to_half_degree(c_lat)
        s_lon = _snap_to_half_degree(c_lon)

        half_lat = max(min_halfwidth, 0.5 * width_lat)
        half_lon = max(min_halfwidth, 0.5 * width_lon)

        bbox_eff = (s_lat - half_lat, s_lon - half_lon, s_lat + half_lat, s_lon + half_lon)
        meta.update(
            {
                "bbox_snapped": "true",
                "bbox_center_input": f"({c_lat},{c_lon})",
                "bbox_center_snapped": f"({s_lat},{s_lon})",
                "bbox_effective": f"{bbox_eff}",
            }
        )
        return bbox_eff, meta

    # Otherwise just ensure ordering and non-zero width if needed
    if width_lat == 0.0:
        lat_min -= min_halfwidth
        lat_max += min_halfwidth
    if width_lon == 0.0:
        lon_min -= min_halfwidth
        lon_max += min_halfwidth

    bbox_eff = (lat_min, lon_min, lat_max, lon_max)
    meta["bbox_effective"] = f"{bbox_eff}"
    return bbox_eff, meta

def _download_dataverse_file(
    file_id: int,
    dest_path: Path,
    ssl_verify: bool = True,
    timeout_s: int = 120,
    chunk_bytes: int = 1024 * 1024,
) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path

    url = DATAVERSE_ACCESS_URL.format(file_id=file_id)
    params = {"format": "original"}

    tmp = dest_path.with_suffix(dest_path.suffix + ".part")

    with requests.get(url, params=params, stream=True, timeout=timeout_s, verify=ssl_verify) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_bytes):
                if chunk:
                    f.write(chunk)

    tmp.replace(dest_path)
    return dest_path


def _extract_zip(zip_path: Path, extract_dir: Path) -> List[Path]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    out_paths: List[Path] = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            # keep only CSVs
            if not name.lower().endswith(".csv"):
                continue
            target = extract_dir / Path(name).name
            if not target.exists():
                z.extract(name, extract_dir)
                extracted = extract_dir / name
                # move to flat name if zip has subfolders
                if extracted.exists() and extracted.is_file() and extracted != target:
                    extracted.replace(target)
            if target.exists():
                out_paths.append(target)
    return out_paths


def _detect_lat_lon_cols(cols: List[str]) -> Tuple[Optional[str], Optional[str]]:
    lower = {c.lower(): c for c in cols}
    lat_candidates = ["lat", "latitude", "y"]
    lon_candidates = ["lon", "longitude", "x"]
    lat_col = next((lower[c] for c in lat_candidates if c in lower), None)
    lon_col = next((lower[c] for c in lon_candidates if c in lower), None)
    return lat_col, lon_col


def _try_parse_month_colname(c: str) -> Optional[pd.Timestamp]:
    # common patterns: "2010-01", "2010-01-01", "201001", "2010_01"
    s = c.strip()
    for fmt in ("%Y-%m", "%Y-%m-%d", "%Y%m", "%Y_%m"):
        try:
            t = pd.to_datetime(s, format=fmt)
            return t.to_period("M").to_timestamp()
        except Exception:
            pass
    # last resort
    t = pd.to_datetime(s, errors="coerce")
    if pd.isna(t):
        return None
    # if it looks like monthly
    return t.to_period("M").to_timestamp()


def _is_wide_monthly(cols: List[str]) -> Tuple[bool, Dict[str, pd.Timestamp]]:
    mapping: Dict[str, pd.Timestamp] = {}
    parsed = 0
    for c in cols:
        tt = _try_parse_month_colname(c)
        if tt is not None:
            mapping[c] = tt
            parsed += 1
    # heuristic: wide format has many month columns
    return (parsed >= 12), mapping


def _aggregate_csv_to_monthly_series(
    csv_path: Path,
    months_needed: List[pd.Timestamp],
    bbox: Optional[Tuple[float, float, float, float]],
    chunksize: int = 200_000,
) -> pd.Series:
    """
    Returns monthly total in km3/month (sum over selected grid cells / rows).
    Robust to wide or long CSV formats.
    """
    if not months_needed:
        return pd.Series(dtype=float)

    # Inspect header only
    header_cols = list(pd.read_csv(csv_path, nrows=0).columns)
    lat_col, lon_col = _detect_lat_lon_cols(header_cols)
    is_wide, wide_map = _is_wide_monthly(header_cols)

    months_set = set(months_needed)

    # bbox filter
    lat_min = lon_min = lat_max = lon_max = None
    if bbox is not None and lat_col and lon_col:
        lat_min, lon_min, lat_max, lon_max = bbox

    if is_wide:
        # Select only needed month columns + lat/lon if present
        month_cols = [c for c, t in wide_map.items() if t in months_set]
        usecols = month_cols.copy()
        if lat_col:
            usecols.append(lat_col)
        if lon_col:
            usecols.append(lon_col)

        acc = {m: 0.0 for m in months_needed}

        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
            if lat_col and lon_col and bbox is not None:
                chunk = chunk[
                    (chunk[lat_col] >= lat_min) & (chunk[lat_col] <= lat_max) &
                    (chunk[lon_col] >= lon_min) & (chunk[lon_col] <= lon_max)
                ]
            if chunk.empty:
                continue

            for c in month_cols:
                m = wide_map[c]
                v = pd.to_numeric(chunk[c], errors="coerce").fillna(0.0).sum()
                acc[m] += float(v)

        s = pd.Series(acc).sort_index()
        s.index.name = "date"
        s.name = "value_km3_per_month"
        return s

    # LONG format fallback
    # Identify time/value columns
    lower = {c.lower(): c for c in header_cols}
    time_col = None
    for cand in ["time", "date", "month", "timestamp"]:
        if cand in lower:
            time_col = lower[cand]
            break

    # sometimes year/month are split
    year_col = lower.get("year")
    mon_col = lower.get("mon") or lower.get("month")  # if "month" isn't the time col

    # value column candidates
    value_col = None
    for cand in ["value", "demand", "twd", "km3permonth"]:
        if cand in lower:
            value_col = lower[cand]
            break
    if value_col is None:
        # try last column heuristic
        value_col = header_cols[-1]

    usecols = []
    for c in [lat_col, lon_col, time_col, year_col, mon_col, value_col]:
        if c and c not in usecols:
            usecols.append(c)

    acc = {m: 0.0 for m in months_needed}

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        if lat_col and lon_col and bbox is not None:
            chunk = chunk[
                (chunk[lat_col] >= lat_min) & (chunk[lat_col] <= lat_max) &
                (chunk[lon_col] >= lon_min) & (chunk[lon_col] <= lon_max)
            ]
        if chunk.empty:
            continue

        if time_col and time_col in chunk.columns:
            t = pd.to_datetime(chunk[time_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
        elif year_col and mon_col and (year_col in chunk.columns) and (mon_col in chunk.columns):
            yr = pd.to_numeric(chunk[year_col], errors="coerce")
            mm = pd.to_numeric(chunk[mon_col], errors="coerce")
            t = pd.to_datetime(
                yr.astype("Int64").astype(str) + "-" + mm.astype("Int64").astype(str).str.zfill(2) + "-01",
                errors="coerce"
            ).dt.to_period("M").dt.to_timestamp()
        else:
            # cannot parse time -> skip chunk
            continue

        v = pd.to_numeric(chunk[value_col], errors="coerce").fillna(0.0)
        tmp = pd.DataFrame({"date": t, "v": v}).dropna(subset=["date"])
        if tmp.empty:
            continue
        tmp = tmp[tmp["date"].isin(months_set)]
        if tmp.empty:
            continue
        g = tmp.groupby("date", as_index=True)["v"].sum()
        for idx, val in g.items():
            acc[idx] += float(val)

    s = pd.Series(acc).sort_index()
    s.index.name = "date"
    s.name = "value_km3_per_month"
    return s


def build_total_demand_hm3_monthly(
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    scenario: str = DEFAULT_SCENARIO,
    gcm: str = DEFAULT_GCM,
    file_id_1: int = DEFAULT_FILE_ID_MONTHLY_1,
    file_id_2: int = DEFAULT_FILE_ID_MONTHLY_2,
    zip_name_1: str = DEFAULT_ZIP_NAME_1,
    zip_name_2: str = DEFAULT_ZIP_NAME_2,
    months_needed: Optional[List[pd.Timestamp]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    allow_download: bool = False,
    ssl_verify: bool = True,
    fail_soft: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Returns:
      - df_demand: columns [date, demand_hm3] (monthly start timestamps)
      - info: dict with provenance strings (paper-friendly)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    info: Dict[str, str] = {
        "source": "Harvard Dataverse (Tethys demand withdrawals sectors monthly, CSV inside ZIP)",
        "scenario": scenario,
        "gcm": gcm,
        "units_in": "km3/month",
        "units_out": "hm3/month",
    }

    # Normalize bbox to match 0.5° grid (prevents all-zero due to point-like bbox)
    if bbox is not None:
        bbox_eff, bbox_meta = _normalize_bbox_for_0p5deg_grid(bbox, min_halfwidth=0.26)
        info.update(bbox_meta)
        bbox = bbox_eff

    # Determine cache key
    bbox_str = "NONE" if bbox is None else f"{bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f}"
    months_str = "ALL" if not months_needed else f"{months_needed[0].strftime('%Y-%m')}_{months_needed[-1].strftime('%Y-%m')}"
    key = _sha1(f"{scenario}|{gcm}|{bbox_str}|{months_str}")
    cached_series_path = cache_dir / f"total_withdrawals_sectors_hm3_monthly_{key}.csv"

    if cached_series_path.exists():
        df = pd.read_csv(cached_series_path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        info["cache"] = str(cached_series_path)
        info["cache_hit"] = "true"
        return df, info

    info["cache_hit"] = "false"

    zip1 = cache_dir / zip_name_1
    zip2 = cache_dir / zip_name_2

    try:
        if (not zip1.exists()) or (zip1.stat().st_size == 0):
            if allow_download:
                _download_dataverse_file(file_id_1, zip1, ssl_verify=ssl_verify)
            else:
                raise FileNotFoundError(f"Missing ZIP: {zip1}")

        if (not zip2.exists()) or (zip2.stat().st_size == 0):
            if allow_download:
                _download_dataverse_file(file_id_2, zip2, ssl_verify=ssl_verify)
            else:
                raise FileNotFoundError(f"Missing ZIP: {zip2}")

        extract_dir = cache_dir / f"_extracted_{scenario}_{gcm}_withdrawals_sectors_monthly"
        files_1 = _extract_zip(zip1, extract_dir)
        files_2 = _extract_zip(zip2, extract_dir)
        csvs = sorted({p for p in (files_1 + files_2) if p.suffix.lower() == ".csv"})

        if not csvs:
            raise RuntimeError(f"No CSV files found after extracting {zip1.name} and {zip2.name}")

        # If months_needed not provided, we compute for all months present in CSVs
        # (but usually caller should pass overlap range).
        # Here we keep it strict: require months_needed to avoid huge work.
        if not months_needed:
            raise ValueError("months_needed is required to avoid processing the full global dataset.")

        total_km3 = pd.Series(0.0, index=pd.Index(months_needed, name="date"))

        for csv_path in csvs:
            s = _aggregate_csv_to_monthly_series(csv_path, months_needed=months_needed, bbox=bbox)
            # align
            s = s.reindex(total_km3.index).fillna(0.0)
            total_km3 = total_km3 + s

        total_hm3 = total_km3 * 1000.0  # km3 -> hm3
        df_out = total_hm3.reset_index()
        df_out.columns = ["date", "demand_hm3"]
        df_out["date"] = pd.to_datetime(df_out["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        df_out = df_out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        # save cache
        df_out.to_csv(cached_series_path, index=False)
        info["cache"] = str(cached_series_path)
        info["cache_written"] = "true"
        info["zip1"] = str(zip1)
        info["zip2"] = str(zip2)
        info["extracted_dir"] = str(extract_dir)

        return df_out, info

    except Exception as e:
        if fail_soft:
            info["error"] = f"{type(e).__name__}: {e}"
            # Return empty demand
            return pd.DataFrame(columns=["date", "demand_hm3"]), info
        raise


def integrate_total_demand_into_canonical(
    canonical_df: pd.DataFrame,
    demand_monthly_df: pd.DataFrame,
    date_col: str = "date",
    out_col: str = "demand_hm3",
) -> pd.DataFrame:
    """
    Left-join demand (monthly) into canonical table.
    canonical_df is expected to have a monthly date column already.
    """
    df = canonical_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    dem = demand_monthly_df.copy()
    if not dem.empty:
        dem["date"] = pd.to_datetime(dem["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        dem = dem.dropna(subset=["date"]).sort_values("date")
    else:
        # ensure the column exists (no crash downstream)
        df[out_col] = np.nan
        return df

    df = df.merge(dem[["date", "demand_hm3"]], left_on=date_col, right_on="date", how="left", suffixes=("", "_dem"))
    # drop duplicate merge key
    if "date_dem" in df.columns:
        df = df.drop(columns=["date_dem"])
    # enforce final name
    if "demand_hm3" in df.columns and out_col != "demand_hm3":
        df[out_col] = df["demand_hm3"]
        df = df.drop(columns=["demand_hm3"])
    return df


def integrate_open_total_demand_auto(
    canonical_df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    date_col: str = "date",
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    scenario: str = DEFAULT_SCENARIO,
    gcm: str = DEFAULT_GCM,
    file_id_1: int = DEFAULT_FILE_ID_MONTHLY_1,
    file_id_2: int = DEFAULT_FILE_ID_MONTHLY_2,
    allow_download: bool = False,
    ssl_verify: bool = True,
    fail_soft: bool = True,
    out_col: str = "demand_hm3",
    bbox_buffer_deg: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    High-level helper used by Streamlit.

    - Computes overlap months with canonical_df.
    - Computes bbox from canonical_df[lat/lon] if available.
    - Builds TOTAL demand (hm3/month) from cached ZIPs.
    - Integrates into canonical_df (replicated per point_id via merge on date).
    """
    df = canonical_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    if df.empty:
        return canonical_df.copy(), {"error": "canonical_df is empty after date parsing."}

    start = df[date_col].min()
    end = df[date_col].max()
    months_needed = _month_range(start, end)

    # Dataset often starts around 2010 in these products; we keep automatic overlap
    # but DO NOT enforce a hard start here (let the data decide).

    bbox = None
    if (lat_col in df.columns) and (lon_col in df.columns):
        lat = pd.to_numeric(df[lat_col], errors="coerce")
        lon = pd.to_numeric(df[lon_col], errors="coerce")
        if lat.notna().any() and lon.notna().any():
            lat_min = float(lat.min()) - float(bbox_buffer_deg)
            lat_max = float(lat.max()) + float(bbox_buffer_deg)
            lon_min = float(lon.min()) - float(bbox_buffer_deg)
            lon_max = float(lon.max()) + float(bbox_buffer_deg)
            bbox = (lat_min, lon_min, lat_max, lon_max)

    dem_df, info = build_total_demand_hm3_monthly(
        cache_dir=cache_dir,
        scenario=scenario,
        gcm=gcm,
        file_id_1=file_id_1,
        file_id_2=file_id_2,
        zip_name_1=f"{scenario}_{gcm}_withdrawals_sectors_monthly_1.zip",
        zip_name_2=f"{scenario}_{gcm}_withdrawals_sectors_monthly_2.zip",
        months_needed=months_needed,
        bbox=bbox,
        allow_download=allow_download,
        ssl_verify=ssl_verify,
        fail_soft=fail_soft,
    )

    info["bbox_used"] = "none" if bbox is None else f"{bbox}"
    info["months"] = f"{months_needed[0].strftime('%Y-%m')}..{months_needed[-1].strftime('%Y-%m')}"

    df_out = integrate_total_demand_into_canonical(df, dem_df, date_col=date_col, out_col=out_col)
    return df_out, info