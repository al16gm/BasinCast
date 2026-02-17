# basincast/demand/open_demand.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, Literal, List

import numpy as np
import pandas as pd

Spatial = Literal["bbox_sum", "bbox_mean"]


def _to_month_start(x) -> pd.Timestamp:
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        return ts
    return ts.to_period("M").to_timestamp()


def _series_to_month_start(s: pd.Series) -> pd.Series:
    s2 = pd.to_datetime(s, errors="coerce")
    return s2.dt.to_period("M").dt.to_timestamp()


def _pick_lat_lon_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    lat_col = "lat" if "lat" in df.columns else ("latitude" if "latitude" in df.columns else None)
    lon_col = "lon" if "lon" in df.columns else ("longitude" if "longitude" in df.columns else None)
    return lat_col, lon_col


def _choose_data_var(ds, var_hint: Optional[str] = None) -> str:
    if var_hint and var_hint in ds.data_vars:
        return var_hint

    keys = list(ds.data_vars.keys())
    if not keys:
        raise ValueError("NetCDF contains no data variables.")

    scored = []
    for k in keys:
        kl = k.lower()
        score = 0
        if "demand" in kl:
            score += 5
        if "withdraw" in kl or "withdrawal" in kl:
            score += 5
        if "water" in kl:
            score += 2
        if "total" in kl:
            score += 2
        if "month" in kl or "monthly" in kl:
            score += 1
        scored.append((score, k))
    scored.sort(reverse=True, key=lambda t: t[0])
    return scored[0][1]


def _find_name(da, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in da.coords:
            return c
    for c in candidates:
        if c in da.dims:
            return c
    return None


def integrate_open_total_demand_auto(
    canonical: pd.DataFrame,
    cache_dir: str,
    spatial: Spatial = "bbox_sum",
    var_hint: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    SAFE: never crashes Streamlit.

    Reads ONLY a cached file:
      <cache_dir>/open_demand_0p5deg_monthly.nc

    If missing/unreadable → returns canonical with demand=NaN + info['ok']='false'
    If ok → merges a monthly total demand series by 'date' (same demand series for all points)
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    nc_path = cache / "open_demand_0p5deg_monthly.nc"

    out = canonical.copy()
    if "date" not in out.columns:
        out["demand"] = np.nan
        return out, {"ok": "false", "message": "Canonical has no 'date' column.", "cached_netcdf": str(nc_path)}

    out["date"] = _series_to_month_start(out["date"])
    out["demand"] = np.nan

    info: Dict[str, str] = {
        "ok": "false",
        "message": "",
        "cached_netcdf": str(nc_path),
        "spatial": str(spatial),
        "variable": "",
        "usable_start": "",
        "usable_end": "",
        "source": "USER_PROVIDED_NETCDF (cached)",
    }

    if not nc_path.exists():
        info["message"] = (
            f"Open-demand NetCDF not found in cache: {nc_path}. "
            "Upload it once (Option B). Continuing without demand."
        )
        return out, info

    # Canonical overlap window
    hist_min = pd.to_datetime(out["date"].min(), errors="coerce")
    hist_max = pd.to_datetime(out["date"].max(), errors="coerce")
    if pd.isna(hist_min) or pd.isna(hist_max):
        info["message"] = "Canonical dates cannot be parsed. Continuing without demand."
        return out, info

    try:
        try:
            import xarray as xr  # optional dependency
        except Exception as e:
            info["message"] = (
                "Open-demand NetCDF exists, but xarray is not installed. "
                "Install: `pip install xarray netCDF4` or use Option A (upload CSV). "
                "Continuing without demand."
            )
            return out, info

        ds = xr.open_dataset(nc_path)
        varname = _choose_data_var(ds, var_hint=var_hint)
        da = ds[varname]

        time_name = _find_name(da, ["time", "Time", "t"])
        lat_name = _find_name(da, ["lat", "latitude", "y"])
        lon_name = _find_name(da, ["lon", "longitude", "x"])

        if time_name is None:
            info["message"] = f"NetCDF variable '{varname}' has no time coordinate. Continuing without demand."
            info["variable"] = varname
            return out, info

        sub = da

        # Optional bbox around points if both canonical and NetCDF have lat/lon
        lat_col, lon_col = _pick_lat_lon_cols(out)
        use_bbox = (lat_col and lon_col and lat_name and lon_name)

        if use_bbox:
            pts = out[[lat_col, lon_col]].dropna().drop_duplicates()
            if not pts.empty:
                lat_min, lat_max = float(pts[lat_col].min()), float(pts[lat_col].max())
                lon_min, lon_max = float(pts[lon_col].min()), float(pts[lon_col].max())

                try:
                    sub = sub.sortby(lat_name)
                except Exception:
                    pass
                try:
                    sub = sub.sortby(lon_name)
                except Exception:
                    pass

                sub = sub.sel(
                    {
                        lat_name: slice(min(lat_min, lat_max), max(lat_min, lat_max)),
                        lon_name: slice(min(lon_min, lon_max), max(lon_min, lon_max)),
                    }
                )

        # Aggregate to a single monthly series
        spatial_dims = [d for d in sub.dims if d != time_name]
        if spatial_dims:
            if spatial == "bbox_mean":
                ts_da = sub.mean(dim=spatial_dims, skipna=True)
            else:
                ts_da = sub.sum(dim=spatial_dims, skipna=True)
        else:
            ts_da = sub

        # Convert to pandas
        tvals = pd.to_datetime(ts_da[time_name].values, errors="coerce")
        yvals = np.asarray(ts_da.values, dtype=float)

        df_dem = pd.DataFrame({"date": tvals, "demand": yvals})
        df_dem["date"] = _series_to_month_start(df_dem["date"])
        df_dem["demand"] = pd.to_numeric(df_dem["demand"], errors="coerce")
        df_dem = df_dem.dropna(subset=["date", "demand"]).copy()
        df_dem = df_dem.groupby("date", as_index=False)["demand"].mean().sort_values("date")

        # Clip to canonical overlap
        hist_min_m = _to_month_start(hist_min)
        hist_max_m = _to_month_start(hist_max)
        df_dem = df_dem[(df_dem["date"] >= hist_min_m) & (df_dem["date"] <= hist_max_m)].copy()

        if df_dem.empty:
            info["message"] = (
                f"Open-demand NetCDF loaded ('{varname}') but no overlap with "
                f"[{hist_min_m.date()} – {hist_max_m.date()}]. Continuing without demand."
            )
            info["variable"] = varname
            return out, info

        merged = out.merge(df_dem, on="date", how="left")
        merged["demand"] = pd.to_numeric(merged["demand"], errors="coerce")

        info["ok"] = "true"
        info["variable"] = varname
        info["usable_start"] = str(pd.to_datetime(df_dem["date"].min()).date())
        info["usable_end"] = str(pd.to_datetime(df_dem["date"].max()).date())
        info["message"] = f"Open demand integrated from cache using variable '{varname}'."

        return merged, info

    except Exception as e:
        info["message"] = f"Open-demand integration failed safely ({type(e).__name__}: {e}). Continuing without demand."
        return out, info