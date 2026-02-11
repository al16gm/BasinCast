from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests


POWER_MONTHLY_POINT_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"


@dataclass(frozen=True)
class PowerParams:
    """
    Default parameter short-names for NASA POWER monthly point API.

    Notes
    -----
    - Units depend on `community` (AG/RE/SB). Keep raw columns and
      compute derived columns explicitly for clarity. :contentReference[oaicite:4]{index=4}
    """
    precip: str = "PRECTOTCORR"  # precipitation (community-dependent units)
    t2m: str = "T2M"             # mean temperature (°C in many community outputs)
    t2m_max: str = "T2M_MAX"
    t2m_min: str = "T2M_MIN"


def _to_year(x: str | datetime | pd.Timestamp) -> int:
    dt = pd.to_datetime(x, errors="raise")
    return int(dt.year)


def _safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _cache_key(lat: float, lon: float, start_year: int, end_year: int, community: str, params: Iterable[str]) -> str:
    # rounded coords to avoid cache explosion with tiny diffs
    lat_r = round(lat, 4)
    lon_r = round(lon, 4)
    p = "_".join(list(params))
    return f"power_monthly_{community}_lat{lat_r}_lon{lon_r}_{start_year}_{end_year}_{p}.csv"


def fetch_power_monthly_point(
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
    community: str = "AG",
    params: Optional[Iterable[str]] = None,
    cache_dir: str | Path = ".cache/power",
    timeout_s: int = 60,
    sleep_s: float = 0.0,
) -> pd.DataFrame:
    """
    Fetch monthly meteo data from NASA POWER (point API) for a location.

    Parameters
    ----------
    lat, lon : float
        Coordinates in WGS84 (degrees).
    start_year, end_year : int
        Year range (inclusive). Monthly API uses years. :contentReference[oaicite:5]{index=5}
    community : str
        One of {"AG","RE","SB"}; affects units/availability. :contentReference[oaicite:6]{index=6}
    params : Iterable[str] | None
        POWER parameter short-names.
    cache_dir : str | Path
        Local cache folder. If cached file exists, it is reused.
    timeout_s : int
        HTTP timeout.
    sleep_s : float
        Optional polite sleep between calls (useful for many points).

    Returns
    -------
    pd.DataFrame
        Columns: date, precip_mm_day, precip_mm_month_est, t2m_c, tmax_c, tmin_c, source
    """
    if not (-90 <= lat <= 90):
        raise ValueError(f"Invalid latitude: {lat}")
    if not (-180 <= lon <= 180):
        raise ValueError(f"Invalid longitude: {lon}")
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")

    p = PowerParams()
    if params is None:
        params = [p.precip, p.t2m, p.t2m_max, p.t2m_min]

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / _cache_key(lat, lon, start_year, end_year, community, params)
    if cache_file.exists():
        df = pd.read_csv(cache_file)
        df["date"] = pd.to_datetime(df["date"])
        return df

    req_params = {
        "parameters": ",".join(list(params)),
        "community": community,
        "longitude": lon,
        "latitude": lat,
        "start": start_year,
        "end": end_year,
        "format": "JSON",
    }

    r = requests.get(POWER_MONTHLY_POINT_URL, params=req_params, timeout=timeout_s)
    if r.status_code != 200:
        raise RuntimeError(
            f"NASA POWER request failed ({r.status_code}). "
            f"URL={r.url} | text={r.text[:300]}"
        )

    payload = r.json()
    properties = payload.get("properties", {})
    param_block = properties.get("parameter", {})
    if not param_block:
        raise RuntimeError("NASA POWER returned empty parameter block. Check params/community.")

    # Build a tidy table by YYYYMM keys
    # param_block looks like: { "T2M": {"201601": 12.3, ...}, ... }
    all_keys = set()
    for par, series in param_block.items():
        all_keys |= set(series.keys())

    rows = []
    for yyyymm in sorted(all_keys):
        try:
            dt = pd.to_datetime(yyyymm + "01", format="%Y%m%d")
        except Exception:
            continue

        row = {"date": dt, "source": "NASA_POWER_MONTHLY"}
        # raw extraction
        for par in params:
            v = _safe_float(param_block.get(par, {}).get(yyyymm))
            row[par] = v
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("date")

    # Rename into BasinCast canonical exog naming
    precip_raw = p.precip
    df["precip_mm_day"] = df[precip_raw]
    # derived monthly estimate
    df["days_in_month"] = df["date"].dt.days_in_month
    df["precip_mm_month_est"] = df["precip_mm_day"] * df["days_in_month"]

    df["t2m_c"] = df[p.t2m]
    df["tmax_c"] = df[p.t2m_max]
    df["tmin_c"] = df[p.t2m_min]

    keep = ["date", "precip_mm_day", "precip_mm_month_est", "t2m_c", "tmax_c", "tmin_c", "source"]
    df_out = df[keep].copy()

    df_out.to_csv(cache_file, index=False)

    if sleep_s > 0:
        import time
        time.sleep(sleep_s)

    return df_out