from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

POWER_MONTHLY_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/monthly/point"


@dataclass(frozen=True)
class PowerMonthlyConfig:
    """
    Config for NASA POWER monthly point requests.

    Notes:
    - Some POWER endpoints accept start/end as YYYY (monthly docs show this),
      but some combinations return 422 in practice. We therefore retry with YYYYMMDD.
    - We request precipitation separately from temperature for easier fallbacks.
    """
    community: str = "AG"
    timeout_s: float = 30.0
    ssl_verify: bool = True
    cache_dir: Optional[Path] = None
    user_agent: str = "BasinCast/0.14 (+https://github.com/al16gm/BasinCast)"

    # Requested parameters (we'll fallback if some combo fails)
    precip_params: Tuple[str, ...] = ("PRECTOTCORR", "PRECTOT")
    temp_params: Tuple[str, ...] = ("T2M", "T2M_MAX", "T2M_MIN")

    # Retry order for start/end formatting
    #  - "YYYY"      -> start=2010 end=2011
    #  - "YYYYMMDD"  -> start=20100101 end=20111231
    date_formats: Tuple[str, ...] = ("YYYY", "YYYYMMDD")


def _ensure_cache_dir(cfg: PowerMonthlyConfig) -> Optional[Path]:
    if cfg.cache_dir is None:
        return None
    p = Path(cfg.cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_key(lat: float, lon: float, start_year: int, end_year: int, params: Sequence[str], community: str) -> str:
    s = f"{lat:.6f}|{lon:.6f}|{start_year}|{end_year}|{community}|{','.join(params)}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _build_start_end(start_year: int, end_year: int, fmt: str) -> Tuple[str, str]:
    if fmt == "YYYY":
        return str(int(start_year)), str(int(end_year))
    if fmt == "YYYYMMDD":
        return f"{int(start_year)}0101", f"{int(end_year)}1231"
    raise ValueError(f"Unknown date format: {fmt}")


def _power_request_json(
    *,
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
    params: Sequence[str],
    cfg: PowerMonthlyConfig,
    date_fmt: str,
) -> Dict:
    start, end = _build_start_end(start_year, end_year, date_fmt)

    q = {
        "parameters": ",".join(params),
        "community": cfg.community,
        "longitude": f"{lon}",
        "latitude": f"{lat}",
        "start": start,
        "end": end,
        "format": "JSON",
    }
    headers = {"User-Agent": cfg.user_agent}

    resp = requests.get(
        POWER_MONTHLY_ENDPOINT,
        params=q,
        timeout=float(cfg.timeout_s),
        verify=bool(cfg.ssl_verify),
        headers=headers,
    )

    # Provide readable error text (POWER returns useful JSON messages)
    if resp.status_code != 200:
        txt = resp.text or ""
        raise RuntimeError(
            f"NASA POWER request failed status={resp.status_code}. "
            f"params={tuple(params)} start={start} end={end} lat={lat} lon={lon}. Response: {txt}"
        )
    return resp.json()


def _try_power_request_with_fallbacks(
    *,
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
    params_try_orders: Sequence[Sequence[str]],
    cfg: PowerMonthlyConfig,
) -> Tuple[Dict, Dict]:
    """
    Returns: (payload, debug_info)
    Tries:
      - different date formats (YYYY then YYYYMMDD)
      - different parameter sets (e.g., [PRECTOTCORR, PRECTOT] then [PRECTOT])
    """
    last_err: Optional[Exception] = None
    for date_fmt in cfg.date_formats:
        for params in params_try_orders:
            try:
                payload = _power_request_json(
                    lat=lat, lon=lon, start_year=start_year, end_year=end_year,
                    params=params, cfg=cfg, date_fmt=date_fmt
                )
                dbg = {
                    "ok": True,
                    "date_fmt": date_fmt,
                    "params": list(params),
                    "url": POWER_MONTHLY_ENDPOINT,
                }
                return payload, dbg
            except Exception as e:
                last_err = e
                continue

    raise RuntimeError(f"All NASA POWER request attempts failed. Last error: {last_err}")


def _extract_monthly_series(payload: Dict, param: str) -> pd.Series:
    """
    POWER monthly JSON typically:
      payload["properties"]["parameter"][param] = { "201001": value, ... }

    We only accept keys like YYYYMM with MM in 01..12.
    """
    try:
        d = payload["properties"]["parameter"][param]
    except Exception:
        return pd.Series(dtype=float)

    vals = {}
    for k, v in d.items():
        ks = str(k)
        if len(ks) != 6 or (not ks.isdigit()):
            continue
        y = int(ks[:4])
        m = int(ks[4:6])
        if m < 1 or m > 12:
            continue
        ts = pd.Timestamp(year=y, month=m, day=1)
        try:
            fv = float(v)
        except Exception:
            fv = np.nan
        # POWER sometimes uses -999 for missing
        if fv <= -900:
            fv = np.nan
        vals[ts] = fv

    s = pd.Series(vals).sort_index()
    s.index.name = "date"
    return s


def _mm_day_to_mm_month(s_mm_day: pd.Series) -> pd.Series:
    """
    Convert a monthly series expressed as mm/day to mm/month by multiplying
    by the number of days in each month. Returns a Series (never Index/array).
    """
    if s_mm_day is None or len(s_mm_day) == 0:
        return pd.Series(dtype=float)

    # Ensure datetime index (month starts)
    idx = pd.to_datetime(s_mm_day.index, errors="coerce")
    s = pd.Series(pd.to_numeric(s_mm_day.to_numpy(), errors="coerce"), index=idx).dropna()
    if s.empty:
        return pd.Series(dtype=float)

    # Days in month as numpy float array
    dim = pd.DatetimeIndex(s.index).days_in_month.astype(float)

    out = s.to_numpy(dtype=float) * dim
    out_s = pd.Series(out, index=pd.DatetimeIndex(s.index), name=s_mm_day.name)
    out_s.index.name = "date"
    return out_s



def fetch_power_monthly(
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
    cfg: Optional[PowerMonthlyConfig] = None,
) -> pd.DataFrame:
    cfg = cfg or PowerMonthlyConfig()
    cache_dir = _ensure_cache_dir(cfg)

    # ----- Cache (best-effort) -----
    precip_key = _cache_key(lat, lon, start_year, end_year, cfg.precip_params, cfg.community)
    temp_key = _cache_key(lat, lon, start_year, end_year, cfg.temp_params, cfg.community)
    precip_cache = (cache_dir / f"power_monthly_precip_{precip_key}.json") if cache_dir else None
    temp_cache = (cache_dir / f"power_monthly_temp_{temp_key}.json") if cache_dir else None

    precip_payload = None
    temp_payload = None
    dbg_precip = {}
    dbg_temp = {}

    # Precip request fallbacks: try both params, then only PRECTOTCORR, then only PRECTOT
    precip_orders = [
        tuple(cfg.precip_params),
        ("PRECTOTCORR",),
        ("PRECTOT",),
    ]

    # Temp request fallbacks: full set, then progressively smaller
    temp_orders = [
        tuple(cfg.temp_params),
        ("T2M", "T2M_MAX"),
        ("T2M", "T2M_MIN"),
        ("T2M",),
    ]

    if precip_cache and precip_cache.exists():
        precip_payload = json.loads(precip_cache.read_text(encoding="utf-8"))
        dbg_precip = {"ok": True, "cache": True, "path": str(precip_cache)}
    else:
        precip_payload, dbg_precip = _try_power_request_with_fallbacks(
            lat=lat, lon=lon, start_year=start_year, end_year=end_year,
            params_try_orders=precip_orders, cfg=cfg
        )
        if precip_cache:
            precip_cache.write_text(json.dumps(precip_payload), encoding="utf-8")

    if temp_cache and temp_cache.exists():
        temp_payload = json.loads(temp_cache.read_text(encoding="utf-8"))
        dbg_temp = {"ok": True, "cache": True, "path": str(temp_cache)}
    else:
        temp_payload, dbg_temp = _try_power_request_with_fallbacks(
            lat=lat, lon=lon, start_year=start_year, end_year=end_year,
            params_try_orders=temp_orders, cfg=cfg
        )
        if temp_cache:
            temp_cache.write_text(json.dumps(temp_payload), encoding="utf-8")

    # ----- Extract series -----
    # Precip: prefer PRECTOTCORR if present & non-null; else PRECTOT
    s_pcorr = _extract_monthly_series(precip_payload, "PRECTOTCORR")
    s_praw = _extract_monthly_series(precip_payload, "PRECTOT")

    s_precip_base = s_pcorr if (not s_pcorr.dropna().empty) else s_praw
    precip_mm_month_est = pd.Series(_mm_day_to_mm_month(s_precip_base), index=s_precip_base.index)

    s_t2m = _extract_monthly_series(temp_payload, "T2M")
    s_tmax = _extract_monthly_series(temp_payload, "T2M_MAX")
    s_tmin = _extract_monthly_series(temp_payload, "T2M_MIN")

    # Common index (union of all)
    idx = precip_mm_month_est.index.union(s_t2m.index).union(s_tmax.index).union(s_tmin.index)
    idx = idx.sort_values()

    df = pd.DataFrame({"date": idx})
    df["precip_mm_month_est"] = df["date"].map(precip_mm_month_est.to_dict())
    df["t2m_c"] = df["date"].map(s_t2m.to_dict())
    df["tmax_c"] = df["date"].map(s_tmax.to_dict())
    df["tmin_c"] = df["date"].map(s_tmin.to_dict())

    df["source"] = "NASA_POWER_MONTHLY"

    # Debug columns are handy to diagnose future 422s (you can drop later)
    df["debug_precip_request"] = json.dumps(dbg_precip, ensure_ascii=False)
    df["debug_temp_request"] = json.dumps(dbg_temp, ensure_ascii=False)

    df = df.sort_values("date").reset_index(drop=True)
    return df


__all__ = ["PowerMonthlyConfig", "fetch_power_monthly"]