from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests


POWER_MONTHLY_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/monthly/point"
DEFAULT_PARAMS = ["PRECTOT", "T2M", "T2M_MAX", "T2M_MIN"]


@dataclass(frozen=True)
class PowerMonthlyConfig:
    community: str = "AG"
    params: Tuple[str, ...] = tuple(DEFAULT_PARAMS)
    timeout_s: int = 60
    cache_dir: str = ".cache/power"
    round_coords: int = 4  # cache key stability


def _month_start(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=month, day=1).to_period("M").to_timestamp()


def _days_in_month(ts: pd.Timestamp) -> int:
    # month start -> next month start - 1 day
    next_m = (ts + pd.DateOffset(months=1)).to_period("M").to_timestamp()
    return int((next_m - ts).days)


def _safe_float(x) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float("nan")


def _extract_param_series(param_block: Dict) -> Dict[pd.Timestamp, float]:
    """
    POWER monthly point can come as:
      - keys: "YYYYMM"
      - or nested "YYYY" -> {"MM": value}
    We normalize into {month_start: value}.
    """
    out: Dict[pd.Timestamp, float] = {}

    # Case 1: YYYYMM keys
    keys = list(param_block.keys())
    if any(isinstance(k, str) and k.isdigit() and len(k) == 6 for k in keys):
        for k, v in param_block.items():
            if not (isinstance(k, str) and k.isdigit() and len(k) == 6):
                continue
            y = int(k[:4])
            m = int(k[4:6])

            # NASA POWER sometimes includes "month 13" = annual aggregate (or other non-month keys).
            # We only keep real months 1..12.
            if not (1 <= m <= 12):
                continue

            out[_month_start(y, m)] = _safe_float(v)
        return out

    # Case 2: YYYY -> month dict
    if any(isinstance(k, str) and k.isdigit() and len(k) == 4 for k in keys):
        for yk, months in param_block.items():
            if not (isinstance(yk, str) and yk.isdigit() and len(yk) == 4):
                continue
            if not isinstance(months, dict):
                continue
            y = int(yk)
            for mk, v in months.items():
                try:
                    m = int(mk)
                except Exception:
                    continue
                if 1 <= m <= 12:
                    out[_month_start(y, m)] = _safe_float(v)
        return out

    # Unknown format -> empty
    return out


def _detect_units_from_response(payload: Dict) -> Dict[str, str]:
    """
    Try to find units in response metadata (if present). If not found, return {}.
    """
    units: Dict[str, str] = {}

    # Some POWER responses include parameter information blocks; we try a few common shapes
    props = payload.get("properties", {}) if isinstance(payload, dict) else {}
    pinfo = props.get("parameter_information") or props.get("parameters") or {}

    if isinstance(pinfo, dict):
        # could be {"T2M": {"units": "C"}, ...}
        for p, meta in pinfo.items():
            if isinstance(meta, dict) and "units" in meta:
                units[str(p)] = str(meta["units"])

    return units


def fetch_power_monthly(
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
    cfg: PowerMonthlyConfig = PowerMonthlyConfig(),
) -> pd.DataFrame:
    """
    Returns monthly dataframe with:
      date, precip_mm_month_est, t2m_c, tmax_c, tmin_c, source
    """
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rlat = round(float(lat), cfg.round_coords)
    rlon = round(float(lon), cfg.round_coords)
    key = f"power_monthly_{cfg.community}_{rlat}_{rlon}_{start_year}_{end_year}_{'-'.join(cfg.params)}.json"
    cache_path = cache_dir / key

    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        params = {
            "parameters": ",".join(cfg.params),
            "community": cfg.community,
            "longitude": str(lon),
            "latitude": str(lat),
            "start": str(start_year),
            "end": str(end_year),
            "format": "JSON",
        }
        resp = requests.get(POWER_MONTHLY_ENDPOINT, params=params, timeout=cfg.timeout_s)
        resp.raise_for_status()
        payload = resp.json()
        cache_path.write_text(json.dumps(payload), encoding="utf-8")

    props = payload.get("properties", {})
    param_block = props.get("parameter", {})

    # Extract raw series
    raw: Dict[str, Dict[pd.Timestamp, float]] = {}
    for p in cfg.params:
        block = param_block.get(p, {})
        if isinstance(block, dict):
            raw[p] = _extract_param_series(block)
        else:
            raw[p] = {}

    # Union of all months
    all_months = set()
    for d in raw.values():
        all_months |= set(d.keys())
    months = sorted(all_months)

    if not months:
        return pd.DataFrame(columns=["date", "precip_mm_month_est", "t2m_c", "tmax_c", "tmin_c", "source"])

    # Units detection (best-effort)
    units = _detect_units_from_response(payload)
    prectot_units = units.get("PRECTOT", "").lower()

    rows = []
    for dt in months:
        prectot = raw.get("PRECTOT", {}).get(dt, float("nan"))
        t2m = raw.get("T2M", {}).get(dt, float("nan"))
        tmax = raw.get("T2M_MAX", {}).get(dt, float("nan"))
        tmin = raw.get("T2M_MIN", {}).get(dt, float("nan"))

        # Convert precipitation to mm/month (robust):
        # - If units say mm/day -> multiply by days
        # - If units say mm -> keep
        # - If units missing -> heuristic: small values look like mm/day
        dim = _days_in_month(dt)
        if "mm/day" in prectot_units or "/day" in prectot_units:
            precip_mm_month = prectot * dim
        elif "mm" in prectot_units:
            precip_mm_month = prectot
        else:
            # heuristic
            if pd.notna(prectot) and 0.0 <= float(prectot) <= 25.0:
                precip_mm_month = float(prectot) * dim
            else:
                precip_mm_month = float(prectot)

        rows.append(
            {
                "date": dt,
                "precip_mm_month_est": precip_mm_month,
                "t2m_c": t2m,
                "tmax_c": tmax,
                "tmin_c": tmin,
                "source": "NASA_POWER_MONTHLY",
            }
        )

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out    