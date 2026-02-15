from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests


@dataclass(frozen=True)
class GeoResult:
    lat: float
    lon: float
    label: str
    confidence: str  # HIGH / MEDIUM / LOW
    raw: Dict[str, Any]


def _cache_path() -> Path:
    p = Path(".cache")
    p.mkdir(parents=True, exist_ok=True)
    return p / "geocode_cache.json"


def _load_cache() -> Dict[str, Any]:
    cp = _cache_path()
    if cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    cp = _cache_path()
    cp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


_LAST_CALL_TS = 0.0


def _rate_limit(min_seconds: float = 1.0) -> None:
    global _LAST_CALL_TS
    now = time.time()
    dt = now - _LAST_CALL_TS
    if dt < min_seconds:
        time.sleep(min_seconds - dt)
    _LAST_CALL_TS = time.time()


def _confidence_from_address(addr: Dict[str, Any]) -> str:
    # HIGH if we have city/town/village (fine-grained)
    if any(k in addr for k in ["city", "town", "village", "municipality", "suburb"]):
        return "HIGH"
    # MEDIUM if we have state/county but not city
    if any(k in addr for k in ["state", "province", "county", "region"]):
        return "MEDIUM"
    # LOW if basically country-only
    return "LOW"


def geocode_place(query: str, *, timeout: int = 20) -> Optional[GeoResult]:
    """
    Forward geocode using Nominatim (OpenStreetMap). No API key required.
    """
    q = (query or "").strip()
    if not q:
        return None

    cache = _load_cache()
    key = f"fwd::{q.lower()}"
    if key in cache:
        c = cache[key]
        return GeoResult(
            lat=float(c["lat"]),
            lon=float(c["lon"]),
            label=str(c["label"]),
            confidence=str(c["confidence"]),
            raw=dict(c.get("raw", {})),
        )

    _rate_limit(1.0)

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": q,
        "format": "json",
        "addressdetails": 1,
        "limit": 1,
    }
    headers = {"User-Agent": "BasinCast/0.10 (open-source research tool)"}
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None

    best = data[0]
    lat = float(best["lat"])
    lon = float(best["lon"])
    label = str(best.get("display_name", q))
    addr = best.get("address", {}) if isinstance(best.get("address"), dict) else {}
    conf = _confidence_from_address(addr)

    out = GeoResult(lat=lat, lon=lon, label=label, confidence=conf, raw=best)

    cache[key] = {"lat": lat, "lon": lon, "label": label, "confidence": conf, "raw": best}
    _save_cache(cache)
    return out


def reverse_geocode(lat: float, lon: float, *, timeout: int = 20) -> Optional[GeoResult]:
    """
    Reverse geocode (lat/lon -> approximate place label).
    """
    cache = _load_cache()
    key = f"rev::{round(float(lat), 4)},{round(float(lon), 4)}"
    if key in cache:
        c = cache[key]
        return GeoResult(
            lat=float(c["lat"]),
            lon=float(c["lon"]),
            label=str(c["label"]),
            confidence=str(c["confidence"]),
            raw=dict(c.get("raw", {})),
        )

    _rate_limit(1.0)

    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": float(lat),
        "lon": float(lon),
        "format": "json",
        "addressdetails": 1,
    }
    headers = {"User-Agent": "BasinCast/0.10 (open-source research tool)"}
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    best = r.json()
    if not best:
        return None

    label = str(best.get("display_name", f"{lat},{lon}"))
    addr = best.get("address", {}) if isinstance(best.get("address"), dict) else {}
    conf = _confidence_from_address(addr)

    out = GeoResult(lat=float(lat), lon=float(lon), label=label, confidence=conf, raw=best)

    cache[key] = {"lat": float(lat), "lon": float(lon), "label": label, "confidence": conf, "raw": best}
    _save_cache(cache)
    return out