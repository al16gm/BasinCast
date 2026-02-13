from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import requests


NOMINATIM_REVERSE = "https://nominatim.openstreetmap.org/reverse"


@dataclass(frozen=True)
class GeocodeConfig:
    user_agent: str
    timeout_s: int = 30
    cache_dir: str = ".cache/geocode"
    sleep_s: float = 1.0  # be polite with public service


def reverse_geocode_nominatim(lat: float, lon: float, cfg: GeocodeConfig) -> Dict[str, str]:
    """
    Reverse geocode lat/lon -> admin fields (best-effort).
    Cached locally. Requires a real User-Agent per policy.
    """
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = f"rev_{round(float(lat), 5)}_{round(float(lon), 5)}.json"
    p = cache_dir / key
    if p.exists():
        payload = json.loads(p.read_text(encoding="utf-8"))
    else:
        headers = {"User-Agent": cfg.user_agent}
        params = {
            "format": "jsonv2",
            "lat": str(lat),
            "lon": str(lon),
            "zoom": "10",
            "addressdetails": "1",
        }
        r = requests.get(NOMINATIM_REVERSE, params=params, headers=headers, timeout=cfg.timeout_s)
        r.raise_for_status()
        payload = r.json()
        p.write_text(json.dumps(payload), encoding="utf-8")
        time.sleep(cfg.sleep_s)

    addr = payload.get("address", {}) if isinstance(payload, dict) else {}
    return {
        "display_name": str(payload.get("display_name", "")),
        "country": str(addr.get("country", "")),
        "state": str(addr.get("state", "")),
        "province": str(addr.get("province", "")),
        "county": str(addr.get("county", "")),
        "city": str(addr.get("city", "") or addr.get("town", "") or addr.get("village", "")),
        "postcode": str(addr.get("postcode", "")),
    }