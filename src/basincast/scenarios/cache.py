from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


@dataclass(frozen=True)
class CacheKey:
    lat: float
    lon: float
    ssp: str
    baseline_start: int
    baseline_end: int
    future_start: int
    future_end: int
    models_hash: str
    method: str = "cmip6_intake_v1"

    def to_string(self) -> str:
        payload = {
            "lat": round(self.lat, 4),
            "lon": round(self.lon, 4),
            "ssp": self.ssp,
            "baseline": [self.baseline_start, self.baseline_end],
            "future": [self.future_start, self.future_end],
            "models_hash": self.models_hash,
            "method": self.method,
        }
        return json.dumps(payload, sort_keys=True)

    def filename(self) -> str:
        h = hashlib.sha256(self.to_string().encode("utf-8")).hexdigest()[:16]
        return f"deltas_{h}.parquet"


class DeltaCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: CacheKey) -> Optional[pd.DataFrame]:
        fp = self.root / key.filename()
        if fp.exists():
            try:
                return pd.read_parquet(fp)
            except Exception:
                return None
        return None

    def put(self, key: CacheKey, df: pd.DataFrame) -> Path:
        fp = self.root / key.filename()
        df.to_parquet(fp, index=False)
        return fp


def hash_models(models: list[str] | None) -> str:
    if not models:
        return "auto"
    s = "|".join(sorted(models))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]