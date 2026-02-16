# app/paper_layer/io_contract.py

from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class SeriesSchema:
    point_id: str = "point_id"
    date: str = "date"
    y: str = "value"  # observed target

@dataclass(frozen=True)
class ForecastSchema:
    date: str = "date"
    y_hat: str = "y_hat"
    h: str = "h"
    model: str = "model"
    family: str = "family"
    kind: str = "kind"  # monthly_path / key_horizons

@dataclass(frozen=True)
class SkillSchema:
    point_id: str = "point_id"
    model: str = "model"
    family: str = "family"
    h: str = "h"
    kge: str = "kge"

KEY_KINDS: List[str] = ["monthly_path", "key_horizons"]