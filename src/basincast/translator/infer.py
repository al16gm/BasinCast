from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class InferredMapping:
    date_col: Optional[str]
    point_id_col: Optional[str]
    value_col: Optional[str]
    unit_col: Optional[str]
    resource_type_col: Optional[str]
    lat_col: Optional[str]
    lon_col: Optional[str]
    utm_x_col: Optional[str]
    utm_y_col: Optional[str]
    suggested_drop_cols: List[str]


def _norm(s: str) -> str:
    return re.sub(r"\s+", "_", str(s).strip().lower())


def infer_mapping(df: pd.DataFrame) -> InferredMapping:
    cols = list(df.columns)
    norm_map = {_norm(c): c for c in cols}  # normalized -> original

    def pick(*candidates: str) -> Optional[str]:
        for cand in candidates:
            if cand in norm_map:
                return norm_map[cand]
        return None

    date_col = pick("date", "datetime", "timestamp", "time", "month")
    point_id_col = pick("control_point", "point_id", "point", "id", "station", "gauge", "cp")
    value_col = pick("value", "y", "target", "level", "storage", "discharge", "flow")
    unit_col = pick("unit", "units")
    resource_type_col = pick("input", "resource_type", "type")

    lat_col = pick("lat", "latitude")
    lon_col = pick("lon", "longitude", "long")
    utm_x_col = pick("utm_x", "x_utm", "easting", "utm_e", "x")
    utm_y_col = pick("utm_y", "y_utm", "northing", "utm_n", "y")

    # Suggest dropping typical "derived feature" columns (noise for users)
    drop = []
    for c in cols:
        n = _norm(c)
        if (
            n.endswith("_lag1")
            or n.endswith("_lag12")
            or n in {"month", "sin", "cos"}
            or n.startswith("value_lag")
            or n.endswith("_lag")
        ):
            drop.append(c)

    return InferredMapping(
        date_col=date_col,
        point_id_col=point_id_col,
        value_col=value_col,
        unit_col=unit_col,
        resource_type_col=resource_type_col,
        lat_col=lat_col,
        lon_col=lon_col,
        utm_x_col=utm_x_col,
        utm_y_col=utm_y_col,
        suggested_drop_cols=drop,
    )


def column_missing_pct(df: pd.DataFrame) -> pd.Series:
    return (df.isna().mean() * 100.0).sort_values(ascending=False)