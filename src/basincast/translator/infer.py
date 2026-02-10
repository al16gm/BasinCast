from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional

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


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _norm(s: str) -> str:
    s = _strip_accents(str(s).strip().lower())
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s


def infer_mapping(df: pd.DataFrame) -> InferredMapping:
    cols = list(df.columns)
    norm_map = {_norm(c): c for c in cols}  # normalized -> original

    def pick(*candidates: str) -> Optional[str]:
        for cand in candidates:
            if cand in norm_map:
                return norm_map[cand]
        return None

    date_col = pick("date", "fecha", "datetime", "timestamp", "time", "mes", "month")
    point_id_col = pick("control_point", "point_id", "punto", "estacion", "station", "gauge", "nombre", "id")
    value_col = pick("value", "valor", "y", "target", "nivel", "level", "storage", "discharge", "caudal", "flow")
    unit_col = pick("unit", "units", "unidad", "unidades")
    resource_type_col = pick("input", "resource_type", "type", "tipologia", "typology", "recurso")

    lat_col = pick("lat", "latitude", "latitud")
    lon_col = pick("lon", "longitude", "long", "longitud")
    utm_x_col = pick("utm_x", "x_utm", "easting", "utm_e", "x")
    utm_y_col = pick("utm_y", "y_utm", "northing", "utm_n", "y")

    drop: List[str] = []
    for c in cols:
        n = _norm(c)
        if (
            n.endswith("_lag1")
            or n.endswith("_lag12")
            or n in {"month", "sin", "cos"}
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