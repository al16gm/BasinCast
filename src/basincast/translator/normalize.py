from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rapidfuzz import fuzz, process


CANON_RESOURCE_TYPES = ["Aquifer", "Reservoir", "River"]


def _clean(s: str) -> str:
    return str(s).strip()


def suggest_value_map(values: List[str], canon: List[str], score_cutoff: int = 80) -> Dict[str, str]:
    """
    Suggest mapping from observed categorical values to canonical labels
    using fuzzy string matching.
    """
    mapping: Dict[str, str] = {}
    for v in values:
        vv = _clean(v)
        if vv == "" or vv.lower() in {"nan", "none"}:
            continue

        match = process.extractOne(vv, canon, scorer=fuzz.WRatio, score_cutoff=score_cutoff)
        if match is None:
            mapping[vv] = vv  # keep as-is (user will fix in UI)
        else:
            best, score, _ = match
            mapping[vv] = best
    return mapping


def apply_value_map(series: pd.Series, value_map: Dict[str, str]) -> pd.Series:
    """Apply mapping to a pandas Series."""
    return series.astype(str).map(lambda x: value_map.get(_clean(x), _clean(x)))