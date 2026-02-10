from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from basincast.translator.parsing import normalize_columns


SUPPORTED_EXCEL = (".xlsx", ".xls")
SUPPORTED_CSV = (".csv",)


@dataclass(frozen=True)
class LoadedTable:
    """User-provided table loaded from Excel/CSV."""
    name: str  # sheet name or filename
    df: pd.DataFrame  # with normalized column names
    raw_to_normalized_columns: Dict[str, str]


def load_user_file(path: str | Path) -> List[LoadedTable]:
    """
    Load a user file (Excel or CSV) and return one or more tables.
    - Excel: one per sheet
    - CSV: single table
    Normalizes column names to prevent invisible errors (e.g., trailing spaces).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix in SUPPORTED_CSV:
        df_raw = pd.read_csv(path)
        df_raw = df_raw.dropna(axis=0, how="all").dropna(axis=1, how="all")
        df, colmap = normalize_columns(df_raw)
        return [LoadedTable(name=path.name, df=df, raw_to_normalized_columns=colmap)]

    if suffix in SUPPORTED_EXCEL:
        xls = pd.ExcelFile(path)
        tables: List[LoadedTable] = []
        for sheet in xls.sheet_names:
            df_raw = pd.read_excel(path, sheet_name=sheet)
            df_raw = df_raw.dropna(axis=0, how="all").dropna(axis=1, how="all")
            df, colmap = normalize_columns(df_raw)
            tables.append(LoadedTable(name=sheet, df=df, raw_to_normalized_columns=colmap))
        return tables

    raise ValueError(f"Unsupported file type: {suffix}. Use CSV or Excel.")


def summarize_table(df: pd.DataFrame, max_cols: int = 30) -> Dict[str, object]:
    cols = list(df.columns.astype(str))
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": cols[:max_cols] + (["..."] if len(cols) > max_cols else []),
        "missing_pct_overall": float(df.isna().mean().mean() * 100.0),
    }