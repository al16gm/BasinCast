from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


SUPPORTED_EXCEL = (".xlsx", ".xls")
SUPPORTED_CSV = (".csv",)


@dataclass(frozen=True)
class LoadedTable:
    """Represents a user-provided table loaded from Excel/CSV."""
    name: str  # sheet name (Excel) or filename (CSV)
    df: pd.DataFrame


def load_user_file(path: str | Path) -> List[LoadedTable]:
    """
    Load a user file (Excel or CSV) and return one or more tables.
    - Excel: returns one table per sheet
    - CSV: returns a single table
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix in SUPPORTED_CSV:
        df = pd.read_csv(path)
        return [LoadedTable(name=path.name, df=df)]

    if suffix in SUPPORTED_EXCEL:
        xls = pd.ExcelFile(path)
        tables: List[LoadedTable] = []
        for sheet in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            # Drop fully empty rows/cols (common in Excel exports)
            df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
            tables.append(LoadedTable(name=sheet, df=df))
        return tables

    raise ValueError(f"Unsupported file type: {suffix}. Use CSV or Excel.")


def summarize_table(df: pd.DataFrame, max_cols: int = 30) -> Dict[str, object]:
    """Small summary for UI."""
    cols = list(df.columns.astype(str))
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": cols[:max_cols] + (["..."] if len(cols) > max_cols else []),
        "missing_pct": float(df.isna().mean().mean() * 100.0),
    }