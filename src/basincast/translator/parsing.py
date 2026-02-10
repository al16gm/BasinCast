from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ParseReport:
    ok_ratio: float
    n_ok: int
    n_bad: int
    strategy: str
    details: Dict[str, Any]


def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, str]]:
    """
    Normalize column names safely:
    - strip leading/trailing whitespace
    - collapse internal whitespace
    - keep mapping raw->normalized
    """
    raw_cols = list(df.columns)
    raw_to_norm: Dict[str, str] = {}

    def _norm(c: Any) -> str:
        s = "" if c is None else str(c)
        s2 = re.sub(r"\s+", " ", s.strip())
        return s2

    new_cols: List[str] = []
    seen: Dict[str, int] = {}

    for c in raw_cols:
        nc = _norm(c)
        if nc in seen:
            seen[nc] += 1
            nc2 = f"{nc}__{seen[nc]}"
        else:
            seen[nc] = 1
            nc2 = nc
        new_cols.append(nc2)
        raw_to_norm[str(c)] = nc2

    out = df.copy()
    out.columns = new_cols
    return out, raw_to_norm


def _sample_series(s: pd.Series, n: int = 1000) -> pd.Series:
    s2 = s.dropna()
    if len(s2) <= n:
        return s2
    return s2.sample(n=n, random_state=7)


def parse_numeric_series(s: pd.Series, locale_hint: str = "auto") -> tuple[pd.Series, ParseReport]:
    """
    Robust numeric parsing:
    - locale_hint: "auto" | "comma_decimal" | "dot_decimal"
    Handles:
      - 1.234,56 (EU)
      - 1,234.56 (US)
      - 12 345,7 (spaces)
      - units/symbols attached (hm3, %, etc.)
    """
    if pd.api.types.is_numeric_dtype(s):
        out = pd.to_numeric(s, errors="coerce")
        bad = int(out.isna().sum() - s.isna().sum())
        rep = ParseReport(
            ok_ratio=float(1 - bad / max(1, len(s.dropna()))),
            n_ok=int(out.notna().sum()),
            n_bad=bad,
            strategy="native_numeric",
            details={},
        )
        return out.astype(float), rep

    s_str = s.astype(str).replace({"nan": np.nan, "None": np.nan})
    sample = _sample_series(s_str)

    def keep_num_chars(x: str) -> str:
        x = x.strip()
        x = x.replace("\u00a0", " ")  # NBSP
        x = re.sub(r"[^0-9\-\+\,\.\s']", "", x)
        x = re.sub(r"\s+", " ", x).strip()
        return x

    def normalize_one_auto(x: str) -> str:
        x = keep_num_chars(x)
        if x == "":
            return ""
        x = x.replace("'", "")  # Swiss thousands

        if "." in x and "," in x:
            # decimal is last separator
            last_dot = x.rfind(".")
            last_com = x.rfind(",")
            if last_com > last_dot:
                # EU: 1.234,56 -> 1234.56
                x = x.replace(".", "")
                x = x.replace(",", ".")
            else:
                # US: 1,234.56 -> 1234.56
                x = x.replace(",", "")
        else:
            if "," in x:
                if re.search(r",\d{1,6}$", x):
                    x = x.replace(".", "")
                    x = x.replace(",", ".")
                else:
                    x = x.replace(",", "")
            elif "." in x:
                # if ends with .ddd (3 digits) it might be thousands; treat as thousands in auto
                if re.search(r"\.\d{3}$", x) and not re.search(r"\.\d{1,2}$", x):
                    x = x.replace(".", "")

        x = x.replace(" ", "")
        return x

    def normalize_one_comma_decimal(x: str) -> str:
        x = keep_num_chars(x).replace("'", "")
        if x == "":
            return ""
        # Assume comma decimal, dot/space thousands
        x = x.replace(".", "")
        x = x.replace(" ", "")
        x = x.replace(",", ".")
        return x

    def normalize_one_dot_decimal(x: str) -> str:
        x = keep_num_chars(x).replace("'", "")
        if x == "":
            return ""
        # Assume dot decimal, comma/space thousands
        x = x.replace(",", "")
        x = x.replace(" ", "")
        return x

    if locale_hint == "comma_decimal":
        norm_fn = normalize_one_comma_decimal
        strategy = "comma_decimal"
    elif locale_hint == "dot_decimal":
        norm_fn = normalize_one_dot_decimal
        strategy = "dot_decimal"
    else:
        norm_fn = normalize_one_auto
        strategy = "auto_decimal_thousands"

    normalized = s_str.map(lambda v: norm_fn(v) if pd.notna(v) else np.nan)
    out = pd.to_numeric(normalized, errors="coerce")

    bad = int(out.isna().sum() - s.isna().sum())
    ok_ratio = float(1 - bad / max(1, len(s.dropna())))

    rep = ParseReport(
        ok_ratio=ok_ratio,
        n_ok=int(out.notna().sum()),
        n_bad=bad,
        strategy=strategy,
        details={
            "example_raw": sample.dropna().head(5).tolist(),
            "example_norm": normalized.dropna().head(5).tolist(),
        },
    )
    return out.astype(float), rep


def _try_parse_dates(s: pd.Series, dayfirst: bool, fmt: Optional[str]) -> pd.Series:
    if fmt:
        return pd.to_datetime(s, errors="coerce", format=fmt)
    return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)


def parse_date_series(s: pd.Series, date_hint: str = "auto") -> tuple[pd.Series, ParseReport]:
    """
    Robust date parsing:
    - date_hint: "auto" | "dayfirst" | "monthfirst" | "excel_serial"
    Handles:
      - dd/mm/yyyy vs mm/dd/yyyy ambiguity (asks user if needed in UI)
      - YYYY-MM / YYYY-MM-DD
      - Excel serial numbers
    """
    # Excel serial hint
    if date_hint == "excel_serial":
        dtser = pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")
        bad = int(dtser.isna().sum() - s.isna().sum())
        rep = ParseReport(
            ok_ratio=float(1 - bad / max(1, len(s.dropna()))),
            n_ok=int(dtser.notna().sum()),
            n_bad=bad,
            strategy="excel_serial",
            details={},
        )
        return dtser, rep

    # If numeric dtype, try excel serial automatically
    if pd.api.types.is_numeric_dtype(s):
        dtser = pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")
        bad = int(dtser.isna().sum() - s.isna().sum())
        rep = ParseReport(
            ok_ratio=float(1 - bad / max(1, len(s.dropna()))),
            n_ok=int(dtser.notna().sum()),
            n_bad=bad,
            strategy="excel_serial_auto",
            details={},
        )
        return dtser, rep

    s_str = s.astype(str).replace({"nan": np.nan, "None": np.nan}).map(lambda x: x.strip() if isinstance(x, str) else x)
    sample = _sample_series(s_str)

    if date_hint == "dayfirst":
        full = _try_parse_dates(s_str, dayfirst=True, fmt=None)
        bad = int(full.isna().sum() - s.isna().sum())
        ok_ratio = float(1 - bad / max(1, len(s.dropna())))
        rep = ParseReport(ok_ratio=ok_ratio, n_ok=int(full.notna().sum()), n_bad=bad, strategy="dayfirst", details={})
        return full, rep

    if date_hint == "monthfirst":
        full = _try_parse_dates(s_str, dayfirst=False, fmt=None)
        bad = int(full.isna().sum() - s.isna().sum())
        ok_ratio = float(1 - bad / max(1, len(s.dropna())))
        rep = ParseReport(ok_ratio=ok_ratio, n_ok=int(full.notna().sum()), n_bad=bad, strategy="monthfirst", details={})
        return full, rep

    candidates: List[tuple[str, dict]] = [
        ("auto_dayfirst_false", {"dayfirst": False, "fmt": None}),
        ("auto_dayfirst_true", {"dayfirst": True, "fmt": None}),
        ("fmt_%Y-%m-%d", {"dayfirst": False, "fmt": "%Y-%m-%d"}),
        ("fmt_%d/%m/%Y", {"dayfirst": True, "fmt": "%d/%m/%Y"}),
        ("fmt_%m/%d/%Y", {"dayfirst": False, "fmt": "%m/%d/%Y"}),
        ("fmt_%Y/%m/%d", {"dayfirst": False, "fmt": "%Y/%m/%d"}),
        ("fmt_%Y%m%d", {"dayfirst": False, "fmt": "%Y%m%d"}),
        ("fmt_%Y-%m", {"dayfirst": False, "fmt": "%Y-%m"}),
        ("fmt_%m-%Y", {"dayfirst": False, "fmt": "%m-%Y"}),
    ]

    scored: List[tuple[str, int, dict]] = []
    for name, cfg in candidates:
        parsed = _try_parse_dates(sample, cfg["dayfirst"], cfg["fmt"])
        scored.append((name, int(parsed.notna().sum()), cfg))

    scored.sort(key=lambda x: x[1], reverse=True)
    best_name, best_ok, best_cfg = scored[0]
    second_ok = scored[1][1] if len(scored) > 1 else -1

    full = _try_parse_dates(s_str, best_cfg["dayfirst"], best_cfg["fmt"])
    bad = int(full.isna().sum() - s.isna().sum())
    ok_ratio = float(1 - bad / max(1, len(s.dropna())))

    rep = ParseReport(
        ok_ratio=ok_ratio,
        n_ok=int(full.notna().sum()),
        n_bad=bad,
        strategy=best_name,
        details={
            "best_sample_ok": best_ok,
            "second_sample_ok": second_ok,
            "sample_n": int(sample.notna().sum()),
            "dayfirst": best_cfg["dayfirst"],
            "format": best_cfg["fmt"],
            "ambiguous": (best_ok - second_ok) <= max(3, int(0.01 * max(1, int(sample.notna().sum())))),
        },
    )
    return full, rep


def coerce_month_start(dtser: pd.Series) -> pd.Series:
    dtser = pd.to_datetime(dtser, errors="coerce")
    return dtser.dt.to_period("M").dt.to_timestamp()