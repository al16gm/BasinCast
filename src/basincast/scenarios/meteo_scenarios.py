from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .scenario_deltas import SCENARIO_DELTAS


def build_meteo_scenario(
    meteo_future: pd.DataFrame,
    scenario: Literal["Base", "Favorable", "Unfavorable"],
    precip_col: str = "precip_mm",
    t2m_col: str = "t2m_c",
) -> pd.DataFrame:
    """
    Apply monthly deltas to a future meteorology dataframe.

    Parameters
    ----------
    meteo_future:
        DataFrame with at least columns [date, precip_col, t2m_col]. Date must be monthly timestamps.
    scenario:
        Scenario name in SCENARIO_DELTAS.
    precip_col, t2m_col:
        Column names.

    Returns
    -------
    DataFrame with adjusted precip and temperature for the scenario.
    """
    if meteo_future is None or meteo_future.empty:
        return pd.DataFrame()

    if scenario not in SCENARIO_DELTAS:
        raise ValueError(f"Unknown scenario: {scenario}. Valid: {list(SCENARIO_DELTAS.keys())}")

    df = meteo_future.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=["date"]).sort_values("date")

    deltas = SCENARIO_DELTAS[scenario]
    precip_mult = np.array(deltas["precip_mult"], dtype=float)
    t2m_add = np.array(deltas["t2m_add"], dtype=float)

    month_idx = df["date"].dt.month.values - 1  # 0..11

    if precip_col in df.columns:
        df[precip_col] = df[precip_col].astype(float) * (1.0 + precip_mult[month_idx])
    if t2m_col in df.columns:
        df[t2m_col] = df[t2m_col].astype(float) + t2m_add[month_idx]

    df["scenario"] = scenario
    return df