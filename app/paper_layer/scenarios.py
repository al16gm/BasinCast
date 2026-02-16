# app/paper_layer/scenarios.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd

@dataclass
class ScenarioDeltas:
    # monthly multiplicative factors or additive deltas for meteo-driven variables
    # for v0.13 we keep it generic: apply to y_hat as "scenario conditioning"
    # base=0, favorable=+a, unfavorable=-a (toy). Replace with location-specific deltas later.
    favorable_monthly_delta: float = 0.05
    unfavorable_monthly_delta: float = -0.05

def apply_scenarios_to_forecast(monthly_forecast: pd.DataFrame, deltas: ScenarioDeltas) -> pd.DataFrame:
    """
    monthly_forecast columns: date, y_hat, h, model, family, kind (monthly_path)
    Returns stacked scenarios with columns: scenario, y_hat_s
    """
    df = monthly_forecast.copy()
    df = df[df["kind"] == "monthly_path"].copy()

    out = []
    base = df.copy()
    base["scenario"] = "Base"
    base["y_hat_s"] = base["y_hat"]
    out.append(base)

    fav = df.copy()
    fav["scenario"] = "Favorable"
    fav["y_hat_s"] = fav["y_hat"] * (1.0 + deltas.favorable_monthly_delta)
    out.append(fav)

    unf = df.copy()
    unf["scenario"] = "Unfavorable"
    unf["y_hat_s"] = unf["y_hat"] * (1.0 + deltas.unfavorable_monthly_delta)
    out.append(unf)

    res = pd.concat(out, ignore_index=True)
    return res