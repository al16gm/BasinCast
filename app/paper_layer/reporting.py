# app/paper_layer/reporting.py

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd

def reliability_horizon(skill_df: pd.DataFrame, model: str, threshold: float) -> int:
    df = skill_df[(skill_df["model"] == model) & (skill_df["kge"].notna())].sort_values("h")
    ok = df[df["kge"] >= threshold]
    return int(ok["h"].max()) if len(ok) else 0

def mini_analysis(leaderboard: pd.DataFrame, skill_df: pd.DataFrame, planning_thr: float, advisory_thr: float) -> str:
    if leaderboard is None or leaderboard.empty:
        return "No leaderboard available (insufficient data)."

    winner = leaderboard.iloc[0]["model"]
    best_mean = leaderboard.iloc[0]["kge_mean"]
    fail_rate = leaderboard.iloc[0].get("fail_rate", np.nan)

    h_plan = reliability_horizon(skill_df, winner, planning_thr)
    h_adv = reliability_horizon(skill_df, winner, advisory_thr)

    return (
        f"Winner model: {winner}. "
        f"Mean KGE (focus horizons) = {best_mean:.3f}. "
        f"Fail rate during backtesting = {fail_rate:.2%} (lower is better). "
        f"Reliability horizon: planning={h_plan} months (KGE≥{planning_thr}), "
        f"advisory={h_adv} months (KGE≥{advisory_thr})."
    )