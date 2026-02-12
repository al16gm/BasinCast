from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SelectionResult:
    selected_family: str
    decision: str
    status: str
    reason: str

    # horizons for reporting (per family)
    per_family_planning_horizon: Dict[str, int]
    per_family_advisory_horizon: Dict[str, int]

    # horizons for selected family
    planning_horizon_selected: int
    advisory_horizon_selected: int

    # for transparent tie-breaking
    decisive_horizon: int
    decisive_kge: float
    decisive_rmse: float


def _safe_float(x: object) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return float("nan")


def _planning_horizon_from_kge(kges: Dict[int, float], threshold: float) -> int:
    ok = []
    for h, v in kges.items():
        v = _safe_float(v)
        if not np.isnan(v) and v >= threshold:
            ok.append(int(h))
    return max(ok) if ok else 0


def _decisive_horizon(planning_h: int, advisory_h: int) -> int:
    # planning dominates; if no planning, use advisory; else 0
    if planning_h > 0:
        return int(planning_h)
    if advisory_h > 0:
        return int(advisory_h)
    return 0


def _value_at(d: Dict[int, float], h: int) -> float:
    return _safe_float(d.get(int(h), float("nan")))


def select_family(
    skill_df: pd.DataFrame | None = None,
    *,
    kge_by_family: Dict[str, Dict[int, float]] | None = None,
    rmse_by_family: Dict[str, Dict[int, float]] | None = None,
    horizons: Tuple[int, ...] | List[int] | None = None,
    planning_kge_threshold: float = 0.6,
    advisory_kge_threshold: float = 0.3,
) -> Dict[str, object]:
    """
    Select best family following BasinCast decision rules:

    1) Maximize PLANNING horizon (max h with KGE>=planning threshold)
    2) If no planning: maximize ADVISORY horizon (max h with KGE>=advisory threshold)
    3) If tie: choose higher KGE at the decisive horizon (planning or advisory horizon)
    4) If tie: choose lower RMSE at the decisive horizon
    5) If tie: prefer BASELINE* (conservative)
    """

    def _safe_float(x: object) -> float:
        try:
            v = float(x)
            return v
        except Exception:
            return float("nan")

    def _planning_horizon_from_kges(kges: Dict[int, float], thr: float) -> int:
        ok = []
        for h, v in kges.items():
            vv = _safe_float(v)
            if not np.isnan(vv) and vv >= thr:
                ok.append(int(h))
        return max(ok) if ok else 0

    def _value_at(d: Dict[int, float], h: int) -> float:
        return _safe_float(d.get(int(h), float("nan")))

    # -------- Build dictionaries from skill_df if provided --------
    if skill_df is not None:
        if not {"family", "horizon", "kge", "rmse"}.issubset(set(skill_df.columns)):
            raise ValueError("skill_df must contain columns: family, horizon, kge, rmse")

        fams = [str(x) for x in skill_df["family"].unique().tolist()]
        kge_by_family = {f: {} for f in fams}
        rmse_by_family = {f: {} for f in fams}

        for _, row in skill_df.iterrows():
            f = str(row["family"])
            h = int(row["horizon"])
            kge_by_family[f][h] = _safe_float(row["kge"])
            rmse_by_family[f][h] = _safe_float(row["rmse"])

        if horizons is None:
            horizons = tuple(sorted({int(h) for h in skill_df["horizon"].unique().tolist()}))

    # -------- Validate inputs --------
    if kge_by_family is None or rmse_by_family is None:
        raise ValueError("Provide either skill_df or (kge_by_family & rmse_by_family).")

    families = list(kge_by_family.keys())
    if horizons is None:
        # fallback: union of keys across families
        hs = set()
        for f in families:
            hs |= set(int(h) for h in kge_by_family.get(f, {}).keys())
        horizons = tuple(sorted(hs)) if hs else (12, 24, 36, 48)

    # -------- Compute planning/advisory horizons per family --------
    planning_h = {f: _planning_horizon_from_kges(kge_by_family.get(f, {}), planning_kge_threshold) for f in families}
    advisory_h = {f: _planning_horizon_from_kges(kge_by_family.get(f, {}), advisory_kge_threshold) for f in families}

    best_plan = max(planning_h.values()) if planning_h else 0
    best_adv = max(advisory_h.values()) if advisory_h else 0

    if best_plan > 0:
        pool = [f for f in families if planning_h[f] == best_plan]
        grade = "PLANNING"
        decisive_h = int(best_plan)
    elif best_adv > 0:
        pool = [f for f in families if advisory_h[f] == best_adv]
        grade = "ADVISORY"
        decisive_h = int(best_adv)
    else:
        # nobody reaches advisory threshold
        # pick baseline if exists, else first
        baseline = [f for f in families if f.upper().startswith("BASELINE")]
        selected = baseline[0] if baseline else families[0]
        return {
            "selected_family": selected,
            "decision": "NOT_RELIABLE",
            "status": "WARN",
            "reason": "SKILL_BELOW_ADVISORY_THRESHOLD",
            "planning_horizon_by_family": planning_h,
            "advisory_horizon_by_family": advisory_h,
            "planning_horizon_selected": planning_h.get(selected, 0),
            "advisory_horizon_selected": advisory_h.get(selected, 0),
            "decisive_horizon": 0,
            "decisive_kge_selected": float("nan"),
            "decisive_rmse_selected": float("nan"),
        }

    # -------- Tie-break inside pool --------
    def _rank(f: str) -> tuple:
        k = _value_at(kge_by_family.get(f, {}), decisive_h)
        r = _value_at(rmse_by_family.get(f, {}), decisive_h)
        # NaNs: push to worst
        k_sort = -1e12 if np.isnan(k) else float(k)
        r_sort = 1e12 if np.isnan(r) else float(r)
        baseline_bonus = 1 if f.upper().startswith("BASELINE") else 0
        # sort descending: higher KGE, lower RMSE, baseline bonus
        return (k_sort, -r_sort, baseline_bonus)

    selected = sorted(pool, key=_rank, reverse=True)[0]

    is_baseline = selected.upper().startswith("BASELINE")
    if grade == "PLANNING":
        decision = "BASELINE_PLANNING" if is_baseline else "MODEL_PLANNING"
        status = "OK"
        reason = ""
    else:
        decision = "BASELINE_ADVISORY" if is_baseline else "MODEL_ADVISORY"
        status = "WARN"
        reason = "ONLY_ADVISORY_GRADE"

    return {
        "selected_family": selected,
        "decision": decision,
        "status": status,
        "reason": reason,
        "planning_horizon_by_family": planning_h,
        "advisory_horizon_by_family": advisory_h,
        "planning_horizon_selected": planning_h.get(selected, 0),
        "advisory_horizon_selected": advisory_h.get(selected, 0),
        "decisive_horizon": decisive_h,
        "decisive_kge_selected": _value_at(kge_by_family.get(selected, {}), decisive_h),
        "decisive_rmse_selected": _value_at(rmse_by_family.get(selected, {}), decisive_h),
    }