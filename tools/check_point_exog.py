import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXOG_METEO_COLS = ["precip_mm_month_est", "t2m_c", "tmax_c", "tmin_c"]
DEMAND_COL = "demand"


def to_month_start(x):
    ts = pd.to_datetime(x, errors="coerce")
    # Series -> use .dt
    if isinstance(ts, pd.Series):
        return ts.dt.to_period("M").dt.to_timestamp()
    # scalar Timestamp -> direct
    if pd.isna(ts):
        return ts
    return ts.to_period("M").to_timestamp()

def count_valid_rows(g: pd.DataFrame, need_meteo: bool, need_demand: bool) -> int:
    gg = g.copy()
    gg["date"] = to_month_start(gg["date"])
    gg = gg.sort_values("date").reset_index(drop=True)

    gg["delta_y"] = pd.to_numeric(gg["value"], errors="coerce").diff()
    gg["value_lag1"] = pd.to_numeric(gg["value"], errors="coerce").shift(1)
    gg["value_lag12"] = pd.to_numeric(gg["value"], errors="coerce").shift(12)
    gg["delta_lag1"] = gg["delta_y"].shift(1)

    ok = gg[["value_lag1", "value_lag12", "delta_lag1"]].notna().all(axis=1)

    if need_meteo:
        for c in EXOG_METEO_COLS:
            if c in gg.columns:
                gg[f"{c}_lag1"] = pd.to_numeric(gg[c], errors="coerce").shift(1)
                ok = ok & gg[f"{c}_lag1"].notna()
            else:
                return 0

    if need_demand:
        if DEMAND_COL in gg.columns:
            gg["demand_lag1"] = pd.to_numeric(gg[DEMAND_COL], errors="coerce").shift(1)
            ok = ok & gg["demand_lag1"].notna()
        else:
            return 0

    return int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Canonical CSV used by core (e.g. outputs/_tmp_canonical_for_core.csv)")
    ap.add_argument("--point_id", required=True, help="Point id to inspect (string)")
    args = ap.parse_args()

    path = Path(args.csv)
    df = pd.read_csv(path)
    df["point_id"] = df["point_id"].astype(str)
    g = df[df["point_id"] == str(args.point_id)].copy()

    if g.empty:
        print(f"Point {args.point_id} not found in {path}")
        return

    g["date"] = to_month_start(g["date"])
    print(f"CSV: {path}")
    print(f"point_id: {args.point_id}")
    print(f"rows: {len(g)} | date range: {g['date'].min().date()} to {g['date'].max().date()}")
    print()

    # Availability
    for c in EXOG_METEO_COLS:
        if c in g.columns:
            nn = int(pd.to_numeric(g[c], errors="coerce").notna().sum())
            print(f"meteo {c}: non-null {nn}/{len(g)}")
        else:
            print(f"meteo {c}: MISSING COLUMN")

    if DEMAND_COL in g.columns:
        nn = int(pd.to_numeric(g[DEMAND_COL], errors="coerce").notna().sum())
        print(f"demand: non-null {nn}/{len(g)}")
    else:
        print("demand: MISSING COLUMN")

    print()
    # Expected valid training rows
    print("Valid rows after lags+dropna (proxy for n_pairs):")
    print("ENDO_ML:", count_valid_rows(g, need_meteo=False, need_demand=False))
    print("EXOG_ML:", count_valid_rows(g, need_meteo=True, need_demand=False))
    print("DEMAND_ML:", count_valid_rows(g, need_meteo=False, need_demand=True))
    print("EXOG_ML_DEMAND:", count_valid_rows(g, need_meteo=True, need_demand=True))


if __name__ == "__main__":
    main()