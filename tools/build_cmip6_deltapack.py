#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Grid:
    lats: np.ndarray  # (P,)
    lons: np.ndarray  # (P,) in 0..360
    lat0: float
    lon0: float
    dlat: float
    dlon: float


def build_regular_grid(step_deg: float) -> Grid:
    # Cell centers
    lats = np.arange(-90 + step_deg / 2, 90, step_deg, dtype=float)
    lons = np.arange(0 + step_deg / 2, 360, step_deg, dtype=float)
    lon2d, lat2d = np.meshgrid(lons, lats)
    return Grid(
        lats=lat2d.ravel(),
        lons=lon2d.ravel(),
        lat0=float(lats[0]),
        lon0=float(lons[0]),
        dlat=float(step_deg),
        dlon=float(step_deg),
    )


def month_start_from_timeobj(t) -> np.datetime64:
    # Works for cftime or datetime-like objects (needs year/month)
    return np.datetime64(f"{int(t.year):04d}-{int(t.month):02d}-01")


def open_cmip6_catalog(cat_url: str):
    import intake
    return intake.open_esm_datastore(cat_url)


def pick_models_intersection(col, ssp_list: List[str], table_id: str = "Amon") -> List[str]:
    """
    Choose models that have BOTH tas and pr for:
      - historical
      - each ssp in ssp_list
    """
    needed = [("historical", "tas"), ("historical", "pr")]
    for ssp in ssp_list:
        needed.extend([(ssp, "tas"), (ssp, "pr")])

    model_sets = []
    for exp, var in needed:
        q = col.search(experiment_id=exp, table_id=table_id, variable_id=var, grid_label="gn")
        if len(q.df) == 0:
            return []
        model_sets.append(set(q.df["source_id"].unique().tolist()))

    inter = set.intersection(*model_sets) if model_sets else set()
    return sorted(list(inter))


def load_ensemble_point_matrix(
    col,
    models: List[str],
    experiment_id: str,
    variable_id: str,
    years: Tuple[int, int],
    grid: Grid,
    statistic: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      values: float32 array (P, Y, 12) where P=points, Y=number of years in range, months 1..12
      year_index: int array (Y,) listing years.
    """
    import warnings
    import pandas as pd
    import xarray as xr
    import gcsfs

    fs = gcsfs.GCSFileSystem(token="anon", asynchronous=False)
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)

    y0, y1 = years
    year_index = np.arange(y0, y1 + 1, dtype=int)
    Y = len(year_index)
    P = len(grid.lats)

    # Accumulators for ensemble mean/median
    acc_sum = np.zeros((P, Y, 12), dtype=np.float64)
    acc_count = np.zeros((P, Y, 12), dtype=np.int32)

    q = col.search(
        experiment_id=experiment_id,
        table_id="Amon",
        variable_id=variable_id,
        grid_label="gn",
        source_id=models,
    )
    if len(q.df) == 0:
        raise RuntimeError(f"No assets for exp={experiment_id}, var={variable_id}")

    # Use zstore column and open zarr directly (robust)
    if "zstore" not in q.df.columns:
        raise RuntimeError("Catalog missing zstore column")

    zstores = q.df["zstore"].dropna().unique().tolist()

    # vectorized nearest selection points
    lat_da = xr.DataArray(grid.lats.astype(float), dims="points")
    lon_da = xr.DataArray(grid.lons.astype(float), dims="points")

    for z in zstores:
        try:
            mapper = fs.get_mapper(z)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                ds = xr.open_zarr(mapper, consolidated=True, decode_times=time_coder)

            if variable_id not in ds:
                continue

            da = ds[variable_id]

            # Slice time
            da = da.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))

            # Nearest point extraction for ALL points at once
            da_pt = da.sel(lat=lat_da, lon=lon_da, method="nearest")

            # Convert time to year/month
            tvals = da_pt["time"].values
            dates = np.array([month_start_from_timeobj(t) for t in tvals], dtype="datetime64[M]")
            yrs = (dates.astype("datetime64[Y]").astype(int) + 1970).astype(int)
            mos = (dates.astype("datetime64[M]").astype(int) % 12 + 1).astype(int)

            vals = da_pt.values  # (T, P)
            if vals.ndim != 2:
                continue

            # Accumulate into (P, Y, 12)
            for ti in range(vals.shape[0]):
                y = int(yrs[ti])
                m = int(mos[ti])
                if y < y0 or y > y1:
                    continue
                yi = y - y0
                mi = m - 1
                v = vals[ti, :].astype(np.float64)
                mask = np.isfinite(v)
                acc_sum[mask, yi, mi] += v[mask]
                acc_count[mask, yi, mi] += 1

        except Exception:
            # Skip broken stores
            continue

    # Finalize
    with np.errstate(invalid="ignore", divide="ignore"):
        out = acc_sum / np.maximum(acc_count, 1)

    out = out.astype(np.float32)
    return out, year_index


def save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Output folder for pack")
    ap.add_argument("--grid_deg", type=float, default=2.5, help="Grid resolution in degrees (recommended: 2.5)")
    ap.add_argument("--cat_url", default="https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    ap.add_argument("--n_models", type=int, default=10, help="Number of models to use (intersection)")
    ap.add_argument("--baseline_years", default="1950,2023", help="Baseline series range (start,end)")
    ap.add_argument("--future_years", default="2024,2050", help="Future series range (start,end)")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    b0, b1 = [int(x) for x in args.baseline_years.split(",")]
    f0, f1 = [int(x) for x in args.future_years.split(",")]
    ssp_list = ["ssp126", "ssp245", "ssp585"]

    grid = build_regular_grid(args.grid_deg)

    col = open_cmip6_catalog(args.cat_url)
    models_all = pick_models_intersection(col, ssp_list)
    if not models_all:
        raise RuntimeError("Could not find an intersection of models with required variables/experiments.")
    models = models_all[: int(args.n_models)]

    meta = {
        "pack_id": "basincast_deltapack_cmip6_v1",
        "grid_deg": float(args.grid_deg),
        "cat_url": args.cat_url,
        "models": models,
        "baseline_years": [b0, b1],
        "future_years": [f0, f1],
        "variables": ["tas", "pr"],
        "experiments": ["historical"] + ssp_list,
        "notes": "Baseline series uses historical + (optionally) scenario continuation handled at runtime if needed.",
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    save_npz(outdir / "grid.npz", lats=grid.lats.astype(np.float32), lons=grid.lons.astype(np.float32))

    # Baseline series (historical only, 1950-2014 is safest; if you want 2015-2023 baseline, blend later)
    # We keep 1950-2023 here; missing months will have NaNs if historical doesn't cover them.
    print("Building baseline (historical) tas/pr ...")
    tas_base, years_base = load_ensemble_point_matrix(col, models, "historical", "tas", (b0, b1), grid)
    pr_base, _ = load_ensemble_point_matrix(col, models, "historical", "pr", (b0, b1), grid)

    save_npz(outdir / f"baseline_tas_{b0}_{b1}.npz", years=years_base.astype(np.int16), values=tas_base)
    save_npz(outdir / f"baseline_pr_{b0}_{b1}.npz", years=years_base.astype(np.int16), values=pr_base)

    for ssp in ssp_list:
        print(f"Building future {ssp} tas/pr ...")
        tas_fut, years_fut = load_ensemble_point_matrix(col, models, ssp, "tas", (f0, f1), grid)
        pr_fut, _ = load_ensemble_point_matrix(col, models, ssp, "pr", (f0, f1), grid)

        save_npz(outdir / f"{ssp}_tas_{f0}_{f1}.npz", years=years_fut.astype(np.int16), values=tas_fut)
        save_npz(outdir / f"{ssp}_pr_{f0}_{f1}.npz", years=years_fut.astype(np.int16), values=pr_fut)

    print("DONE. Pack written to:", outdir)
    import sys, os
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()