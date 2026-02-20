from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .cache import CacheKey, DeltaCache, hash_models


ScenarioFamily = Literal["Favorable", "Base", "Unfavorable"]

FAMILY_TO_SSP = {
    "Favorable": "ssp126",
    "Base": "ssp245",
    "Unfavorable": "ssp585",
}


@dataclass
class CMIP6IntakeDeltaProvider:
    """
    CMIP6 delta provider using cloud catalogs (intake-esm) + local cache.

    Produces year-month deltas relative to a dynamic baseline period.

    Notes
    -----
    - Temperature deltas are additive (°C).
    - Precipitation deltas are multiplicative (% as ratio - 1).
    """
    cache_dir: Path
    models: Optional[list[str]] = None
    statistic: Literal["mean", "median"] = "mean"
    grid_method: Literal["nearest"] = "nearest"  # keep it simple/fast
    cat_url: Optional[str] = None  # allow override

    def __post_init__(self) -> None:
        self.cache = DeltaCache(Path(self.cache_dir))

    def get_deltas(
        self,
        lat: float,
        lon: float,
        family: ScenarioFamily,
        baseline_start: int,
        baseline_end: int,
        future_start: int,
        future_end: int,
    ) -> pd.DataFrame:
        ssp = FAMILY_TO_SSP[family]
        key = CacheKey(
            lat=lat, lon=lon, ssp=ssp,
            baseline_start=baseline_start, baseline_end=baseline_end,
            future_start=future_start, future_end=future_end,
            models_hash=hash_models(self.models),
        )
        cached = self.cache.get(key)
        if cached is not None and not cached.empty:
            return cached

        df = self._compute_deltas_intake(
            lat=lat, lon=lon, ssp=ssp,
            baseline_start=baseline_start, baseline_end=baseline_end,
            future_start=future_start, future_end=future_end,
        )
        self.cache.put(key, df)
        return df

    def _compute_deltas_intake(
        self,
        lat: float,
        lon: float,
        ssp: str,
        baseline_start: int,
        baseline_end: int,
        future_start: int,
        future_end: int,
    ) -> pd.DataFrame:
        import intake
        import intake_esm
        import xarray as xr

        # Default catalog: Google CMIP6 cloud catalog via intake-esm docs/notebooks
        # Users can override cat_url if needed.
        cat_url = self.cat_url or "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
        col = intake.open_esm_datastore(cat_url)

        # We need:
        # - historical tas/pr for baseline
        # - scenario (ssp*) tas/pr for future
        # monthly (Amon)
        query_common = dict(
            table_id="Amon",
            variable_id=["tas", "pr"],
            grid_label="gn",
        )
        if self.models:
            query_common["source_id"] = self.models

        hist = col.search(experiment_id="historical", **query_common)
        fut = col.search(experiment_id=ssp, **query_common)

        if len(hist.df) == 0 or len(fut.df) == 0:
            raise RuntimeError("CMIP6 catalog query returned no matches (check catalog availability / filters).")

        # Load to dict of datasets
        # aggregate=False because we may have multiple models; we'll reduce later.
        d_hist = hist.to_dataset_dict(zarr_kwargs={"consolidated": True}, aggregate=False)
        d_fut = fut.to_dataset_dict(zarr_kwargs={"consolidated": True}, aggregate=False)

        # Helper to extract point timeseries from many datasets
        def extract_point_series(ds_dict: dict, var: str, y0: int, y1: int) -> pd.DataFrame:
            frames = []
            for _, ds in ds_dict.items():
                if var not in ds:
                    continue
                da = ds[var]

                # Standardize coords
                # CMIP6 uses lat/lon coordinates; select nearest
                da_pt = da.sel(lat=lat, lon=lon, method=self.grid_method)

                # Ensure time is decoded and monthly
                da_pt = da_pt.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))

                # Convert to pandas
                s = da_pt.to_series()
                s.index = pd.to_datetime(s.index)
                s = s.resample("MS").mean()
                frames.append(s.rename("value"))

            if not frames:
                return pd.DataFrame(columns=["date", "value"])

            df = pd.concat(frames, axis=1)
            # Ensemble statistic across models
            if self.statistic == "median":
                v = df.median(axis=1)
            else:
                v = df.mean(axis=1)

            out = v.reset_index()
            out.columns = ["date", "value"]
            out["date"] = pd.to_datetime(out["date"]).dt.to_period("M").dt.to_timestamp()
            out = out.dropna(subset=["date"]).sort_values("date")
            return out

        tas_base = extract_point_series(d_hist, "tas", baseline_start, baseline_end)
        pr_base = extract_point_series(d_hist, "pr", baseline_start, baseline_end)

        tas_fut = extract_point_series(d_fut, "tas", future_start, future_end)
        pr_fut = extract_point_series(d_fut, "pr", future_start, future_end)

        if tas_base.empty or pr_base.empty or tas_fut.empty or pr_fut.empty:
            raise RuntimeError("CMIP6 time series extraction returned empty results.")

        # Convert units:
        # tas in K -> °C
        tas_base["value"] = tas_base["value"].astype(float) - 273.15
        tas_fut["value"] = tas_fut["value"].astype(float) - 273.15

        # pr in kg m-2 s-1 -> mm/day or mm/month:
        # For monthly deltas, ratio is unitless; we can keep in original units,
        # but avoid zeros -> use climatology with epsilon.
        # We'll compute climatologies month-wise/year-wise on the extracted values.
        # (Still ok.)

        # Baseline monthly climatology (1988-2023 or dynamic)
        tas_base["year"] = tas_base["date"].dt.year
        tas_base["month"] = tas_base["date"].dt.month
        pr_base["year"] = pr_base["date"].dt.year
        pr_base["month"] = pr_base["date"].dt.month

        tas_fut["year"] = tas_fut["date"].dt.year
        tas_fut["month"] = tas_fut["date"].dt.month
        pr_fut["year"] = pr_fut["date"].dt.year
        pr_fut["month"] = pr_fut["date"].dt.month

        tas_base_clim = tas_base.groupby("month")["value"].mean().rename("tas_base").reset_index()
        pr_base_clim = pr_base.groupby("month")["value"].mean().rename("pr_base").reset_index()

        # Future year-month means
        tas_fut_ym = tas_fut.groupby(["year", "month"])["value"].mean().rename("tas_fut").reset_index()
        pr_fut_ym = pr_fut.groupby(["year", "month"])["value"].mean().rename("pr_fut").reset_index()

        # Merge and compute deltas
        df = tas_fut_ym.merge(pr_fut_ym, on=["year", "month"], how="inner")
        df = df.merge(tas_base_clim, on="month", how="left")
        df = df.merge(pr_base_clim, on="month", how="left")

        df["delta_temp_add"] = df["tas_fut"] - df["tas_base"]

        eps = 1e-9
        df["delta_precip_mult"] = (df["pr_fut"] / (df["pr_base"].abs() + eps)) - 1.0

        out = df[["year", "month", "delta_temp_add", "delta_precip_mult"]].copy()
        out = out.sort_values(["year", "month"]).reset_index(drop=True)
        return out