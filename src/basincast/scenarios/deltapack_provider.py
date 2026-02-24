from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd

ScenarioFamily = Literal["Favorable", "Base", "Unfavorable"]
FAMILY_TO_SSP = {"Favorable": "ssp126", "Base": "ssp245", "Unfavorable": "ssp585"}


@dataclass
class DeltaPackProvider:
    root: Path
    grid_lats: np.ndarray  # (P,)
    grid_lons: np.ndarray  # (P,) in 0..360
    meta: Dict

    @classmethod
    def load(cls, root: Path) -> "DeltaPackProvider":
        root = Path(root).resolve()
        meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        grid = np.load(root / "grid.npz")
        return cls(root=root, grid_lats=grid["lats"].astype(float), grid_lons=grid["lons"].astype(float), meta=meta)

    def _nearest_cell(self, lat: float, lon: float) -> int:
        lon2 = float(lon) % 360.0
        d2 = (self.grid_lats - float(lat)) ** 2 + (self.grid_lons - lon2) ** 2
        return int(np.argmin(d2))

    def _load_npz(self, name: str):
        return np.load(self.root / name)

    def get_deltas(
        self,
        lat: float,
        lon: float,
        family: ScenarioFamily,
        baseline_start_year: int,
        baseline_end_year: int,
        future_start_year: int,
        future_end_year: int,
    ) -> pd.DataFrame:
        """
        Returns year-month deltas:
          delta_temp_add (°C, additive)
          delta_precip_mult (ratio - 1, multiplicative)
        """
        ssp = FAMILY_TO_SSP[family]
        cell = self._nearest_cell(lat, lon)

        b0, b1 = self.meta["baseline_years"]
        f0, f1 = self.meta["future_years"]

        # Clamp to pack coverage
        bs = max(int(baseline_start_year), int(b0))
        be = min(int(baseline_end_year), int(b1))
        fs = max(int(future_start_year), int(f0))
        fe = min(int(future_end_year), int(f1))

        if bs > be:
            raise ValueError("Baseline years are outside pack coverage.")
        if fs > fe:
            raise ValueError("Future years are outside pack coverage.")

        # Load baseline series
        tas_b = self._load_npz(f"baseline_tas_{b0}_{b1}.npz")
        pr_b = self._load_npz(f"baseline_pr_{b0}_{b1}.npz")
        yb = tas_b["years"].astype(int)
        tas_bv = tas_b["values"][cell, :, :]  # (Y, 12)
        pr_bv = pr_b["values"][cell, :, :]

        # Baseline climatology for months (mean across years bs..be)
        mask_b = (yb >= bs) & (yb <= be)
        if not mask_b.any():
            raise ValueError("No baseline years found in pack for this cell.")
        tas_base_clim = np.nanmean(tas_bv[mask_b, :], axis=0)
        pr_base_clim = np.nanmean(pr_bv[mask_b, :], axis=0)

        # Load future series for SSP
        tas_f = self._load_npz(f"{ssp}_tas_{f0}_{f1}.npz")
        pr_f = self._load_npz(f"{ssp}_pr_{f0}_{f1}.npz")
        yf = tas_f["years"].astype(int)
        tas_fv = tas_f["values"][cell, :, :]  # (Y, 12)
        pr_fv = pr_f["values"][cell, :, :]

        mask_f = (yf >= fs) & (yf <= fe)
        if not mask_f.any():
            raise ValueError("No future years found in pack for this cell.")

        years = yf[mask_f]
        tas_mat = tas_fv[mask_f, :]
        pr_mat = pr_fv[mask_f, :]

        # Units: tas is Kelvin in CMIP6 -> convert to °C
        tas_base_clim_c = tas_base_clim - 273.15
        tas_mat_c = tas_mat - 273.15

        rows = []
        eps = 1e-9
        for yi, y in enumerate(years):
            for m in range(1, 13):
                mi = m - 1
                dT = float(tas_mat_c[yi, mi] - tas_base_clim_c[mi])
                dP = float((pr_mat[yi, mi] / (abs(pr_base_clim[mi]) + eps)) - 1.0)
                rows.append({"year": int(y), "month": int(m), "delta_temp_add": dT, "delta_precip_mult": dP})

        return pd.DataFrame(rows).sort_values(["year", "month"]).reset_index(drop=True)