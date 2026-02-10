# BasinCast

**Climate-informed multi-horizon forecasting for regulated basins.**  
BasinCast is an open-source framework and web app to generate **monthly forecasts (1–48 months)** of hydrological states (aquifers, reservoirs, rivers) using **ENDO vs EXOG** feature families and **scenario-conditioned** climate forcings.

> **Status:** active development (pre-release).  
> **License:** MIT.

---

## Cite BasinCast

If you use BasinCast in academic work, please cite the Zenodo archive:

- **Concept DOI (all versions):**(https://doi.org/10.5281/zenodo.18596221

> For journal articles, we recommend citing the **specific version DOI** (e.g., v1.0.0) once the paper-ready release is available.

You can also use GitHub’s citation box (powered by CITATION.cff).

---

## Key features

- **No-code workflow:** upload your endogenous historical series + point metadata (lat/lon).
- **Automatic meteorology by location:** download official reanalysis / gridded observations (default: **Copernicus ERA5**) and derive monthly predictors.
- **Two feature families:**
  - **ENDO:** memory-only (lags + seasonality)
  - **EXOG:** ENDO + climate/demand forcings
- **Multi-horizon forecasting:** recursive strategy in incremental space (Δy → y) with guardrails.
- **Diagnostics:** hindcast skill (KGE/NSE/RMSE) + reliability horizon.
- **Reproducible:** Docker-ready + versioned releases with Zenodo DOI.

---

## Quick start

### Option A — Run the web app (recommended)
- **Public app:** YOUR_APP_URL (coming soon)

### Option B — Local installation (Python)

```bash
git clone https://github.com/YOUR_GITHUB_USER/BasinCast.git
cd BasinCast

# (recommended) create a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app/streamlit_app.py
