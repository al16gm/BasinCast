Case Study 1: California Central Valley (Surface Water System)
1. Location and Hydrological Context
This case study evaluates the San Joaquin River Basin in the Central Valley of California, USA. The system is heavily dependent on snowmelt from the high-elevation Sierra Nevada range. Specifically, we test BasinCast on two interlinked surface water assets governed by the Friant Dam (completed in 1942 as part of the Central Valley Project):
Millerton Lake Reservoir: A multi-purpose surface storage facility (capacity: ~520,500 acre-feet).
San Joaquin River (below Friant): The downstream regulated river reach.
2. Why is this an interesting testbed for BasinCast?
The Central Valley of California is one of the most heavily engineered and anthropogenically altered hydrological systems in the world. Testing BasinCast here pushes the boundaries of data-driven forecasting because natural rainfall-runoff mechanics are completely decoupled from downstream flows:
Extreme Anthropogenic Stress: Millerton Lake is the primary source for two massive agricultural conveyance lifelines (the Friant-Kern and Madera Canals), requiring the model to learn complex seasonal irrigation allocations.
Institutional Rule Curves: The downstream river discharge is dictated by statutory environmental releases (San Joaquin River Restoration Program) rather than precipitation.
Compound Extreme Events: California experiences severe multi-year droughts (e.g., 2020-2022) abruptly interrupted by atmospheric river surges (e.g., Spring 2023). This forces the Model Zoo (specifically `RandomForest` and `BayesianRidge`) to handle non-linear threshold behaviors and emergency spillway gate operations.
Demonstration of Advisory Horizons: It allows us to prove that BasinCast can autonomously forecast 21 months ahead (two full irrigation campaigns) for reservoir storage, providing actionable intelligence for agricultural water districts.
3. Data Acquisition and Harmonization
To ensure reproducibility, all data for this case study was acquired from public APIs and harmonized into BasinCast's standard 25-column feature schema:
Target Telemetry (Ground Truth):
Reservoir Storage: Retrieved from the California Data Exchange Center (CDEC) API (Station ID: `MIL`). Aggregated to monthly Acre-Feet (af).
Streamflow: Retrieved from the USGS National Water Information System (NWIS) REST API (Station ID: `11251000`). Aggregated to monthly mean Cubic feet per second (cfs).
Hydrometeorological Forcings: High-resolution gridded climate data ($0.1^\circ$) for precipitation (`precip_mean`), maximum, and minimum temperatures (`tmax_mean`, `tmin_mean`) were harvested from the PRISM Climate Group (Oregon State University) and ECMWF ERA5-Land reanalysis, projected over the upstream Sierra Nevada catchment bounds.
Anthropogenic Demands: Historical monthly canal diversions (Friant-Kern and Madera canals) were extracted from USBR/CDEC records and formatted into the `agricultural_hm3` and `total_demand_hm3` columns to explicitly account for human water abstractions.
4. How to run this Case Study
To replicate the benchmarking for this case study using the BasinCast engine:
Navigate to the `/data` folder and load `MIL_CDEC_ready.csv` or `11251000_ready.csv`.
Run the `BasinCast_Core.py` pipeline.
Select the "Autonomous Multi-Step" evaluation mode setting the split date to `2019-12-31`.
