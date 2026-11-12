# Case Study 2: Seine-Normandie Paris Basin (Groundwater System)

## 1. Location and Hydrogeological Context
This case study evaluates the **Champigny Limestone and Chalk Aquifer** within the Seine-Normandie basin, located in the Île-de-France region (France). Specifically, *BasinCast* is tested on the unconfined peri-urban piezometer of **Montereau-sur-le-Jard**.
The area represents a transitional landscape where groundwater levels directly influence soil moisture for peri-urban green infrastructure, municipal parks, and local agricultural plots. 

## 2. Why is this an interesting testbed for BasinCast?
While the California case study demonstrates surface water flashiness and human regulation, the French aquifer perfectly illustrates the concept of **hydrodynamic memory and physical low-pass filtering**:
* **Deep Vadose Zone:** With a water table depth ranging between 39 and 44 meters, the thick unsaturated zone mechanically dampens high-frequency sub-seasonal storm pulses.
* **Autoregressive Dominance:** It allows us to demonstrate how *BasinCast*'s Endogenous (`ENDO`) feature space and Incremental ($\Delta y$) modeling formulation excel in environments with high storage inertia. 
* **Demonstration of Planning Horizons:** This asset serves as the ultimate benchmark for autonomous multi-step predictability. It proves that due to the slow diffusion gradients of porous media, *BasinCast* can sustain a high-fidelity Planning Horizon ($KGE \ge 0.60$) for **49 continuous months** without requiring any future meteorological forecasts, making it highly valuable for long-term municipal and ecological groundwater stewardship.

## 3. Data Acquisition and Harmonization
Consistent with the open-data philosophy of the framework, all features were programmatically retrieved and normalized:

* **Target Telemetry (Ground Truth):** Piezometric head (Water Table Depth in meters) was retrieved from the French National Groundwater portal, **Hub'eau Piezométrie API** (Managed by BRGM). Station ID: `02206X0022/S1`. Missing values (< 1% of the historical record) were patched using Piecewise Cubic Hermite Interpolating Polynomials (PCHIP) to preserve monotonic trends.
* **Hydrometeorological Forcings:** Gridded climate variables (Precipitation, Maximum Temperature, Minimum Temperature) were extracted from the **ECMWF ERA5-Land** reanalysis platform using the Copernicus Climate Data Store (CDS) API, bounded by the local UTM Zone 31N coordinates.
* **Anthropogenic Demands:** Localized urban green-space irrigation requirements and minor municipal abstractions were aggregated into the `domestic_hm3` and `total_demand_hm3` input vectors to capture local summer depletion trends.

## 4. How to run this Case Study
To replicate the piezometric forecasting for this case study using the *BasinCast* engine:
1. Navigate to the `/data` folder and load `02206X0022_S1_ready.csv`.
2. Run the `BasinCast_Core.py` pipeline.
3. Observe how the *Model Zoo* autonomously converges towards the `BayesianRidge` algorithm to optimally handle the continuous, linear nature of the aquifer's hydrodynamic memory.