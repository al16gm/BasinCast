import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from pyproj import CRS, Transformer

from basincast.translator.reader import load_user_file, summarize_table
from basincast.translator.infer import infer_mapping, column_missing_pct


st.set_page_config(page_title="BasinCast Translator (v0.2)", layout="wide")
st.title("BasinCast — Input Translator (v0.2)")
st.write("Upload an Excel/CSV. BasinCast will **auto-detect** fields, you **confirm**, then we export a canonical dataset.")


uploaded = st.file_uploader("Upload a file (.xlsx, .xls, .csv)", type=["xlsx", "xls", "csv"])

if uploaded is None:
    st.stop()

suffix = Path(uploaded.name).suffix
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.getbuffer())
    tmp_path = tmp.name

tables = load_user_file(tmp_path)
st.success(f"Loaded {len(tables)} table(s).")

table_names = [t.name for t in tables]
selected_name = st.selectbox("Select table/sheet", table_names, index=0)
df = next(t.df for t in tables if t.name == selected_name).copy()

info = summarize_table(df)
st.subheader("Table summary")
st.json(info)

st.subheader("Preview")
st.dataframe(df.head(25), width="stretch")

st.subheader("Auto-detected mapping (editable)")
inf = infer_mapping(df)

cols = ["(none)"] + list(df.columns.astype(str))

date_col = st.selectbox("Date column", cols, index=cols.index(inf.date_col) if inf.date_col in cols else 0)
point_col = st.selectbox("Point ID column (optional)", cols, index=cols.index(inf.point_id_col) if inf.point_id_col in cols else 0)
value_col = st.selectbox("Value column (ENDO target)", cols, index=cols.index(inf.value_col) if inf.value_col in cols else 0)
unit_col = st.selectbox("Unit column (optional)", cols, index=cols.index(inf.unit_col) if inf.unit_col in cols else 0)
rtype_col = st.selectbox("Resource type column (optional)", cols, index=cols.index(inf.resource_type_col) if inf.resource_type_col in cols else 0)

st.markdown("### Coordinates")
coord_mode = st.radio("Coordinate type", ["Auto", "Lat/Lon", "UTM", "No coordinates"], index=0)

lat_col = lon_col = utm_x_col = utm_y_col = "(none)"
utm_zone = 30
utm_hemisphere = "N"

if coord_mode in ["Auto", "Lat/Lon"]:
    lat_col = st.selectbox("Latitude column", cols, index=cols.index(inf.lat_col) if inf.lat_col in cols else 0)
    lon_col = st.selectbox("Longitude column", cols, index=cols.index(inf.lon_col) if inf.lon_col in cols else 0)

if coord_mode in ["Auto", "UTM"]:
    utm_x_col = st.selectbox("UTM X (Easting) column", cols, index=cols.index(inf.utm_x_col) if inf.utm_x_col in cols else 0)
    utm_y_col = st.selectbox("UTM Y (Northing) column", cols, index=cols.index(inf.utm_y_col) if inf.utm_y_col in cols else 0)
    utm_zone = st.number_input("UTM zone (integer)", min_value=1, max_value=60, value=30, step=1)
    utm_hemisphere = st.selectbox("Hemisphere", ["N", "S"], index=0)

st.markdown("### Noise filtering")
missing = column_missing_pct(df)
with st.expander("Show missing % by column"):
    st.dataframe(missing.to_frame("missing_pct"), width="stretch")

default_drop = set(inf.suggested_drop_cols)
auto_drop_high_missing = set(missing[missing > 95.0].index.tolist())
suggested_drop = sorted(list(default_drop.union(auto_drop_high_missing)))

drop_cols = st.multiselect(
    "Columns to hide/ignore (recommended pre-filled)",
    list(df.columns.astype(str)),
    default=suggested_drop,
)

if drop_cols:
    df_view = df.drop(columns=drop_cols, errors="ignore")
else:
    df_view = df

st.subheader("Preview after hiding ignored columns")
st.dataframe(df_view.head(25), width="stretch")

st.markdown("## Build canonical dataset")
if st.button("✅ Build & Export canonical CSV", type="primary"):
    if date_col == "(none)" or value_col == "(none)":
        st.error("You must select at least Date column and Value column.")
        st.stop()

    d = df.copy()

    # Parse dates
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])

    # Point id
    if point_col != "(none)":
        d["point_id"] = d[point_col].astype(str)
    else:
        single_id = st.text_input("No point_id column detected. Enter a point id:", value="POINT_001")
        d["point_id"] = single_id

    # Value
    d["value"] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=["value"])

    # Unit/type
    d["unit"] = d[unit_col].astype(str) if unit_col != "(none)" else ""
    d["resource_type"] = d[rtype_col].astype(str) if rtype_col != "(none)" else ""

    # Coordinates handling
    d["lat"] = None
    d["lon"] = None

    has_latlon = (lat_col != "(none)" and lon_col != "(none)")
    has_utm = (utm_x_col != "(none)" and utm_y_col != "(none)")

    if coord_mode == "Lat/Lon" or (coord_mode == "Auto" and has_latlon):
        d["lat"] = pd.to_numeric(d[lat_col], errors="coerce")
        d["lon"] = pd.to_numeric(d[lon_col], errors="coerce")

    elif coord_mode == "UTM" or (coord_mode == "Auto" and (not has_latlon) and has_utm):
        x = pd.to_numeric(d[utm_x_col], errors="coerce")
        y = pd.to_numeric(d[utm_y_col], errors="coerce")

        epsg = (32600 + int(utm_zone)) if utm_hemisphere == "N" else (32700 + int(utm_zone))
        transformer = Transformer.from_crs(CRS.from_epsg(epsg), CRS.from_epsg(4326), always_xy=True)

        lon, lat = transformer.transform(x.to_numpy(), y.to_numpy())
        d["lat"] = lat
        d["lon"] = lon

    # Keep only canonical columns
    canonical = d[[date_col, "point_id", "value", "unit", "resource_type", "lat", "lon"]].copy()
    canonical = canonical.rename(columns={date_col: "date"})
    canonical = canonical.sort_values(["point_id", "date"])

    # Build mapping for reproducibility + original labels
    mapping = {
        "source_file": uploaded.name,
        "sheet": selected_name,
        "date_col": date_col,
        "point_id_col": point_col,
        "value_col": value_col,
        "unit_col": unit_col,
        "resource_type_col": rtype_col,
        "coord_mode": coord_mode,
        "lat_col": lat_col,
        "lon_col": lon_col,
        "utm_x_col": utm_x_col,
        "utm_y_col": utm_y_col,
        "utm_zone": int(utm_zone),
        "utm_hemisphere": utm_hemisphere,
        "ignored_columns": drop_cols,
    }

    st.success(f"Canonical dataset built: {canonical.shape[0]} rows | {canonical['point_id'].nunique()} point(s)")
    st.dataframe(canonical.head(25), width="stretch")

    csv_bytes = canonical.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download canonical_timeseries.csv", data=csv_bytes, file_name="canonical_timeseries.csv", mime="text/csv")

    json_bytes = json.dumps(mapping, indent=2).encode("utf-8")
    st.download_button("⬇️ Download mapping.json", data=json_bytes, file_name="mapping.json", mime="application/json")