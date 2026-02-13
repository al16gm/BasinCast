import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from pyproj import CRS, Transformer

from basincast.translator.reader import load_user_file, summarize_table
from basincast.translator.infer import infer_mapping, column_missing_pct
from basincast.translator.parsing import (
    parse_date_series,
    parse_numeric_series,
    coerce_month_start,
)
from basincast.translator.normalize import (
    CANON_RESOURCE_TYPES,
    suggest_value_map,
    apply_value_map,
)

from basincast.translator.meteo import (
    METEO_COLS,
    MeteoMapping,
    canonical_has_meteo,
    read_table_from_upload,
    build_user_meteo_table,
    fetch_nasa_power_for_canonical,
    to_month_start,
)

st.set_page_config(page_title="BasinCast Translator (v0.4)", layout="wide")
st.title("BasinCast — Input Translator (v0.4)")
st.write(
    "Upload Excel/CSV. BasinCast auto-detects fields; you confirm; then we export a canonical dataset safely.\n\n"
    "**New in v0.4:** mandatory confirmation + error-row exports + temporal integrity checks."
)

QUALITY_THRESHOLD = 0.99


def build_parse_error_rows(
    df_raw: pd.DataFrame,
    date_col: str,
    value_col: str,
    date_month: pd.Series,
    value_num: pd.Series,
) -> pd.DataFrame:
    """
    Return rows where input had something, but parsing failed.
    """
    out = df_raw.copy()
    out["_date_parsed"] = date_month
    out["_value_parsed"] = value_num

    reasons = []
    for i in range(len(out)):
        r = []
        raw_date = out.iloc[i][date_col]
        raw_val = out.iloc[i][value_col]

        date_fail = pd.notna(raw_date) and pd.isna(out.iloc[i]["_date_parsed"])
        val_fail = pd.notna(raw_val) and pd.isna(out.iloc[i]["_value_parsed"])

        if date_fail:
            r.append("date_parse_failed")
        if val_fail:
            r.append("value_parse_failed")
        reasons.append(";".join(r))

    out["_parse_error_reason"] = reasons
    err = out[out["_parse_error_reason"] != ""].copy()
    return err


def temporal_integrity_reports(canonical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      dup_rows: duplicated (point_id, date) rows
      missing_months: rows describing missing months per point_id
    """
    c = canonical.copy()
    c["date"] = pd.to_datetime(c["date"], errors="coerce")

    # duplicates
    dup_mask = c.duplicated(subset=["point_id", "date"], keep=False)
    dup_rows = c.loc[dup_mask].sort_values(["point_id", "date"]).copy()

    # missing months per point
    missing_records = []
    for pid, g in c.dropna(subset=["date"]).groupby("point_id"):
        dates = pd.to_datetime(g["date"]).sort_values()
        if dates.empty:
            continue
        start = dates.min()
        end = dates.max()
        expected = pd.date_range(start=start, end=end, freq="MS")
        observed = pd.DatetimeIndex(dates.unique())
        missing = expected.difference(observed)
        for m in missing:
            missing_records.append({"point_id": pid, "missing_month": m})

    missing_months = pd.DataFrame(missing_records)
    return dup_rows, missing_months


if "confirmed_mapping" not in st.session_state:
    st.session_state["confirmed_mapping"] = False

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
table = next(t for t in tables if t.name == selected_name)
df = table.df.copy()

st.subheader("Table summary")
st.json(summarize_table(df))

with st.expander("Show raw→normalized column name map (handles trailing spaces, invisible whitespace)"):
    st.json(table.raw_to_normalized_columns)

st.subheader("Preview")
st.dataframe(df.head(25), width="stretch")

# --- Auto-detection ---
st.subheader("1) Auto-detected mapping (editable)")
inf = infer_mapping(df)
all_cols = ["(none)"] + list(df.columns.astype(str))

date_col = st.selectbox("Date column (required)", all_cols, index=all_cols.index(inf.date_col) if inf.date_col in all_cols else 0)
point_col = st.selectbox("Point ID column (optional)", all_cols, index=all_cols.index(inf.point_id_col) if inf.point_id_col in all_cols else 0)
value_col = st.selectbox("Value column (required, ENDO target)", all_cols, index=all_cols.index(inf.value_col) if inf.value_col in all_cols else 0)
unit_col = st.selectbox("Unit column (optional)", all_cols, index=all_cols.index(inf.unit_col) if inf.unit_col in all_cols else 0)
rtype_col = st.selectbox("Resource type column (optional)", all_cols, index=all_cols.index(inf.resource_type_col) if inf.resource_type_col in all_cols else 0)

if date_col == "(none)" or value_col == "(none)":
    st.warning("Select at least Date column and Value column to continue.")
    st.stop()

# --- Robust parsing options (only if needed) ---
st.subheader("2) Robust parsing (dates & numbers)")
date_hint = "auto"
num_hint = "auto"

with st.expander("Advanced parsing options (use only if parsing fails)"):
    date_hint = st.radio("Date interpretation", ["auto", "dayfirst", "monthfirst", "excel_serial"], index=0)
    num_hint = st.radio("Number format", ["auto", "comma_decimal", "dot_decimal"], index=0)

date_parsed, date_rep = parse_date_series(df[date_col], date_hint=date_hint)
date_month = coerce_month_start(date_parsed)

value_num, value_rep = parse_numeric_series(df[value_col], locale_hint=num_hint)

c1, c2 = st.columns(2)
with c1:
    st.metric("Date parse OK ratio", f"{date_rep.ok_ratio:.3f}")
    st.caption(f"Strategy: {date_rep.strategy} | Ambiguous: {date_rep.details.get('ambiguous', False)}")
with c2:
    st.metric("Value parse OK ratio", f"{value_rep.ok_ratio:.3f}")
    st.caption(f"Strategy: {value_rep.strategy}")

# --- Parsing error rows export ---
st.subheader("2.1) Pre-flight error rows (download if any)")
error_rows = build_parse_error_rows(df, date_col, value_col, date_month, value_num)

if len(error_rows) == 0:
    st.success("No parsing errors detected ✅")
else:
    st.warning(f"Found {len(error_rows)} row(s) with parsing errors. Download and fix them, or adjust Advanced parsing options.")
    st.dataframe(error_rows.head(25), width="stretch")
    st.download_button(
        "⬇️ Download parse_error_rows.csv",
        data=error_rows.to_csv(index=False).encode("utf-8"),
        file_name="parse_error_rows.csv",
        mime="text/csv",
    )

if date_rep.details.get("ambiguous", False) and date_hint == "auto":
    st.warning("Date format looks ambiguous (dd/mm vs mm/dd). If dates look wrong, choose dayfirst/monthfirst in Advanced parsing options.")

if date_rep.ok_ratio < QUALITY_THRESHOLD or value_rep.ok_ratio < QUALITY_THRESHOLD:
    st.error(
        "Parsing quality is below threshold. Adjust Advanced parsing options until OK ratio is high "
        f"(>= {QUALITY_THRESHOLD:.2f}). Export is blocked."
    )
    with st.expander("Debug details"):
        st.write("Numeric examples:")
        st.json(value_rep.details)
        st.write("Date parsing details:")
        st.json(date_rep.details)
    st.stop()

# Effective rows: avoid blank Excel rows killing missing stats
mask_eff = date_month.notna() & value_num.notna()
df_eff = df.loc[mask_eff].copy()
df_eff["_date"] = date_month.loc[mask_eff]
df_eff["_value"] = value_num.loc[mask_eff]

st.subheader("3) Quality summary on effective rows")
n_points = df_eff[point_col].nunique() if point_col != "(none)" and point_col in df_eff.columns else 1
st.write(f"Effective rows: **{df_eff.shape[0]}** | Points detected: **{n_points}**")

# --- Coordinates ---
st.subheader("4) Coordinates")
coord_mode = st.radio("Coordinate type", ["Auto", "Lat/Lon", "UTM", "No coordinates"], index=0)

lat_col = lon_col = utm_x_col = utm_y_col = "(none)"
utm_zone = 30
utm_hemisphere = "N"

if coord_mode in ["Auto", "Lat/Lon"]:
    lat_col = st.selectbox("Latitude column", all_cols, index=all_cols.index(inf.lat_col) if inf.lat_col in all_cols else 0)
    lon_col = st.selectbox("Longitude column", all_cols, index=all_cols.index(inf.lon_col) if inf.lon_col in all_cols else 0)

if coord_mode in ["Auto", "UTM"]:
    utm_x_col = st.selectbox("UTM X (Easting) column", all_cols, index=all_cols.index(inf.utm_x_col) if inf.utm_x_col in all_cols else 0)
    utm_y_col = st.selectbox("UTM Y (Northing) column", all_cols, index=all_cols.index(inf.utm_y_col) if inf.utm_y_col in all_cols else 0)
    utm_zone = st.number_input("UTM zone", min_value=1, max_value=60, value=30, step=1)
    utm_hemisphere = st.selectbox("Hemisphere", ["N", "S"], index=0)

# --- Noise filtering (on effective rows) ---
st.subheader("5) Noise filtering (recommended)")
missing_eff = column_missing_pct(df_eff.drop(columns=["_date", "_value"], errors="ignore"))

role_cols = {c for c in [date_col, point_col, value_col, unit_col, rtype_col, lat_col, lon_col, utm_x_col, utm_y_col] if c and c != "(none)"}
default_drop = set(inf.suggested_drop_cols)
auto_drop_high_missing = set(missing_eff[missing_eff > 95.0].index.tolist())
suggested_drop = sorted(list((default_drop.union(auto_drop_high_missing)) - role_cols))

drop_cols = st.multiselect(
    "Columns to hide/ignore (UI only — does NOT change export columns)",
    list(df.columns.astype(str)),
    default=suggested_drop,
)

df_view = df.drop(columns=drop_cols, errors="ignore")
st.dataframe(df_view.head(25), width="stretch")

# --- Value mapping for resource_type and units ---
st.subheader("6) Value mapping (user confirms)")
resource_type_value_map = {}
unit_value_map = {}

if rtype_col != "(none)" and rtype_col in df_eff.columns:
    observed = sorted([v for v in df_eff[rtype_col].dropna().astype(str).map(str.strip).unique().tolist() if v != ""])
    resource_type_value_map = suggest_value_map(observed, CANON_RESOURCE_TYPES, score_cutoff=75)
    st.write("Map observed resource types → canonical (edit if needed):")
    map_df = pd.DataFrame({"observed": list(resource_type_value_map.keys()), "canonical": list(resource_type_value_map.values())})
    edited = st.data_editor(
        map_df,
        hide_index=True,
        column_config={"canonical": st.column_config.SelectboxColumn("canonical", options=CANON_RESOURCE_TYPES)},
        width="stretch",
    )
    resource_type_value_map = dict(zip(edited["observed"].astype(str), edited["canonical"].astype(str)))

if unit_col != "(none)" and unit_col in df_eff.columns:
    observed_u = sorted([v for v in df_eff[unit_col].dropna().astype(str).map(str.strip).unique().tolist() if v != ""])
    unit_value_map = {u: u for u in observed_u}
    st.write("Units (keep or standardize):")
    unit_df = pd.DataFrame({"observed": list(unit_value_map.keys()), "canonical": list(unit_value_map.values())})
    edited_u = st.data_editor(unit_df, hide_index=True, width="stretch")
    unit_value_map = dict(zip(edited_u["observed"].astype(str), edited_u["canonical"].astype(str)))

# --- Build canonical dataset (pre-check) ---
st.subheader("7) Confirmation (mandatory before export)")

# build canonical preview (without downloading yet)
d = df_eff.copy()

if point_col != "(none)" and point_col in d.columns:
    d["point_id"] = d[point_col].astype(str).map(str.strip)
else:
    d["point_id"] = "POINT_001"

d["date"] = d["_date"]
d["value"] = d["_value"]
d["unit"] = d[unit_col].astype(str).map(str.strip) if unit_col != "(none)" and unit_col in d.columns else ""
d["resource_type"] = d[rtype_col].astype(str).map(str.strip) if rtype_col != "(none)" and rtype_col in d.columns else ""

if resource_type_value_map and "resource_type" in d.columns:
    d["resource_type"] = apply_value_map(d["resource_type"], resource_type_value_map)
if unit_value_map and "unit" in d.columns:
    d["unit"] = apply_value_map(d["unit"], unit_value_map)

d["lat"] = np.nan
d["lon"] = np.nan

has_latlon = (lat_col != "(none)" and lon_col != "(none)" and lat_col in d.columns and lon_col in d.columns)
has_utm = (utm_x_col != "(none)" and utm_y_col != "(none)" and utm_x_col in d.columns and utm_y_col in d.columns)

if coord_mode == "Lat/Lon" or (coord_mode == "Auto" and has_latlon):
    lat_num, _ = parse_numeric_series(d[lat_col], locale_hint=num_hint)
    lon_num, _ = parse_numeric_series(d[lon_col], locale_hint=num_hint)
    d["lat"] = lat_num
    d["lon"] = lon_num
elif coord_mode == "UTM" or (coord_mode == "Auto" and (not has_latlon) and has_utm):
    x_num, _ = parse_numeric_series(d[utm_x_col], locale_hint=num_hint)
    y_num, _ = parse_numeric_series(d[utm_y_col], locale_hint=num_hint)
    epsg = (32600 + int(utm_zone)) if utm_hemisphere == "N" else (32700 + int(utm_zone))
    transformer = Transformer.from_crs(CRS.from_epsg(epsg), CRS.from_epsg(4326), always_xy=True)
    lon, lat = transformer.transform(x_num.to_numpy(), y_num.to_numpy())
    d["lat"] = lat
    d["lon"] = lon

canonical = d[["date", "point_id", "value", "unit", "resource_type", "lat", "lon"]].copy()
canonical = canonical.sort_values(["point_id", "date"])

st.session_state["canonical"] = canonical
# Save mapping (optional) only if the variable exists in the current script scope
try:
    st.session_state["mapping_json"] = mapping_json
except NameError:
    # Some versions name it differently or build mapping later
    st.session_state["mapping_json"] = None


# Integrity reports
dup_rows, missing_months = temporal_integrity_reports(canonical)

# Confirmation summary
summary = {
    "source_file": uploaded.name,
    "sheet": selected_name,
    "role_columns": {
        "date": date_col,
        "value": value_col,
        "point_id": point_col,
        "unit": unit_col,
        "resource_type": rtype_col,
        "coord_mode": coord_mode,
        "lat_col": lat_col,
        "lon_col": lon_col,
        "utm_x_col": utm_x_col,
        "utm_y_col": utm_y_col,
        "utm_zone": int(utm_zone),
        "utm_hemisphere": utm_hemisphere,
    },
    "parsing": {
        "date_hint": date_hint,
        "number_locale_hint": num_hint,
        "quality_threshold": QUALITY_THRESHOLD,
        "date_strategy": date_rep.strategy,
        "value_strategy": value_rep.strategy,
        "date_ok_ratio": float(date_rep.ok_ratio),
        "value_ok_ratio": float(value_rep.ok_ratio),
    },
    "data": {
        "effective_rows": int(df_eff.shape[0]),
        "points_detected": int(canonical["point_id"].nunique()),
        "date_min": str(pd.to_datetime(canonical["date"]).min().date()),
        "date_max": str(pd.to_datetime(canonical["date"]).max().date()),
    },
    "integrity": {
        "duplicate_point_date_rows": int(len(dup_rows)),
        "missing_months_rows": int(len(missing_months)),
    },
}

st.write("**Review & confirm before export:**")
st.json(summary)

cA, cB = st.columns(2)
with cA:
    st.write("Canonical preview (first 25 rows):")
    st.dataframe(canonical.head(25), width="stretch")

with cB:
    st.write("Integrity checks:")
    if len(dup_rows) == 0:
        st.success("No duplicates (point_id, date) ✅")
    else:
        st.warning(f"Duplicates found: {len(dup_rows)} row(s)")
        st.dataframe(dup_rows.head(25), width="stretch")
        st.download_button(
            "⬇️ Download duplicates_point_date.csv",
            data=dup_rows.to_csv(index=False).encode("utf-8"),
            file_name="duplicates_point_date.csv",
            mime="text/csv",
        )

    if len(missing_months) == 0:
        st.success("No missing months detected ✅")
    else:
        st.warning(f"Missing months detected: {len(missing_months)} month(s)")
        st.dataframe(missing_months.head(25), width="stretch")
        st.download_button(
            "⬇️ Download missing_months.csv",
            data=missing_months.to_csv(index=False).encode("utf-8"),
            file_name="missing_months.csv",
            mime="text/csv",
        )

import streamlit as st
import pandas as pd

st.markdown("---")
st.header("🌦️ Meteorología (opcional)")

if "canonical_df" not in st.session_state:
    st.info("Primero sube tu fichero y genera el CANONICAL en los pasos anteriores.")
    st.stop()

canonical_df = st.session_state["canonical_df"].copy()
canonical_df["date"] = to_month_start(canonical_df["date"])
canonical_df["point_id"] = canonical_df["point_id"].astype(str)

# 1) Si ya viene meteo en el archivo
if canonical_has_meteo(canonical_df):
    st.success("✅ Tu fichero YA contiene meteorología (meteo detectada en el canonical). No hay que hacer nada.")
    st.session_state["canonical_with_meteo"] = canonical_df

else:
    st.warning("Tu fichero NO trae meteorología. Vamos a resolverlo con el flujo for dummies.")

    choice = st.radio(
        "¿Cómo quieres aportar meteorología?",
        [
            "A) Puedo subir otro fichero con meteorología histórica",
            "B) No tengo meteorología → descargar de NASA POWER (sin API key)",
        ],
        index=1,
    )

    # ---------------------------------------
    # A) Upload meteo
    # ---------------------------------------
    if choice.startswith("A)"):
        st.subheader("A) Subir meteorología histórica (archivo aparte)")
        meteo_upload = st.file_uploader("Sube CSV/XLSX con meteo", type=["csv", "xlsx", "xls"], key="meteo_upload")

        if meteo_upload is not None:
            raw = read_table_from_upload(meteo_upload)
            st.write("Vista previa:")
            st.dataframe(raw.head(20), use_container_width=True)

            cols = list(raw.columns)
            if len(cols) < 2:
                st.error("El fichero meteo no tiene columnas suficientes.")
                st.stop()

            # guesses
            date_guess = "date" if "date" in [c.lower() for c in cols] else cols[0]
            precip_guess = next((c for c in cols if "precip" in c.lower() or "rain" in c.lower()), cols[0])
            t2m_guess = next((c for c in cols if "t2m" in c.lower() or ("temp" in c.lower() and "max" not in c.lower() and "min" not in c.lower())), cols[0])
            tmax_guess = next((c for c in cols if "tmax" in c.lower() or "max" in c.lower()), cols[0])
            tmin_guess = next((c for c in cols if "tmin" in c.lower() or "min" in c.lower()), cols[0])
            pid_guess = next((c for c in cols if "point" in c.lower() or "id" == c.lower() or "point_id" in c.lower()), "")

            st.markdown("### 🔧 Mapeo de columnas (confirma/corrige)")
            c1, c2, c3 = st.columns(3)
            with c1:
                date_col = st.selectbox("Columna FECHA", cols, index=cols.index(date_guess) if date_guess in cols else 0)
                precip_col = st.selectbox("Columna PRECIP", cols, index=cols.index(precip_guess) if precip_guess in cols else 0)
            with c2:
                t2m_col = st.selectbox("Columna T2M", cols, index=cols.index(t2m_guess) if t2m_guess in cols else 0)
                tmax_col = st.selectbox("Columna TMAX", cols, index=cols.index(tmax_guess) if tmax_guess in cols else 0)
            with c3:
                tmin_col = st.selectbox("Columna TMIN", cols, index=cols.index(tmin_guess) if tmin_guess in cols else 0)

            mode = st.radio(
                "¿Tu fichero meteo identifica el point_id?",
                ["Sí, tiene point_id", "No: es de un solo punto", "No: es común para todos los puntos"],
                index=0 if pid_guess else 2,
            )

            point_id_col = ""
            single_point = ""

            unique_points = sorted(canonical_df["point_id"].unique().tolist())

            if mode == "Sí, tiene point_id":
                point_id_col = st.selectbox("Columna POINT_ID", ["(elige)"] + cols, index=(cols.index(pid_guess) + 1) if pid_guess in cols else 0)
                if point_id_col == "(elige)":
                    st.error("Selecciona la columna point_id.")
                    st.stop()
                mode_key = "HAS_POINT_ID"

            elif mode == "No: es de un solo punto":
                single_point = st.selectbox("¿Para qué point_id aplica?", unique_points)
                mode_key = "SINGLE_POINT"
            else:
                mode_key = "COMMON_ALL"

            mapping = MeteoMapping(
                date_col=date_col,
                precip_col=precip_col,
                t2m_col=t2m_col,
                tmax_col=tmax_col,
                tmin_col=tmin_col,
                point_id_col=point_id_col if mode_key == "HAS_POINT_ID" else "",
            )

            if st.button("✅ Aplicar meteorología subida"):
                try:
                    meteo_df = build_user_meteo_table(
                        raw=raw,
                        canonical_point_ids=unique_points,
                        mapping=mapping,
                        mode=mode_key,
                        single_point_id=single_point,
                    )
                    canonical_with_meteo = canonical_df.merge(
                        meteo_df[["point_id", "date"] + METEO_COLS + ["source"]],
                        on=["point_id", "date"],
                        how="left",
                    )

                    st.session_state["meteo_df"] = meteo_df
                    st.session_state["canonical_with_meteo"] = canonical_with_meteo
                    st.success("✅ Meteorología integrada en canonical_with_meteo")
                except Exception as e:
                    st.error(f"Error integrando meteorología: {e}")

    # ---------------------------------------
    # B) NASA POWER
    # ---------------------------------------
    else:
        st.subheader("B) Descargar de NASA POWER (sin API key)")

        if not {"lat", "lon"}.issubset(set(canonical_df.columns)):
            st.error("Tu canonical no tiene lat/lon. Vuelve al paso de mapeo y asegúrate de incluir coordenadas.")
            st.stop()

        # check missing coords
        coords_ok = canonical_df[["lat", "lon"]].notna().all(axis=1).all()
        if not coords_ok:
            st.warning("Faltan coordenadas en algunas filas. Rellena lat/lon antes de descargar NASA.")

        if st.button("🌍 Descargar meteorología NASA POWER y crear canonical_with_meteo"):
            try:
                with st.spinner("Descargando NASA POWER... (puede tardar 5–30s por punto, queda cacheado)"):
                    meteo_df, canonical_with_meteo = fetch_nasa_power_for_canonical(canonical_df)

                st.session_state["meteo_df"] = meteo_df
                st.session_state["canonical_with_meteo"] = canonical_with_meteo
                st.success("✅ NASA POWER descargado e integrado en canonical_with_meteo")
            except Exception as e:
                st.error(f"Error NASA POWER: {e}")

# Descargas finales
if "canonical_with_meteo" in st.session_state:
    st.markdown("### 📥 Descargas")
    cwm = st.session_state["canonical_with_meteo"]
    st.download_button(
        "Download canonical_with_meteo.csv",
        data=cwm.to_csv(index=False).encode("utf-8"),
        file_name="canonical_with_meteo.csv",
        mime="text/csv",
    )

if "meteo_df" in st.session_state:
    met = st.session_state["meteo_df"]
    st.download_button(
        "Download meteo_power_monthly.csv",
        data=met.to_csv(index=False).encode("utf-8"),
        file_name="meteo_power_monthly.csv",
        mime="text/csv",
    )

# Mandatory confirmation
st.write("---")
st.write("### ✅ Step 7.1 — Confirm mapping")
if st.button("I confirm the mapping and parsing settings are correct", type="secondary"):
    st.session_state["confirmed_mapping"] = True

if not st.session_state["confirmed_mapping"]:
    st.warning("Export is locked until you confirm the mapping above.")
    st.stop()

# --- Export ---
st.subheader("8) Export canonical dataset")
mapping = {
    "source_file": uploaded.name,
    "sheet": selected_name,
    "column_name_map_raw_to_normalized": table.raw_to_normalized_columns,
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
    "ignored_columns_ui": drop_cols,
    "parsing": {
        "date_hint": date_hint,
        "number_locale_hint": num_hint,
        "quality_threshold": QUALITY_THRESHOLD,
        "date_strategy": date_rep.strategy,
        "value_strategy": value_rep.strategy,
        "date_ok_ratio": float(date_rep.ok_ratio),
        "value_ok_ratio": float(value_rep.ok_ratio),
    },
    "resource_type_value_map": resource_type_value_map,
    "unit_value_map": unit_value_map,
    "lag_policy": {"train": "drop", "forecast": "repeat_first"},
    "row_filter": {"rule": "keep rows where date and value parse correctly", "effective_rows": int(df_eff.shape[0])},
    "integrity": {
        "duplicate_point_date_rows": int(len(dup_rows)),
        "missing_months_rows": int(len(missing_months)),
    },
}

st.success("Confirmed ✅ Export is now enabled.")
st.download_button(
    "⬇️ Download canonical_timeseries.csv",
    data=canonical.to_csv(index=False).encode("utf-8"),
    file_name="canonical_timeseries.csv",
    mime="text/csv",
)
st.download_button(
    "⬇️ Download mapping.json",
    data=json.dumps(mapping, indent=2, ensure_ascii=False).encode("utf-8"),
    file_name="mapping.json",
    mime="application/json",
)