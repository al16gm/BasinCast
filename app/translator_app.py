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

from basincast.translator.i18n import language_selector, tr

APP_VERSION = "v0.7"

st.set_page_config(page_title=f"BasinCast Translator ({APP_VERSION})", layout="wide")

# Language selector (sidebar)
LANG = language_selector(default="en")

st.title(tr("BasinCast — Traductor de Entrada", "BasinCast — Input Translator", LANG) + f" ({APP_VERSION})")
st.write(
    tr(
        "Sube un Excel/CSV. BasinCast detecta campos; tú confirmas; y exportamos un dataset CANONICAL seguro.\n\n"
        "**Incluye:** parsing robusto de fechas/números, ocultar ruido, UTM→Lat/Lon, checks de integridad y meteorología opcional.",
        "Upload an Excel/CSV. BasinCast auto-detects fields; you confirm; then we export a safe CANONICAL dataset.\n\n"
        "**Includes:** robust date/number parsing, noise hiding, UTM→Lat/Lon, integrity checks, and optional meteorology.",
        LANG,
    )
)

QUALITY_THRESHOLD = 0.99


def build_parse_error_rows(
    df_raw: pd.DataFrame,
    date_col: str,
    value_col: str,
    date_month: pd.Series,
    value_num: pd.Series,
) -> pd.DataFrame:
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
    c = canonical.copy()
    c["date"] = pd.to_datetime(c["date"], errors="coerce")

    dup_mask = c.duplicated(subset=["point_id", "date"], keep=False)
    dup_rows = c.loc[dup_mask].sort_values(["point_id", "date"]).copy()

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


# -----------------------------
# Session defaults
# -----------------------------
if "confirmed_mapping" not in st.session_state:
    st.session_state["confirmed_mapping"] = False


# -----------------------------
# Upload
# -----------------------------
uploaded = st.file_uploader(
    tr("Sube un archivo (.xlsx, .xls, .csv)", "Upload a file (.xlsx, .xls, .csv)", LANG),
    type=["xlsx", "xls", "csv"],
)
if uploaded is None:
    st.stop()

suffix = Path(uploaded.name).suffix
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.getbuffer())
    tmp_path = tmp.name

tables = load_user_file(tmp_path)
st.success(tr(f"Cargadas {len(tables)} tabla(s).", f"Loaded {len(tables)} table(s).", LANG))

table_names = [t.name for t in tables]
selected_name = st.selectbox(tr("Selecciona tabla/hoja", "Select table/sheet", LANG), table_names, index=0)
table = next(t for t in tables if t.name == selected_name)
df = table.df.copy()

st.subheader(tr("Resumen de la tabla", "Table summary", LANG))
st.json(summarize_table(df))

with st.expander(tr("Ver mapa raw→normalizado de nombres de columnas", "Show raw→normalized column name map", LANG)):
    st.json(table.raw_to_normalized_columns)

st.subheader(tr("Vista previa", "Preview", LANG))
st.dataframe(df.head(25), use_container_width=True)

# -----------------------------
# 1) Auto-detection
# -----------------------------
st.subheader(tr("1) Mapeo autodetectado (editable)", "1) Auto-detected mapping (editable)", LANG))
inf = infer_mapping(df)
all_cols = ["(none)"] + list(df.columns.astype(str))

main_date_col = st.selectbox(tr("Columna FECHA (obligatoria)", "Date column (required)", LANG), all_cols,
                            index=all_cols.index(inf.date_col) if inf.date_col in all_cols else 0)
main_point_col = st.selectbox(tr("Columna ID de punto (opcional)", "Point ID column (optional)", LANG), all_cols,
                             index=all_cols.index(inf.point_id_col) if inf.point_id_col in all_cols else 0)
main_value_col = st.selectbox(tr("Columna VALOR (obligatoria, ENDO)", "Value column (required, ENDO target)", LANG), all_cols,
                             index=all_cols.index(inf.value_col) if inf.value_col in all_cols else 0)
main_unit_col = st.selectbox(tr("Columna UNIDAD (opcional)", "Unit column (optional)", LANG), all_cols,
                            index=all_cols.index(inf.unit_col) if inf.unit_col in all_cols else 0)
main_rtype_col = st.selectbox(tr("Columna TIPO DE RECURSO (opcional)", "Resource type column (optional)", LANG), all_cols,
                             index=all_cols.index(inf.resource_type_col) if inf.resource_type_col in all_cols else 0)

if main_date_col == "(none)" or main_value_col == "(none)":
    st.warning(tr("Selecciona al menos FECHA y VALOR para continuar.", "Select at least Date and Value to continue.", LANG))
    st.stop()

# -----------------------------
# 2) Parsing (robust)
# -----------------------------
st.subheader(tr("2) Parsing robusto (fechas y números)", "2) Robust parsing (dates & numbers)", LANG))
date_hint = "auto"
num_hint = "auto"

with st.expander(tr("Opciones avanzadas (solo si falla el parsing)", "Advanced options (use only if parsing fails)", LANG)):
    date_hint = st.radio(tr("Interpretación de fecha", "Date interpretation", LANG),
                         ["auto", "dayfirst", "monthfirst", "excel_serial"], index=0)
    num_hint = st.radio(tr("Formato numérico", "Number format", LANG),
                        ["auto", "comma_decimal", "dot_decimal"], index=0)

date_parsed, date_rep = parse_date_series(df[main_date_col], date_hint=date_hint)
date_month = coerce_month_start(date_parsed)

value_num, value_rep = parse_numeric_series(df[main_value_col], locale_hint=num_hint)

c1, c2 = st.columns(2)
with c1:
    st.metric(tr("Ratio OK (fecha)", "Date parse OK ratio", LANG), f"{date_rep.ok_ratio:.3f}")
    st.caption(f"{tr('Estrategia', 'Strategy', LANG)}: {date_rep.strategy} | {tr('Ambigua', 'Ambiguous', LANG)}: {date_rep.details.get('ambiguous', False)}")
with c2:
    st.metric(tr("Ratio OK (valor)", "Value parse OK ratio", LANG), f"{value_rep.ok_ratio:.3f}")
    st.caption(f"{tr('Estrategia', 'Strategy', LANG)}: {value_rep.strategy}")

st.subheader(tr("2.1) Filas con errores de parsing (descarga si hay)", "2.1) Pre-flight error rows (download if any)", LANG))
error_rows = build_parse_error_rows(df, main_date_col, main_value_col, date_month, value_num)

if len(error_rows) == 0:
    st.success(tr("No se detectan errores de parsing ✅", "No parsing errors detected ✅", LANG))
else:
    st.warning(tr(f"Hay {len(error_rows)} fila(s) con error. Descárgalas y corrige, o ajusta opciones avanzadas.",
                  f"Found {len(error_rows)} row(s) with parsing errors. Download and fix them, or adjust Advanced options.",
                  LANG))
    st.dataframe(error_rows.head(25), use_container_width=True)
    st.download_button(
        tr("⬇️ Descargar parse_error_rows.csv", "⬇️ Download parse_error_rows.csv", LANG),
        data=error_rows.to_csv(index=False).encode("utf-8"),
        file_name="parse_error_rows.csv",
        mime="text/csv",
    )

if date_rep.details.get("ambiguous", False) and date_hint == "auto":
    st.warning(tr("Formato de fecha ambiguo (dd/mm vs mm/dd). Si se ven mal, elige dayfirst/monthfirst.",
                  "Date format looks ambiguous (dd/mm vs mm/dd). If wrong, choose dayfirst/monthfirst.",
                  LANG))

if date_rep.ok_ratio < QUALITY_THRESHOLD or value_rep.ok_ratio < QUALITY_THRESHOLD:
    st.error(
        tr(
            f"La calidad de parsing es baja. Ajusta opciones hasta ratio OK >= {QUALITY_THRESHOLD:.2f}. Export bloqueado.",
            f"Parsing quality is below threshold. Adjust options until OK ratio >= {QUALITY_THRESHOLD:.2f}. Export blocked.",
            LANG,
        )
    )
    with st.expander(tr("Detalles (debug)", "Debug details", LANG)):
        st.write(tr("Ejemplos numéricos:", "Numeric examples:", LANG))
        st.json(value_rep.details)
        st.write(tr("Detalles parsing fecha:", "Date parsing details:", LANG))
        st.json(date_rep.details)
    st.stop()

mask_eff = date_month.notna() & value_num.notna()
df_eff = df.loc[mask_eff].copy()
df_eff["_date"] = date_month.loc[mask_eff]
df_eff["_value"] = value_num.loc[mask_eff]

st.subheader(tr("3) Resumen de calidad (filas efectivas)", "3) Quality summary on effective rows", LANG))
n_points = df_eff[main_point_col].nunique() if main_point_col != "(none)" and main_point_col in df_eff.columns else 1
st.write(tr(f"Filas efectivas: **{df_eff.shape[0]}** | Puntos detectados: **{n_points}**",
            f"Effective rows: **{df_eff.shape[0]}** | Points detected: **{n_points}**",
            LANG))

# -----------------------------
# 4) Coordinates
# -----------------------------
st.subheader(tr("4) Coordenadas", "4) Coordinates", LANG))
coord_mode = st.radio(tr("Tipo de coordenadas", "Coordinate type", LANG),
                      ["Auto", "Lat/Lon", "UTM", "No coordinates"], index=0)

lat_col = lon_col = utm_x_col = utm_y_col = "(none)"
utm_zone = 30
utm_hemisphere = "N"

if coord_mode in ["Auto", "Lat/Lon"]:
    lat_col = st.selectbox(tr("Columna LATITUD", "Latitude column", LANG), all_cols,
                           index=all_cols.index(inf.lat_col) if inf.lat_col in all_cols else 0)
    lon_col = st.selectbox(tr("Columna LONGITUD", "Longitude column", LANG), all_cols,
                           index=all_cols.index(inf.lon_col) if inf.lon_col in all_cols else 0)

if coord_mode in ["Auto", "UTM"]:
    utm_x_col = st.selectbox(tr("UTM X (Easting)", "UTM X (Easting)", LANG), all_cols,
                             index=all_cols.index(inf.utm_x_col) if inf.utm_x_col in all_cols else 0)
    utm_y_col = st.selectbox(tr("UTM Y (Northing)", "UTM Y (Northing)", LANG), all_cols,
                             index=all_cols.index(inf.utm_y_col) if inf.utm_y_col in all_cols else 0)
    utm_zone = st.number_input(tr("Zona UTM", "UTM zone", LANG), min_value=1, max_value=60, value=30, step=1)
    utm_hemisphere = st.selectbox(tr("Hemisferio", "Hemisphere", LANG), ["N", "S"], index=0)

# -----------------------------
# 5) Noise filtering (UI only)
# -----------------------------
st.subheader(tr("5) Ocultar ruido (recomendado)", "5) Noise filtering (recommended)", LANG))
missing_eff = column_missing_pct(df_eff.drop(columns=["_date", "_value"], errors="ignore"))

role_cols = {c for c in [main_date_col, main_point_col, main_value_col, main_unit_col, main_rtype_col, lat_col, lon_col, utm_x_col, utm_y_col] if c and c != "(none)"}
default_drop = set(inf.suggested_drop_cols)
auto_drop_high_missing = set(missing_eff[missing_eff > 95.0].index.tolist())
suggested_drop = sorted(list((default_drop.union(auto_drop_high_missing)) - role_cols))

drop_cols = st.multiselect(
    tr("Columnas a ocultar/ignorar (solo UI)", "Columns to hide/ignore (UI only)", LANG),
    list(df.columns.astype(str)),
    default=suggested_drop,
)

df_view = df.drop(columns=drop_cols, errors="ignore")
st.dataframe(df_view.head(25), use_container_width=True)

# -----------------------------
# 6) Value mapping (confirm)
# -----------------------------
st.subheader(tr("6) Mapeo de valores (confirma)", "6) Value mapping (confirm)", LANG))
resource_type_value_map = {}
unit_value_map = {}

if main_rtype_col != "(none)" and main_rtype_col in df_eff.columns:
    observed = sorted([v for v in df_eff[main_rtype_col].dropna().astype(str).map(str.strip).unique().tolist() if v != ""])
    resource_type_value_map = suggest_value_map(observed, CANON_RESOURCE_TYPES, score_cutoff=75)
    st.write(tr("Mapea tipos observados → canónico (edita si hace falta):",
                "Map observed resource types → canonical (edit if needed):",
                LANG))
    map_df = pd.DataFrame({"observed": list(resource_type_value_map.keys()), "canonical": list(resource_type_value_map.values())})
    edited = st.data_editor(
        map_df,
        hide_index=True,
        column_config={"canonical": st.column_config.SelectboxColumn("canonical", options=CANON_RESOURCE_TYPES)},
        use_container_width=True,
    )
    resource_type_value_map = dict(zip(edited["observed"].astype(str), edited["canonical"].astype(str)))

if main_unit_col != "(none)" and main_unit_col in df_eff.columns:
    observed_u = sorted([v for v in df_eff[main_unit_col].dropna().astype(str).map(str.strip).unique().tolist() if v != ""])
    unit_value_map = {u: u for u in observed_u}
    st.write(tr("Unidades (mantener o estandarizar):", "Units (keep or standardize):", LANG))
    unit_df = pd.DataFrame({"observed": list(unit_value_map.keys()), "canonical": list(unit_value_map.values())})
    edited_u = st.data_editor(unit_df, hide_index=True, use_container_width=True)
    unit_value_map = dict(zip(edited_u["observed"].astype(str), edited_u["canonical"].astype(str)))

# -----------------------------
# 7) Build canonical + integrity
# -----------------------------
st.subheader(tr("7) Confirmación (obligatoria antes de exportar)", "7) Confirmation (mandatory before export)", LANG))

d = df_eff.copy()

if main_point_col != "(none)" and main_point_col in d.columns:
    d["point_id"] = d[main_point_col].astype(str).map(str.strip)
else:
    d["point_id"] = "POINT_001"

d["date"] = d["_date"]
d["value"] = d["_value"]
d["unit"] = d[main_unit_col].astype(str).map(str.strip) if main_unit_col != "(none)" and main_unit_col in d.columns else ""
d["resource_type"] = d[main_rtype_col].astype(str).map(str.strip) if main_rtype_col != "(none)" and main_rtype_col in d.columns else ""

if resource_type_value_map:
    d["resource_type"] = apply_value_map(d["resource_type"], resource_type_value_map)
if unit_value_map:
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
canonical = canonical.sort_values(["point_id", "date"]).reset_index(drop=True)

# Store both keys to avoid future mismatches
st.session_state["canonical"] = canonical
st.session_state["canonical_df"] = canonical

dup_rows, missing_months = temporal_integrity_reports(canonical)

summary = {
    "source_file": uploaded.name,
    "sheet": selected_name,
    "role_columns": {
        "date": main_date_col,
        "value": main_value_col,
        "point_id": main_point_col,
        "unit": main_unit_col,
        "resource_type": main_rtype_col,
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

st.write(tr("**Revisa y confirma antes de exportar:**", "**Review & confirm before export:**", LANG))
st.json(summary)

cA, cB = st.columns(2)
with cA:
    st.write(tr("Vista previa CANONICAL (25 filas):", "Canonical preview (first 25 rows):", LANG))
    st.dataframe(canonical.head(25), use_container_width=True)

with cB:
    st.write(tr("Checks de integridad:", "Integrity checks:", LANG))
    if len(dup_rows) == 0:
        st.success(tr("Sin duplicados (point_id, date) ✅", "No duplicates (point_id, date) ✅", LANG))
    else:
        st.warning(tr(f"Duplicados: {len(dup_rows)} fila(s)", f"Duplicates found: {len(dup_rows)} row(s)", LANG))
        st.dataframe(dup_rows.head(25), use_container_width=True)
        st.download_button(
            tr("⬇️ Descargar duplicates_point_date.csv", "⬇️ Download duplicates_point_date.csv", LANG),
            data=dup_rows.to_csv(index=False).encode("utf-8"),
            file_name="duplicates_point_date.csv",
            mime="text/csv",
        )

    if len(missing_months) == 0:
        st.success(tr("No faltan meses ✅", "No missing months detected ✅", LANG))
    else:
        st.warning(tr(f"Meses faltantes: {len(missing_months)}", f"Missing months detected: {len(missing_months)}", LANG))
        st.dataframe(missing_months.head(25), use_container_width=True)
        st.download_button(
            tr("⬇️ Descargar missing_months.csv", "⬇️ Download missing_months.csv", LANG),
            data=missing_months.to_csv(index=False).encode("utf-8"),
            file_name="missing_months.csv",
            mime="text/csv",
        )

st.write("---")
confirm = st.checkbox(tr("✅ Confirmo que el mapeo/parsing es correcto (obligatorio)", "✅ I confirm mapping/parsing is correct (required)", LANG))
st.session_state["confirmed_mapping"] = bool(confirm)

if not st.session_state["confirmed_mapping"]:
    st.warning(tr("Export bloqueado hasta confirmar.", "Export is locked until you confirm.", LANG))
    st.stop()

# -----------------------------
# 8) Export canonical + mapping
# -----------------------------
st.subheader(tr("8) Exportar CANONICAL + mapping", "8) Export canonical + mapping", LANG))

mapping_json_out = {
    "source_file": uploaded.name,
    "sheet": selected_name,
    "column_name_map_raw_to_normalized": table.raw_to_normalized_columns,
    "date_col": main_date_col,
    "point_id_col": main_point_col,
    "value_col": main_value_col,
    "unit_col": main_unit_col,
    "resource_type_col": main_rtype_col,
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
    "ui": {"language": LANG},
}

st.success(tr("Confirmado ✅ Export habilitado.", "Confirmed ✅ Export enabled.", LANG))
st.download_button(
    tr("⬇️ Descargar canonical_timeseries.csv", "⬇️ Download canonical_timeseries.csv", LANG),
    data=canonical.to_csv(index=False).encode("utf-8"),
    file_name="canonical_timeseries.csv",
    mime="text/csv",
)
st.download_button(
    tr("⬇️ Descargar mapping.json", "⬇️ Download mapping.json", LANG),
    data=json.dumps(mapping_json_out, indent=2, ensure_ascii=False).encode("utf-8"),
    file_name="mapping.json",
    mime="application/json",
)

# -----------------------------
# 9) Meteorology (optional)
# -----------------------------
st.markdown("---")
st.header(tr("🌦️ Meteorología (opcional)", "🌦️ Meteorology (optional)", LANG))

canonical_df = st.session_state["canonical_df"].copy()
canonical_df["date"] = to_month_start(canonical_df["date"])
canonical_df["point_id"] = canonical_df["point_id"].astype(str)

if canonical_has_meteo(canonical_df):
    st.success(tr("✅ Tu fichero YA contiene meteorología. No hay que hacer nada.",
                  "✅ Your file already contains meteorology. Nothing to do.",
                  LANG))
    st.session_state["canonical_with_meteo"] = canonical_df

else:
    st.warning(tr("Tu fichero NO trae meteorología. Elige una opción:",
                  "Your file does NOT include meteorology. Choose an option:",
                  LANG))

    choice = st.radio(
        tr("¿Cómo quieres aportar meteorología?", "How do you want to provide meteorology?", LANG),
        [
            tr("A) Puedo subir otro fichero con meteorología histórica", "A) I can upload a meteorology historical file", LANG),
            tr("B) No tengo meteorología → descargar de NASA POWER (sin API key)", "B) I don't have meteorology → fetch from NASA POWER (no API key)", LANG),
            tr("C) No tengo meteorología y quiero continuar SOLO con ENDO (menos fiable)", "C) Continue ENDO-only (less reliable)", LANG),
        ],
        index=1,
        key="meteo_choice",
    )

    # A) Upload meteo
    if choice.startswith("A)") or choice.startswith("A"):
        st.subheader(tr("A) Subir meteorología histórica (archivo aparte)", "A) Upload historical meteorology (separate file)", LANG))
        meteo_upload = st.file_uploader(tr("Sube CSV/XLSX con meteo", "Upload CSV/XLSX with meteorology", LANG),
                                        type=["csv", "xlsx", "xls"], key="meteo_upload")

        if meteo_upload is not None:
            raw = read_table_from_upload(meteo_upload)
            st.write(tr("Vista previa:", "Preview:", LANG))
            st.dataframe(raw.head(20), use_container_width=True)

            cols = list(raw.columns)
            if len(cols) < 2:
                st.error(tr("El fichero meteo no tiene columnas suficientes.", "Meteorology file has not enough columns.", LANG))
            else:
                met_date_guess = next((c for c in cols if c.lower() in ("date", "fecha")), cols[0])
                met_precip_guess = next((c for c in cols if "precip" in c.lower() or "rain" in c.lower() or "lluv" in c.lower()), cols[0])
                met_t2m_guess = next((c for c in cols if "t2m" in c.lower() or ("temp" in c.lower() and "max" not in c.lower() and "min" not in c.lower())), cols[0])
                met_tmax_guess = next((c for c in cols if "tmax" in c.lower() or "max" in c.lower()), cols[0])
                met_tmin_guess = next((c for c in cols if "tmin" in c.lower() or "min" in c.lower()), cols[0])
                met_pid_guess = next((c for c in cols if "point_id" in c.lower() or c.lower() == "id" or "point" in c.lower()), "")

                st.markdown(tr("### 🔧 Mapeo de columnas (confirma/corrige)", "### 🔧 Column mapping (confirm/fix)", LANG))
                c1, c2, c3 = st.columns(3)
                with c1:
                    met_date_col = st.selectbox(tr("Columna FECHA", "DATE column", LANG), cols,
                                                index=cols.index(met_date_guess) if met_date_guess in cols else 0, key="met_date_col")
                    met_precip_col = st.selectbox(tr("Columna PRECIP", "PRECIP column", LANG), cols,
                                                  index=cols.index(met_precip_guess) if met_precip_guess in cols else 0, key="met_precip_col")
                with c2:
                    met_t2m_col = st.selectbox(tr("Columna T2M", "T2M column", LANG), cols,
                                               index=cols.index(met_t2m_guess) if met_t2m_guess in cols else 0, key="met_t2m_col")
                    met_tmax_col = st.selectbox(tr("Columna TMAX", "TMAX column", LANG), cols,
                                                index=cols.index(met_tmax_guess) if met_tmax_guess in cols else 0, key="met_tmax_col")
                with c3:
                    met_tmin_col = st.selectbox(tr("Columna TMIN", "TMIN column", LANG), cols,
                                                index=cols.index(met_tmin_guess) if met_tmin_guess in cols else 0, key="met_tmin_col")

                mode = st.radio(
                    tr("¿Tu fichero meteo identifica el point_id?", "Does your meteo file contain point_id?", LANG),
                    [
                        tr("Sí, tiene point_id", "Yes, it has point_id", LANG),
                        tr("No: es de un solo punto", "No: it is for a single point", LANG),
                        tr("No: es común para todos los puntos", "No: it is common for all points", LANG),
                    ],
                    index=0 if met_pid_guess else 2,
                    key="met_mode",
                )

                point_id_col = ""
                single_point = ""
                unique_points = sorted(canonical_df["point_id"].unique().tolist())

                if mode.startswith("Sí") or mode.startswith("Yes"):
                    point_id_col = st.selectbox(
                        tr("Columna POINT_ID", "POINT_ID column", LANG),
                        ["(choose)"] + cols,
                        index=(cols.index(met_pid_guess) + 1) if met_pid_guess in cols else 0,
                        key="met_point_id_col",
                    )
                    if point_id_col in ("(choose)", "(elige)"):
                        st.error(tr("Selecciona la columna point_id.", "Select the point_id column.", LANG))
                        st.stop()
                    mode_key = "HAS_POINT_ID"
                elif "solo" in mode.lower() or "single" in mode.lower():
                    single_point = st.selectbox(tr("¿Para qué point_id aplica?", "Which point_id does it apply to?", LANG),
                                                unique_points, key="met_single_point")
                    mode_key = "SINGLE_POINT"
                else:
                    mode_key = "COMMON_ALL"

                met_map = MeteoMapping(
                    date_col=met_date_col,
                    precip_col=met_precip_col,
                    t2m_col=met_t2m_col,
                    tmax_col=met_tmax_col,
                    tmin_col=met_tmin_col,
                    point_id_col=point_id_col if mode_key == "HAS_POINT_ID" else "",
                )

                if st.button(tr("✅ Aplicar meteorología subida", "✅ Apply uploaded meteorology", LANG), key="apply_uploaded_meteo"):
                    try:
                        meteo_df = build_user_meteo_table(
                            raw=raw,
                            canonical_point_ids=unique_points,
                            mapping=met_map,
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
                        st.success(tr("✅ Meteorología integrada en canonical_with_meteo", "✅ Meteorology integrated into canonical_with_meteo", LANG))
                    except Exception as e:
                        st.error(tr(f"Error integrando meteorología: {e}", f"Error integrating meteorology: {e}", LANG))

    # B) NASA POWER
    elif choice.startswith("B)") or choice.startswith("B"):
        st.subheader(tr("B) Descargar de NASA POWER (sin API key)", "B) Fetch from NASA POWER (no API key)", LANG))

        if not {"lat", "lon"}.issubset(set(canonical_df.columns)):
            st.warning(tr("Tu canonical no tiene lat/lon. Vuelve y mapéalas, o sube meteo, o usa ENDO-only.",
                          "Your canonical has no lat/lon. Map coordinates, upload meteo, or use ENDO-only.",
                          LANG))
        else:
            coords_ok = canonical_df[["lat", "lon"]].notna().all(axis=1).all()
            if not coords_ok:
                st.warning(tr("Faltan coordenadas en algunas filas. NASA POWER requiere lat/lon completos.",
                              "Some coordinates are missing. NASA POWER requires complete lat/lon.",
                              LANG))
            else:
                if st.button(tr("🌍 Descargar NASA POWER y crear canonical_with_meteo", "🌍 Fetch NASA POWER and build canonical_with_meteo", LANG),
                             key="fetch_nasa_power"):
                    try:
                        with st.spinner(tr("Descargando NASA POWER... (se cachea)", "Fetching NASA POWER... (cached)", LANG)):
                            meteo_df, canonical_with_meteo = fetch_nasa_power_for_canonical(canonical_df)
                        st.session_state["meteo_df"] = meteo_df
                        st.session_state["canonical_with_meteo"] = canonical_with_meteo
                        st.success(tr("✅ NASA POWER integrado en canonical_with_meteo", "✅ NASA POWER integrated into canonical_with_meteo", LANG))
                    except Exception as e:
                        st.error(tr(f"Error NASA POWER: {e}", f"NASA POWER error: {e}", LANG))

    # C) ENDO-only
    else:
        st.info(tr("Continuando SOLO con ENDO. Menos fiable, pero funciona.",
                   "Continuing ENDO-only. Less reliable, but it works.",
                   LANG))
        st.session_state["canonical_with_meteo"] = canonical_df.copy()

# Downloads for meteo outputs
if "canonical_with_meteo" in st.session_state:
    st.markdown(tr("### 📥 Descargas (Meteorología)", "### 📥 Downloads (Meteorology)", LANG))
    cwm = st.session_state["canonical_with_meteo"]
    st.download_button(
        tr("Descargar canonical_with_meteo.csv", "Download canonical_with_meteo.csv", LANG),
        data=cwm.to_csv(index=False).encode("utf-8"),
        file_name="canonical_with_meteo.csv",
        mime="text/csv",
    )

if "meteo_df" in st.session_state:
    met = st.session_state["meteo_df"]
    st.download_button(
        tr("Descargar meteo_power_monthly.csv", "Download meteo_power_monthly.csv", LANG),
        data=met.to_csv(index=False).encode("utf-8"),
        file_name="meteo_power_monthly.csv",
        mime="text/csv",
    )