import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from pyproj import CRS, Transformer
from basincast.meteo.geocode import geocode_place, reverse_geocode

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

APP_VERSION = "v0.10"

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

def aggregate_to_monthly_canonical(canonical_with_raw: pd.DataFrame, value_policy: str) -> pd.DataFrame:
    """
    canonical_with_raw: must include columns:
      point_id, date (month start), value, unit, resource_type, lat, lon, date_raw
    value_policy: LAST | MEAN | SUM
    """
    c = canonical_with_raw.copy()
    c["date"] = pd.to_datetime(c["date"], errors="coerce")
    c["date_raw"] = pd.to_datetime(c["date_raw"], errors="coerce")

    keys = ["point_id", "date"]

    # Helper for "first non-null"
    def first_valid(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else np.nan

    if value_policy.upper() == "LAST":
        # take last row within (point_id, month) according to date_raw
        c = c.sort_values(["point_id", "date", "date_raw"])
        idx = c.groupby(keys, sort=False)["date_raw"].idxmax()
        out = c.loc[idx].copy()
    else:
        agg = {
            "unit": first_valid,
            "resource_type": first_valid,
            "lat": first_valid,
            "lon": first_valid,
            "date_raw": "max",
        }
        if value_policy.upper() == "MEAN":
            agg["value"] = "mean"
        elif value_policy.upper() == "SUM":
            agg["value"] = "sum"
        else:
            raise ValueError(f"Unknown value_policy: {value_policy}")

        out = c.groupby(keys, as_index=False).agg(agg)

    out = out.sort_values(["point_id", "date"]).reset_index(drop=True)
    return out

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
date_raw = pd.to_datetime(date_parsed, errors="coerce")

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
df_eff["_date_raw"] = date_raw.loc[mask_eff]

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

# Guardamos también la fecha original (puede ser diaria)
d["date_raw"] = pd.to_datetime(d["_date_raw"], errors="coerce")

canonical = d[["date", "date_raw", "point_id", "value", "unit", "resource_type", "lat", "lon"]].copy()
canonical["date"] = pd.to_datetime(canonical["date"], errors="coerce")
canonical = canonical.sort_values(["point_id", "date", "date_raw"]).reset_index(drop=True)

# --- 7.1 Temporal harmonization (daily/irregular -> monthly) ---
st.subheader("7.1) Temporal harmonization (if your data is daily/irregular)")

dup_count = int(canonical.duplicated(subset=["point_id", "date"], keep=False).sum())

if dup_count > 0:
    st.warning(
        f"Detected multiple records per month (duplicates by point_id+month): {dup_count}. "
        "This usually means your input is daily/weekly/irregular. We should aggregate to one value per month."
    )

    # --- Smart default for monthly aggregation ---
    # If resource_type suggests monthly volumes, default to SUM.
    resource_guess = ""
    try:
        resource_guess = str(canonical["resource_type"].dropna().astype(str).iloc[0]).strip().lower()
    except Exception:
        resource_guess = ""

    # Heuristic defaults
    default_policy_label = "LAST (recommended for levels/storage)"
    if any(k in resource_guess for k in ["river", "demand", "flow", "inflow", "volume", "hm3", "m3"]):
        default_policy_label = "SUM (monthly total)"
    elif any(k in resource_guess for k in ["reservoir", "storage", "level", "stage", "elevation"]):
        default_policy_label = "LAST (recommended for levels/storage)"
    else:
        # unknown -> keep LAST but we will tell user
        default_policy_label = "LAST (recommended for levels/storage)"

    options = [
        "LAST (recommended for levels/storage)",
        "MEAN (monthly average)",
        "SUM (monthly total)",
    ]
    default_index = options.index(default_policy_label) if default_policy_label in options else 0

    value_policy = st.selectbox(
        "How should we aggregate your ENDO value to monthly?",
        options,
        index=default_index,
        key="endo_monthly_policy",
    )

    if default_policy_label.startswith("SUM"):
        st.info("ℹ️ Default chosen: **SUM (monthly total)** because your resource looks like a monthly volume series (e.g., River). You can change it above.")
    elif default_policy_label.startswith("LAST"):
        st.info("ℹ️ Default chosen: **LAST** because your resource looks like a level/storage-type series. You can change it above.")

    do_agg = st.checkbox("✅ Convert to monthly now (recommended)", value=True, key="do_monthly_agg")

    if do_agg:
        before_rows = len(canonical)

        pol = "LAST"
        if value_policy.startswith("MEAN"):
            pol = "MEAN"
        elif value_policy.startswith("SUM"):
            pol = "SUM"

        canonical = aggregate_to_monthly_canonical(canonical, pol)
        after_rows = len(canonical)

        st.success(f"Monthly aggregation applied ✅  {before_rows} rows → {after_rows} monthly rows")
else:
    st.success("Your dataset already looks monthly (1 row per point_id and month) ✅")

# Ya no necesitamos date_raw en el export final
canonical = canonical.drop(columns=["date_raw"], errors="ignore")
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
st.header(tr("meteo_header"))

canonical_df = st.session_state["canonical_df"].copy()
canonical_df["date"] = to_month_start(canonical_df["date"])
canonical_df["point_id"] = canonical_df["point_id"].astype(str)

# If file already has meteo columns
if canonical_has_meteo(canonical_df):
    st.success(tr("meteo_already_present"))
    st.session_state["canonical_with_meteo"] = canonical_df

else:
    st.warning(tr("meteo_missing"))

    choice = st.radio(
        tr("meteo_choice_prompt"),
        [
            tr("meteo_choice_a"),
            tr("meteo_choice_b"),
            tr("meteo_choice_c"),
        ],
        index=1,
        key="meteo_choice",
    )

    # ---------------------------------------
    # A) User uploads meteo file
    # ---------------------------------------
    if choice == tr("meteo_choice_a"):
        st.subheader(tr("meteo_a_title"))
        meteo_upload = st.file_uploader(tr("meteo_a_uploader"), type=["csv", "xlsx", "xls"], key="meteo_upload")

        if meteo_upload is not None:
            raw = read_table_from_upload(meteo_upload)
            st.write(tr("preview"))
            st.dataframe(raw.head(20), use_container_width=True)

            cols = list(raw.columns)
            if len(cols) < 2:
                st.error(tr("meteo_a_not_enough_cols"))
            else:
                # guesses
                met_date_guess = next((c for c in cols if c.lower() in ("date", "fecha")), cols[0])
                met_precip_guess = next((c for c in cols if "precip" in c.lower() or "rain" in c.lower() or "lluv" in c.lower()), cols[0])
                met_t2m_guess = next((c for c in cols if "t2m" in c.lower() or ("temp" in c.lower() and "max" not in c.lower() and "min" not in c.lower())), cols[0])
                met_tmax_guess = next((c for c in cols if "tmax" in c.lower() or "max" in c.lower()), cols[0])
                met_tmin_guess = next((c for c in cols if "tmin" in c.lower() or "min" in c.lower()), cols[0])
                met_pid_guess = next((c for c in cols if "point_id" in c.lower() or c.lower() == "id" or "point" in c.lower()), "")

                st.markdown(tr("meteo_mapping_title"))
                c1, c2, c3 = st.columns(3)
                with c1:
                    met_date_col = st.selectbox(tr("meteo_col_date"), cols, index=cols.index(met_date_guess) if met_date_guess in cols else 0, key="met_date_col")
                    met_precip_col = st.selectbox(tr("meteo_col_precip"), cols, index=cols.index(met_precip_guess) if met_precip_guess in cols else 0, key="met_precip_col")
                with c2:
                    met_t2m_col = st.selectbox(tr("meteo_col_t2m"), cols, index=cols.index(met_t2m_guess) if met_t2m_guess in cols else 0, key="met_t2m_col")
                    met_tmax_col = st.selectbox(tr("meteo_col_tmax"), cols, index=cols.index(met_tmax_guess) if met_tmax_guess in cols else 0, key="met_tmax_col")
                with c3:
                    met_tmin_col = st.selectbox(tr("meteo_col_tmin"), cols, index=cols.index(met_tmin_guess) if met_tmin_guess in cols else 0, key="met_tmin_col")

                mode = st.radio(
                    tr("meteo_has_point_id_q"),
                    [tr("yes_has_point_id"), tr("no_single_point"), tr("no_common_all")],
                    index=0 if met_pid_guess else 2,
                    key="met_mode",
                )

                point_id_col = ""
                single_point = ""
                unique_points = sorted(canonical_df["point_id"].unique().tolist())

                if mode == tr("yes_has_point_id"):
                    point_id_col = st.selectbox(
                        tr("meteo_col_point_id"),
                        ["(choose)"] + cols,
                        index=(cols.index(met_pid_guess) + 1) if met_pid_guess in cols else 0,
                        key="met_point_id_col",
                    )
                    if point_id_col == "(choose)":
                        st.error(tr("meteo_need_point_id"))
                        st.stop()
                    mode_key = "HAS_POINT_ID"
                elif mode == tr("no_single_point"):
                    single_point = st.selectbox(tr("meteo_for_which_point"), unique_points, key="met_single_point")
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

                if st.button(tr("apply_uploaded_meteo_btn"), key="apply_uploaded_meteo"):
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
                        st.success(tr("meteo_integrated_ok"))
                    except Exception as e:
                        st.error(f"{tr('meteo_integrated_err')}: {e}")

    # ---------------------------------------
    # B) NASA POWER (needs coords -> if missing, geocode)
    # ---------------------------------------
    elif choice == tr("meteo_choice_b"):
        st.subheader(tr("meteo_b_title"))

        # Build per-point coord status
        pts = canonical_df.groupby("point_id", as_index=False).agg({"lat": "first", "lon": "first"})
        pts["coords_ok"] = pts[["lat", "lon"]].notna().all(axis=1)
        n_ok = int(pts["coords_ok"].sum())
        n_total = int(len(pts))
        n_missing = n_total - n_ok

        st.write(tr("meteo_coords_status").format(n_ok=n_ok, n_total=n_total))

        if n_missing > 0:
            st.warning(tr("meteo_coords_missing").format(n_missing=n_missing))
            st.dataframe(pts, use_container_width=True)

            st.markdown(tr("meteo_fix_coords_title"))

            # Option 1: global location text
            global_loc = st.text_input(tr("meteo_global_location"), value="", key="global_loc")

            # Option 2: upload locations file (point_id + location OR lat/lon)
            loc_upload = st.file_uploader(tr("meteo_locations_upload"), type=["csv", "xlsx", "xls"], key="loc_upload")
            loc_df = None
            if loc_upload is not None:
                loc_df = read_table_from_upload(loc_upload)
                st.write(tr("preview"))
                st.dataframe(loc_df.head(20), use_container_width=True)

                cols = list(loc_df.columns)
                pid_guess = next((c for c in cols if "point" in c.lower() or "id" == c.lower() or "point_id" in c.lower()), cols[0])
                loc_guess = next((c for c in cols if "loc" in c.lower() or "city" in c.lower() or "municip" in c.lower() or "place" in c.lower() or "name" in c.lower()), "")
                lat_guess = next((c for c in cols if c.lower() in ("lat", "latitude")), "")
                lon_guess = next((c for c in cols if c.lower() in ("lon", "longitude", "lng")), "")

                c1, c2 = st.columns(2)
                with c1:
                    loc_pid_col = st.selectbox(tr("loc_col_point_id"), cols, index=cols.index(pid_guess) if pid_guess in cols else 0, key="loc_pid_col")
                    loc_text_col = st.selectbox(tr("loc_col_location"), ["(none)"] + cols, index=(cols.index(loc_guess) + 1) if loc_guess in cols else 0, key="loc_text_col")
                with c2:
                    loc_lat_col = st.selectbox(tr("loc_col_lat"), ["(none)"] + cols, index=(cols.index(lat_guess) + 1) if lat_guess in cols else 0, key="loc_lat_col")
                    loc_lon_col = st.selectbox(tr("loc_col_lon"), ["(none)"] + cols, index=(cols.index(lon_guess) + 1) if lon_guess in cols else 0, key="loc_lon_col")

                st.caption(tr("meteo_locations_rule"))

        # Optional: show approximate location if coords OK
        if n_ok > 0 and st.checkbox(tr("meteo_show_reverse"), value=False, key="rev_geo_chk"):
            rows = []
            for _, r in pts[pts["coords_ok"]].head(8).iterrows():  # cap to avoid slow UI
                rr = reverse_geocode(float(r["lat"]), float(r["lon"]))
                rows.append(
                    {
                        "point_id": r["point_id"],
                        "lat": r["lat"],
                        "lon": r["lon"],
                        "approx_location": rr.label if rr else "",
                        "confidence": rr.confidence if rr else "",
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Fix coords button (if missing)
        if n_missing > 0 and st.button(tr("meteo_resolve_coords_btn"), key="resolve_coords"):
            try:
                pts2 = pts.copy()

                # 1) If locations file provided: use per-point lat/lon or geocode per-point location text
                if "loc_upload" in st.session_state and st.session_state["loc_upload"] is not None:
                    if loc_df is None:
                        loc_df = read_table_from_upload(st.session_state["loc_upload"])
                    # Rebuild selected cols
                    loc_pid_col = st.session_state.get("loc_pid_col")
                    loc_text_col = st.session_state.get("loc_text_col")
                    loc_lat_col = st.session_state.get("loc_lat_col")
                    loc_lon_col = st.session_state.get("loc_lon_col")

                    loc_map = {}
                    for _, rr in loc_df.iterrows():
                        pid = str(rr.get(loc_pid_col, "")).strip()
                        if not pid:
                            continue

                        latv = None
                        lonv = None

                        if loc_lat_col and loc_lat_col != "(none)" and loc_lon_col and loc_lon_col != "(none)":
                            try:
                                latv = float(str(rr.get(loc_lat_col)).replace(",", "."))
                                lonv = float(str(rr.get(loc_lon_col)).replace(",", "."))
                            except Exception:
                                latv = lonv = None

                        if (latv is None or lonv is None) and loc_text_col and loc_text_col != "(none)":
                            q = str(rr.get(loc_text_col, "")).strip()
                            gr = geocode_place(q) if q else None
                            if gr:
                                latv, lonv = gr.lat, gr.lon

                        if latv is not None and lonv is not None:
                            loc_map[pid] = (latv, lonv)

                    # apply map
                    def _fill(row):
                        if row["coords_ok"]:
                            return row
                        pid = str(row["point_id"])
                        if pid in loc_map:
                            row["lat"] = loc_map[pid][0]
                            row["lon"] = loc_map[pid][1]
                            row["coords_ok"] = True
                        return row

                    pts2 = pts2.apply(_fill, axis=1)

                # 2) If still missing, apply global location
                still_missing = pts2[~pts2["coords_ok"]]
                if not still_missing.empty:
                    gl = (st.session_state.get("global_loc") or "").strip()
                    if gl:
                        gr = geocode_place(gl)
                        if gr:
                            pts2.loc[~pts2["coords_ok"], "lat"] = gr.lat
                            pts2.loc[~pts2["coords_ok"], "lon"] = gr.lon
                            pts2["coords_ok"] = pts2[["lat", "lon"]].notna().all(axis=1)

                # Update canonical_df lat/lon per point_id
                pt_lookup = dict(zip(pts2["point_id"].astype(str), zip(pts2["lat"], pts2["lon"])))
                canonical_df2 = canonical_df.copy()
                canonical_df2["lat"] = canonical_df2["point_id"].map(lambda x: pt_lookup.get(str(x), (np.nan, np.nan))[0])
                canonical_df2["lon"] = canonical_df2["point_id"].map(lambda x: pt_lookup.get(str(x), (np.nan, np.nan))[1])

                st.session_state["canonical_df"] = canonical_df2
                st.success(tr("meteo_coords_resolved_ok"))
                st.rerun()
            except Exception as e:
                st.error(f"{tr('meteo_coords_resolved_err')}: {e}")

        # Now NASA POWER button (only if coords ok)
        pts3 = st.session_state["canonical_df"].groupby("point_id", as_index=False).agg({"lat": "first", "lon": "first"})
        coords_ok_all = pts3[["lat", "lon"]].notna().all(axis=1).all()

        if coords_ok_all:
            if st.button(tr("meteo_fetch_nasa_btn"), key="fetch_nasa_power"):
                try:
                    with st.spinner(tr("meteo_fetching_spinner")):
                        meteo_df, canonical_with_meteo = fetch_nasa_power_for_canonical(st.session_state["canonical_df"])
                    st.session_state["meteo_df"] = meteo_df
                    st.session_state["canonical_with_meteo"] = canonical_with_meteo
                    st.success(tr("meteo_nasa_ok"))
                except Exception as e:
                    st.error(f"{tr('meteo_nasa_err')}: {e}")
        else:
            st.info(tr("meteo_need_coords_or_other_option"))

    # ---------------------------------------
    # C) ENDO only
    # ---------------------------------------
    else:
        st.info(tr("meteo_endo_only_info"))
        st.session_state["canonical_with_meteo"] = canonical_df.copy()

# Downloads
if "canonical_with_meteo" in st.session_state:
    st.markdown("### 📥 Downloads")
    cwm = st.session_state["canonical_with_meteo"]
    st.download_button(
        tr("download_canonical_with_meteo"),
        data=cwm.to_csv(index=False).encode("utf-8"),
        file_name="canonical_with_meteo.csv",
        mime="text/csv",
    )

if "meteo_df" in st.session_state:
    met = st.session_state["meteo_df"]
    st.download_button(
        tr("download_meteo_df"),
        data=met.to_csv(index=False).encode("utf-8"),
        file_name="meteo_power_monthly.csv",
        mime="text/csv",
    )