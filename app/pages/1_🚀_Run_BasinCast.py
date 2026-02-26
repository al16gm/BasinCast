import streamlit as st
import sys
from pathlib import Path
import tempfile
import subprocess
import datetime
import hashlib
import zipfile
import shutil
from urllib.request import Request, urlopen

import pandas as pd
import numpy as np
from pyproj import CRS, Transformer

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="BasinCast · Run", page_icon="🚀", layout="wide")

st.markdown(
    """
<style>
.stButton > button {
  padding: 0.6rem 1.1rem !important;
  font-size: 1.05rem !important;
  border-radius: 12px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🚀 Run BasinCast")
st.caption("A guided workflow: upload → confirm mapping → add meteo/demand → run → open dashboard.")

# -----------------------------
# Language
# -----------------------------
LANG = st.sidebar.selectbox("Language / Idioma", ["en", "es"], index=0)


def tr(es: str, en: str) -> str:
    return es if LANG == "es" else en


# -----------------------------
# Path bootstrap (Cloud-safe)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../app/pages -> repo root
SRC_DIR = PROJECT_ROOT / "src"
APP_DIR = PROJECT_ROOT / "app"
for p in [SRC_DIR, APP_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# -----------------------------
# Import the "engine" modules used by translator_app (no UI)
# -----------------------------
from basincast.translator.reader import load_user_file, summarize_table
from basincast.translator.infer import infer_mapping
from basincast.translator.parsing import parse_date_series, parse_numeric_series, coerce_month_start
from basincast.translator.normalize import CANON_RESOURCE_TYPES, suggest_value_map, apply_value_map
from basincast.translator.meteo import (
    METEO_COLS,
    MeteoMapping,
    canonical_has_meteo,
    read_table_from_upload,
    build_user_meteo_table,
    fetch_nasa_power_for_canonical,
    to_month_start,
)

QUALITY_THRESHOLD = 0.99

# -----------------------------
# DeltaPack (CMIP6) auto-install
# -----------------------------
DELTAPACK_URL = "https://github.com/al16gm/BasinCast/releases/download/deltapack-cmip6-v1/deltapack_cmip6_v1.zip"
DELTAPACK_SHA256 = "be47a159d36f2f36f77f4d9d1fb9250c22fd87b5ea979bf119804fc117882ca2"
DELTAPACK_CACHE_DIR = PROJECT_ROOT / "outputs" / "cache" / "deltapack_cmip6_v1"
DELTAPACK_ZIP_NAME = "deltapack_cmip6_v1.zip"

RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "BasinCast/1.0"})
    with urlopen(req) as r, open(dst, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def ensure_deltapack_available(status_box=None) -> Path:
    if (DELTAPACK_CACHE_DIR / "metadata.json").exists():
        return DELTAPACK_CACHE_DIR

    cache_root = DELTAPACK_CACHE_DIR.parent
    cache_root.mkdir(parents=True, exist_ok=True)

    zip_path = cache_root / DELTAPACK_ZIP_NAME
    tmp_extract = cache_root / "_tmp_extract_deltapack"

    if status_box is not None:
        status_box.info(tr("Descargando DeltaPack (CMIP6)...", "Downloading DeltaPack (CMIP6)..."))

    if zip_path.exists():
        try:
            zip_path.unlink()
        except Exception:
            pass

    _download_file(DELTAPACK_URL, zip_path)

    if status_box is not None:
        status_box.info(tr("Verificando SHA256...", "Verifying SHA256..."))

    got = _sha256_file(zip_path)
    if got.lower() != DELTAPACK_SHA256.lower():
        raise RuntimeError(f"DeltaPack SHA256 mismatch. Expected={DELTAPACK_SHA256} Got={got}")

    if tmp_extract.exists():
        shutil.rmtree(tmp_extract, ignore_errors=True)
    tmp_extract.mkdir(parents=True, exist_ok=True)

    if status_box is not None:
        status_box.info(tr("Descomprimiendo...", "Extracting..."))

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp_extract)

    extracted_root = tmp_extract
    candidates = [p for p in tmp_extract.iterdir() if p.is_dir()]
    if len(candidates) == 1 and (candidates[0] / "metadata.json").exists():
        extracted_root = candidates[0]

    if not (extracted_root / "metadata.json").exists():
        raise RuntimeError("DeltaPack extracted but metadata.json not found.")

    if DELTAPACK_CACHE_DIR.exists():
        shutil.rmtree(DELTAPACK_CACHE_DIR, ignore_errors=True)

    shutil.move(str(extracted_root), str(DELTAPACK_CACHE_DIR))
    shutil.rmtree(tmp_extract, ignore_errors=True)

    if status_box is not None:
        status_box.success(tr("DeltaPack listo ✅", "DeltaPack ready ✅"))

    return DELTAPACK_CACHE_DIR


def _write_upload_to_temp(uploaded) -> Path:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        return Path(tmp.name)


def _guess_col(cols: list[str], candidates: list[str]) -> str | None:
    low = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def _norm_coord_mode(s: str) -> str:
    """
    Normalize to: NONE / LATLON / UTM / AUTO
    """
    if s is None:
        return "NONE"
    x = str(s).strip().upper()
    x = x.replace(" ", "")
    x = x.replace("/", "")
    if x in ["NONE", "NOCOORDINATES", "SINCOORDENADAS"]:
        return "NONE"
    if x in ["LATLON", "LATLONWGS84", "LATLONDEG"]:
        return "LATLON"
    if x in ["UTM"]:
        return "UTM"
    if x in ["AUTO", "AUTORECOMMENDED", "AUTORECOMENDADO", "AUTORECOMMENDED)"]:
        return "AUTO"
    # tolerate common inputs
    if "AUTO" in x:
        return "AUTO"
    if "UTM" in x:
        return "UTM"
    if "LAT" in x and "LON" in x:
        return "LATLON"
    return "NONE"


def _build_canonical(
    df_raw: pd.DataFrame,
    date_col: str,
    value_col: str,
    point_col: str | None,
    unit_col: str | None,
    rtype_col: str | None,
    coord_mode: str,
    lat_col: str | None,
    lon_col: str | None,
    utm_e_col: str | None,
    utm_n_col: str | None,
    utm_zone: int | None,
    utm_hemi: str | None,
) -> tuple[pd.DataFrame, dict]:
    """
    Always returns (canonical_df, report_dict).
    coord_mode supports: NONE / LATLON / UTM / AUTO
    AUTO: try lat/lon first, else UTM if available.
    """
    mode = _norm_coord_mode(coord_mode)

    # Parsing
    date_parsed, date_rep = parse_date_series(df_raw[date_col], date_hint="auto")
    date_month = coerce_month_start(date_parsed)
    value_num, value_rep = parse_numeric_series(df_raw[value_col], locale_hint="auto")

    rep = {
        "date_ok_ratio": float(date_rep.ok_ratio),
        "date_strategy": str(date_rep.strategy),
        "value_ok_ratio": float(value_rep.ok_ratio),
        "value_strategy": str(value_rep.strategy),
        "coord_mode": mode,
    }
    rep["blocked"] = (rep["date_ok_ratio"] < QUALITY_THRESHOLD) or (rep["value_ok_ratio"] < QUALITY_THRESHOLD)

    # Canonical
    can = pd.DataFrame()
    can["date"] = to_month_start(date_month)
    can["value"] = value_num
    can["point_id"] = df_raw[point_col].astype(str) if point_col else "1"
    can["unit"] = df_raw[unit_col].astype(str) if unit_col else "hm3"
    can["resource_type"] = df_raw[rtype_col].astype(str) if rtype_col else "River"

    # Normalize resource_type by fuzzy map (optional)
    try:
        observed = sorted(
            [v for v in can["resource_type"].dropna().astype(str).map(str.strip).unique().tolist() if v != ""]
        )
        vmap = suggest_value_map(observed, CANON_RESOURCE_TYPES, score_cutoff=75)
        if vmap:
            can["resource_type"] = apply_value_map(can["resource_type"].astype(str), vmap)
            rep["resource_type_value_map"] = vmap
    except Exception:
        pass

    # -----------------------------
    # Coordinates (AUTO resolves + converts internally)
    # -----------------------------
    can["lat"] = np.nan
    can["lon"] = np.nan

    def _parse_num(s):
        x, _ = parse_numeric_series(s, locale_hint="auto")
        return x

    def _apply_latlon(lc, lo):
        lat_num = _parse_num(df_raw[lc])
        lon_num = _parse_num(df_raw[lo])
        can["lat"] = lat_num.to_numpy(dtype=float)
        can["lon"] = lon_num.to_numpy(dtype=float)

    def _apply_utm(ec, nc, zone, hemi):
        e_num = _parse_num(df_raw[ec])
        n_num = _parse_num(df_raw[nc])
        mask = e_num.notna() & n_num.notna()

        lat_arr = np.full(len(df_raw), np.nan, dtype=float)
        lon_arr = np.full(len(df_raw), np.nan, dtype=float)

        z = int(zone)
        h = str(hemi).upper().strip()
        epsg = (32600 + z) if h.startswith("N") else (32700 + z)

        transformer = Transformer.from_crs(CRS.from_epsg(epsg), CRS.from_epsg(4326), always_xy=True)

        if mask.any():
            lon_v, lat_v = transformer.transform(
                e_num[mask].to_numpy(dtype=float),
                n_num[mask].to_numpy(dtype=float),
            )
            lat_arr[mask.to_numpy()] = lat_v
            lon_arr[mask.to_numpy()] = lon_v

        can["lat"] = lat_arr
        can["lon"] = lon_arr

    if mode == "LATLON":
        if lat_col and lon_col:
            _apply_latlon(lat_col, lon_col)

    elif mode == "UTM":
        if utm_e_col and utm_n_col and utm_zone and utm_hemi:
            _apply_utm(utm_e_col, utm_n_col, utm_zone, utm_hemi)

    elif mode == "AUTO":
        # Prefer lat/lon if selectable and parses to any valid rows
        used = False
        if lat_col and lon_col:
            lat_num = _parse_num(df_raw[lat_col])
            lon_num = _parse_num(df_raw[lon_col])
            if lat_num.notna().any() and lon_num.notna().any():
                can["lat"] = lat_num.to_numpy(dtype=float)
                can["lon"] = lon_num.to_numpy(dtype=float)
                used = True

        if (not used) and utm_e_col and utm_n_col and utm_zone and utm_hemi:
            _apply_utm(utm_e_col, utm_n_col, utm_zone, utm_hemi)

    # Clean
    can = can.dropna(subset=["date", "value"]).copy()
    can["point_id"] = can["point_id"].astype(str)

    # Report coord validity
    if {"lat", "lon"}.issubset(set(can.columns)):
        valid_ll = can[["lat", "lon"]].notna().all(axis=1)
        rep["latlon_valid_ratio"] = float(valid_ll.mean()) if len(valid_ll) else 0.0
    else:
        rep["latlon_valid_ratio"] = 0.0

    return can, rep


# -----------------------------
# Wizard state
# -----------------------------
STEPS = [tr("Datos", "Data"), tr("Confirmar", "Confirm"), tr("Meteo/Demanda", "Meteo/Demand"), tr("Ejecutar", "Run")]
if "bc_step" not in st.session_state:
    st.session_state["bc_step"] = 0


def goto_step(i: int):
    st.session_state["bc_step"] = int(max(0, min(len(STEPS) - 1, i)))


def next_step():
    goto_step(st.session_state["bc_step"] + 1)


def prev_step():
    goto_step(st.session_state["bc_step"] - 1)


for k in [
    "bc_raw_df",
    "bc_tables",
    "bc_table_name",
    "canonical_df",
    "canonical_with_meteo",
    "canonical_with_meteo_and_demand",
    "bc_build_report",
]:
    if k not in st.session_state:
        st.session_state[k] = None

st.markdown(f"### {tr('Paso', 'Step')} {st.session_state['bc_step'] + 1}/4 — **{STEPS[st.session_state['bc_step']]}**")

nav1, nav2, _ = st.columns([1, 1, 2])
with nav1:
    st.button(
        tr("⬅ Anterior", "⬅ Back"),
        on_click=prev_step,
        disabled=(st.session_state["bc_step"] == 0),
        use_container_width=True,
    )
with nav2:
    st.button(
        tr("Siguiente ➡", "Next ➡"),
        on_click=next_step,
        disabled=(st.session_state["bc_step"] == len(STEPS) - 1),
        use_container_width=True,
    )

st.divider()

# -----------------------------
# STEP 1 — Data
# -----------------------------
if st.session_state["bc_step"] == 0:
    st.subheader(tr("Sube tu archivo", "Upload your file"))

    uploaded = st.file_uploader(
        tr("Archivo (.xlsx/.xls/.csv)", "File (.xlsx/.xls/.csv)"),
        type=["xlsx", "xls", "csv"],
    )
    if uploaded is None:
        st.info(tr("Sube un archivo para continuar.", "Upload a file to continue."))
        st.stop()

    tmp_path = _write_upload_to_temp(uploaded)
    tables = load_user_file(str(tmp_path))
    st.session_state["bc_tables"] = tables

    names = [t.name for t in tables]
    chosen = st.selectbox(tr("Hoja/tabla", "Sheet/table"), names, index=0)
    st.session_state["bc_table_name"] = chosen
    table = next(t for t in tables if t.name == chosen)
    df = table.df.copy()
    st.session_state["bc_raw_df"] = df

    s = summarize_table(df)
    st.success(
        tr(
            f"Tabla cargada ✅ Filas={s.get('n_rows')} | Columnas={s.get('n_cols')}",
            f"Loaded ✅ Rows={s.get('n_rows')} | Cols={s.get('n_cols')}",
        )
    )

    with st.expander(tr("Avanzado: vista previa", "Advanced: preview"), expanded=False):
        st.json(s)
        st.dataframe(df.head(30), use_container_width=True)

# -----------------------------
# STEP 2 — Confirm + coords
# -----------------------------
elif st.session_state["bc_step"] == 1:
    df = st.session_state.get("bc_raw_df", None)
    if df is None:
        st.info(tr("Primero sube un archivo en el paso 1.", "Upload a file first (Step 1)."))
        st.stop()

    st.subheader(tr("Confirmación de columnas", "Confirm columns"))
    inf = infer_mapping(df)
    all_cols = ["(none)"] + list(df.columns.astype(str))

    date_col = st.selectbox(
        tr("Fecha (obligatoria)", "Date (required)"),
        all_cols,
        index=all_cols.index(inf.date_col) if getattr(inf, "date_col", None) in all_cols else 0,
    )
    value_col = st.selectbox(
        tr("Valor (obligatoria)", "Value (required)"),
        all_cols,
        index=all_cols.index(inf.value_col) if getattr(inf, "value_col", None) in all_cols else 0,
    )

    point_col = st.selectbox(
        tr("Point ID (opcional)", "Point ID (optional)"),
        all_cols,
        index=all_cols.index(inf.point_id_col) if getattr(inf, "point_id_col", None) in all_cols else 0,
    )
    unit_col = st.selectbox(
        tr("Unidad (opcional)", "Unit (optional)"),
        all_cols,
        index=all_cols.index(inf.unit_col) if getattr(inf, "unit_col", None) in all_cols else 0,
    )
    rtype_col = st.selectbox(
        tr("Tipo recurso (opcional)", "Resource type (optional)"),
        all_cols,
        index=all_cols.index(inf.resource_type_col) if getattr(inf, "resource_type_col", None) in all_cols else 0,
    )

    if date_col == "(none)" or value_col == "(none)":
        st.warning(tr("Selecciona al menos FECHA y VALOR.", "Select at least DATE and VALUE."))
        st.stop()

    # ---- Coordinates (AUTO + UTM + Lat/Lon) ----
    st.subheader(tr("Coordenadas", "Coordinates"))
    cols = list(df.columns.astype(str))

    lat_guess = getattr(inf, "lat_col", None) if getattr(inf, "lat_col", None) in cols else _guess_col(cols, ["lat", "latitude"])
    lon_guess = getattr(inf, "lon_col", None) if getattr(inf, "lon_col", None) in cols else _guess_col(cols, ["lon", "lng", "longitude"])
    e_guess = _guess_col(cols, ["utm_e", "easting", "east", "utm_x", "x"])
    n_guess = _guess_col(cols, ["utm_n", "northing", "north", "utm_y", "y"])

    coord_mode_ui = st.radio(
        tr("Sistema", "System"),
        ["Auto (recommended)", "Lat/Lon", "UTM", tr("Sin coordenadas", "No coordinates")],
        index=0,
        horizontal=True,
        key="coord_mode_confirm",
    )

    lat_col = lon_col = utm_e_col = utm_n_col = None
    utm_zone = 30
    utm_hemi = "N"

    if coord_mode_ui in ["Auto (recommended)", "Lat/Lon"]:
        lat_pick = st.selectbox(
            tr("Latitud", "Latitude"),
            all_cols,
            index=all_cols.index(lat_guess) if lat_guess in all_cols else 0,
            key="lat_col_confirm",
        )
        lon_pick = st.selectbox(
            tr("Longitud", "Longitude"),
            all_cols,
            index=all_cols.index(lon_guess) if lon_guess in all_cols else 0,
            key="lon_col_confirm",
        )
        lat_col = None if lat_pick == "(none)" else lat_pick
        lon_col = None if lon_pick == "(none)" else lon_pick

    if coord_mode_ui in ["Auto (recommended)", "UTM"]:
        utm_e_pick = st.selectbox(
            tr("Easting (UTM)", "Easting (UTM)"),
            all_cols,
            index=all_cols.index(e_guess) if e_guess in all_cols else 0,
            key="utm_e_col_confirm",
        )
        utm_n_pick = st.selectbox(
            tr("Northing (UTM)", "Northing (UTM)"),
            all_cols,
            index=all_cols.index(n_guess) if n_guess in all_cols else 0,
            key="utm_n_col_confirm",
        )
        utm_e_col = None if utm_e_pick == "(none)" else utm_e_pick
        utm_n_col = None if utm_n_pick == "(none)" else utm_n_pick

        c1, c2 = st.columns(2)
        with c1:
            utm_zone = st.number_input(
                tr("Zona UTM (1–60)", "UTM zone (1–60)"),
                min_value=1,
                max_value=60,
                value=30,
                step=1,
                key="utm_zone_confirm",
            )
        with c2:
            utm_hemi = st.selectbox(tr("Hemisferio", "Hemisphere"), ["N", "S"], index=0, key="utm_hemi_confirm")

    if coord_mode_ui == "Auto (recommended)":
        coord_mode_for_build = "AUTO"
    elif coord_mode_ui == "Lat/Lon":
        coord_mode_for_build = "LATLON"
    elif coord_mode_ui == "UTM":
        coord_mode_for_build = "UTM"
    else:
        coord_mode_for_build = "NONE"

    can, rep = _build_canonical(
        df_raw=df,
        date_col=date_col,
        value_col=value_col,
        point_col=None if point_col == "(none)" else point_col,
        unit_col=None if unit_col == "(none)" else unit_col,
        rtype_col=None if rtype_col == "(none)" else rtype_col,
        coord_mode=coord_mode_for_build,
        lat_col=lat_col,
        lon_col=lon_col,
        utm_e_col=utm_e_col,
        utm_n_col=utm_n_col,
        utm_zone=int(utm_zone) if utm_zone is not None else None,
        utm_hemi=str(utm_hemi) if utm_hemi is not None else None,
    )

    st.session_state["canonical_df"] = can
    st.session_state["bc_build_report"] = rep

    c1, c2 = st.columns(2)
    with c1:
        st.metric(tr("OK ratio fecha", "Date OK ratio"), f"{rep['date_ok_ratio']:.3f}")
        st.caption(tr("Estrategia", "Strategy") + f": {rep['date_strategy']}")
    with c2:
        st.metric(tr("OK ratio valor", "Value OK ratio"), f"{rep['value_ok_ratio']:.3f}")
        st.caption(tr("Estrategia", "Strategy") + f": {rep['value_strategy']}")

    st.caption(tr(f"Lat/Lon válidas: {rep.get('latlon_valid_ratio', 0.0):.1%}",
                  f"Valid lat/lon: {rep.get('latlon_valid_ratio', 0.0):.1%}"))

    if rep["blocked"]:
        st.error(
            tr(
                f"Parsing insuficiente. Ajusta el archivo hasta ratio ≥ {QUALITY_THRESHOLD:.2f}.",
                f"Parsing quality below threshold. Fix until OK ratio ≥ {QUALITY_THRESHOLD:.2f}.",
            )
        )
        st.stop()

    npts = can["point_id"].nunique()
    dmin = pd.to_datetime(can["date"].min(), errors="coerce")
    dmax = pd.to_datetime(can["date"].max(), errors="coerce")
    st.success(
        tr(
            f"Canónico listo ✅ Puntos={npts} | Periodo={dmin.date()}→{dmax.date()}",
            f"Canonical ready ✅ Points={npts} | Period={dmin.date()}→{dmax.date()}",
        )
    )

# -----------------------------
# STEP 3 — Meteo/Demand
# -----------------------------
elif st.session_state["bc_step"] == 2:
    can = st.session_state.get("canonical_df", None)
    if can is None or can.empty:
        st.info(tr("Completa el paso 2 primero.", "Complete Step 2 first."))
        st.stop()

    st.subheader(tr("Meteorología", "Meteorology"))

    has_m = canonical_has_meteo(can)

    # coords_ok: each point_id must have at least ONE row with valid lat/lon
    tmp = can.copy()
    tmp["point_id"] = tmp["point_id"].astype(str)
    valid_ll = tmp[["lat", "lon"]].notna().all(axis=1)
    ok_by_point = valid_ll.groupby(tmp["point_id"]).any()
    coords_ok = bool(ok_by_point.all())

    if not coords_ok:
        missing_pts = ok_by_point[~ok_by_point].index.tolist()
        st.error(tr(f"Faltan lat/lon para estos puntos: {missing_pts}",
                    f"Missing lat/lon for these points: {missing_pts}"))

    if has_m:
        st.success(tr("Meteorología ya presente ✅", "Meteorology already present ✅"))
        st.session_state["canonical_with_meteo"] = can
    else:
        st.warning(tr("No hay meteorología en el canónico.", "No meteorology in canonical."))
        meteo_choice = st.radio(
            tr("Cómo aportar meteorología", "How to provide meteorology"),
            [
                tr("Descargar NASA POWER (recomendado)", "Download NASA POWER (recommended)"),
                tr("Subir fichero meteo (CSV/XLSX)", "Upload meteo file (CSV/XLSX)"),
                tr("Continuar SOLO ENDO (sin meteo)", "Continue ENDO-only (no meteo)"),
            ],
            index=0,
        )

        if "NASA" in meteo_choice:
            if not coords_ok:
                st.error(tr("Faltan lat/lon (o no se han convertido).", "Missing lat/lon (or conversion failed)."))
            else:
                if st.button(tr("Descargar NASA POWER ahora", "Download NASA POWER now"), use_container_width=True):
                    with st.spinner(tr("Descargando meteorología...", "Downloading meteorology...")):
                        _, cwm = fetch_nasa_power_for_canonical(can)
                    st.session_state["canonical_with_meteo"] = cwm
                    st.success(tr("Meteorología NASA añadida ✅", "NASA meteorology added ✅"))

        elif "Subir" in meteo_choice:
            meteo_upload = st.file_uploader(tr("Archivo meteo (.csv/.xlsx)", "Meteo file (.csv/.xlsx)"), type=["csv", "xlsx", "xls"])
            if meteo_upload is not None:
                raw = read_table_from_upload(meteo_upload)
                cols = list(raw.columns.astype(str))

                with st.expander(tr("Avanzado: mapeo meteo", "Advanced: meteo mapping"), expanded=False):
                    date_col = st.selectbox(tr("Columna fecha", "Date column"), cols, index=0)
                    pid_col = st.selectbox(tr("Columna point_id", "Point id column"), cols, index=0)

                    def _g(names):
                        x = _guess_col(cols, names)
                        return x if x is not None else cols[0]

                    precip_col = st.selectbox("Precip column", cols, index=cols.index(_g(["precip_mm_month_est", "precip", "pr", "prectotcorr"])))
                    t2m_col = st.selectbox("T2M column", cols, index=cols.index(_g(["t2m_c", "t2m", "temp", "tas"])))
                    tmax_col = st.selectbox("TMAX column", cols, index=cols.index(_g(["tmax_c", "tmax", "t2m_max", "tasmax"])))
                    tmin_col = st.selectbox("TMIN column", cols, index=cols.index(_g(["tmin_c", "tmin", "t2m_min", "tasmin"])))

                mm = MeteoMapping(
                    date_col=date_col,
                    precip_col=precip_col,
                    t2m_col=t2m_col,
                    tmax_col=tmax_col,
                    tmin_col=tmin_col,
                    point_id_col=pid_col,
                )

                meteo_df = build_user_meteo_table(
                    raw=raw,
                    canonical_point_ids=sorted(can["point_id"].unique().tolist()),
                    mapping=mm,
                    mode="HAS_POINT_ID",
                    single_point_id="",
                )

                cwm = can.merge(
                    meteo_df[["point_id", "date"] + METEO_COLS + ["source"]],
                    on=["point_id", "date"],
                    how="left",
                )
                st.session_state["canonical_with_meteo"] = cwm
                st.success(tr("Meteo integrada ✅", "Meteo integrated ✅"))

        else:
            st.info(tr("Continuando sin meteo (ENDO-only).", "Continuing without meteo (ENDO-only)."))
            st.session_state["canonical_with_meteo"] = can

    st.divider()
    st.subheader(tr("Demanda (opcional)", "Demand (optional)"))

    base_exog = st.session_state.get("canonical_with_meteo", None)
    if base_exog is None or (isinstance(base_exog, pd.DataFrame) and base_exog.empty):
        base_exog = can

    demand_choice = st.radio(
        tr("Demanda exógena", "Demand exogenous"),
        [
            tr("Subir demanda (CSV/XLSX)", "A) Upload demand (CSV/XLSX)"),
            tr("Demanda Open Data (Dataverse ZIP cache)", "B) Open demand (Dataverse ZIP cached)"),
            tr("Sin demanda", "C) No demand"),
        ],
        index=2,
        horizontal=True,
        key="demand_choice_wizard",
    )

    # Reset on change (avoid stale merges)
    prev_choice = st.session_state.get("demand_choice_prev_wizard", None)
    if prev_choice is not None and prev_choice != demand_choice:
        st.session_state["canonical_with_meteo_and_demand"] = None
    st.session_state["demand_choice_prev_wizard"] = demand_choice

    if ("Sin demanda" in demand_choice) or ("No demand" in demand_choice) or demand_choice.startswith("C)"):
        st.session_state["canonical_with_meteo_and_demand"] = base_exog
        st.success(tr("Demanda: no aplicada ✅", "Demand: not applied ✅"))

    elif ("Open Data" in demand_choice) or ("Dataverse" in demand_choice) or demand_choice.startswith("B)"):
        st.caption(tr(
            "Usa ZIPs cacheados si existen. La descarga automática es muy grande (desactivada por defecto).",
            "Uses cached ZIPs if present. Auto-download is huge (disabled by default)."
        ))

        from basincast.demand.open_demand import build_total_demand_hm3_monthly, integrate_total_demand_into_canonical

        cache_dir = Path("outputs") / "cache" / "demand"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Same “simple” default as translator_app
        scenario = "ssp1_rcp26"
        gcm = "gfdl"
        file_id_1 = 6062173
        file_id_2 = 6062170
        zip_name_1 = f"{scenario}_{gcm}_withdrawals_sectors_monthly_1.zip"
        zip_name_2 = f"{scenario}_{gcm}_withdrawals_sectors_monthly_2.zip"

        zip1 = cache_dir / zip_name_1
        zip2 = cache_dir / zip_name_2

        st.write({"zip1_exists": zip1.exists(), "zip2_exists": zip2.exists(), "cache_dir": str(cache_dir)})

        if not (zip1.exists() and zip2.exists()):
            c1, c2 = st.columns(2)
            with c1:
                up1 = st.file_uploader(tr("Sube ZIP 1 (...monthly_1.zip)", "Upload ZIP 1 (...monthly_1.zip)"), type=["zip"], key="open_demand_zip1_upload")
            with c2:
                up2 = st.file_uploader(tr("Sube ZIP 2 (...monthly_2.zip)", "Upload ZIP 2 (...monthly_2.zip)"), type=["zip"], key="open_demand_zip2_upload")

            if up1 is not None:
                zip1.write_bytes(up1.getbuffer())
                st.success(tr("ZIP1 guardado ✅", "ZIP1 saved ✅") + f" {zip1.name}")
            if up2 is not None:
                zip2.write_bytes(up2.getbuffer())
                st.success(tr("ZIP2 guardado ✅", "ZIP2 saved ✅") + f" {zip2.name}")

        allow_download = st.checkbox(
            tr("Permitir descarga automática desde Dataverse (MUY grande)", "Allow auto-download from Dataverse (VERY large)"),
            value=False,
            key="open_demand_allow_download",
        )
        ssl_verify = st.checkbox("SSL verify", value=True, key="open_demand_ssl_verify")

        if st.button(tr("✅ Integrar demanda Open Data", "✅ Integrate Open demand"), key="apply_open_demand"):
            # Months needed based on canonical overlap
            base_exog = base_exog.copy()
            base_exog["date"] = pd.to_datetime(base_exog["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
            hist_min = pd.to_datetime(base_exog["date"].min(), errors="coerce")
            hist_max = pd.to_datetime(base_exog["date"].max(), errors="coerce")

            overlap_start = max(hist_min.to_period("M").to_timestamp(), pd.Timestamp("2010-01-01"))
            overlap_end = hist_max.to_period("M").to_timestamp()
            months_needed = list(pd.date_range(start=overlap_start, end=overlap_end, freq="MS"))

            # bbox requires lat/lon
            lat = pd.to_numeric(base_exog.get("lat", pd.Series(dtype=float)), errors="coerce")
            lon = pd.to_numeric(base_exog.get("lon", pd.Series(dtype=float)), errors="coerce")
            if (not lat.notna().any()) or (not lon.notna().any()):
                st.warning(tr("No hay lat/lon válidas -> no se puede construir bbox.", "No valid lat/lon -> cannot build bbox."))
            else:
                bbox = (float(lat.min()), float(lon.min()), float(lat.max()), float(lon.max()))

                with st.spinner(tr("Construyendo demanda Open Data...", "Building Open demand...")):
                    dem_df, info = build_total_demand_hm3_monthly(
                        cache_dir=cache_dir,
                        scenario=scenario,
                        gcm=gcm,
                        file_id_1=file_id_1,
                        file_id_2=file_id_2,
                        zip_name_1=zip_name_1,
                        zip_name_2=zip_name_2,
                        months_needed=months_needed,
                        bbox=bbox,
                        allow_download=bool(allow_download),
                        ssl_verify=bool(ssl_verify),
                        fail_soft=True,
                    )

                if dem_df is None or dem_df.empty:
                    st.warning(tr("No se pudo integrar la demanda Open Data.", "Could not integrate Open demand.") + f" Info: {info}")
                else:
                    # harmonize column name if needed
                    if ("demand_hm3" not in dem_df.columns) and ("demand" in dem_df.columns):
                        dem_df = dem_df.rename(columns={"demand": "demand_hm3"})

                    merged = integrate_total_demand_into_canonical(
                        base_exog,
                        dem_df,
                        date_col="date",
                        out_col="demand",
                    )
                    st.session_state["canonical_with_meteo_and_demand"] = merged
                    st.success(tr("Demanda Open Data integrada ✅", "Open demand integrated ✅") + f" Non-null: {int(merged['demand'].notna().sum())}")

    else:
        # Upload demand
        d_up = st.file_uploader(tr("Archivo demanda (date, point_id, demand)", "Demand file (date, point_id, demand)"), type=["csv", "xlsx", "xls"])
        if d_up is not None:
            tmp = _write_upload_to_temp(d_up)
            dtabs = load_user_file(str(tmp))
            ddf = dtabs[0].df.copy()

            cols = list(ddf.columns.astype(str))
            low = {c.lower().strip(): c for c in cols}
            date_key = low.get("date", low.get("fecha", None))
            pid_key = low.get("point_id", low.get("pointid", None))
            dem_key = low.get("demand", low.get("demanda", None))

            if date_key is None or pid_key is None or dem_key is None:
                st.error(tr("Faltan columnas: date/point_id/demand.", "Missing columns: date/point_id/demand."))
                st.code(", ".join(cols))
            else:
                ddf = ddf.rename(columns={date_key: "date", pid_key: "point_id", dem_key: "demand"})
                ddf["date"] = pd.to_datetime(ddf["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
                ddf["point_id"] = ddf["point_id"].astype(str)
                ddf["demand"] = pd.to_numeric(ddf["demand"], errors="coerce")

                merged = base_exog.merge(ddf[["point_id", "date", "demand"]], on=["point_id", "date"], how="left")
                st.session_state["canonical_with_meteo_and_demand"] = merged
                st.success(tr("Demanda integrada ✅", "Demand integrated ✅"))

# -----------------------------
# STEP 4 — Run core
# -----------------------------
else:
    st.subheader(tr("Ejecutar BasinCast Core", "Run BasinCast Core"))

    final_df = st.session_state.get("canonical_with_meteo_and_demand", None)
    if final_df is None or (isinstance(final_df, pd.DataFrame) and final_df.empty):
        final_df = st.session_state.get("canonical_with_meteo", None)
    if final_df is None or (isinstance(final_df, pd.DataFrame) and final_df.empty):
        final_df = st.session_state.get("canonical_df", None)

    if final_df is None or final_df.empty:
        st.info(tr("Completa los pasos anteriores.", "Complete previous steps."))
        st.stop()

    mode = st.radio(tr("Modo", "Mode"), [tr("Rápido (web)", "Fast (web)"), tr("Completo (paper)", "Full (paper)")], index=0)
    holdout = 48 if ("Rápido" in mode or "Fast" in mode) else 96
    inner_val = 24 if ("Rápido" in mode or "Fast" in mode) else 36

    st.write(
        tr(
            f"Holdout={holdout} meses | Inner validation={inner_val} meses",
            f"Holdout={holdout} months | Inner validation={inner_val} months",
        )
    )

    ensure_pack = st.checkbox(tr("Asegurar DeltaPack CMIP6 (recomendado)", "Ensure CMIP6 DeltaPack (recommended)"), value=True)
    show_logs = st.checkbox(tr("Mostrar logs (avanzado)", "Show logs (advanced)"), value=False)

    status = st.empty()
    log_box = st.empty()

    if st.button(tr("▶ RUN", "▶ RUN"), use_container_width=True):
        if ensure_pack:
            try:
                ensure_deltapack_available(status_box=status)
            except Exception as e:
                st.warning(
                    tr(
                        f"DeltaPack no disponible: {e!r}. El run seguirá, pero escenarios pueden omitirse.",
                        f"DeltaPack unavailable: {e!r}. Run will continue but scenarios may be skipped.",
                    )
                )

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RUNS_DIR / f"run_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        inp = run_dir / "_tmp_canonical_for_core.csv"
        final_df.to_csv(inp, index=False)

        core_py = PROJECT_ROOT / "app" / "run_core_v0_6.py"
        cmd = [
            sys.executable,
            str(core_py),
            "--input",
            str(inp),
            "--outdir",
            str(run_dir),
            "--holdout_months",
            str(int(holdout)),
            "--inner_val_months",
            str(int(inner_val)),
        ]

        status.info(tr("Ejecutando core...", "Running core..."))

        lines = []
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        while True:
            line = p.stdout.readline()
            if not line and p.poll() is not None:
                break
            if line:
                s = line.rstrip("\n")
                lines.append(s)
                if len(lines) > 200:
                    lines = lines[-200:]
                if show_logs:
                    log_box.code("\n".join(lines), language="text")

        rc = p.wait()
        if rc != 0:
            st.error(tr("Core falló. Activa logs para ver el error.", "Core failed. Enable logs to see the error."))
            st.stop()

        st.success(tr("Run completado ✅", "Run completed ✅"))
        st.session_state["core_last_outdir"] = str(run_dir)

        st.page_link("pages/2_📊_Dashboard.py", label=tr("Abrir Dashboard", "Open Dashboard"), icon="📊")
        st.caption(str(run_dir))