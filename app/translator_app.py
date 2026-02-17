# translator_app.py
# BasinCast Translator — Clean consolidated version (v0.13.4-ready)
# - No duplicated meteo blocks
# - Demand A/B/C fixed (no dem_mode NameError)
# - Paper-friendly tables + monthly forecast path preserved
# - For dummies: clear UI + safe defaults

import json
import os
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Literal, Optional

import numpy as np
import pandas as pd
import streamlit as st


import hashlib
from datetime import datetime

from pyproj import CRS, Transformer

import plotly.graph_objects as go

# -----------------------------
# BasinCast imports (existing project modules)
# -----------------------------
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
from basincast.translator.i18n import language_selector, tr

from basincast.translator.meteo import (
    METEO_COLS,
    MeteoMapping,
    canonical_has_meteo,
    read_table_from_upload,
    build_user_meteo_table,
    fetch_nasa_power_for_canonical,
    to_month_start,
)

from basincast.meteo.geocode import geocode_place, reverse_geocode


# -----------------------------
# App config
# -----------------------------
APP_VERSION = "v0.13.4"
st.set_page_config(page_title=f"BasinCast Translator ({APP_VERSION})", layout="wide")

LANG = language_selector(default="en")

st.title(tr("BasinCast — Traductor de Entrada", "BasinCast — Input Translator", LANG) + f" ({APP_VERSION})")
st.write(
    tr(
        "Sube un Excel/CSV. BasinCast detecta campos; tú confirmas; y exportamos un dataset CANONICAL seguro.\n\n"
        "**Incluye:** parsing robusto de fechas/números, ocultar ruido, UTM→Lat/Lon, checks de integridad y meteorología/demanda opcional.",
        "Upload an Excel/CSV. BasinCast auto-detects fields; you confirm; then we export a safe CANONICAL dataset.\n\n"
        "**Includes:** robust date/number parsing, noise hiding, UTM→Lat/Lon, integrity checks, optional meteorology and demand.",
        LANG,
    )
)

QUALITY_THRESHOLD = 0.99


# -----------------------------
# Small utilities
# -----------------------------
def _to_month_start(s: pd.Series) -> pd.Series:
    s2 = pd.to_datetime(s, errors="coerce")
    return s2.dt.to_period("M").dt.to_timestamp()


def _month_start_ts(x) -> pd.Timestamp:
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        return ts
    return ts.to_period("M").to_timestamp()


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
    canonical_with_raw must include columns:
      point_id, date (month start), value, unit, resource_type, lat, lon, date_raw
    value_policy: LAST | MEAN | SUM
    """
    c = canonical_with_raw.copy()
    c["date"] = pd.to_datetime(c["date"], errors="coerce")
    c["date_raw"] = pd.to_datetime(c["date_raw"], errors="coerce")

    keys = ["point_id", "date"]

    def first_valid(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else np.nan

    if value_policy.upper() == "LAST":
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


def _guess_date_col(columns):
    keys_exact = ["date", "fecha", "time", "datetime", "month", "mes"]
    for k in keys_exact:
        for c in columns:
            if str(c).lower().strip() == k:
                return c
    for c in columns:
        cl = str(c).lower()
        if ("date" in cl) or ("fecha" in cl) or ("time" in cl) or ("mes" in cl) or ("month" in cl):
            return c
    return columns[0] if columns else None


def _guess_demand_col(columns):
    keys_exact = [
        "demand", "demanda", "total_demand", "total_demand_hm3", "withdrawal", "water_withdrawal",
        "consumption", "use", "uso"
    ]
    for k in keys_exact:
        for c in columns:
            if str(c).lower().strip() == k:
                return c
    for c in columns:
        cl = str(c).lower()
        if ("demand" in cl) or ("demanda" in cl) or ("withdraw" in cl) or ("consump" in cl) or ("uso" in cl):
            return c
    return columns[1] if len(columns) > 1 else (columns[0] if columns else None)


def _prepare_demand_monthly(demand_df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    d = demand_df[[date_col, value_col]].copy()
    d = d.rename(columns={date_col: "date", value_col: "demand"})
    d["date"] = _to_month_start(d["date"])
    d["demand"] = pd.to_numeric(d["demand"], errors="coerce")
    d = d.dropna(subset=["date", "demand"]).copy()
    d = d.groupby("date", as_index=False)["demand"].mean()
    d = d.sort_values("date").reset_index(drop=True)
    return d


# -----------------------------
# Monthly forecast helpers (viz)
# -----------------------------
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge

EXOG_COLS_VIZ = ["precip_mm_month_est", "t2m_c", "tmax_c", "tmin_c"]


def _make_model_for_viz(model_type: str):
    mt = (model_type or "").strip().lower()
    if mt == "rf":
        return RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    if mt == "gbr":
        return GradientBoostingRegressor(random_state=42)
    return BayesianRidge()


def _build_train_table(history: pd.DataFrame, family: str) -> tuple[pd.DataFrame, list[str]]:
    g = history.copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    g = g.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)

    g["value"] = pd.to_numeric(g["value"], errors="coerce")
    g = g.dropna(subset=["value"]).reset_index(drop=True)

    g["delta_y"] = g["value"].diff()
    g["value_lag1"] = g["value"].shift(1)
    g["value_lag12"] = g["value"].shift(12)
    g["value_lag12"] = g["value_lag12"].fillna(g["value_lag1"])
    g["delta_y_lag1"] = g["delta_y"].shift(1).fillna(0.0)

    m = g["date"].dt.month.astype(int)
    g["month_sin"] = np.sin(2 * np.pi * m / 12.0)
    g["month_cos"] = np.cos(2 * np.pi * m / 12.0)

    feat_cols = ["value_lag1", "value_lag12", "delta_y_lag1", "month_sin", "month_cos"]

    if family == "EXOG_ML":
        missing = [c for c in EXOG_COLS_VIZ if c not in g.columns]
        if missing:
            raise ValueError(f"Missing EXOG columns for EXOG_ML: {missing}")

        for c in EXOG_COLS_VIZ:
            g[c] = pd.to_numeric(g[c], errors="coerce")
            g[f"{c}_lag1"] = g[c].shift(1).fillna(g[c])
            feat_cols.append(f"{c}_lag1")

    out = g.dropna(subset=["delta_y"] + feat_cols).reset_index(drop=True)
    return out[["date", "delta_y"] + feat_cols], feat_cols


def _forecast_monthly_path(
    history: pd.DataFrame,
    model_type: str,
    family: str,
    max_h: int = 48,
    non_negative: bool = True,
) -> pd.DataFrame:
    hist = history.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    hist = hist.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)

    last_date = hist["date"].max()
    last_value = float(pd.to_numeric(hist.iloc[-1]["value"], errors="coerce"))

    train_tbl, feat_cols = _build_train_table(hist, family=family)
    model = _make_model_for_viz(model_type)
    model.fit(train_tbl[feat_cols], train_tbl["delta_y"])

    exog_df = None
    clim = None
    if family == "EXOG_ML":
        exog_df = hist[["date"] + EXOG_COLS_VIZ].copy()
        exog_df["date"] = pd.to_datetime(exog_df["date"]).dt.to_period("M").dt.to_timestamp()
        clim = exog_df.groupby(exog_df["date"].dt.month)[EXOG_COLS_VIZ].mean(numeric_only=True)

    y_prev = last_value
    vals = hist["value"].astype(float).to_list()
    if len(vals) >= 12:
        lag12 = vals[-12:]
    else:
        lag12 = [vals[0]] * (12 - len(vals)) + vals

    delta_prev = 0.0
    out_rows = []

    for h in range(1, int(max_h) + 1):
        fc_date = (pd.Timestamp(last_date) + pd.DateOffset(months=h)).to_period("M").to_timestamp()
        m = int(fc_date.month)

        row = {
            "value_lag1": float(y_prev),
            "value_lag12": float(lag12[0]),
            "delta_y_lag1": float(delta_prev),
            "month_sin": float(np.sin(2 * np.pi * m / 12.0)),
            "month_cos": float(np.cos(2 * np.pi * m / 12.0)),
        }

        if family == "EXOG_ML":
            exog_lag_date = (fc_date - pd.DateOffset(months=1)).to_period("M").to_timestamp()
            rr = exog_df.loc[exog_df["date"] == exog_lag_date] if exog_df is not None else pd.DataFrame()
            if rr is not None and len(rr) > 0:
                vals_ex = rr.iloc[0][EXOG_COLS_VIZ].to_dict()
            else:
                vals_ex = clim.loc[int(exog_lag_date.month)].to_dict() if clim is not None else {c: 0.0 for c in EXOG_COLS_VIZ}
            for c in EXOG_COLS_VIZ:
                row[f"{c}_lag1"] = float(vals_ex.get(c, 0.0))

        X = pd.DataFrame([row], columns=feat_cols)
        delta_pred = float(model.predict(X)[0])
        y_fc = y_prev + delta_pred
        if non_negative:
            y_fc = max(0.0, float(y_fc))

        out_rows.append({"date": fc_date, "horizon": h, "y_forecast": float(y_fc)})

        lag12.append(y_fc)
        lag12 = lag12[-12:]
        delta_prev = delta_pred
        y_prev = y_fc

    return pd.DataFrame(out_rows)


def _seasonal_recursive_monthly(history: pd.DataFrame, horizon_max: int = 48) -> pd.DataFrame:
    """
    Simple monthly seasonal naive:
    y(t+h) = y(t+h-12) if exists, else last observed
    """
    hist = history.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    hist = hist.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)

    last_date = hist["date"].max()
    obs_map = dict(zip(hist["date"].tolist(), hist["value"].astype(float).tolist()))
    last_val = float(hist["value"].iloc[-1])

    rows = []
    for h in range(1, int(horizon_max) + 1):
        fc_date = (pd.Timestamp(last_date) + pd.DateOffset(months=h)).to_period("M").to_timestamp()
        ref = (fc_date - pd.DateOffset(months=12)).to_period("M").to_timestamp()
        y = float(obs_map.get(ref, last_val))
        rows.append({"date": fc_date, "horizon": h, "y_forecast": y})
    return pd.DataFrame(rows)


def _infer_selected_model_type(metrics_row: dict, skill_point: pd.DataFrame, selected_family: str) -> str:
    mt = str(metrics_row.get("selected_model_type", "") or "").strip()
    if mt:
        return mt
    if str(selected_family) == "BASELINE_SEASONAL":
        return "seasonal_naive"
    if (not skill_point.empty) and ("model_type" in skill_point.columns) and ("family" in skill_point.columns):
        gg = skill_point[skill_point["family"].astype(str) == str(selected_family)].copy()
        if not gg.empty:
            vals = gg["model_type"].dropna().astype(str).tolist()
            if vals:
                return pd.Series(vals).mode().iloc[0]
    return "bayes_ridge"


def _ensure_monthly_forecast_path(
    selected_family: str,
    selected_model_type: str,
    g_obs: pd.DataFrame,
    g_fc_key: pd.DataFrame,
    horizon_max: int = 48,
) -> pd.DataFrame:
    """
    Always returns a monthly path h=1..horizon_max.
    If core provides only key horizons, we build monthly with model-based path for viz.
    """
    # If already monthly (h=1..H)
    if "horizon" in g_fc_key.columns:
        h = pd.to_numeric(g_fc_key["horizon"], errors="coerce")
        if h.notna().any():
            if int(h.min()) == 1 and int(h.max()) >= horizon_max and int(h.nunique()) >= int(0.8 * horizon_max):
                df = g_fc_key.copy()
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
                df = df.dropna(subset=["date"]).sort_values("date")
                return df[["date", "horizon", "y_forecast"]].copy()

    # Otherwise build from scratch
    fam = str(selected_family or "")
    if fam == "BASELINE_SEASONAL":
        return _seasonal_recursive_monthly(g_obs, horizon_max=horizon_max)

    if fam == "EXOG_ML":
        missing_exog = [c for c in EXOG_COLS_VIZ if c not in g_obs.columns]
        if missing_exog:
            st.warning(tr(
                f"Faltan EXOG para visualizar EXOG_ML ({missing_exog}). Muestro ENDO_ML.",
                f"Missing EXOG columns for EXOG_ML viz ({missing_exog}). Falling back to ENDO_ML.",
                LANG
            ))
            fam = "ENDO_ML"

    return _forecast_monthly_path(g_obs, model_type=selected_model_type, family=fam, max_h=horizon_max)


# -----------------------------
# Paper-friendly helpers
# -----------------------------
def _paper_leaderboard_from_skill(g_sk: pd.DataFrame, horizons_focus=(1, 12)) -> pd.DataFrame:
    if g_sk is None or g_sk.empty:
        return pd.DataFrame()

    df = g_sk.copy()
    if "horizon" in df.columns:
        df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
    if "kge" in df.columns:
        df["kge"] = pd.to_numeric(df["kge"], errors="coerce")

    hmin, hmax = horizons_focus
    if "horizon" in df.columns:
        df = df[(df["horizon"] >= hmin) & (df["horizon"] <= hmax)].copy()
    if df.empty:
        return pd.DataFrame()

    if "family" not in df.columns:
        df["family"] = "UNKNOWN"
    if "model_type" not in df.columns:
        df["model_type"] = "UNKNOWN"
    else:
        df["model_type"] = df["model_type"].fillna("UNKNOWN").astype(str)

    agg = (
        df.groupby(["family", "model_type"], as_index=False)
          .agg(
              kge_mean=("kge", "mean"),
              kge_std=("kge", "std"),
              kge_min=("kge", "min"),
              kge_max=("kge", "max"),
              n=("kge", "count"),
          )
          .sort_values(["kge_mean", "kge_min"], ascending=[False, False])
          .reset_index(drop=True)
    )
    agg["rank"] = np.arange(1, len(agg) + 1)
    return agg


def _paper_mini_analysis(leaderboard: pd.DataFrame, planning_thr=0.60, advisory_thr=0.30) -> str:
    if leaderboard is None or leaderboard.empty:
        return "No leaderboard available (insufficient skill data)."

    w = leaderboard.iloc[0].to_dict()
    winner = f"{w.get('family','?')} / {w.get('model_type','?')}"
    msg = f"Winner: {winner}. Mean KGE (focus horizons) = {w.get('kge_mean', np.nan):.3f}."

    if len(leaderboard) > 1:
        r = leaderboard.iloc[1].to_dict()
        runner = f"{r.get('family','?')} / {r.get('model_type','?')}"
        gap = float(w.get("kge_mean", np.nan)) - float(r.get("kge_mean", np.nan))
        if np.isfinite(gap):
            msg += f" Runner-up: {runner} (ΔKGE_mean = {gap:.3f})."

    if float(w.get("kge_mean", -999)) >= planning_thr:
        msg += f" Strong planning-grade performance (KGE_mean ≥ {planning_thr})."
    elif float(w.get("kge_mean", -999)) >= advisory_thr:
        msg += f" Advisory-grade performance (KGE_mean ≥ {advisory_thr})."
    else:
        msg += " Low reliability under the current thresholds."

    return msg


def _paper_scenario_bands(monthly_fc: pd.DataFrame, favorable_pct=0.05, unfavorable_pct=0.05) -> pd.DataFrame:
    if monthly_fc is None or monthly_fc.empty:
        return pd.DataFrame()

    df = monthly_fc.copy()
    df["scenario_base"] = df["y_forecast"]
    df["scenario_favorable"] = df["y_forecast"] * (1.0 + float(favorable_pct))
    df["scenario_unfavorable"] = df["y_forecast"] * (1.0 - float(unfavorable_pct))
    return df


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
    st.caption(f"{tr('Estrategia', 'Strategy', LANG)}: {date_rep.strategy}")
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

if date_rep.ok_ratio < QUALITY_THRESHOLD or value_rep.ok_ratio < QUALITY_THRESHOLD:
    st.error(
        tr(
            f"La calidad de parsing es baja. Ajusta opciones hasta ratio OK >= {QUALITY_THRESHOLD:.2f}. Export bloqueado.",
            f"Parsing quality is below threshold. Adjust options until OK ratio >= {QUALITY_THRESHOLD:.2f}. Export blocked.",
            LANG,
        )
    )
    st.stop()

mask_eff = date_month.notna() & value_num.notna()
df_eff = df.loc[mask_eff].copy()
df_eff["_date"] = date_month.loc[mask_eff]
df_eff["_value"] = value_num.loc[mask_eff]
df_eff["_date_raw"] = date_raw.loc[mask_eff]


# -----------------------------
# 3) Quality summary
# -----------------------------
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

    resource_guess = ""
    try:
        resource_guess = str(canonical["resource_type"].dropna().astype(str).iloc[0]).strip().lower()
    except Exception:
        resource_guess = ""

    default_policy_label = "LAST (recommended for levels/storage)"
    if any(k in resource_guess for k in ["river", "demand", "flow", "inflow", "volume", "hm3", "m3"]):
        default_policy_label = "SUM (monthly total)"
    elif any(k in resource_guess for k in ["reservoir", "storage", "level", "stage", "elevation"]):
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

    do_agg = st.checkbox("✅ Convert to monthly now (recommended)", value=True, key="do_monthly_agg")

    if do_agg:
        pol = "LAST"
        if value_policy.startswith("MEAN"):
            pol = "MEAN"
        elif value_policy.startswith("SUM"):
            pol = "SUM"

        before_rows = len(canonical)
        canonical = aggregate_to_monthly_canonical(canonical, pol)
        after_rows = len(canonical)

        st.success(f"Monthly aggregation applied ✅  {before_rows} rows → {after_rows} monthly rows")
else:
    st.success("Your dataset already looks monthly (1 row per point_id and month) ✅")

canonical = canonical.drop(columns=["date_raw"], errors="ignore")
canonical = canonical.sort_values(["point_id", "date"]).reset_index(drop=True)

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

    if len(missing_months) == 0:
        st.success(tr("No faltan meses ✅", "No missing months detected ✅", LANG))
    else:
        st.warning(tr(f"Meses faltantes: {len(missing_months)}", f"Missing months detected: {len(missing_months)}", LANG))
        st.dataframe(missing_months.head(25), use_container_width=True)

st.write("---")
confirm = st.checkbox(tr("✅ Confirmo que el mapeo/parsing es correcto (obligatorio)", "✅ I confirm mapping/parsing is correct (required)", LANG))
if not confirm:
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

    # A) Upload meteo
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
                st.stop()

            # Guesses
            met_date_guess = _guess_date_col(cols)
            met_precip_guess = next((c for c in cols if "precip" in str(c).lower() or "rain" in str(c).lower() or "lluv" in str(c).lower()), cols[0])
            met_t2m_guess = next((c for c in cols if "t2m" in str(c).lower() or ("temp" in str(c).lower() and "max" not in str(c).lower() and "min" not in str(c).lower())), cols[0])
            met_tmax_guess = next((c for c in cols if "tmax" in str(c).lower() or "max" in str(c).lower()), cols[0])
            met_tmin_guess = next((c for c in cols if "tmin" in str(c).lower() or "min" in str(c).lower()), cols[0])
            met_pid_guess = next((c for c in cols if "point_id" in str(c).lower() or str(c).lower() == "id" or "point" in str(c).lower()), "")

            NONE_OPT = "__NONE__"

            def _fmt_none(x: str) -> str:
                if x == NONE_OPT:
                    return tr("— No tengo este valor —", "— I don't have this value —", LANG)
                return str(x)

            opt_cols = [NONE_OPT] + cols

            c1, c2, c3 = st.columns(3)
            with c1:
                met_date_col = st.selectbox(
                    tr("Date column", "Date column", LANG),
                    cols,
                    index=cols.index(met_date_guess) if met_date_guess in cols else 0,
                    key="met_date_col",
                )
                met_precip_sel = st.selectbox(
                    tr("Precipitation column", "Precipitation column", LANG),
                    opt_cols,
                    index=(1 + cols.index(met_precip_guess)) if met_precip_guess in cols else 0,
                    format_func=_fmt_none,
                    key="met_precip_col",
                )
            with c2:
                met_t2m_sel = st.selectbox(
                    tr("Mean temperature (T2M) column", "Mean temperature (T2M) column", LANG),
                    opt_cols,
                    index=(1 + cols.index(met_t2m_guess)) if met_t2m_guess in cols else 0,
                    format_func=_fmt_none,
                    key="met_t2m_col",
                )
                met_tmax_sel = st.selectbox(
                    tr("Max temperature (Tmax) column", "Max temperature (Tmax) column", LANG),
                    opt_cols,
                    index=(1 + cols.index(met_tmax_guess)) if met_tmax_guess in cols else 0,
                    format_func=_fmt_none,
                    key="met_tmax_col",
                )
            with c3:
                met_tmin_sel = st.selectbox(
                    tr("Min temperature (Tmin) column", "Min temperature (Tmin) column", LANG),
                    opt_cols,
                    index=(1 + cols.index(met_tmin_guess)) if met_tmin_guess in cols else 0,
                    format_func=_fmt_none,
                    key="met_tmin_col",
                )

            fill_missing_with_nasa = st.checkbox(
                tr("Completar valores faltantes con NASA POWER (recomendado)",
                   "Fill missing values with NASA POWER (recommended)", LANG),
                value=True,
                key="met_fill_missing_with_nasa",
            )

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

            raw2 = raw.copy()

            def _ensure_col(sel: str, placeholder_name: str) -> str:
                if sel == NONE_OPT:
                    if placeholder_name not in raw2.columns:
                        raw2[placeholder_name] = np.nan
                    return placeholder_name
                return sel

            met_precip_col = _ensure_col(met_precip_sel, "_missing_precip")
            met_t2m_col = _ensure_col(met_t2m_sel, "_missing_t2m")
            met_tmax_col = _ensure_col(met_tmax_sel, "_missing_tmax")
            met_tmin_col = _ensure_col(met_tmin_sel, "_missing_tmin")

            met_map = MeteoMapping(
                date_col=met_date_col,
                precip_col=met_precip_col,
                t2m_col=met_t2m_col,
                tmax_col=met_tmax_col,
                tmin_col=met_tmin_col,
                point_id_col=point_id_col if mode_key == "HAS_POINT_ID" else "",
            )

            pts_tmp = canonical_df.groupby("point_id", as_index=False).agg({"lat": "first", "lon": "first"})
            coords_ok_all = pts_tmp[["lat", "lon"]].notna().all(axis=1).all()

            if st.button(tr("apply_uploaded_meteo_btn"), key="apply_uploaded_meteo"):
                try:
                    user_provided_any = any(sel != NONE_OPT for sel in [met_precip_sel, met_t2m_sel, met_tmax_sel, met_tmin_sel])

                    if (not user_provided_any) and fill_missing_with_nasa and coords_ok_all:
                        with st.spinner(tr("meteo_fetching_spinner")):
                            meteo_df, canonical_with_meteo = fetch_nasa_power_for_canonical(canonical_df)
                        st.session_state["meteo_df"] = meteo_df
                        st.session_state["canonical_with_meteo"] = canonical_with_meteo
                        st.success(tr("No aportaste variables meteo; he usado NASA POWER automáticamente.",
                                      "You didn't provide meteo variables; NASA POWER was used automatically.", LANG))
                        st.rerun()

                    meteo_df = build_user_meteo_table(
                        raw=raw2,
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

                    if {"t2m_c", "tmax_c", "tmin_c"}.issubset(set(canonical_with_meteo.columns)):
                        m = (
                            canonical_with_meteo["t2m_c"].isna()
                            & canonical_with_meteo["tmax_c"].notna()
                            & canonical_with_meteo["tmin_c"].notna()
                        )
                        canonical_with_meteo.loc[m, "t2m_c"] = (
                            canonical_with_meteo.loc[m, "tmax_c"] + canonical_with_meteo.loc[m, "tmin_c"]
                        ) / 2.0

                    needs_fill = canonical_with_meteo[METEO_COLS].isna().any().any()
                    filled_any = pd.Series(False, index=canonical_with_meteo.index)

                    if fill_missing_with_nasa and coords_ok_all and needs_fill:
                        with st.spinner(tr("meteo_fetching_spinner")):
                            _met_nasa, cwm_nasa = fetch_nasa_power_for_canonical(canonical_df)
                        for c in METEO_COLS:
                            mask = canonical_with_meteo[c].isna() & cwm_nasa[c].notna()
                            if mask.any():
                                canonical_with_meteo.loc[mask, c] = cwm_nasa.loc[mask, c]
                                filled_any |= mask

                    user_any_before = canonical_with_meteo[METEO_COLS].notna().any(axis=1)
                    source = np.where(user_any_before, "USER_UPLOAD", "NONE")
                    source = np.where(filled_any & user_any_before, "MIXED_NASA", source)
                    source = np.where(filled_any & ~user_any_before, "NASA_POWER_MONTHLY", source)
                    canonical_with_meteo["source"] = source

                    meteo_df_final = canonical_with_meteo[["point_id", "date"] + METEO_COLS + ["source"]].copy()
                    st.session_state["meteo_df"] = meteo_df_final
                    st.session_state["canonical_with_meteo"] = canonical_with_meteo

                    st.success(tr("meteo_integrated_ok"))
                    st.rerun()

                except Exception as e:
                    st.error(f"{tr('meteo_integrated_err')}: {e}")

    # B) NASA POWER (with coord resolving)
    elif choice == tr("meteo_choice_b"):
        st.subheader(tr("meteo_b_title"))

        pts = canonical_df.groupby("point_id", as_index=False).agg({"lat": "first", "lon": "first"})
        pts["coords_ok"] = pts[["lat", "lon"]].notna().all(axis=1)
        n_ok = int(pts["coords_ok"].sum())
        n_total = int(len(pts))
        n_missing = n_total - n_ok

        st.write(tr("meteo_coords_status").format(n_ok=n_ok, n_total=n_total))

        global_loc = ""
        loc_df = None
        loc_pid_col = loc_text_col = loc_lat_col = loc_lon_col = None

        if n_missing > 0:
            st.warning(tr("meteo_coords_missing").format(n_missing=n_missing))
            st.dataframe(pts, use_container_width=True)

            st.markdown(tr("meteo_fix_coords_title"))
            global_loc = st.text_input(tr("meteo_global_location"), value="", key="global_loc")

            loc_upload = st.file_uploader(tr("meteo_locations_upload"), type=["csv", "xlsx", "xls"], key="loc_upload")
            if loc_upload is not None:
                loc_df = read_table_from_upload(loc_upload)
                st.write(tr("preview"))
                st.dataframe(loc_df.head(20), use_container_width=True)

                cols = list(loc_df.columns)
                pid_guess = next((c for c in cols if "point" in str(c).lower() or str(c).lower() == "id" or "point_id" in str(c).lower()), cols[0])
                loc_guess = next((c for c in cols if "loc" in str(c).lower() or "city" in str(c).lower() or "municip" in str(c).lower() or "place" in str(c).lower() or "name" in str(c).lower()), "")
                lat_guess = next((c for c in cols if str(c).lower() in ("lat", "latitude")), "")
                lon_guess = next((c for c in cols if str(c).lower() in ("lon", "longitude", "lng")), "")

                c1, c2 = st.columns(2)
                with c1:
                    loc_pid_col = st.selectbox(tr("loc_col_point_id"), cols, index=cols.index(pid_guess) if pid_guess in cols else 0, key="loc_pid_col")
                    loc_text_col = st.selectbox(tr("loc_col_location"), ["(none)"] + cols, index=(cols.index(loc_guess) + 1) if loc_guess in cols else 0, key="loc_text_col")
                with c2:
                    loc_lat_col = st.selectbox(tr("loc_col_lat"), ["(none)"] + cols, index=(cols.index(lat_guess) + 1) if lat_guess in cols else 0, key="loc_lat_col")
                    loc_lon_col = st.selectbox(tr("loc_col_lon"), ["(none)"] + cols, index=(cols.index(lon_guess) + 1) if lon_guess in cols else 0, key="loc_lon_col")

        if n_ok > 0 and st.checkbox(tr("meteo_show_reverse"), value=False, key="rev_geo_chk"):
            rows = []
            for _, r in pts[pts["coords_ok"]].head(8).iterrows():
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

        if n_missing > 0 and st.button(tr("meteo_resolve_coords_btn"), key="resolve_coords"):
            try:
                pts2 = pts.copy()

                # 1) Per-point fill from locations file
                if loc_df is not None and loc_pid_col:
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

                # 2) Global fallback
                still_missing = pts2[~pts2["coords_ok"]]
                gl = (global_loc or "").strip()
                if (not still_missing.empty) and gl:
                    gr = geocode_place(gl)
                    if gr:
                        pts2.loc[~pts2["coords_ok"], "lat"] = gr.lat
                        pts2.loc[~pts2["coords_ok"], "lon"] = gr.lon
                        pts2["coords_ok"] = pts2[["lat", "lon"]].notna().all(axis=1)

                pt_lookup = dict(zip(pts2["point_id"].astype(str), zip(pts2["lat"], pts2["lon"])))
                canonical_df2 = canonical_df.copy()
                canonical_df2["lat"] = canonical_df2["point_id"].map(lambda x: pt_lookup.get(str(x), (np.nan, np.nan))[0])
                canonical_df2["lon"] = canonical_df2["point_id"].map(lambda x: pt_lookup.get(str(x), (np.nan, np.nan))[1])

                st.session_state["canonical_df"] = canonical_df2
                st.success(tr("meteo_coords_resolved_ok"))
                st.rerun()
            except Exception as e:
                st.error(f"{tr('meteo_coords_resolved_err')}: {e}")

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

    # C) ENDO only
    else:
        st.info(tr("meteo_endo_only_info"))
        st.session_state["canonical_with_meteo"] = canonical_df.copy()

# Downloads (meteo)
if "canonical_with_meteo" in st.session_state and st.session_state["canonical_with_meteo"] is not None:
    st.markdown("### 📥 Downloads")
    cwm = st.session_state["canonical_with_meteo"]
    st.download_button(
        tr("download_canonical_with_meteo"),
        data=cwm.to_csv(index=False).encode("utf-8"),
        file_name="canonical_with_meteo.csv",
        mime="text/csv",
    )
if "meteo_df" in st.session_state and st.session_state["meteo_df"] is not None:
    met = st.session_state["meteo_df"]
    st.download_button(
        tr("download_meteo_df"),
        data=met.to_csv(index=False).encode("utf-8"),
        file_name="meteo_power_monthly.csv",
        mime="text/csv",
    )


# -----------------------------
# 9.5) Demand (optional) — Bulletproof (never breaks the app)
# -----------------------------
st.markdown("---")
st.header("📦 Demand (optional)")

from pathlib import Path

base_exog = st.session_state.get("canonical_with_meteo", None)
if base_exog is None:
    base_exog = st.session_state["canonical_df"].copy()
else:
    base_exog = base_exog.copy()

base_exog["date"] = to_month_start(base_exog["date"])
base_exog["point_id"] = base_exog["point_id"].astype(str)

demand_choice = st.radio(
    tr("Demanda exógena", "Demand exogenous", LANG),
    [
        "A) Upload demand (user)",
        "B) Open demand (Dataverse ZIP cached) — TOTAL withdrawals demand (simple)",
        "C) No demand (continue)",
    ],
    index=2,
    key="demand_choice_v014",
)

# ---- Persistent state keys (do NOT overwrite every rerun) ----
if "canonical_with_meteo_and_demand" not in st.session_state:
    st.session_state["canonical_with_meteo_and_demand"] = None
if "open_demand_info" not in st.session_state:
    st.session_state["open_demand_info"] = None

# If the user changed the option, clear previous merge to avoid mixing states
_prev_choice = st.session_state.get("demand_choice_prev_v014", None)
if _prev_choice is not None and _prev_choice != demand_choice:
    st.session_state["canonical_with_meteo_and_demand"] = None
    st.session_state["open_demand_info"] = None
st.session_state["demand_choice_prev_v014"] = demand_choice

# -----------------------------
# Helpers (local fallbacks) — so this block never NameErrors
# -----------------------------
def _guess_date_col_fallback(cols):
    cands = [c for c in cols if "date" in str(c).lower() or "fecha" in str(c).lower() or "time" in str(c).lower()]
    return cands[0] if cands else cols[0]

def _guess_demand_col_fallback(cols):
    keys = ["demand", "demanda", "total", "withdraw", "twd", "hm3", "km3", "value", "valor"]
    for k in keys:
        hit = next((c for c in cols if k in str(c).lower()), None)
        if hit is not None:
            return hit
    return cols[1] if len(cols) > 1 else cols[0]

# A) User upload
if demand_choice.startswith("A)"):
    st.subheader("A) Upload demand file (safe)")

    dem_apply_mode = st.radio(
        tr("¿La demanda tiene point_id?", "Does demand file have point_id?", LANG),
        ["COMMON_ALL (same demand for all points)", "HAS_POINT_ID (demand file has point_id column)"],
        index=0,
        key="dem_apply_mode_v014",
    )

    demand_upload = st.file_uploader(
        tr("Sube fichero de demanda (CSV o Excel)", "Upload demand file (CSV or Excel)", LANG),
        type=["csv", "xlsx", "xls"],
        key="demand_upload_v014",
    )

    if demand_upload is not None:
        try:
            raw_dem = read_table_from_upload(demand_upload)
            st.write(tr("Vista previa", "Preview", LANG))
            st.dataframe(raw_dem.head(20), use_container_width=True)

            cols = list(raw_dem.columns)
            if len(cols) < 2:
                st.error("Demand file must have at least 2 columns (date + value).")
            else:
                # robust guessing (uses your helpers if they exist, else fallback)
                try:
                    dem_date_guess = _guess_date_col(cols)
                except Exception:
                    dem_date_guess = _guess_date_col_fallback(cols)

                try:
                    dem_val_guess = _guess_demand_col(cols)
                except Exception:
                    dem_val_guess = _guess_demand_col_fallback(cols)

                dem_date_col = st.selectbox(
                    tr("Columna fecha (demanda)", "Date column (demand)", LANG),
                    cols,
                    index=cols.index(dem_date_guess) if dem_date_guess in cols else 0,
                    key="dem_date_col_v014",
                )
                dem_val_col = st.selectbox(
                    tr("Columna valor demanda", "Demand value column", LANG),
                    cols,
                    index=cols.index(dem_val_guess) if dem_val_guess in cols else (1 if len(cols) > 1 else 0),
                    key="dem_val_col_v014",
                )
                st.caption(tr(
                    f"Auto-detected: date='{dem_date_guess}', demand='{dem_val_guess}'",
                    f"Auto-detected: date='{dem_date_guess}', demand='{dem_val_guess}'",
                    LANG,
                ))

                dem_pid_col = None
                if dem_apply_mode.startswith("HAS_POINT_ID"):
                    pid_guess = next(
                        (c for c in cols if "point_id" in str(c).lower() or str(c).lower() == "id" or "point" in str(c).lower()),
                        cols[0]
                    )
                    dem_pid_col = st.selectbox(
                        tr("Columna point_id (demanda)", "Point id column (demand)", LANG),
                        cols,
                        index=cols.index(pid_guess) if pid_guess in cols else 0,
                        key="dem_pid_col_v014",
                    )

                if st.button(tr("✅ Integrar demanda", "✅ Integrate demand into canonical", LANG), key="apply_demand_upload_v014"):
                    ddem = raw_dem.copy()
                    ddem = ddem.rename(columns={dem_date_col: "date", dem_val_col: "demand"})
                    ddem["date"] = to_month_start(ddem["date"])
                    ddem["demand"] = pd.to_numeric(ddem["demand"], errors="coerce")
                    ddem = ddem.dropna(subset=["date", "demand"]).copy()

                    if dem_pid_col is not None:
                        ddem["point_id"] = ddem[dem_pid_col].astype(str).str.strip()
                        ddem = ddem.groupby(["point_id", "date"], as_index=False)["demand"].mean()
                        merged = base_exog.merge(ddem[["point_id", "date", "demand"]], on=["point_id", "date"], how="left")
                        apply_mode = "HAS_POINT_ID"
                    else:
                        ddem = ddem.groupby(["date"], as_index=False)["demand"].mean()
                        merged = base_exog.merge(ddem[["date", "demand"]], on=["date"], how="left")
                        apply_mode = "COMMON_ALL"

                    st.session_state["canonical_with_meteo_and_demand"] = merged
                    st.session_state["open_demand_info"] = {
                        "source": "user_upload",
                        "apply_mode": apply_mode,
                        "file_name": getattr(demand_upload, "name", "UNKNOWN"),
                        "date_col": str(dem_date_col),
                        "value_col": str(dem_val_col),
                        "point_id_col": str(dem_pid_col) if dem_pid_col is not None else None,
                    }

                    st.success(tr(
                        f"Demand integrated ✅  Non-null rows: {int(merged['demand'].notna().sum())}",
                        f"Demand integrated ✅  Non-null rows: {int(merged['demand'].notna().sum())}",
                        LANG,
                    ))
        except Exception as e:
            st.error(f"Demand integration failed (safe): {e}")

# B) Open demand (Dataverse ZIP cached)
elif demand_choice.startswith("B)"):
    st.subheader("B) Open demand (Dataverse ZIP cached) — TOTAL withdrawals demand (simple)")

    cache_dir = Path("outputs") / "cache" / "demand"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Fixed "simple" choice (your current validated pair)
    scenario = "ssp1_rcp26"
    gcm = "gfdl"
    file_id_1 = 6062173
    file_id_2 = 6062170
    zip_name_1 = f"{scenario}_{gcm}_withdrawals_sectors_monthly_1.zip"
    zip_name_2 = f"{scenario}_{gcm}_withdrawals_sectors_monthly_2.zip"

    zip1 = cache_dir / zip_name_1
    zip2 = cache_dir / zip_name_2

    hist_min = pd.to_datetime(base_exog["date"].min(), errors="coerce")
    hist_max = pd.to_datetime(base_exog["date"].max(), errors="coerce")
    if pd.isna(hist_min) or pd.isna(hist_max):
        st.warning("Historical dates are invalid. Continuing safely without demand.")
        st.session_state["canonical_with_meteo_and_demand"] = None
        st.session_state["open_demand_info"] = None
    else:
        st.caption(f"Your historical period: {hist_min.date()} to {hist_max.date()}")

        overlap_start = max(hist_min.to_period("M").to_timestamp(), pd.Timestamp("2010-01-01"))
        overlap_end = hist_max.to_period("M").to_timestamp()
        st.caption(f"Demand overlap used: {overlap_start.date()} to {overlap_end.date()}")

        st.caption(f"Cache folder: {cache_dir}")
        st.write(
            tr("Estado de caché ZIP:", "ZIP cache status:", LANG),
            {
                "zip1_exists": zip1.exists(),
                "zip2_exists": zip2.exists(),
                "zip1_path": str(zip1),
                "zip2_path": str(zip2),
            },
        )

        if zip1.exists() and zip2.exists():
            st.success("Cached ZIPs found ✅ (no upload needed).")
        else:
            st.warning("Missing one or both ZIPs. Upload them once below OR enable auto-download (large files).")

            c_up1, c_up2 = st.columns(2)
            with c_up1:
                up1 = st.file_uploader("Upload ZIP 1 (…monthly_1.zip)", type=["zip"], key="open_demand_zip1_upload_v014")
            with c_up2:
                up2 = st.file_uploader("Upload ZIP 2 (…monthly_2.zip)", type=["zip"], key="open_demand_zip2_upload_v014")

            if up1 is not None:
                try:
                    with open(zip1, "wb") as f:
                        f.write(up1.getbuffer())
                    st.success(f"Saved ZIP1 ✅ {zip1}")
                except Exception as e:
                    st.error(f"Failed to save ZIP1 (safe): {e}")

            if up2 is not None:
                try:
                    with open(zip2, "wb") as f:
                        f.write(up2.getbuffer())
                    st.success(f"Saved ZIP2 ✅ {zip2}")
                except Exception as e:
                    st.error(f"Failed to save ZIP2 (safe): {e}")

        allow_download = st.checkbox(
            "Allow auto-download from Dataverse if ZIPs missing (VERY LARGE, ~2–3GB).",
            value=False,
            key="open_demand_allow_download_v014",
        )

        ssl_verify = st.checkbox(
            "SSL verify (uncheck only if corporate SSL blocks requests)",
            value=True,
            key="open_demand_ssl_verify_v014",
        )

        if st.button("✅ Integrate OPEN demand into canonical (safe)", key="apply_open_demand_v014"):
            try:
                from basincast.demand.open_demand import build_total_demand_hm3_monthly, integrate_total_demand_into_canonical

                # Guard: if zips missing and we don't allow download, don't attempt
                if (not zip1.exists() or not zip2.exists()) and (not allow_download):
                    st.warning("ZIPs missing and auto-download disabled. Upload the ZIPs or enable download.")
                    st.session_state["canonical_with_meteo_and_demand"] = None
                    st.session_state["open_demand_info"] = None
                else:
                    months_needed = list(pd.date_range(start=overlap_start, end=overlap_end, freq="MS"))

                    bbox = None
                    if ("lat" in base_exog.columns) and ("lon" in base_exog.columns):
                        lat = pd.to_numeric(base_exog["lat"], errors="coerce")
                        lon = pd.to_numeric(base_exog["lon"], errors="coerce")
                        if lat.notna().any() and lon.notna().any():
                            bbox = (float(lat.min()), float(lon.min()), float(lat.max()), float(lon.max()))
                    if bbox is None:
                        st.warning("No valid lat/lon in canonical table. Skipping open demand safely (bbox is required).")
                        st.session_state["canonical_with_meteo_and_demand"] = None
                        st.session_state["open_demand_info"] = None
                    else:
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

                        # Store provenance for manifest
                        info = dict(info or {})
                        info.update({
                            "source": "dataverse_zip_csv",
                            "scenario": scenario,
                            "gcm": gcm,
                            "file_id_1": int(file_id_1),
                            "file_id_2": int(file_id_2),
                            "zip1": str(zip1),
                            "zip2": str(zip2),
                        })
                        st.session_state["open_demand_info"] = info

                        if dem_df is None or dem_df.empty:
                            st.warning(f"Open demand not integrated (safe). Info: {info}")
                            st.session_state["canonical_with_meteo_and_demand"] = None
                        else:
                            merged = integrate_total_demand_into_canonical(
                                base_exog,
                                dem_df,      # expects demand_hm3
                                date_col="date",
                                out_col="demand",
                            )
                            st.session_state["canonical_with_meteo_and_demand"] = merged
                            st.success(tr(
                                f"Open demand integrated ✅  Non-null rows: {int(merged['demand'].notna().sum())}",
                                f"Open demand integrated ✅  Non-null rows: {int(merged['demand'].notna().sum())}",
                                LANG,
                            ))
                            with st.expander("Open demand provenance (paper-friendly)"):
                                st.json(info)

            except Exception as e:
                st.error(f"Open demand integration failed (safe): {e}")
                st.session_state["canonical_with_meteo_and_demand"] = None
                st.session_state["open_demand_info"] = None

# C) No demand
else:
    st.info(tr("Continuamos sin demanda.", "Continuing without demand.", LANG))
    st.session_state["canonical_with_meteo_and_demand"] = None
    st.session_state["open_demand_info"] = None

cwd = st.session_state.get("canonical_with_meteo_and_demand", None)
if cwd is not None:
    st.markdown("### 📥 Downloads")
    st.download_button(
        "Download canonical_with_meteo_and_demand.csv",
        data=cwd.to_csv(index=False).encode("utf-8"),
        file_name="canonical_with_meteo_and_demand.csv",
        mime="text/csv",
    )


# -----------------------------
# 10) Run BasinCast + Visualize
# -----------------------------
st.markdown("---")
st.header(tr("📈 Ejecutar BasinCast + Visualizar (v0.14)", "📈 Run BasinCast Core + Visualize (v0.14)", LANG))

import json
import hashlib
from datetime import datetime
from typing import Optional

# Prefer canonical_with_meteo_and_demand > canonical_with_meteo > canonical_df
df_run = None
run_mode = "ENDO_ONLY"

if st.session_state.get("canonical_with_meteo_and_demand", None) is not None:
    df_run = st.session_state["canonical_with_meteo_and_demand"].copy()
    run_mode = "CANONICAL_WITH_METEO_DEMAND"
elif st.session_state.get("canonical_with_meteo", None) is not None:
    df_run = st.session_state["canonical_with_meteo"].copy()
    run_mode = "CANONICAL_WITH_METEO"
else:
    df_run = st.session_state["canonical_df"].copy()
    run_mode = "ENDO_ONLY"

df_run["date"] = pd.to_datetime(df_run["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
df_run = df_run.dropna(subset=["date", "point_id", "value"]).sort_values(["point_id", "date"]).reset_index(drop=True)

st.caption(
    tr(
        f"Modo: **{run_mode}** | Puntos: **{df_run['point_id'].nunique()}** | Filas: **{len(df_run)}**",
        f"Run mode: **{run_mode}** | Points: **{df_run['point_id'].nunique()}** | Rows: **{len(df_run)}**",
        LANG,
    )
)

def _pick_latest_csv(folder: Path, prefix: str) -> Optional[Path]:
    files = list(folder.glob(f"{prefix}*.csv"))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]

def _make_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")

def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _try_git_commit_hash() -> str:
    try:
        import subprocess
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(Path.cwd()),
        )
        if res.returncode == 0:
            return (res.stdout or "").strip()
    except Exception:
        pass
    return "UNKNOWN"

def _resolve_core_outputs(outdir: Path) -> tuple[Path, Path, Path]:
    fixed_metrics = outdir / "metrics_v0_6.csv"
    fixed_skill = outdir / "skill_v0_6.csv"
    fixed_fc = outdir / "forecasts_v0_6.csv"

    if fixed_metrics.exists() and fixed_skill.exists() and fixed_fc.exists():
        return fixed_metrics, fixed_skill, fixed_fc

    m = _pick_latest_csv(outdir, "metrics_")
    s = _pick_latest_csv(outdir, "skill_")
    f = _pick_latest_csv(outdir, "forecasts_")
    if m is None or s is None or f is None:
        raise RuntimeError(f"Core did not create expected outputs in {outdir}")
    return m, s, f

def _run_core_cli(df_input: pd.DataFrame, outdir: Path, holdout_months: int, inner_val_months: int) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    tmp_inp = outdir / "_tmp_canonical_for_core.csv"
    df_input.to_csv(tmp_inp, index=False)

    script_candidates = [
        Path("app/run_core_v0_6.py"),
        Path("app/run_core.py"),
    ]
    core_script = next((p for p in script_candidates if p.exists()), None)
    if core_script is None:
        raise FileNotFoundError("Cannot find app/run_core_v0_6.py or app/run_core.py")

    cmd = [
        sys.executable,
        str(core_script),
        "--input", str(tmp_inp),
        "--outdir", str(outdir),
        "--holdout_months", str(int(holdout_months)),
        "--inner_val_months", str(int(inner_val_months)),
    ]

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    stdout = res.stdout or ""
    stderr = res.stderr or ""

    if res.returncode != 0:
        raise RuntimeError(f"Core failed (returncode={res.returncode}). STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}")

    metrics_path, skill_path, forecasts_path = _resolve_core_outputs(outdir)
    metrics_df = pd.read_csv(metrics_path)
    skill_df = pd.read_csv(skill_path)
    forecasts_df = pd.read_csv(forecasts_path)

    return {
        "stdout": stdout,
        "stderr": stderr,
        "metrics": metrics_df,
        "skill": skill_df,
        "forecasts": forecasts_df,
        "paths": {
            "metrics": str(metrics_path),
            "skill": str(skill_path),
            "forecasts": str(forecasts_path),
        },
    }

c1, c2, c3 = st.columns(3)
with c1:
    holdout_months = st.number_input(
        tr("Holdout (meses)", "Holdout months (paper-2 protocol)", LANG),
        min_value=24, max_value=240, value=96, step=12
    )
with c2:
    inner_val_months = st.number_input(
        tr("Validación interna (meses)", "Inner validation months", LANG),
        min_value=12, max_value=120, value=36, step=12
    )
with c3:
    default_root = Path("outputs") / "runs"
    default_root.mkdir(parents=True, exist_ok=True)

    run_root = Path(st.text_input(
        tr("Carpeta raíz de runs", "Runs root folder", LANG),
        value=str(default_root),
    ))

    # pending run id (so you see the folder BEFORE running)
    if "pending_run_id" not in st.session_state:
        st.session_state["pending_run_id"] = _make_run_id()

    regen = st.button(tr("🔄 Nuevo run folder", "🔄 New run folder", LANG), key="regen_run_id_v014")
    if regen:
        st.session_state["pending_run_id"] = _make_run_id()

    run_outdir = run_root / st.session_state["pending_run_id"]
    st.caption(f"Next run folder: {run_outdir}")

run_btn = st.button(tr("▶ Ejecutar BasinCast Core", "▶ Run BasinCast Core", LANG), type="primary")

if run_btn:
    try:
        with st.spinner(tr("Ejecutando BasinCast Core...", "Running BasinCast Core...", LANG)):
            out = _run_core_cli(df_run, run_outdir, holdout_months, inner_val_months)

        # ---- Write manifest.json (paper-friendly reproducibility) ----
        try:
            tmp_inp = Path(run_outdir) / "_tmp_canonical_for_core.csv"
            canonical_sha256 = _sha256_file(tmp_inp) if tmp_inp.exists() else "MISSING"

            meteo_info = st.session_state.get("meteo_info", None)
            demand_info = st.session_state.get("open_demand_info", None)

            manifest = {
                "run_id": st.session_state.get("pending_run_id", "UNKNOWN"),
                "timestamp_local": datetime.now().isoformat(timespec="seconds"),
                "git_commit": _try_git_commit_hash(),
                "core": {
                    "holdout_months": int(holdout_months),
                    "inner_val_months": int(inner_val_months),
                },
                "inputs": {
                    "canonical_csv": str(tmp_inp),
                    "canonical_sha256": canonical_sha256,
                    "run_mode": str(run_mode),
                },
                "provenance": {
                    "meteo": meteo_info,
                    "demand": demand_info,
                },
                "outputs": out.get("paths", {}),
            }

            manifest_path = Path(run_outdir) / "manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)

            st.caption(f"Manifest saved: {manifest_path}")

            # Include manifest in displayed paths
            out["paths"]["manifest"] = str(manifest_path)
        except Exception as e:
            st.warning(f"Manifest could not be written (safe): {e}")

        st.session_state["core_stdout"] = out["stdout"]
        st.session_state["core_stderr"] = out["stderr"]
        st.session_state["core_metrics"] = out["metrics"]
        st.session_state["core_skill"] = out["skill"]
        st.session_state["core_forecasts"] = out["forecasts"]
        st.session_state["core_paths"] = out["paths"]
        st.session_state["last_run_outdir"] = str(run_outdir)

        # After a run, auto-generate a fresh run id for the NEXT run (prevents overwriting)
        st.session_state["pending_run_id"] = _make_run_id()

        st.success(tr(
            "Core ejecutado. Outputs cargados en la app.",
            "Core run completed. Outputs loaded into the app.",
            LANG
        ))
    except Exception as e:
        st.error(f"Core run failed: {e}")

if "core_stdout" in st.session_state:
    with st.expander(tr("Logs del Core (stdout/stderr)", "Core logs (stdout/stderr)", LANG)):
        st.code(st.session_state.get("core_stdout", "")[:8000])
        if st.session_state.get("core_stderr", ""):
            st.code(st.session_state.get("core_stderr", "")[:8000])

# Visualize
if "core_metrics" in st.session_state and "core_skill" in st.session_state and "core_forecasts" in st.session_state:
    metrics_df = st.session_state["core_metrics"].copy()
    skill_df = st.session_state["core_skill"].copy()
    forecasts_df = st.session_state["core_forecasts"].copy()

    if "point_id" not in metrics_df.columns:
        st.error(tr(
            "metrics.csv no tiene point_id. No puedo visualizar.",
            "metrics output has no point_id column. Cannot visualize.",
            LANG
        ))
        st.stop()

    point_ids = sorted(metrics_df["point_id"].astype(str).unique().tolist())
    pid = st.selectbox(tr("Selecciona point_id", "Select point_id", LANG), point_ids, index=0)

    g_obs = df_run[df_run["point_id"].astype(str) == str(pid)].copy()
    g_obs["date"] = pd.to_datetime(g_obs["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    g_obs = g_obs.dropna(subset=["date"]).sort_values("date")

    g_fc_key = forecasts_df[forecasts_df["point_id"].astype(str) == str(pid)].copy()
    if "date" in g_fc_key.columns:
        g_fc_key["date"] = pd.to_datetime(g_fc_key["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()

    g_sk = skill_df[skill_df["point_id"].astype(str) == str(pid)].copy()
    mrow = metrics_df[metrics_df["point_id"].astype(str) == str(pid)].head(1).copy()

    st.subheader(tr("🧾 Resumen de decisión", "🧾 Decision summary", LANG))

    decision = ""
    selected_family = ""
    planning_h = 0
    advisory_h = 0

    if not mrow.empty:
        m = mrow.iloc[0].to_dict()
        decision = str(m.get("decision", ""))
        selected_family = str(m.get("selected_family", ""))
        planning_h = int(m.get("planning_horizon", 0) or 0)
        advisory_h = int(m.get("advisory_horizon", 0) or 0)
        selected_model_type = _infer_selected_model_type(m, g_sk, selected_family)

        colA, colB, colC, colD = st.columns(4)
        colA.metric(tr("Decisión", "Decision", LANG), decision)
        colB.metric(tr("Familia", "Selected family", LANG), selected_family)
        colC.metric(tr("Horiz. planificación", "Planning horizon", LANG), planning_h)
        colD.metric(tr("Horiz. advisory", "Advisory horizon", LANG), advisory_h)
    else:
        selected_family = "UNKNOWN"
        selected_model_type = "UNKNOWN"

    HORIZON_MAX = 48
    monthly_fc = _ensure_monthly_forecast_path(
        selected_family=selected_family,
        selected_model_type=selected_model_type,
        g_obs=g_obs,
        g_fc_key=g_fc_key,
        horizon_max=HORIZON_MAX,
    )
    monthly_fc["date"] = pd.to_datetime(monthly_fc["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    monthly_fc["horizon"] = pd.to_numeric(monthly_fc["horizon"], errors="coerce").astype(int)
    monthly_fc = monthly_fc.dropna(subset=["date", "horizon", "y_forecast"]).sort_values("date")

    # Paper-friendly tables
    st.subheader(tr("📋 Tablas paper-friendly (por qué gana un modelo)", "📋 Paper-friendly tables (why a model wins)", LANG))
    leaderboard = _paper_leaderboard_from_skill(g_sk, horizons_focus=(1, 12))
    if leaderboard.empty:
        st.info(tr(
            "No puedo construir leaderboard (skill insuficiente o sin columnas esperadas).",
            "Cannot build leaderboard (insufficient skill or missing columns).",
            LANG
        ))
    else:
        st.dataframe(leaderboard, use_container_width=True)
        st.caption(_paper_mini_analysis(leaderboard, planning_thr=0.60, advisory_thr=0.30))
        with st.expander(tr("Ver skill detallado (transparencia)", "Show detailed skill table (transparency)", LANG)):
            if {"family", "horizon"}.issubset(set(g_sk.columns)):
                st.dataframe(g_sk.sort_values(["family", "horizon"]), use_container_width=True)
            else:
                st.dataframe(g_sk, use_container_width=True)

    # Scenarios
    st.subheader(tr("🌦️ Escenarios climáticos (bandas simples)", "🌦️ Climate scenarios (simple bands)", LANG))
    fav_pct = st.slider(tr("% favorable", "% favorable", LANG), min_value=0, max_value=30, value=5, step=1) / 100.0
    unf_pct = st.slider(tr("% unfavorable", "% unfavorable", LANG), min_value=0, max_value=30, value=5, step=1) / 100.0
    scen = _paper_scenario_bands(monthly_fc, favorable_pct=fav_pct, unfavorable_pct=unf_pct)

    # ---- Monthly plot (ONLY ONCE) ----
    st.subheader(tr("📉 Serie temporal + predicción (mensual)", "📉 Time series + forecasts (monthly)", LANG))

    def _x_iso(x) -> Optional[str]:
        ts = pd.to_datetime(x, errors="coerce")
        if pd.isna(ts):
            return None
        ts = ts.to_period("M").to_timestamp()
        return ts.strftime("%Y-%m-%d")

    def _add_vline_safe(_fig, _x_iso_str: str, _label: str):
        _fig.add_shape(
            type="line",
            x0=_x_iso_str, x1=_x_iso_str,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(width=2, dash="dash"),
        )
        _fig.add_annotation(
            x=_x_iso_str, y=1.02,
            xref="x", yref="paper",
            text=_label,
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
        )

    def _add_vrect_safe(_fig, _x0_iso: str, _x1_iso: str, _opacity: float, _label: str):
        _fig.add_shape(
            type="rect",
            x0=_x0_iso, x1=_x1_iso,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(width=0),
            fillcolor="rgba(0,0,0,1)",
            opacity=float(_opacity),
            layer="below",
        )
        _fig.add_annotation(
            x=_x1_iso, y=1.02,
            xref="x", yref="paper",
            text=_label,
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
        )

    fig = go.Figure()

    g_obs_plot = g_obs.copy()
    g_obs_plot["date"] = pd.to_datetime(g_obs_plot["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    g_obs_plot = g_obs_plot.dropna(subset=["date"]).sort_values("date")
    fig.add_trace(go.Scatter(x=g_obs_plot["date"], y=g_obs_plot["value"], mode="lines", name="Observed"))

    mf = monthly_fc.copy()
    mf["date"] = pd.to_datetime(mf["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    mf = mf.dropna(subset=["date"]).sort_values("date")

    last_obs_date = pd.to_datetime(g_obs_plot["date"].max(), errors="coerce")
    adv_h = int(advisory_h or 0)
    plan_h = int(planning_h or 0)
    reliable_h = plan_h if plan_h > 0 else adv_h

    cutoff_iso = None
    if pd.notna(last_obs_date) and reliable_h > 0:
        cutoff_iso = _x_iso(last_obs_date + pd.DateOffset(months=reliable_h))

    if cutoff_iso is not None:
        cutoff_ts = pd.to_datetime(cutoff_iso)
        mf_rel = mf[mf["date"] <= cutoff_ts].copy()
        mf_bey = mf[mf["date"] > cutoff_ts].copy()

        if not mf_rel.empty:
            fig.add_trace(go.Scatter(
                x=mf_rel["date"], y=mf_rel["y_forecast"],
                mode="lines", name=f"Forecast (reliable ≤ {reliable_h}m)"
            ))
        if not mf_bey.empty:
            fig.add_trace(go.Scatter(
                x=mf_bey["date"], y=mf_bey["y_forecast"],
                mode="lines", name="Forecast (beyond reliable)",
                line=dict(dash="dot")
            ))

        x0_iso = _x_iso(mf["date"].min())
        x1_iso = _x_iso(mf["date"].max())
        if x0_iso and x1_iso:
            _add_vrect_safe(fig, x0_iso, cutoff_iso, 0.10, "Usable (reliable)")
            _add_vrect_safe(fig, cutoff_iso, x1_iso, 0.05, "Beyond reliable")
    else:
        fig.add_trace(go.Scatter(x=mf["date"], y=mf["y_forecast"], mode="lines", name="Forecast (monthly)"))

    if scen is not None and (not scen.empty):
        scen_plot = scen.copy()
        scen_plot["date"] = pd.to_datetime(scen_plot["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        scen_plot = scen_plot.dropna(subset=["date"]).sort_values("date")
        fig.add_trace(go.Scatter(x=scen_plot["date"], y=scen_plot["scenario_favorable"], mode="lines", name="Scenario favorable"))
        fig.add_trace(go.Scatter(x=scen_plot["date"], y=scen_plot["scenario_unfavorable"], mode="lines", name="Scenario unfavorable"))

    if pd.notna(last_obs_date) and adv_h > 0:
        adv_iso = _x_iso(last_obs_date + pd.DateOffset(months=adv_h))
        if adv_iso:
            _add_vline_safe(fig, adv_iso, f"Advisory horizon ({adv_h}m)")
    if pd.notna(last_obs_date) and plan_h > 0:
        plan_iso = _x_iso(last_obs_date + pd.DateOffset(months=plan_h))
        if plan_iso:
            _add_vline_safe(fig, plan_iso, f"Planning horizon ({plan_h}m)")

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Date",
        yaxis_title="Value",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Skill curve (SECOND FIGURE)
    st.subheader(tr("📊 Skill (KGE) vs horizonte", "📊 Skill (KGE) vs horizon", LANG))
    if (not g_sk.empty) and {"horizon", "kge", "family"}.issubset(set(g_sk.columns)):
        gk = g_sk.copy()
        gk["horizon"] = pd.to_numeric(gk["horizon"], errors="coerce")
        gk["kge"] = pd.to_numeric(gk["kge"], errors="coerce")
        gk = gk.dropna(subset=["horizon", "kge"]).sort_values(["family", "horizon"])

        fig2 = go.Figure()
        for fam in sorted(gk["family"].astype(str).unique()):
            gg = gk[gk["family"].astype(str) == fam]
            fig2.add_trace(go.Scatter(x=gg["horizon"], y=gg["kge"], mode="lines+markers", name=str(fam)))

        fig2.add_shape(type="line", x0=gk["horizon"].min(), x1=gk["horizon"].max(),
                       y0=0.60, y1=0.60, xref="x", yref="y", line=dict(width=1, dash="dash"))
        fig2.add_shape(type="line", x0=gk["horizon"].min(), x1=gk["horizon"].max(),
                       y0=0.30, y1=0.30, xref="x", yref="y", line=dict(width=1, dash="dash"))

        fig2.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Horizon (months)",
            yaxis_title="KGE",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info(tr(
            "Skill no tiene columnas horizon/kge/family en el formato esperado.",
            "Skill table does not contain horizon/kge/family columns in expected format.",
            LANG
        ))

    # Downloads
    st.subheader(tr("📥 Descargas (outputs)", "📥 Downloads (outputs)", LANG))
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button("metrics.csv", data=metrics_df.to_csv(index=False).encode("utf-8"),
                           file_name="metrics.csv", mime="text/csv")
    with d2:
        st.download_button("skill.csv", data=skill_df.to_csv(index=False).encode("utf-8"),
                           file_name="skill.csv", mime="text/csv")
    with d3:
        st.download_button("forecasts_key_horizons.csv", data=g_fc_key.to_csv(index=False).encode("utf-8"),
                           file_name="forecasts_key_horizons.csv", mime="text/csv")
    with d4:
        st.download_button("forecasts_monthly.csv", data=monthly_fc.to_csv(index=False).encode("utf-8"),
                           file_name="forecasts_monthly.csv", mime="text/csv")

    if "core_paths" in st.session_state:
        st.caption(f"Files read from: {st.session_state['core_paths']}")
    if "last_run_outdir" in st.session_state:
        st.caption(f"Last run folder: {st.session_state['last_run_outdir']}")