import streamlit as st
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.graph_objects as go

st.set_page_config(page_title="BasinCast · Dashboard", page_icon="📊", layout="wide")
st.title("📊 BasinCast Dashboard")
st.caption("Map points, filter results, and inspect the latest BasinCast run.")

# -----------------------------
# Language (ES/EN)
# -----------------------------
LANG = st.sidebar.selectbox("Language / Idioma", ["English", "Español"], index=0)

def tr(es: str, en: str) -> str:
    return es if LANG == "Español" else en

# Ensure src/ is importable (Cloud-safe)
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../app/pages -> repo root
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"


def _safe_read_csv(p: Path) -> pd.DataFrame:
    if (not p.exists()) or p.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(p)


def _find_latest_run_dir(runs_dir: Path) -> Path | None:
    if not runs_dir.exists():
        return None
    cands = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _to_month_ts(x):
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_period("M").to_timestamp()


def _to_iso_date(x) -> str | None:
    ts = _to_month_ts(x)
    if ts is None:
        return None
    return ts.strftime("%Y-%m-%d")


def load_run(outdir: Path) -> dict:
    outdir = Path(outdir)
    manifest_path = outdir / "manifest.json"

    paths = {
        "manifest": manifest_path,
        "canonical": outdir / "_tmp_canonical_for_core.csv",
        "metrics": outdir / "metrics_v0_6.csv",
        "skill": outdir / "skill_v0_6.csv",
        "forecasts": outdir / "forecasts_v0_6.csv",
        "scenarios": outdir / "forecasts_scenarios_v0_6.csv",
    }

    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for k in ["metrics", "skill", "forecasts", "manifest", "canonical", "scenarios"]:
                if k in manifest and isinstance(manifest[k], str):
                    paths[k] = Path(manifest[k])
        except Exception:
            manifest = {}

    canonical_df = _safe_read_csv(paths["canonical"])
    metrics_df = _safe_read_csv(paths["metrics"])
    skill_df = _safe_read_csv(paths["skill"])
    forecasts_df = _safe_read_csv(paths["forecasts"])
    scenarios_df = _safe_read_csv(paths["scenarios"])

    for df in [canonical_df, forecasts_df, scenarios_df]:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()

    return {
        "outdir": outdir,
        "paths": paths,
        "manifest": manifest,
        "canonical": canonical_df,
        "metrics": metrics_df,
        "skill": skill_df,
        "forecasts": forecasts_df,
        "scenarios": scenarios_df,
    }


# -----------------------------
# Sidebar: interpretation + run selection + filters
# -----------------------------
st.sidebar.subheader(tr("Interpretación", "Interpretation"))
st.sidebar.write(
    tr(
        "- **Previsión operativa (ganadora)**: mejor modelo en backtesting (sin deltas CMIP6)\n"
        "- **Curvas CMIP6 SSP**: trayectorias condicionadas por escenario (DeltaPack)\n"
        "- **KGE por horizonte**: confianza por meses en el futuro",
        "- **Operational forecast (winner)**: best model on backtesting (no CMIP6 deltas)\n"
        "- **CMIP6 SSP curves**: scenario-conditioned trajectories (DeltaPack)\n"
        "- **KGE vs horizon**: confidence by months ahead",
    )
)

session_outdir = st.session_state.get("core_last_outdir", None)
default_run_dir = Path(session_outdir) if session_outdir else _find_latest_run_dir(RUNS_DIR)

if default_run_dir is None:
    st.sidebar.warning(tr("No hay runs. Ejecuta primero BasinCast en 'Run'.", "No runs found. Run BasinCast first in 'Run'."))
    st.stop()

st.sidebar.caption(tr("Carpeta del run", "Run folder"))
st.sidebar.code(str(default_run_dir), language=None)

if st.sidebar.button(tr("🔄 Refrescar", "🔄 Refresh")):
    st.rerun()

data = load_run(default_run_dir)
canonical = data["canonical"]
metrics = data["metrics"]
skill = data["skill"]
forecasts = data["forecasts"]
scenarios = data["scenarios"]

if canonical.empty or ("lat" not in canonical.columns) or ("lon" not in canonical.columns) or ("point_id" not in canonical.columns):
    st.error(tr("No se ha encontrado el canónico o faltan lat/lon/point_id.", "Canonical not found or missing lat/lon/point_id."))
    st.stop()

point_ids = sorted(canonical["point_id"].astype(str).unique().tolist())
sel_points = st.sidebar.multiselect(tr("Selecciona point_id(s)", "Select point_id(s)"), point_ids, default=point_ids[:1])
if not sel_points:
    sel_points = point_ids[:1]
focus_pid = sel_points[0]

# -----------------------------
# KPI row
# -----------------------------
mrow = metrics[metrics["point_id"].astype(str) == str(focus_pid)].copy() if (not metrics.empty and "point_id" in metrics.columns) else pd.DataFrame()

def _get_val(df: pd.DataFrame, col: str, default="—"):
    if df is None or df.empty or col not in df.columns:
        return default
    return str(df[col].iloc[0])

def _get_int(df: pd.DataFrame, col: str, default=0) -> int:
    if df is None or df.empty or col not in df.columns:
        return int(default)
    try:
        return int(float(df[col].iloc[0]))
    except Exception:
        return int(default)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric(tr("Familia ganadora", "Winner family"), _get_val(mrow, "selected_family"))
with k2:
    st.metric(tr("Modelo ganador", "Winner model"), _get_val(mrow, "selected_model_type"))
with k3:
    st.metric(tr("Horizonte planning (m)", "Planning horizon (m)"), _get_val(mrow, "planning_horizon"))
with k4:
    st.metric(tr("Horizonte advisory (m)", "Advisory horizon (m)"), _get_val(mrow, "advisory_horizon"))

st.divider()

# -----------------------------
# Layout: Map + Tabs (Time series / KGE)
# -----------------------------
col_map, col_chart = st.columns([1.25, 1.0], gap="large")

# Map
with col_map:
    st.subheader(tr("Mapa de puntos", "Map of points"))
    map_df = canonical[canonical["point_id"].astype(str).isin([str(x) for x in sel_points])][["point_id", "lat", "lon"]].dropna().drop_duplicates().copy()
    map_df["point_id"] = map_df["point_id"].astype(str)

    view = pdk.ViewState(
        latitude=float(map_df["lat"].mean()),
        longitude=float(map_df["lon"].mean()),
        zoom=6,
        pitch=0,
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_radius=5500,
        get_fill_color=[0, 255, 200, 190],    # cyan visible
        get_line_color=[255, 255, 255, 220],  # white halo
        line_width_min_pixels=2,
        pickable=True,
        auto_highlight=True,
    )

    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        initial_view_state=view,
        layers=[layer],
        tooltip={"text": "point_id: {point_id}\nlat: {lat}\nlon: {lon}"},
    )
    st.pydeck_chart(deck, use_container_width=True)

# Charts
with col_chart:
    tabs = st.tabs([tr("Serie temporal", "Time series"), tr("Skill (KGE)", "Skill (KGE)")])

    # ---------------------
    # Tab 1: Time series
    # ---------------------
    with tabs[0]:
        st.subheader(tr(f"Serie temporal — point_id={focus_pid}", f"Time series — point_id={focus_pid}"))
        fig = go.Figure()

        obs = canonical[canonical["point_id"].astype(str) == str(focus_pid)][["date", "value"]].dropna().sort_values("date").copy()
        if not obs.empty:
            fig.add_trace(go.Scatter(x=obs["date"], y=obs["value"], mode="lines", name=tr("Observado", "Observed")))

        fc = forecasts[forecasts["point_id"].astype(str) == str(focus_pid)].copy() if ("point_id" in forecasts.columns) else pd.DataFrame()
        if not fc.empty and "y_forecast" in fc.columns:
            fc = fc.sort_values("date")
            fig.add_trace(go.Scatter(x=fc["date"], y=fc["y_forecast"], mode="lines", name=tr("Previsión operativa (ganadora)", "Operational forecast (winner)")))

        # ---------------------
        # P10–P90 band (from RMSE) — only for operational forecast
        # ---------------------
        show_p1090 = st.checkbox(
            tr("Mostrar P10–P90 (RMSE)", "Show P10–P90 (RMSE)"),
            value=True,
            key=f"p1090_{focus_pid}",
        )

        if show_p1090 and (not fc.empty) and ("y_forecast" in fc.columns):
            # 1) Build a horizon column if missing
            fc_band = fc.sort_values("date").copy()
            if "horizon" in fc_band.columns:
                fc_band["horizon"] = pd.to_numeric(fc_band["horizon"], errors="coerce")
            else:
                fc_band["horizon"] = np.arange(1, len(fc_band) + 1, dtype=int)

            # 2) Try to get RMSE by horizon from skill table
            rmse_default = None
            rmse_by_h = {}

            if (not skill.empty) and ("rmse" in skill.columns):
                sk0 = skill.copy()
                if "point_id" in sk0.columns:
                    sk0 = sk0[sk0["point_id"].astype(str) == str(focus_pid)].copy()

                # Prefer winner family/model if present
                win_family = _get_val(mrow, "selected_family", None)
                win_model = _get_val(mrow, "selected_model_type", None)
                if win_family and ("family" in sk0.columns):
                    sk0 = sk0[sk0["family"].astype(str) == str(win_family)].copy()
                if win_model and ("model_type" in sk0.columns):
                    sk0 = sk0[sk0["model_type"].astype(str) == str(win_model)].copy()

                sk0["horizon"] = pd.to_numeric(sk0["horizon"], errors="coerce")
                sk0["rmse"] = pd.to_numeric(sk0["rmse"], errors="coerce")
                sk0 = sk0.dropna(subset=["horizon", "rmse"]).copy()

                if not sk0.empty:
                    rmse_by_h = dict(zip(sk0["horizon"].astype(int), sk0["rmse"].astype(float)))
                    rmse_default = float(np.nanmedian(sk0["rmse"].to_numpy()))

            if rmse_default is None or (not np.isfinite(rmse_default)) or rmse_default <= 0:
                st.warning(tr(
                    "No se puede dibujar P10–P90: falta RMSE en skill_v0_6.csv.",
                    "Cannot draw P10–P90: RMSE is missing in skill_v0_6.csv."
                ))
            else:
                z = 1.2815515655446004  # Normal quantile for 10/90

                band_df = fc_band[["date", "y_forecast", "horizon"]].copy()
                band_df["sigma"] = band_df["horizon"].map(lambda h: rmse_by_h.get(int(h), rmse_default))
                band_df["p10"] = (band_df["y_forecast"] - z * band_df["sigma"]).clip(lower=0.0)
                band_df["p90"] = (band_df["y_forecast"] + z * band_df["sigma"])

                # Draw band behind lines (higher opacity so it is visible)
                fig.add_trace(go.Scatter(
                    x=band_df["date"],
                    y=band_df["p90"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    name="P90",
                ))
                fig.add_trace(go.Scatter(
                    x=band_df["date"],
                    y=band_df["p10"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(120, 80, 200, 0.25)",
                    name="P10–P90 (RMSE)",
                    hoverinfo="skip",
                ))

        sc = scenarios[scenarios["point_id"].astype(str) == str(focus_pid)].copy() if ("point_id" in scenarios.columns) else pd.DataFrame()
        if not sc.empty and {"scenario", "y_forecast", "date"}.issubset(set(sc.columns)):
            order = [
                ("Favorable", tr("CMIP6 SSP126", "CMIP6 SSP126")),
                ("Base", tr("CMIP6 SSP245", "CMIP6 SSP245")),
                ("Unfavorable", tr("CMIP6 SSP585", "CMIP6 SSP585")),
            ]
            for scen_key, label in order:
                gg = sc[sc["scenario"].astype(str) == scen_key].sort_values("date")
                if not gg.empty:
                    fig.add_trace(go.Scatter(x=gg["date"], y=gg["y_forecast"], mode="lines", name=label))

        # Vertical lines (NO annotation_text in add_vline to avoid Plotly Timestamp bug)
        cutoff_iso = _to_iso_date(_get_val(mrow, "cutoff_date", None))
        last_obs_iso = _to_iso_date(_get_val(mrow, "last_observed_date", None))

        if cutoff_iso:
            fig.add_vline(x=cutoff_iso, line_dash="dot", line_width=2)
            fig.add_annotation(x=cutoff_iso, y=1.03, xref="x", yref="paper",
                               text=tr("Cutoff", "Cutoff"), showarrow=False)
        if last_obs_iso:
            fig.add_vline(x=last_obs_iso, line_dash="dash", line_width=2)
            fig.add_annotation(x=last_obs_iso, y=1.08, xref="x", yref="paper",
                               text=tr("Último observado", "Last observed"), showarrow=False)

        # Planning/advisory horizon boundaries (from last observed)
        adv_h = _get_int(mrow, "advisory_horizon", 0)
        plan_h = _get_int(mrow, "planning_horizon", 0)

        last_obs_ts = _to_month_ts(_get_val(mrow, "last_observed_date", None))
        if last_obs_ts is not None and pd.notna(last_obs_ts):
            if plan_h > 0:
                plan_date = (last_obs_ts + pd.DateOffset(months=int(plan_h))).to_period("M").to_timestamp()
                plan_iso = plan_date.strftime("%Y-%m-%d")
                fig.add_vline(x=plan_iso, line_dash="dot", line_width=1)
                fig.add_annotation(x=plan_iso, y=0.98, xref="x", yref="paper",
                                   text=tr(f"Planning ({plan_h}m)", f"Planning ({plan_h}m)"),
                                   showarrow=False)
            if adv_h > 0:
                adv_date = (last_obs_ts + pd.DateOffset(months=int(adv_h))).to_period("M").to_timestamp()
                adv_iso = adv_date.strftime("%Y-%m-%d")
                fig.add_vline(x=adv_iso, line_dash="dash", line_width=1)
                fig.add_annotation(x=adv_iso, y=0.93, xref="x", yref="paper",
                                   text=tr(f"Advisory ({adv_h}m)", f"Advisory ({adv_h}m)"),
                                   showarrow=False)

        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------
    # Tab 2: KGE vs horizon
    # ---------------------
    with tabs[1]:
        st.subheader(tr("Skill (KGE) por horizonte", "Skill (KGE) vs horizon"))

        if skill.empty or ("point_id" not in skill.columns):
            st.info(tr("No hay tabla skill disponible en este run.", "No skill table available for this run."))
        else:
            sk = skill[skill["point_id"].astype(str) == str(focus_pid)].copy()
            if sk.empty or ("kge" not in sk.columns) or ("horizon" not in sk.columns):
                st.info(tr("No hay KGE/horizon para este punto.", "No KGE/horizon for this point."))
            else:
                sk["horizon"] = pd.to_numeric(sk["horizon"], errors="coerce")
                sk["kge"] = pd.to_numeric(sk["kge"], errors="coerce")
                sk = sk.dropna(subset=["horizon", "kge"]).sort_values("horizon").copy()

                fig2 = go.Figure()
                if "family" in sk.columns:
                    for fam in sorted(sk["family"].astype(str).unique().tolist()):
                        gg = sk[sk["family"].astype(str) == fam]
                        if not gg.empty:
                            fig2.add_trace(go.Scatter(x=gg["horizon"], y=gg["kge"], mode="lines+markers", name=fam))
                else:
                    fig2.add_trace(go.Scatter(x=sk["horizon"], y=sk["kge"], mode="lines+markers", name="KGE"))

                plan_th = float(mrow["planning_kge_threshold"].iloc[0]) if (not mrow.empty and "planning_kge_threshold" in mrow.columns) else 0.6
                adv_th = float(mrow["advisory_kge_threshold"].iloc[0]) if (not mrow.empty and "advisory_kge_threshold" in mrow.columns) else 0.3

                fig2.add_hline(y=plan_th, line_dash="dash", annotation_text=tr("Umbral planning", "Planning threshold"))
                fig2.add_hline(y=adv_th, line_dash="dot", annotation_text=tr("Umbral advisory", "Advisory threshold"))

                fig2.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
                fig2.update_xaxes(title_text=tr("Horizonte (meses)", "Horizon (months)"))
                fig2.update_yaxes(title_text="KGE", range=[-1, 1])

                st.plotly_chart(fig2, use_container_width=True)

                plan_h = _get_int(mrow, "planning_horizon", 0)
                adv_h = _get_int(mrow, "advisory_horizon", 0)
                st.success(tr(
                    f"Horizonte fiable (planning): {plan_h} meses · Horizonte advisory: {adv_h} meses",
                    f"Reliable horizon (planning): {plan_h} months · Advisory horizon: {adv_h} months",
                ))

st.divider()

# Downloads
st.subheader(tr("Descargas", "Downloads"))
d1, d2, d3, d4 = st.columns(4)
with d1:
    p = data["paths"]["metrics"]
    if p.exists():
        st.download_button("metrics_v0_6.csv", p.read_bytes(), file_name="metrics_v0_6.csv")
with d2:
    p = data["paths"]["forecasts"]
    if p.exists():
        st.download_button("forecasts_v0_6.csv", p.read_bytes(), file_name="forecasts_v0_6.csv")
with d3:
    p = data["paths"]["scenarios"]
    if p.exists():
        st.download_button("forecasts_scenarios_v0_6.csv", p.read_bytes(), file_name="forecasts_scenarios_v0_6.csv")
with d4:
    p = data["paths"]["manifest"]
    if p.exists():
        st.download_button("manifest.json", p.read_bytes(), file_name="manifest.json")