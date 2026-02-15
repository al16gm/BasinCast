from __future__ import annotations

import streamlit as st

_LANG_OPTIONS = {"English": "en", "Español": "es"}

# Key-based strings (new style). Backward compatible with tr("ES", "EN").
_STRINGS: dict[str, dict[str, str]] = {
    # --- Meteorology block (v0.10) ---
    "meteo_header": {"en": "🌦️ Meteorology (optional)", "es": "🌦️ Meteorología (opcional)"},
    "meteo_already_present": {"en": "✅ Your file already contains meteorology. Nothing to do.", "es": "✅ Tu fichero ya contiene meteorología. No hay que hacer nada."},
    "meteo_missing": {"en": "Your file does NOT include meteorology. Choose an option:", "es": "Tu fichero NO trae meteorología. Elige una opción:"},
    "meteo_choice_prompt": {"en": "How do you want to provide meteorology?", "es": "¿Cómo quieres aportar meteorología?"},
    "meteo_choice_a": {"en": "A) I can upload a historical meteorology file", "es": "A) Puedo subir otro fichero con meteorología histórica"},
    "meteo_choice_b": {"en": "B) I don't have meteorology → download from NASA POWER (no API key)", "es": "B) No tengo meteorología → descargar de NASA POWER (sin API key)"},
    "meteo_choice_c": {"en": "C) I don't have meteorology → continue ENDO-only (less reliable)", "es": "C) No tengo meteorología → continuar SOLO con ENDO (menos fiable)"},

    "meteo_a_title": {"en": "A) Upload historical meteorology", "es": "A) Subir meteorología histórica"},
    "meteo_a_uploader": {"en": "Upload CSV/XLSX with meteorology", "es": "Sube CSV/XLSX con meteo"},
    "meteo_a_not_enough_cols": {"en": "The meteorology file does not have enough columns.", "es": "El fichero meteo no tiene columnas suficientes."},
    "preview": {"en": "Preview", "es": "Vista previa"},
    "meteo_mapping_title": {"en": "### 🔧 Column mapping (confirm / correct)", "es": "### 🔧 Mapeo de columnas (confirma/corrige)"},
    "meteo_col_date": {"en": "DATE column", "es": "Columna FECHA"},
    "meteo_col_precip": {"en": "PRECIP column", "es": "Columna PRECIP"},
    "meteo_col_t2m": {"en": "T2M column", "es": "Columna T2M"},
    "meteo_col_tmax": {"en": "TMAX column", "es": "Columna TMAX"},
    "meteo_col_tmin": {"en": "TMIN column", "es": "Columna TMIN"},
    "meteo_has_point_id_q": {"en": "Does your meteorology file identify point_id?", "es": "¿Tu fichero meteo identifica el point_id?"},
    "yes_has_point_id": {"en": "Yes, it has point_id", "es": "Sí, tiene point_id"},
    "no_single_point": {"en": "No: it is for one point", "es": "No: es de un solo punto"},
    "no_common_all": {"en": "No: it is common to all points", "es": "No: es común para todos los puntos"},
    "meteo_col_point_id": {"en": "POINT_ID column", "es": "Columna POINT_ID"},
    "meteo_need_point_id": {"en": "Select the point_id column.", "es": "Selecciona la columna point_id."},
    "meteo_for_which_point": {"en": "Which point_id does it apply to?", "es": "¿Para qué point_id aplica?"},
    "apply_uploaded_meteo_btn": {"en": "✅ Apply uploaded meteorology", "es": "✅ Aplicar meteorología subida"},
    "meteo_integrated_ok": {"en": "✅ Meteorology integrated into canonical_with_meteo", "es": "✅ Meteorología integrada en canonical_with_meteo"},
    "meteo_integrated_err": {"en": "Error integrating meteorology", "es": "Error integrando meteorología"},

    "meteo_b_title": {"en": "B) Download from NASA POWER (no API key)", "es": "B) Descargar de NASA POWER (sin API key)"},
    "meteo_coords_status": {"en": "Coordinates OK for {n_ok}/{n_total} points.", "es": "Coordenadas OK en {n_ok}/{n_total} puntos."},
    "meteo_coords_missing": {"en": "Missing coordinates for {n_missing} point(s). Fix before NASA POWER.", "es": "Faltan coordenadas en {n_missing} punto(s). Hay que resolverlo antes de NASA POWER."},
    "meteo_fix_coords_title": {"en": "### Fix coordinates (no API key)", "es": "### Arreglar coordenadas (sin API key)"},
    "meteo_global_location": {"en": "Global location text (applies to missing points), e.g. 'Madrid, Spain'", "es": "Ubicación global (aplica a puntos sin coords), ej: 'Madrid, España'"},
    "meteo_locations_upload": {"en": "Upload locations file (point_id + location OR lat/lon)", "es": "Sube fichero de ubicaciones (point_id + location O lat/lon)"},
    "loc_col_point_id": {"en": "point_id column", "es": "Columna point_id"},
    "loc_col_location": {"en": "location text column (optional)", "es": "Columna location (opcional)"},
    "loc_col_lat": {"en": "lat column (optional)", "es": "Columna lat (opcional)"},
    "loc_col_lon": {"en": "lon column (optional)", "es": "Columna lon (opcional)"},
    "meteo_locations_rule": {"en": "Rule: provide either (lat+lon) OR (location text).", "es": "Regla: aporta (lat+lon) O (texto de ubicación)."},
    "meteo_show_reverse": {"en": "Show approximate location (reverse geocode) for first points", "es": "Mostrar ubicación aproximada (reverse geocode) para los primeros puntos"},
    "meteo_resolve_coords_btn": {"en": "🧭 Resolve missing coordinates now", "es": "🧭 Resolver coordenadas faltantes ahora"},
    "meteo_coords_resolved_ok": {"en": "✅ Coordinates updated. You can now fetch NASA POWER.", "es": "✅ Coordenadas actualizadas. Ya puedes descargar NASA POWER."},
    "meteo_coords_resolved_err": {"en": "Error resolving coordinates", "es": "Error resolviendo coordenadas"},
    "meteo_fetch_nasa_btn": {"en": "🌍 Fetch NASA POWER and build canonical_with_meteo", "es": "🌍 Descargar NASA POWER y crear canonical_with_meteo"},
    "meteo_fetching_spinner": {"en": "Fetching NASA POWER... (cached)", "es": "Descargando NASA POWER... (queda cacheado)"},
    "meteo_nasa_ok": {"en": "✅ NASA POWER integrated into canonical_with_meteo", "es": "✅ NASA POWER integrado en canonical_with_meteo"},
    "meteo_nasa_err": {"en": "NASA POWER error", "es": "Error NASA POWER"},
    "meteo_need_coords_or_other_option": {"en": "NASA POWER needs coordinates. Either resolve coords, upload meteorology, or choose ENDO-only.", "es": "NASA POWER requiere coordenadas. O resuelves coords, o subes meteo, o eliges SOLO ENDO."},

    "meteo_endo_only_info": {"en": "Continuing ENDO-only. BasinCast works, but EXOG is unavailable.", "es": "Continuando SOLO con ENDO. BasinCast funciona, pero EXOG no estará disponible."},
    "download_canonical_with_meteo": {"en": "Download canonical_with_meteo.csv", "es": "Descargar canonical_with_meteo.csv"},
    "download_meteo_df": {"en": "Download meteo_power_monthly.csv", "es": "Descargar meteo_power_monthly.csv"},
}


def get_lang(default: str = "en") -> str:
    if "lang" not in st.session_state:
        st.session_state["lang"] = default
    return str(st.session_state["lang"])


def language_selector(default: str = "en") -> str:
    """
    Sidebar language selector. Stores the selected language in st.session_state['lang'].
    """
    if "lang" not in st.session_state:
        st.session_state["lang"] = default

    current = "English" if st.session_state["lang"] == "en" else "Español"
    idx = 0 if current == "English" else 1

    st.sidebar.subheader("Settings / Ajustes")
    choice = st.sidebar.selectbox("Language / Idioma", ["English", "Español"], index=idx)
    st.session_state["lang"] = _LANG_OPTIONS[choice]
    return str(st.session_state["lang"])


def tr(es_or_key: str, en: str | None = None, lang: str | None = None) -> str:
    """
    Backward-compatible translator.

    Old usage:
      tr("texto ES", "text EN")

    New key-based usage:
      tr("meteo_header")
    """
    if lang is None:
        lang = get_lang()

    # Old style
    if en is not None:
        return en if lang == "en" else es_or_key

    # New key style
    key = es_or_key
    rec = _STRINGS.get(key)
    if not rec:
        # If missing key, return the key itself so UI doesn't crash
        return key
    return rec["en"] if lang == "en" else rec["es"]