from __future__ import annotations

import streamlit as st

_LANG_OPTIONS = {"English": "en", "Español": "es"}


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


def tr(es: str, en: str, lang: str | None = None) -> str:
    """
    Tiny translator helper.
    Usage: tr("texto ES", "text EN")
    """
    if lang is None:
        lang = get_lang()
    return en if lang == "en" else es