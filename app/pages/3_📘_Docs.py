import streamlit as st
from pathlib import Path

st.set_page_config(page_title="BasinCast · Docs", page_icon="📘", layout="wide")
st.title("Documentation")

readme = Path(__file__).resolve().parents[2] / "README.md"
if readme.exists():
    st.markdown(readme.read_text(encoding="utf-8"))
else:
    st.warning("README.md not found in repository root.")