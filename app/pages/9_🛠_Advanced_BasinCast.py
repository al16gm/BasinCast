import streamlit as st
import sys
from pathlib import Path
import importlib

# Page config (safe)
st.set_page_config(page_title="Advanced BasinCast · Run", page_icon="🛠", layout="wide")

st.title("🛠 Advanced BasinCast")
st.caption("Upload your dataset, complete missing meteorology/demand if needed, then run the model.")

# Make sure repo paths work in multipage + Streamlit Cloud
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../app/pages -> repo root
SRC_DIR = PROJECT_ROOT / "src"
APP_DIR = PROJECT_ROOT / "app"

for p in [SRC_DIR, APP_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Quick start (nice UX without duplicating logic)
with st.expander("Quick start", expanded=True):
    st.markdown(
        """
1) Upload your **canonical** file (CSV/XLSX)  
2) If meteorology is missing, select **NASA POWER** (or upload meteo)  
3) Optionally upload **demand**  
4) Click **RUN** and open the **Dashboard**
"""
    )

# IMPORTANT:
# In multipage Streamlit, imported modules are cached.
# When the user interacts (NASA POWER choice), Streamlit reruns this page,
# but the imported translator_app would not rerun unless we reload it.
import translator_app  # noqa: E402
importlib.reload(translator_app)