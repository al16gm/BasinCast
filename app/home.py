import streamlit as st

st.set_page_config(page_title="BasinCast", page_icon="🌊", layout="wide")

st.title("BasinCast")
st.caption("Climate-informed, data-driven monthly forecasting for regulated basins.")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.subheader("🚀 Run")
    st.write("Upload your data or load a demo basin and run BasinCast.")
    st.page_link("pages/1_🚀_Run_BasinCast.py", label="Open Run", icon="🚀")

with c2:
    st.subheader("📊 Dashboard")
    st.write("Map points, filter results, compare scenarios.")
    st.page_link("pages/2_📊_Dashboard.py", label="Open Dashboard", icon="📊")

with c3:
    st.subheader("📘 Docs")
    st.write("README + data schema inside the app.")
    st.page_link("pages/3_📘_Docs.py", label="Open Docs", icon="📘")

with c4:
    st.subheader("👤 About")
    st.write("Research profile, methodology, citation.")
    st.page_link("pages/4_👤_About.py", label="Open About", icon="👤")

st.divider()
st.subheader("What you will see")
st.markdown(
    "- **Operational forecast (winner)**: best model on backtesting (no CMIP6 deltas)\n"
    "- **CMIP6 SSP scenarios**: scenario-conditioned trajectories using DeltaPack\n"
    "- **P10–P90**: uncertainty band reconstructed from RMSE\n"
)