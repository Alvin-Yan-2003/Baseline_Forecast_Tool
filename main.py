"""
main.py — AFC Baseline Forecast Streamlit App
Entry point: streamlit run main.py

File structure:
    main.py            ← this file (entry point + tab routing)
    config.py          ← theme, CSS, colors, shared data helpers
    charts.py          ← all Plotly chart builders
    sidebar.py         ← file upload, params, run button
    page_analysis.py   ← Tab 1: Data analysis & recommendations
    page_model.py      ← Tab 2-5: Reusable model result page
    page_summary.py    ← Tab 6: Model comparison & best model
    baseline_forecast_tool.py  ← forecast engine (4 HW models)
"""

import sys
import os

# Make sure current directory is on path
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

# ── Page config (must be FIRST st call) ──────────────────────
st.set_page_config(
    page_title="AFC Baseline Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import modules ────────────────────────────────────────────
from sidebar import build_sidebar, execute_forecast
import page_analysis
import page_model
import page_summary
import page_multi


# ══════════════════════════════════════════════════════════════
# SIDEBAR — always rendered
# ══════════════════════════════════════════════════════════════
sidebar_state = build_sidebar()

# Trigger forecast run (single-SKU mode only)
if sidebar_state["run_clicked"] and sidebar_state.get("run_mode") != "Multi-SKU":
    execute_forecast(sidebar_state)

# Pass sidebar multi_selected into session so page_multi can pick it up
if sidebar_state.get("multi_selected"):
    st.session_state["sidebar_multi_selected_names"] = sidebar_state["multi_selected"]

if "show_expost" not in st.session_state:
    st.session_state["show_expost"] = True


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
TAB_LABELS = [
    "🔬 Analysis",
    "📈 Trend+Seasonal",
    "📉 Trend",
    "🌀 Seasonal",
    "➡️ Constant",
    "🏆 Summary",
    "📦 Multi-SKU",
]

tabs = st.tabs(TAB_LABELS)

with tabs[0]:
    page_analysis.render()

with tabs[1]:
    page_model.render("trend_seasonal")

with tabs[2]:
    page_model.render("trend")

with tabs[3]:
    page_model.render("seasonal")

with tabs[4]:
    page_model.render("constant")

with tabs[5]:
    page_summary.render()

with tabs[6]:
    page_multi.render()