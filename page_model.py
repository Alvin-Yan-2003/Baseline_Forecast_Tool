"""
page_model.py — Reusable model result page
Used by: Trend+Seasonal, Trend, Seasonal, Constant tabs
"""

import numpy as np
import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta

from baseline_forecast_tool import select_best
from config import (
    MODEL_COLORS, MODEL_LABELS,
    section, metric_card_html,
    build_yoy_df, build_qoq_df, build_mom_df,
    render_yoy_table, render_qoq_table, render_mom_table,
    C_HIST, C_TEXT, C_MUTED,
)
from charts import forecast_chart, yoy_combo_chart, qoq_area_chart, score_scatter


# ══════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════

def _fc_dates(hist_dates, h_steps):
    return [hist_dates[-1] + relativedelta(months=i + 1) for i in range(h_steps)]


def _mape_str(v):
    return f"{v*100:.1f}%" if not (isinstance(v, float) and np.isnan(v)) else "N/A"


def _score_str(v):
    return f"{v:.4f}"


# ══════════════════════════════════════════════════════════════
# MAIN RENDER FUNCTION
# ══════════════════════════════════════════════════════════════

def render(model_key: str):
    """
    Render a full model result page.

    model_key: one of "trend_seasonal", "seasonal", "trend", "constant"
    """
    model_label = MODEL_LABELS.get(model_key, model_key)
    model_color = MODEL_COLORS.get(model_key, "#888")
    model_icons = {
        "trend_seasonal": "📈",
        "seasonal":       "🌀",
        "trend":          "📉",
        "constant":       "➡️",
    }
    icon = model_icons.get(model_key, "📊")

    st.markdown(
        f'<div class="page-header" style="border-left-color:{model_color};">'
        f'<h1>{icon} {model_label} Model</h1>'
        f'<p>Grid search results, forecast 24 tháng, và growth analysis</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Guard ──────────────────────────────────────────────────
    if "results" not in st.session_state:
        st.markdown(
            '<div class="info-box">ℹ️ Nhấn <strong>▶ RUN FORECAST</strong> '
            "ở sidebar để chạy grid search.</div>",
            unsafe_allow_html=True,
        )
        return

    all_results = st.session_state["results"]
    history     = st.session_state["history"]
    hist_dates  = st.session_state["hist_dates"]
    h_steps     = st.session_state["h_steps"]
    top_n       = st.session_state.get("top_n", 10)
    show_expost = st.session_state.get("show_expost", True)
    sku_name    = st.session_state.get("sku_name", "SKU")

    # Filter to this model only
    model_results = [r for r in all_results if r["model"] == model_key]
    if not model_results:
        st.markdown(
            f'<div class="warn-box">⚠️ Không có results cho {model_label}. '
            f"Kiểm tra param range.</div>",
            unsafe_allow_html=True,
        )
        return

    top_model = select_best(model_results, history, top_n=top_n)
    best      = top_model[0]
    fc_dates  = _fc_dates(hist_dates, h_steps)

    # ── KPI row ───────────────────────────────────────────────
    st.markdown(section("Best Parameter Combination"), unsafe_allow_html=True)
    cols = st.columns(6)
    kpis = [
        ("Score",   _score_str(best["score"]),  "lower = better",     model_color),
        ("MAPE",    _mape_str(best["mape"]),     "ex-post accuracy",   model_color),
        ("Alpha α", f"{best['alpha']:.2f}",      "level smoothing",    C_TEXT),
        ("Beta β",  f"{best['beta']:.2f}",       "trend smoothing",    C_TEXT),
        ("Gamma γ", f"{best['gamma']:.2f}",      "seasonal smoothing", C_TEXT),
        ("Combos",  f"{len(model_results):,}",   "evaluated",          C_MUTED),
    ]
    for col, (lbl, val, sub, color) in zip(cols, kpis):
        col.markdown(metric_card_html(lbl, val, sub, color), unsafe_allow_html=True)

    # ── Forecast chart ────────────────────────────────────────
    st.markdown(section("Forecast vs History"), unsafe_allow_html=True)
    st.plotly_chart(
        forecast_chart(
            history, hist_dates,
            best["forecast"], fc_dates,
            model_label, model_color,
            best.get("expost"), show_expost,
            height=400,
        ),
        use_container_width=True,
        key=f"{model_key}_mdl_1",
    )

    # ── Growth tables ─────────────────────────────────────────
    st.markdown(section("Growth Analysis"), unsafe_allow_html=True)

    yoy_df = build_yoy_df(history, hist_dates, best["forecast"], fc_dates)
    qoq_df = build_qoq_df(history, hist_dates, best["forecast"], fc_dates)
    mom_df = build_mom_df(history, hist_dates, best["forecast"], fc_dates)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**📅 Annual YoY Growth**")
        st.markdown(render_yoy_table(yoy_df), unsafe_allow_html=True)

    with col_b:
        st.markdown("**📆 Quarterly QoQ Growth**")
        st.markdown(render_qoq_table(qoq_df, max_rows=16), unsafe_allow_html=True)

    with col_c:
        st.markdown("**🗓️ Monthly MoM (recent 24)**")
        st.markdown(render_mom_table(mom_df, max_rows=24), unsafe_allow_html=True)

    # ── YoY bar + QoQ area ────────────────────────────────────
    st.markdown(section("Volume Charts"), unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(yoy_combo_chart(yoy_df, height=340), use_container_width=True,
        key=f"{model_key}_mdl_2")
    with col_r:
        st.plotly_chart(qoq_area_chart(qoq_df, height=340), use_container_width=True,
        key=f"{model_key}_mdl_3")

    # ── Forecast data table ───────────────────────────────────
    st.markdown(section("Monthly Forecast Detail"), unsafe_allow_html=True)

    import pandas as pd
    fc_rows = []
    prev = history[-1]
    for dt, v in zip(fc_dates, best["forecast"]):
        mom = v / prev - 1 if prev else None
        fc_rows.append({
            "Period":  dt.strftime("%b %Y"),
            "Year":    dt.year,
            "Month":   dt.strftime("%b"),
            "Quarter": f"Q{(dt.month-1)//3+1}",
            "Forecast": round(v, 1),
            "MoM %":   f"{mom*100:+.1f}%" if mom is not None else "—",
        })
        prev = v
    df_fc = pd.DataFrame(fc_rows)
    # YoY column
    from config import yr_sums as _yr_sums
    all_v_yr = _yr_sums(list(history) + list(best["forecast"]),
                         list(hist_dates) + list(fc_dates))
    df_fc["YoY vs prior"] = df_fc.apply(
        lambda row: _get_yoy_for_month(row["Period"], row["Year"],
                                        all_v_yr, hist_dates[-1].year), axis=1
    )

    st.dataframe(
        df_fc.set_index("Period"),
        use_container_width=True,
        height=min(400, len(fc_rows) * 36 + 38),
    )

    # ── Top N param combinations ──────────────────────────────
    st.markdown(section(f"Top {min(top_n, 15)} Parameter Combinations"), unsafe_allow_html=True)

    param_rows = []
    for i, r in enumerate(top_model[:15], 1):
        param_rows.append({
            "Rank":  i,
            "α":     r["alpha"],
            "β":     r["beta"],
            "γ":     r["gamma"],
            "MAPE":  _mape_str(r["mape"]),
            "Score": round(r["score"], 5),
        })
    df_params = pd.DataFrame(param_rows).set_index("Rank")
    st.dataframe(
        df_params.style.background_gradient(subset=["Score"], cmap="RdYlGn_r"),
        use_container_width=True,
        height=min(450, len(param_rows) * 36 + 38),
    )

    # ── Score scatter (MAPE vs Score) ─────────────────────────
    with st.expander("🔍 Score vs MAPE scatter (all combinations)", expanded=False):
        st.plotly_chart(
            score_scatter(top_model, height=360),
            use_container_width=True,
        key=f"{model_key}_mdl_4",
        )

    # ── Model notes ───────────────────────────────────────────
    st.markdown(section("Model Notes"), unsafe_allow_html=True)
    _render_model_notes(model_key, best, history)


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _get_yoy_for_month(period_str, year, yr_dict, last_hist_yr):
    """Return YoY % string for a given period."""
    prev_yr = year - 1
    curr    = yr_dict.get(year)
    prev    = yr_dict.get(prev_yr)
    if curr is None or prev is None or prev == 0:
        return "—"
    v = curr / prev - 1
    sign = "+" if v > 0 else ""
    return f"{sign}{v*100:.1f}%"


def _render_model_notes(model_key, best, history):
    notes = {
        "trend_seasonal": (
            "**Trend + Seasonal (α, β, γ)** — Winters multiplicative model.<br>"
            "Phù hợp khi data có cả trend dài hạn và seasonal pattern rõ (ví dụ: AFC peak tháng 5, tháng 4).<br>"
            "• **α** controls tốc độ cập nhật level — α cao → phản ứng nhanh với biến động<br>"
            "• **β** controls tốc độ học trend — β thấp → trend mượt hơn<br>"
            "• **γ** controls seasonal index update — γ cao → seasonal factor cập nhật nhanh<br>"
            "⚠️ Model phức tạp nhất, cần đủ data (≥2 năm) để init tốt."
        ),
        "seasonal": (
            "**Seasonal (α, γ)** — Winters seasonal-only model.<br>"
            "Phù hợp khi seasonal pattern rõ nhưng không có trend dài hạn đáng kể.<br>"
            "• **α** controls level smoothing<br>"
            "• **γ** controls seasonal index — thường nên ở range 0.70–0.99 với AFC<br>"
            "✓ Thường là best model cho AFC do seasonality mạnh (peak Mar-May, low Sep-Oct)."
        ),
        "trend": (
            "**Trend (α, β)** — Holt's double exponential smoothing.<br>"
            "Phù hợp khi data có trend nhưng không có seasonal pattern rõ.<br>"
            "• Forecast sẽ là đường thẳng extrapolate từ recent trend<br>"
            "⚠️ Không capture seasonal → không recommend cho AFC nếu có monthly variation rõ."
        ),
        "constant": (
            "**Constant (α)** — Simple exponential smoothing.<br>"
            "Forecast phẳng (flat) dựa trên weighted average của history.<br>"
            "• α thấp → forecast gần với long-run mean<br>"
            "• α cao → forecast gần với recent observations<br>"
            "⚠️ Không capture trend hoặc seasonal → chỉ dùng khi data thực sự stable và không có pattern."
        ),
    }
    st.markdown(
        f'<div class="info-box">{notes.get(model_key, "")}</div>',
        unsafe_allow_html=True,
    )