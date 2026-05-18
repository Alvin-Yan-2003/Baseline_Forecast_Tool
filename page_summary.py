"""
page_summary.py — Tab 6: Model Comparison & Recommendation
Business-driven optimal model selection with full comparison
"""

import io
import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta

from baseline_forecast_tool import select_best, export_to_excel
from baseline_forecast_tool import (
    _yoy_growth, _qoq_growth, _mom_growth, _var, _seasonality_strength,
)
from config import (
    MODEL_COLORS, MODEL_LABELS,
    section, metric_card_html,
    build_yoy_df, build_qoq_df, build_mom_df,
    render_yoy_table, render_qoq_table, render_mom_table,
    yr_sums,
    C_HIST, C_FC, C_TS, C_S, C_T, C_CONST, C_MUTED, C_TEXT,
    C_WARN, C_BORDER, C_CARD, C_BG,
)
from charts import (
    forecast_chart, yoy_combo_chart, qoq_area_chart,
    model_overlay_chart, radar_chart, mape_bar_chart,
)


def _fc_dates(hist_dates, h_steps):
    return [hist_dates[-1] + relativedelta(months=i + 1) for i in range(h_steps)]


def _mstr(v):
    return f"{v*100:.1f}%" if not (isinstance(v, float) and np.isnan(v)) else "N/A"


def _pct_delta(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    sign = "+" if v > 0 else ""
    color = "#10B981" if v > 0.005 else ("#EF4444" if v < -0.005 else "#94A3B8")
    return f'<span style="color:{color};font-weight:600">{sign}{v*100:.1f}%</span>'


# ══════════════════════════════════════════════════════════════
# RADAR SCORES
# ══════════════════════════════════════════════════════════════

def _compute_radar(model_key, all_results, history, hist_dates, fc_dates):
    model_top = [r for r in all_results if r["model"] == model_key]
    if not model_top:
        return None
    best  = select_best(model_top, history, top_n=1)[0]
    fc    = best["forecast"]
    all_v = list(history) + list(fc)

    mape    = best["mape"] if not np.isnan(best["mape"]) else 1.0
    var_yoy = _var(_yoy_growth(all_v, 12))
    var_qoq = _var(_qoq_growth(all_v))
    seas_s  = _seasonality_strength(history)
    has_seas = 1 if best["gamma"] > 0 else 0

    growth_s   = max(0, 10 - var_yoy * 120)
    mape_s     = max(0, 10 * (1 - mape))
    seasonal_s = has_seas * seas_s * 10
    trend_s    = max(0, 10 - var_qoq * 120)
    biz_s      = max(0, 10 - best["score"] * 0.8)

    return [round(growth_s, 2), round(mape_s, 2),
            round(seasonal_s, 2), round(trend_s, 2), round(biz_s, 2)]


# ══════════════════════════════════════════════════════════════
# SCORECARD TABLE
# ══════════════════════════════════════════════════════════════

def _build_scorecard(all_results, history, hist_dates, fc_dates):
    rows = []
    for model_key in ["trend_seasonal", "seasonal", "trend", "constant"]:
        model_top = [r for r in all_results if r["model"] == model_key]
        if not model_top:
            continue
        best  = select_best(model_top, history, top_n=1)[0]
        fc    = best["forecast"]
        all_v = list(history) + list(fc)
        all_d = list(hist_dates) + list(fc_dates)
        yd    = yr_sums(all_v, all_d)
        yrs   = sorted(yd)

        var_yoy = _var(_yoy_growth(all_v, 12))
        var_qoq = _var(_qoq_growth(all_v))

        row = {
            "Model":   MODEL_LABELS[model_key],
            "α":       best["alpha"],
            "β":       best["beta"],
            "γ":       best["gamma"],
            "MAPE":    best["mape"],
            "Score":   best["score"],
            "YoY Var": var_yoy,
            "QoQ Var": var_qoq,
            "_key":    model_key,
            "_best":   best,
        }
        # Add FC year columns — chỉ year đầu tiên của forecast (e.g. 2026)
        hist_last_yr = hist_dates[-1].year
        fc_start_yr  = min(d.year for d in fc_dates)
        target_yr    = fc_start_yr  # chỉ show năm đầu forecast
        if target_yr in yd:
            prev = yd.get(target_yr - 1)
            row[f"FC {target_yr}"]  = yd.get(target_yr, 0)
            row[f"YoY {target_yr}"] = yd[target_yr] / prev - 1 if prev else None
        rows.append(row)
    return rows


# ══════════════════════════════════════════════════════════════
# MAIN RENDER
# ══════════════════════════════════════════════════════════════

def render():
    st.markdown(
        '<div class="page-header" style="border-left-color:#10B981;">'
        '<h1>🏆 Model Comparison &amp; Recommendation</h1>'
        '<p>Tổng hợp 4 models — chọn model tối ưu theo business criteria (growth stability + MAPE)</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Guard ──────────────────────────────────────────────────
    if "results" not in st.session_state:
        st.markdown(
            '<div class="info-box">ℹ️ Nhấn <strong>▶ RUN FORECAST</strong> ở sidebar.</div>',
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
    top         = st.session_state["top"]
    fc_dates    = _fc_dates(hist_dates, h_steps)

    best_overall = top[0]
    model_key    = best_overall["model"]
    model_color  = MODEL_COLORS.get(model_key, "#10B981")
    model_label  = MODEL_LABELS.get(model_key, model_key)
    model_icons  = {"trend_seasonal":"📈","seasonal":"🌀","trend":"📉","constant":"➡️"}
    icon         = model_icons.get(model_key, "📊")

    # ── Winner banner ─────────────────────────────────────────
    st.markdown(f"""
    <div class="rec-card" style="border-color:{model_color}50;
         background:linear-gradient(135deg,{model_color}15,#111827 60%);">
        <h2 style="color:{model_color}; font-size:20px">
            {icon} Best Model: {model_label}
        </h2>
        <p>SKU: <strong style="color:#E2E8F0">{sku_name}</strong> —
        Optimal theo business scoring
        (YoY growth stability → QoQ → MoM → MAPE)</p>
        <div class="rec-grid">
            <div class="rec-item">
                <div class="ri-label">Alpha α</div>
                <div class="ri-value" style="color:{model_color}">{best_overall['alpha']:.2f}</div>
            </div>
            <div class="rec-item">
                <div class="ri-label">Beta β</div>
                <div class="ri-value" style="color:{model_color}">{best_overall['beta']:.2f}</div>
            </div>
            <div class="rec-item">
                <div class="ri-label">Gamma γ</div>
                <div class="ri-value" style="color:{model_color}">{best_overall['gamma']:.2f}</div>
            </div>
            <div class="rec-item">
                <div class="ri-label">MAPE</div>
                <div class="ri-value" style="color:{model_color}">{_mstr(best_overall['mape'])}</div>
            </div>
            <div class="rec-item">
                <div class="ri-label">Score</div>
                <div class="ri-value" style="color:{model_color}">{best_overall['score']:.4f}</div>
            </div>
            <div class="rec-item">
                <div class="ri-label">Horizon</div>
                <div class="ri-value" style="color:{model_color}">{h_steps}M</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Best model forecast chart ─────────────────────────────
    st.markdown(section(f"Best Model Forecast — {model_label}"), unsafe_allow_html=True)
    st.plotly_chart(
        forecast_chart(history, hist_dates, best_overall["forecast"], fc_dates,
                       model_label, model_color, best_overall.get("expost"), show_expost, 420),
        use_container_width=True,
    )

    # ── Growth tables for best ────────────────────────────────
    st.markdown(section("Growth Analysis — Best Model"), unsafe_allow_html=True)
    yoy_df = build_yoy_df(history, hist_dates, best_overall["forecast"], fc_dates)
    qoq_df = build_qoq_df(history, hist_dates, best_overall["forecast"], fc_dates)
    mom_df = build_mom_df(history, hist_dates, best_overall["forecast"], fc_dates)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**📅 Annual YoY**")
        st.markdown(render_yoy_table(yoy_df), unsafe_allow_html=True)
    with col_b:
        st.markdown("**📆 Quarterly QoQ**")
        st.markdown(render_qoq_table(qoq_df, max_rows=16), unsafe_allow_html=True)
    with col_c:
        st.markdown("**🗓️ Monthly MoM (24M)**")
        st.markdown(render_mom_table(mom_df, max_rows=24), unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(yoy_combo_chart(yoy_df, height=320), use_container_width=True)
    with col_r:
        st.plotly_chart(qoq_area_chart(qoq_df, height=320), use_container_width=True)

    # ── ALL MODELS OVERLAY ────────────────────────────────────
    st.markdown(section("All Models Forecast Overlay"), unsafe_allow_html=True)
    model_forecasts = {}
    model_mapes     = {}
    for mk in ["trend_seasonal", "seasonal", "trend", "constant"]:
        mr = [r for r in all_results if r["model"] == mk]
        if mr:
            bm = select_best(mr, history, top_n=1)[0]
            model_forecasts[mk] = bm["forecast"]
            model_mapes[mk]     = bm["mape"] if not np.isnan(bm["mape"]) else 1.0

    st.plotly_chart(
        model_overlay_chart(history, hist_dates, model_forecasts, fc_dates, 440),
        use_container_width=True,
    )

    # ── SCORECARD TABLE ───────────────────────────────────────
    st.markdown(section("Model Scorecard"), unsafe_allow_html=True)
    scorecard_rows = _build_scorecard(all_results, history, hist_dates, fc_dates)

    # Build display dataframe
    display_rows = []
    for row in scorecard_rows:
        dr = {
            "Model":   row["Model"],
            "α":       f"{row['α']:.2f}",
            "β":       f"{row['β']:.2f}",
            "γ":       f"{row['γ']:.2f}",
            "MAPE":    _mstr(row["MAPE"]),
            "Score":   f"{row['Score']:.4f}",
            "YoY Var": f"{row['YoY Var']:.5f}",
            "QoQ Var": f"{row['QoQ Var']:.5f}",
        }
        # FC year columns
        for k, v in row.items():
            if k.startswith("FC "):
                dr[k] = f"{v:,.0f}" if v is not None else "—"
            elif k.startswith("YoY "):
                dr[k] = f"{v*100:+.1f}%" if v is not None and not (isinstance(v,float) and np.isnan(v)) else "—"
        display_rows.append(dr)

    sc_df = pd.DataFrame(display_rows)
    # Highlight best model row
    best_model_label = MODEL_LABELS.get(best_overall["model"], "")

    def highlight_best(row):
        return ["background-color: #0d2318; color: #10B981" if row.name == best_model_label
                else "" for _ in row]

    st.dataframe(
        sc_df.set_index("Model").style.apply(highlight_best, axis=1),
        use_container_width=True,
        height=220,
    )

    # ── RADAR + MAPE bar ──────────────────────────────────────
    st.markdown(section("Multi-Dimension Comparison"), unsafe_allow_html=True)
    col_rad, col_mape = st.columns([3, 2])

    model_scores = {}
    for mk in ["trend_seasonal", "seasonal", "trend", "constant"]:
        r = _compute_radar(mk, all_results, history, hist_dates, fc_dates)
        if r:
            model_scores[mk] = r

    with col_rad:
        st.plotly_chart(radar_chart(model_scores, height=400), use_container_width=True)

    with col_mape:
        st.plotly_chart(mape_bar_chart(model_mapes, height=400), use_container_width=True)

    # ── BUSINESS DECISION GUIDE ───────────────────────────────
    st.markdown(section("Business Decision Guide"), unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <strong>📋 Scoring Priority — Mondelez Business Logic:</strong><br><br>
    <strong>1️⃣ YoY Growth Stability (weight 10×)</strong><br>
    Forecast năm tiếp theo phải <em>reasonable</em> so với năm hiện tại.
    Không có ±50% đột biến vô lý. Model tốt = YoY variance thấp.<br><br>
    <strong>2️⃣ QoQ Growth Stability (weight 5×)</strong><br>
    Growth giữa các quý phải smooth và consistent — tránh oscillate mạnh Q1→Q2→Q3.<br><br>
    <strong>3️⃣ MoM Growth Stability (weight 2×)</strong><br>
    Monthly pattern phải phản ánh đúng historical seasonality, không spike bất thường.<br><br>
    <strong>4️⃣ MAPE (weight 1×)</strong><br>
    Quan trọng nhưng không phải số 1. Model MAPE thấp nhưng forecast phẳng
    = không hữu ích cho S&OP planning.<br><br>
    <strong>⚠️ Lưu ý:</strong> Model <em>Constant</em> thường có MAPE tốt nhất nhưng
    forecast phẳng hoàn toàn — không phản ánh seasonality của AFC.
    Chỉ chấp nhận nếu business cần flat assumption.
    </div>
    """, unsafe_allow_html=True)

    # ── TOP 10 OVERALL ────────────────────────────────────────
    st.markdown(section(f"Top {min(top_n, 10)} Best Combinations (All Models)"), unsafe_allow_html=True)
    top10_rows = []
    for i, r in enumerate(top[:10], 1):
        top10_rows.append({
            "Rank":  i,
            "Model": MODEL_LABELS.get(r["model"], r["model"]),
            "α":     r["alpha"],
            "β":     r["beta"],
            "γ":     r["gamma"],
            "MAPE":  _mstr(r["mape"]),
            "Score": round(r["score"], 5),
        })
    df_top10 = pd.DataFrame(top10_rows).set_index("Rank")
    st.dataframe(df_top10, use_container_width=True, height=min(400, len(top10_rows)*36+38))

    # ── EXPORT ────────────────────────────────────────────────
    st.markdown(section("Export Results"), unsafe_allow_html=True)
    col_ex1, col_ex2 = st.columns([2, 3])

    with col_ex1:
        if st.button("⬇️  Export to Excel", type="secondary", use_container_width=True):
            with st.spinner("Generating Excel…"):
                all_sku_results = [{
                    "name":       sku_name,
                    "history":    history,
                    "hist_dates": hist_dates,
                    "fc_dates":   fc_dates,
                    "top":        top,
                }]
                with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                    tmp_path = tmp.name
                export_to_excel(tmp_path, all_sku_results)
                buf = io.BytesIO()
                with open(tmp_path, "rb") as f:
                    buf.write(f.read())
                os.unlink(tmp_path)
                buf.seek(0)
            st.download_button(
                label="📥 Download .xlsx",
                data=buf,
                file_name=f"forecast_{sku_name.replace(' ','_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    with col_ex2:
        st.markdown(
            '<div class="info-box">Excel output gồm: <strong>Summary</strong> sheet, '
            '<strong>Results</strong> sheet (top 10 mọi model với full monthly values), '
            'và <strong>Growth Analysis</strong> sheet.</div>',
            unsafe_allow_html=True,
        )