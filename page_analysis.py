"""
page_analysis.py — Tab 1: Data Analysis
Trend, seasonal, pattern analysis + parameter recommendations
"""

import numpy as np
import streamlit as st

from config import (
    C_TS, C_S, C_T, C_CONST, C_HIST, C_FC, C_WARN, C_MUTED,
    C_TEXT, C_BORDER,
    section, metric_card_html,
)
from charts import (
    history_with_trend,
    decomposition_chart,
    monthly_pattern_chart,
    yoy_combo_chart,
)
from config import build_yoy_df
from baseline_forecast_tool import _seasonality_strength


def render():
    st.markdown(
        '<div class="page-header">'
        '<h1>🔬 Data Analysis</h1>'
        '<p>Historical pattern, seasonality, trend — và gợi ý thông số tối ưu</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Guard: need data loaded ───────────────────────────────
    if "history" not in st.session_state:
        st.markdown(
            '<div class="warn-box">⚠️ <strong>Chưa có dữ liệu.</strong> '
            "Upload file ở sidebar để bắt đầu phân tích.</div>",
            unsafe_allow_html=True,
        )
        return

    history    = st.session_state["history"]
    hist_dates = st.session_state["hist_dates"]
    sku_name   = st.session_state.get("sku_name", "SKU")

    hist_arr = np.array(history, float)
    n        = len(history)

    # ── Derived stats ─────────────────────────────────────────
    mean_vol    = np.mean(hist_arr)
    std_vol     = np.std(hist_arr)
    cv          = std_vol / mean_vol if mean_vol != 0 else 0
    seas_str    = _seasonality_strength(history, 12)
    trend_slope = (
        (np.mean(history[n//2:]) - np.mean(history[:n//2])) / np.mean(history[:n//2])
        if np.mean(history[:n//2]) != 0 else 0
    )
    peak_idx    = int(np.argmax(hist_arr))
    trough_idx  = int(np.argmin(hist_arr))

    # ── KPI cards ─────────────────────────────────────────────
    st.markdown(section("Key Statistics"), unsafe_allow_html=True)

    cols = st.columns(7)
    kpis = [
        ("Months",          str(n),                      "data points"),
        ("Mean Volume",     f"{mean_vol:,.0f}",           "avg monthly"),
        ("Std Dev",         f"{std_vol:,.0f}",            "dispersion"),
        ("CV",              f"{cv:.2f}",                  ">0.3 = volatile"),
        ("Seasonality",     f"{seas_str:.2f}",            ">0.3 = strong"),
        ("Trend (H1→H2)",   f"{trend_slope*100:+.1f}%",  "first vs second half"),
        ("Peak Month",      hist_dates[peak_idx].strftime("%b %Y"), f"val={hist_arr[peak_idx]:,.0f}"),
    ]
    accent_colors = [C_HIST, C_HIST, C_MUTED,
                     C_WARN if cv > 0.3 else C_TS,
                     C_TS if seas_str > 0.3 else C_MUTED,
                     C_FC if abs(trend_slope) > 0.05 else C_MUTED,
                     C_FC]
    for col, (lbl, val, sub), color in zip(cols, kpis, accent_colors):
        col.markdown(metric_card_html(lbl, val, sub, color), unsafe_allow_html=True)

    # ── History + trend line ──────────────────────────────────
    st.markdown(section(f"Historical Volume — {sku_name}"), unsafe_allow_html=True)
    st.plotly_chart(
        history_with_trend(history, hist_dates, height=370),
        use_container_width=True,
        key="ana_chart_1",
    )

    # ── Seasonal decomposition ────────────────────────────────
    st.markdown(section("Seasonal Decomposition (Multiplicative)"), unsafe_allow_html=True)

    if n >= 24:
        fig_dec, s_factors = decomposition_chart(history, hist_dates, period=12, height=540)
        st.plotly_chart(fig_dec, use_container_width=True,
        key="ana_chart_2")

        # Seasonal index table
        with st.expander("📋 Seasonal Index Values (by month position)", expanded=False):
            MNAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"]
            rows = ""
            for m in range(12):
                idx  = s_factors.get(m, 1.0)
                flag = "🔺" if idx > 1.05 else ("🔻" if idx < 0.95 else "—")
                rows += (
                    f"<tr><td>{MNAMES[m]}</td>"
                    f"<td style='color:{'#10B981' if idx>1.05 else ('#EF4444' if idx<0.95 else '#94A3B8')}'>"
                    f"{idx:.3f}</td><td>{flag}</td></tr>"
                )
            st.markdown(
                f'<table class="growth-table" style="max-width:340px">'
                f'<thead><tr><th>Month</th><th>Seasonal Index</th><th></th></tr></thead>'
                f'<tbody>{rows}</tbody></table>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f'<div class="warn-box">⚠️ Cần ≥24 tháng để decompose. '
            f"Hiện có {n} tháng.</div>",
            unsafe_allow_html=True,
        )
        s_factors = {}

    # ── Monthly pattern + YoY side by side ───────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(section("Monthly Average Pattern"), unsafe_allow_html=True)
        st.plotly_chart(
            monthly_pattern_chart(history, hist_dates, height=320),
            use_container_width=True,
        key="ana_chart_3",
        )

    with col_r:
        st.markdown(section("Annual YoY Growth"), unsafe_allow_html=True)
        # build a dummy yoy_df for history only
        from config import yr_sums, build_yoy_df
        from datetime import datetime
        # Fake single-point forecast at last point so build_yoy_df works
        dummy_fc   = [history[-1]]
        dummy_fc_d = [hist_dates[-1] + __import__("dateutil.relativedelta",
                       fromlist=["relativedelta"]).relativedelta(months=1)]
        yoy_df = build_yoy_df(history, hist_dates, dummy_fc, dummy_fc_d)
        yoy_df = yoy_df[yoy_df["Type"] == "History"]          # history only
        st.plotly_chart(yoy_combo_chart(yoy_df, height=320), use_container_width=True,
        key="ana_chart_4")

    # ── Parameter recommendations ─────────────────────────────
    st.markdown(section("📌 Parameter Recommendations"), unsafe_allow_html=True)

    # Alpha
    if cv < 0.20:   rec_a, ra_reason = "0.01 – 0.10", "Data ổn định (CV thấp) → α nhỏ, học chậm"
    elif cv < 0.40: rec_a, ra_reason = "0.10 – 0.25", "CV trung bình → α cân bằng"
    else:            rec_a, ra_reason = "0.25 – 0.50", "Data volatile (CV cao) → α lớn, phản ứng nhanh"

    # Beta
    if abs(trend_slope) < 0.03: rec_b, rb_reason = "0.01 – 0.10", "Trend yếu → β nhỏ hoặc dùng model Seasonal"
    elif abs(trend_slope) < 0.10: rec_b, rb_reason = "0.10 – 0.25", "Trend vừa → β moderate"
    else:                          rec_b, rb_reason = "0.20 – 0.40", "Trend mạnh → β cao hơn"

    # Gamma
    if seas_str < 0.20:   rec_g, rg_reason = "Không cần", "Seasonality yếu → dùng model không có γ"
    elif seas_str < 0.40: rec_g, rg_reason = "0.40 – 0.70", "Seasonal vừa → γ moderate"
    else:                  rec_g, rg_reason = "0.70 – 0.99", "Seasonal mạnh → γ cao để capture pattern"

    col1, col2, col3 = st.columns(3)
    param_cards = [
        (col1, "Alpha (α)", rec_a, ra_reason, C_HIST),
        (col2, "Beta (β)",  rec_b, rb_reason, C_T),
        (col3, "Gamma (γ)", rec_g, rg_reason, C_TS),
    ]
    for col, param, rec, reason, color in param_cards:
        col.markdown(
            metric_card_html(param, rec, reason, color),
            unsafe_allow_html=True,
        )

    # ── Model recommendation ──────────────────────────────────
    st.markdown(section("Recommended Model"), unsafe_allow_html=True)

    has_trend    = abs(trend_slope) > 0.05
    has_seasonal = seas_str > 0.30

    if has_trend and has_seasonal:
        rec_model  = "Trend + Seasonal"
        rec_color  = C_TS
        rec_icon   = "📈"
        rec_reason = (
            f"Data có <strong>cả trend</strong> ({trend_slope*100:+.1f}% H1→H2) "
            f"<strong>và seasonality mạnh</strong> (strength={seas_str:.2f}). "
            "→ Dùng model Trend+Seasonal với α, β, γ."
        )
    elif has_seasonal:
        rec_model  = "Seasonal"
        rec_color  = C_S
        rec_icon   = "🌀"
        rec_reason = (
            f"Data có <strong>seasonal pattern rõ</strong> (strength={seas_str:.2f}) "
            f"nhưng trend không đáng kể ({trend_slope*100:+.1f}%). "
            "→ Dùng model Seasonal (α + γ)."
        )
    elif has_trend:
        rec_model  = "Trend"
        rec_color  = C_T
        rec_icon   = "📉"
        rec_reason = (
            f"Có trend ({trend_slope*100:+.1f}%) nhưng seasonal pattern yếu (strength={seas_str:.2f}). "
            "→ Dùng model Trend (α + β)."
        )
    else:
        rec_model  = "Seasonal hoặc Constant"
        rec_color  = C_MUTED
        rec_icon   = "➡️"
        rec_reason = (
            "Cả trend và seasonality đều không rõ ràng. "
            "Kiểm tra kỹ monthly pattern trước khi chọn Constant."
        )

    st.markdown(
        f"""<div class="rec-card" style="border-color:{rec_color}50;
             background:linear-gradient(135deg,{rec_color}12,#111827);">
            <h2 style="color:{rec_color}">{rec_icon} Recommended: {rec_model}</h2>
            <p>{rec_reason}</p>
            <div class="rec-grid" style="grid-template-columns:repeat(3,1fr); margin-top:14px;">
                <div class="rec-item">
                    <div class="ri-label">Seasonality Strength</div>
                    <div class="ri-value" style="color:{rec_color}">{seas_str:.2f}</div>
                </div>
                <div class="rec-item">
                    <div class="ri-label">CV (Volatility)</div>
                    <div class="ri-value" style="color:{rec_color}">{cv:.2f}</div>
                </div>
                <div class="rec-item">
                    <div class="ri-label">Trend H1→H2</div>
                    <div class="ri-value" style="color:{rec_color}">{trend_slope*100:+.1f}%</div>
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Data preview ─────────────────────────────────────────
    with st.expander("📋 Raw Data Preview", expanded=False):
        import pandas as pd
        df_raw = pd.DataFrame({
            "Date":   [d.strftime("%b %Y") for d in hist_dates],
            "Volume": [round(v, 2) for v in history],
        })
        st.dataframe(df_raw, use_container_width=True, height=300)