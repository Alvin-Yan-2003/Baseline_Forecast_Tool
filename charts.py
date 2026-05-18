"""
charts.py — Plotly chart builders
Mondelez Vietnam / AFC Baseline Forecast App
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def _hex_alpha(hex_color: str, alpha_hex: str) -> str:
    """Convert '#RRGGBB' + 2-char hex alpha → 'rgba(R,G,B,A)' for Plotly compatibility."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    a = round(int(alpha_hex, 16) / 255, 3)
    return f"rgba({r},{g},{b},{a})"


from collections import defaultdict

from config import (
    C_HIST, C_FC, C_TS, C_S, C_T, C_CONST, C_BEST,
    C_WARN, C_BORDER, C_TEXT, C_MUTED, C_BG, C_CARD, C_SUB,
    MODEL_COLORS, MODEL_LABELS,
    plotly_base, yr_sums, qtr_sums,
)


# ══════════════════════════════════════════════════════════════
# 1. HISTORY + FORECAST LINE CHART
# ══════════════════════════════════════════════════════════════

def forecast_chart(history, hist_dates, forecast, fc_dates,
                   model_label, model_color,
                   expost=None, show_expost=True, height=400):
    fig = go.Figure()

    xs_h = [d.strftime("%b %Y") for d in hist_dates]
    xs_f = [d.strftime("%b %Y") for d in fc_dates]

    # History area
    fig.add_trace(go.Scatter(
        x=xs_h, y=history,
        name="History",
        mode="lines+markers",
        line=dict(color=C_HIST, width=2.5),
        marker=dict(size=4, color=C_HIST),
        fill="tozeroy",
        fillcolor=_hex_alpha(C_HIST, "18"),
    ))

    # Ex-post fitted
    if show_expost and expost and len(expost) > 0:
        n_skip = len(history) - len(expost)
        fig.add_trace(go.Scatter(
            x=xs_h[n_skip:], y=expost,
            name="Fitted (in-sample)",
            mode="lines",
            line=dict(color=model_color, width=1.5, dash="dot"),
            opacity=0.75,
        ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=xs_f, y=forecast,
        name=f"Forecast — {model_label}",
        mode="lines+markers",
        line=dict(color=model_color, width=2.5, dash="dash"),
        marker=dict(size=6, color=model_color, symbol="diamond"),
        fill="tozeroy",
        fillcolor=_hex_alpha(model_color, "14"),
    ))

    # Vertical divider
    fig.add_vline(
        x=xs_h[-1],
        line=dict(color="#475569", width=1.2, dash="dash"),
    )
    fig.add_annotation(
        x=xs_h[-1], y=1, yref="paper",
        text="  Forecast →", showarrow=False,
        font=dict(size=10, color="#64748B"),
        xanchor="left", yanchor="top",
    )

    plotly_base(fig, f"{model_label} — Forecast vs History", height)
    fig.update_xaxes(tickangle=40, nticks=24)
    return fig


# ══════════════════════════════════════════════════════════════
# 2. SEASONAL DECOMPOSITION
# ══════════════════════════════════════════════════════════════

def decomposition_chart(history, hist_dates, period=12, height=540):
    hist = np.array(history, float)
    n    = len(hist)
    half = period // 2

    # Trend via centred MA
    trend = np.full(n, np.nan)
    for i in range(half, n - half):
        trend[i] = np.mean(hist[i - half: i + half + 1])

    # Seasonal factors
    by_m = defaultdict(list)
    for i in range(n):
        if not np.isnan(trend[i]) and trend[i] != 0:
            by_m[i % period].append(hist[i] / trend[i])
    s_fac = {m: np.mean(v) for m, v in by_m.items()}

    seasonal = np.array([s_fac.get(i % period, np.nan) for i in range(n)])

    residual = np.full(n, np.nan)
    for i in range(n):
        denom = trend[i] * seasonal[i]
        if not np.isnan(denom) and denom != 0:
            residual[i] = hist[i] / denom

    xs = [d.strftime("%b %Y") for d in hist_dates]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=("Observed + Trend (Moving Avg)",
                        "Seasonal Index (multiplicative)",
                        "Residual"),
    )

    # Row 1
    fig.add_trace(go.Scatter(x=xs, y=hist, name="Observed",
                              line=dict(color=C_HIST, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=trend, name="Trend (MA)",
                              line=dict(color=C_FC, width=2.5, dash="dash")), row=1, col=1)

    # Row 2
    fig.add_trace(go.Scatter(x=xs, y=seasonal, name="Seasonal Index",
                              mode="lines+markers",
                              line=dict(color=C_TS, width=2),
                              marker=dict(size=4)), row=2, col=1)
    fig.add_hline(y=1.0, line=dict(color="#475569", dash="dash"), row=2, col=1)

    # Row 3
    fig.add_trace(go.Scatter(x=xs, y=residual, name="Residual",
                              line=dict(color=C_S, width=1.5)), row=3, col=1)
    fig.add_hline(y=1.0, line=dict(color="#475569", dash="dash"), row=3, col=1)

    fig.update_layout(
        height=height,
        plot_bgcolor=C_BG, paper_bgcolor=C_BG,
        font=dict(color=C_MUTED, family="IBM Plex Sans", size=11),
        legend=dict(bgcolor=C_CARD, bordercolor=C_BORDER, borderwidth=1,
                    font=dict(size=11, color=C_TEXT)),
        margin=dict(l=8, r=8, t=40, b=8),
        showlegend=True,
    )
    for row in [1, 2, 3]:
        fig.update_xaxes(gridcolor="#1E2130", linecolor=C_BORDER,
                          tickangle=40, row=row, col=1)
        fig.update_yaxes(gridcolor="#1E2130", linecolor=C_BORDER, row=row, col=1)

    return fig, s_fac


# ══════════════════════════════════════════════════════════════
# 3. MONTHLY AVERAGE BAR CHART
# ══════════════════════════════════════════════════════════════

def monthly_pattern_chart(history, hist_dates, height=320):
    by_m  = defaultdict(list)
    for dt, v in zip(hist_dates, history):
        by_m[dt.month].append(v)

    MNAMES = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    means  = [np.mean(by_m.get(m, [0])) for m in range(1, 13)]
    stds   = [np.std(by_m.get(m, [0]))  for m in range(1, 13)]
    avg    = np.mean(history)

    bar_colors = []
    for v in means:
        if v == max(means):   bar_colors.append(C_FC)
        elif v == min(means): bar_colors.append(C_CONST)
        else:                  bar_colors.append("#3B4F6B")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=MNAMES, y=means,
        error_y=dict(type="data", array=stds, visible=True, color="#475569"),
        marker_color=bar_colors,
        name="Monthly Avg ± 1σ",
        hovertemplate="<b>%{x}</b><br>Avg: %{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=avg, line=dict(color="#64748B", dash="dash", width=1),
                  annotation_text=f"  Annual avg {avg:,.0f}",
                  annotation_font=dict(color="#64748B", size=10))

    plotly_base(fig, "Monthly Average Pattern (all years)", height)
    return fig


# ══════════════════════════════════════════════════════════════
# 4. YoY BAR + LINE COMBO
# ══════════════════════════════════════════════════════════════

def yoy_combo_chart(yoy_df, height=360):
    """Bar = annual volume, line = YoY % on secondary axis."""
    types  = yoy_df["Type"].tolist()
    colors = [C_HIST if t == "History" else C_FC for t in types]

    yoy_vals = yoy_df["YoY"].tolist()
    pct_show = [v if (v is not None and not (isinstance(v, float) and np.isnan(v)))
                else None for v in yoy_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yoy_df["Year"].astype(str), y=yoy_df["Total"],
        marker_color=colors, name="Annual Volume",
        hovertemplate="<b>%{x}</b><br>Volume: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=yoy_df["Year"].astype(str), y=pct_show,
        mode="lines+markers+text",
        name="YoY %", yaxis="y2",
        line=dict(color=C_WARN, width=2.5),
        marker=dict(size=8, color=C_WARN),
        text=[f"{v*100:+.1f}%" if v is not None else "" for v in pct_show],
        textposition="top center",
        textfont=dict(size=10, color=C_WARN),
        hovertemplate="<b>%{x}</b><br>YoY: %{y:.1%}<extra></extra>",
    ))

    plotly_base(fig, "Annual Volume & YoY Growth", height)
    max_pct = max([abs(v) for v in pct_show if v is not None], default=0.3)
    fig.update_layout(
        yaxis2=dict(
            overlaying="y", side="right", showgrid=False,
            tickformat=".0%", color=C_WARN,
            range=[-(max_pct * 1.8), max_pct * 2.5],
        ),
        bargap=0.35,
    )
    return fig


# ══════════════════════════════════════════════════════════════
# 5. QoQ AREA CHART
# ══════════════════════════════════════════════════════════════

def qoq_area_chart(qoq_df, height=300):
    hist_mask = qoq_df["Type"] == "History"
    fc_mask   = qoq_df["Type"] == "Forecast"

    fig = go.Figure()
    if hist_mask.any():
        fig.add_trace(go.Scatter(
            x=qoq_df.loc[hist_mask, "Quarter"],
            y=qoq_df.loc[hist_mask, "Total"],
            name="History", mode="lines+markers",
            line=dict(color=C_HIST, width=2),
            marker=dict(size=5),
            fill="tozeroy", fillcolor=_hex_alpha(C_HIST, "20"),
        ))
    if fc_mask.any():
        fig.add_trace(go.Scatter(
            x=qoq_df.loc[fc_mask, "Quarter"],
            y=qoq_df.loc[fc_mask, "Total"],
            name="Forecast", mode="lines+markers",
            line=dict(color=C_FC, width=2, dash="dash"),
            marker=dict(size=5, symbol="diamond"),
            fill="tozeroy", fillcolor=_hex_alpha(C_FC, "20"),
        ))
    plotly_base(fig, "Quarterly Volume", height)
    fig.update_xaxes(tickangle=40)
    return fig


# ══════════════════════════════════════════════════════════════
# 6. ALL-MODEL OVERLAY
# ══════════════════════════════════════════════════════════════

def model_overlay_chart(history, hist_dates, model_forecasts, fc_dates, height=420):
    """
    model_forecasts: dict { model_key: forecast_list }
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[d.strftime("%b %Y") for d in hist_dates], y=history,
        name="History", mode="lines+markers",
        line=dict(color=C_HIST, width=3),
        marker=dict(size=5),
    ))

    for model_key, fc in model_forecasts.items():
        lbl = MODEL_LABELS.get(model_key, model_key)
        col = MODEL_COLORS.get(model_key, "#888")
        fig.add_trace(go.Scatter(
            x=[d.strftime("%b %Y") for d in fc_dates], y=fc,
            name=lbl, mode="lines+markers",
            line=dict(color=col, width=2, dash="dash"),
            marker=dict(size=4),
        ))

    fig.add_vline(
        x=hist_dates[-1].strftime("%b %Y"),
        line=dict(color="#475569", dash="dash", width=1),
    )
    plotly_base(fig, "Forecast Comparison — All Models (Best Params)", height)
    fig.update_xaxes(tickangle=40, nticks=22)
    return fig


# ══════════════════════════════════════════════════════════════
# 7. RADAR COMPARISON
# ══════════════════════════════════════════════════════════════

def radar_chart(model_scores, height=420):
    """
    model_scores: dict { model_key: [5 scores 0-10] }
    dims: Growth Stability, MAPE, Seasonal Fit, Trend Capture, Business Score
    """
    cats = ["Growth<br>Stability", "MAPE<br>Accuracy",
            "Seasonal<br>Fit", "Trend<br>Capture", "Business<br>Score"]

    fig = go.Figure()
    for model_key, scores in model_scores.items():
        col = MODEL_COLORS.get(model_key, "#888")
        lbl = MODEL_LABELS.get(model_key, model_key)
        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],           # close the polygon
            theta=cats + [cats[0]],
            fill="toself",
            name=lbl,
            line=dict(color=col, width=2),
            fillcolor=_hex_alpha(col, "28"),
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10],
                            gridcolor="#2D3250", linecolor="#2D3250",
                            tickfont=dict(color=C_MUTED, size=9)),
            angularaxis=dict(gridcolor="#2D3250", linecolor="#2D3250",
                             tickfont=dict(color=C_TEXT, size=11)),
            bgcolor=C_BG,
        ),
        paper_bgcolor=C_BG, plot_bgcolor=C_BG,
        legend=dict(bgcolor=C_CARD, bordercolor=C_BORDER, borderwidth=1,
                    font=dict(color=C_TEXT, size=11)),
        height=height,
        margin=dict(l=40, r=40, t=30, b=30),
    )
    return fig


# ══════════════════════════════════════════════════════════════
# 8. MAPE BAR — model comparison
# ══════════════════════════════════════════════════════════════

def mape_bar_chart(model_mapes, height=300):
    """model_mapes: dict { model_key: mape_float }"""
    labels = [MODEL_LABELS.get(k, k) for k in model_mapes]
    vals   = [v * 100 for v in model_mapes.values()]
    colors = [MODEL_COLORS.get(k, "#888") for k in model_mapes]

    fig = go.Figure(go.Bar(
        x=labels, y=vals,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(size=11, color=C_TEXT),
        hovertemplate="<b>%{x}</b><br>MAPE: %{y:.1f}%<extra></extra>",
    ))
    plotly_base(fig, "MAPE Comparison (Best Params per Model)", height)
    fig.update_yaxes(title_text="MAPE %")
    return fig


# ══════════════════════════════════════════════════════════════
# 9. SCORE HEATMAP — top N combinations
# ══════════════════════════════════════════════════════════════

def score_scatter(top_results, height=380):
    """Scatter: x=MAPE, y=score, color=model, size=gamma."""
    xs, ys, cs, szs, texts = [], [], [], [], []
    for r in top_results:
        mape = r["mape"] if not (isinstance(r["mape"], float) and np.isnan(r["mape"])) else 0
        xs.append(mape * 100)
        ys.append(r["score"])
        cs.append(MODEL_COLORS.get(r["model"], "#888"))
        szs.append(8 + r.get("gamma", 0) * 14)
        texts.append(
            f"{MODEL_LABELS.get(r['model'], r['model'])}<br>"
            f"α={r['alpha']:.2f} β={r['beta']:.2f} γ={r['gamma']:.2f}<br>"
            f"MAPE={mape:.1f}% Score={r['score']:.4f}"
        )

    fig = go.Figure(go.Scatter(
        x=xs, y=ys,
        mode="markers",
        marker=dict(color=cs, size=szs, opacity=0.75,
                    line=dict(color=C_BORDER, width=0.5)),
        text=texts,
        hovertemplate="%{text}<extra></extra>",
    ))
    plotly_base(fig, "Score vs MAPE (all top combinations — bubble = γ size)", height)
    fig.update_xaxes(title_text="MAPE %")
    fig.update_yaxes(title_text="Business Score (lower = better)")
    return fig


# ══════════════════════════════════════════════════════════════
# 10. LINEAR TREND LINE ON HISTORY
# ══════════════════════════════════════════════════════════════

def history_with_trend(history, hist_dates, height=360):
    xs = [d.strftime("%b %Y") for d in hist_dates]
    n  = len(history)

    # Linear trend
    z    = np.polyfit(range(n), history, 1)
    tl   = np.polyval(z, range(n))
    slope_pct = (tl[-1] - tl[0]) / tl[0] * 100 if tl[0] != 0 else 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=history, name="Volume",
        mode="lines+markers",
        line=dict(color=C_HIST, width=2.5),
        marker=dict(size=4, color=C_HIST),
        fill="tozeroy", fillcolor=_hex_alpha(C_HIST, "15"),
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=tl, name=f"Linear Trend ({slope_pct:+.1f}%)",
        mode="lines",
        line=dict(color=C_FC, width=2, dash="dash"),
    ))
    # 12-month rolling average
    if n >= 12:
        roll = pd.Series(history).rolling(12, center=True).mean().tolist()
        fig.add_trace(go.Scatter(
            x=xs, y=roll, name="12M Rolling Avg",
            mode="lines",
            line=dict(color=C_WARN, width=1.5, dash="dot"),
        ))

    plotly_base(fig, "Historical Volume + Trend", height)
    fig.update_xaxes(tickangle=40, nticks=20)
    return fig