"""
config.py — Shared theme, colors, chart utilities
Mondelez Vietnam / AFC Baseline Forecast App
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict

# ══════════════════════════════════════════════════════════════
# COLORS
# ══════════════════════════════════════════════════════════════

C_HIST   = "#4A9EFF"
C_FC     = "#FF7043"
C_TS     = "#00C49F"
C_S      = "#FFBB28"
C_T      = "#A78BFA"
C_CONST  = "#F472B6"
C_BEST   = "#10B981"
C_WARN   = "#F59400"
C_DANGER = "#EF4444"
C_BG     = "#0F1117"
C_CARD   = "#1E2130"
C_BORDER = "#2D3250"
C_TEXT   = "#E2E8F0"
C_MUTED  = "#94A3B8"
C_SUB    = "#64748B"

MODEL_COLORS = {
    "trend_seasonal": C_TS,
    "seasonal":       C_S,
    "trend":          C_T,
    "constant":       C_CONST,
}

MODEL_LABELS = {
    "trend_seasonal": "Trend + Seasonal",
    "seasonal":       "Seasonal",
    "trend":          "Trend",
    "constant":       "Constant",
}

MODEL_ICONS = {
    "trend_seasonal": "📈",
    "seasonal":       "🌀",
    "trend":          "📉",
    "constant":       "➡️",
}

# ══════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

[data-testid="stSidebar"] {
    background: #131722 !important;
    border-right: 1px solid #2D3250;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #4A9EFF;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 1.4rem;
}
.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* ── Page header ── */
.page-header {
    background: linear-gradient(135deg, #1a1f35 0%, #0F1117 100%);
    border: 1px solid #2D3250;
    border-left: 4px solid #4A9EFF;
    border-radius: 8px;
    padding: 18px 26px;
    margin-bottom: 20px;
}
.page-header h1 {
    font-size: 20px; font-weight: 700; color: #E2E8F0;
    margin: 0 0 4px 0;
    font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.5px;
}
.page-header p { font-size: 12px; color: #64748B; margin: 0; }

/* ── Section title ── */
.section-title {
    font-size: 10px; font-weight: 600;
    letter-spacing: 2px; text-transform: uppercase;
    color: #4A9EFF;
    font-family: 'IBM Plex Mono', monospace;
    margin: 24px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #2D3250;
}

/* ── Metric card ── */
.metric-card {
    background: #1E2130; border: 1px solid #2D3250;
    border-radius: 8px; padding: 14px 18px; flex: 1; min-width: 120px;
}
.metric-card .mc-label {
    font-size: 9px; color: #64748B;
    text-transform: uppercase; letter-spacing: 1.5px;
    font-family: 'IBM Plex Mono', monospace; margin-bottom: 5px;
}
.metric-card .mc-value {
    font-size: 22px; font-weight: 700; color: #E2E8F0;
    font-family: 'IBM Plex Mono', monospace; line-height: 1;
}
.metric-card .mc-sub { font-size: 10px; color: #94A3B8; margin-top: 3px; }

/* ── Info boxes ── */
.info-box {
    background: #1a2640; border: 1px solid #2563EB40;
    border-left: 3px solid #4A9EFF; border-radius: 6px;
    padding: 11px 15px; font-size: 12px; color: #94A3B8; margin: 10px 0;
}
.warn-box {
    background: #2a1f10; border: 1px solid #F5940060;
    border-left: 3px solid #F59400; border-radius: 6px;
    padding: 11px 15px; font-size: 12px; color: #CBD5E1; margin: 10px 0;
}
.success-box {
    background: #0d2318; border: 1px solid #10B98160;
    border-left: 3px solid #10B981; border-radius: 6px;
    padding: 11px 15px; font-size: 12px; color: #CBD5E1; margin: 10px 0;
}

/* ── Growth tables ── */
.growth-table {
    width: 100%; border-collapse: collapse;
    font-size: 12px; margin-top: 4px;
}
.growth-table th {
    background: #1E2130; color: #64748B;
    font-size: 9px; letter-spacing: 1px;
    text-transform: uppercase; padding: 7px 10px;
    text-align: center; border-bottom: 1px solid #2D3250;
    font-family: 'IBM Plex Mono', monospace;
}
.growth-table td {
    padding: 6px 10px; text-align: center;
    border-bottom: 1px solid #1a1f2e; color: #CBD5E1;
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
}
.growth-table tr:last-child td { border-bottom: none; }
.growth-table tr:hover td { background: #1a2035; }
.pos  { color: #10B981 !important; font-weight: 600; }
.neg  { color: #EF4444 !important; font-weight: 600; }
.neu  { color: #94A3B8 !important; }

/* ── Badges ── */
.badge {
    display: inline-block; padding: 1px 8px;
    border-radius: 100px; font-size: 10px;
    font-family: 'IBM Plex Mono', monospace; font-weight: 600;
}
.b-hist { background: #1D3461; color: #4A9EFF; }
.b-fc   { background: #2a1f10; color: #F59400; }
.b-best { background: #0d2318; color: #10B981; }

/* ── Recommendation card ── */
.rec-card {
    background: #13192b;
    border: 1px solid #2D3250; border-radius: 10px;
    padding: 24px 28px; margin: 14px 0;
}
.rec-card h2 {
    font-size: 20px; font-weight: 800; color: #F1F5F9;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -0.5px; margin: 0 0 6px 0;
    text-shadow: 0 0 20px rgba(255,255,255,0.08);
}
.rec-card .rc-accent {
    display: inline-block; padding: 2px 10px; border-radius: 4px;
    font-size: 20px; font-weight: 800; margin-right: 6px;
    font-family: 'IBM Plex Mono', monospace;
}
.rec-card p { color: #CBD5E1; margin: 0; font-size: 13px; line-height: 1.6; }
.rec-card p strong { color: #F1F5F9; }
.rec-divider { border: none; border-top: 1px solid #2D3250; margin: 14px 0 12px 0; }
.rec-grid {
    display: grid; grid-template-columns: repeat(6, 1fr);
    gap: 8px; margin-top: 0;
}
.rec-item {
    text-align: center;
    background: #1E2130; border: 1px solid #2D3250;
    border-radius: 6px; padding: 8px 4px;
}
.rec-item .ri-label {
    font-size: 9px; color: #94A3B8; letter-spacing: 1.2px;
    text-transform: uppercase; font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 4px;
}
.rec-item .ri-value {
    font-size: 18px; font-weight: 700; color: #F1F5F9;
    font-family: 'IBM Plex Mono', monospace; line-height: 1.2;
}

/* ── Sidebar combo counter ── */
.combo-box {
    background: #1a1f35; border: 1px solid #2D3250;
    border-radius: 6px; padding: 10px;
    font-size: 11px; color: #64748B;
    font-family: 'IBM Plex Mono', monospace;
}
.combo-box span { color: #4A9EFF; }
</style>
"""

# ══════════════════════════════════════════════════════════════
# PLOTLY LAYOUT DEFAULTS
# ══════════════════════════════════════════════════════════════

def plotly_base(fig, title="", height=400):
    fig.update_layout(
        title=dict(text=title,
                   font=dict(size=13, color=C_TEXT, family="IBM Plex Mono"),
                   x=0, xanchor='left', pad=dict(l=4)),
        height=height,
        plot_bgcolor=C_BG, paper_bgcolor=C_BG,
        font=dict(color=C_MUTED, family="IBM Plex Sans", size=11),
        legend=dict(bgcolor=C_CARD, bordercolor=C_BORDER, borderwidth=1,
                    font=dict(size=11, color=C_TEXT)),
        xaxis=dict(gridcolor="#1E2130", linecolor=C_BORDER,
                   showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#1E2130", linecolor=C_BORDER,
                   showgrid=True, zeroline=False),
        margin=dict(l=8, r=8, t=40, b=8),
        hovermode="x unified",
    )
    return fig


# ══════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════

def yr_sums(vals, dates):
    d = defaultdict(float)
    for dt, v in zip(dates, vals):
        d[dt.year] += v
    return dict(sorted(d.items()))


def qtr_sums(vals, dates):
    d = defaultdict(float)
    for dt, v in zip(dates, vals):
        q = (dt.month - 1) // 3 + 1
        d[(dt.year, q)] += v
    return dict(sorted(d.items()))


def mon_vals(vals, dates):
    return dict(sorted(zip(dates, vals)))


def build_yoy_df(history, hist_dates, forecast, fc_dates):
    all_v  = list(history) + list(forecast)
    all_d  = list(hist_dates) + list(fc_dates)
    yd     = yr_sums(all_v, all_d)
    yrs    = sorted(yd)
    rows   = []
    for i, yr in enumerate(yrs):
        prev = yd[yrs[i-1]] if i > 0 else None
        pct  = yd[yr] / prev - 1 if prev else None
        rows.append({
            "Year":  yr,
            "Total": yd[yr],
            "YoY":   pct,
            "Type":  "History" if yr <= hist_dates[-1].year else "Forecast",
        })
    return pd.DataFrame(rows)


def build_qoq_df(history, hist_dates, forecast, fc_dates):
    all_v  = list(history) + list(forecast)
    all_d  = list(hist_dates) + list(fc_dates)
    qd     = qtr_sums(all_v, all_d)
    qtrs   = sorted(qd)
    last_h = (hist_dates[-1].year, (hist_dates[-1].month - 1) // 3 + 1)
    rows   = []
    for i, q in enumerate(qtrs):
        prev = qd[qtrs[i-1]] if i > 0 else None
        pct  = qd[q] / prev - 1 if prev else None
        rows.append({
            "Quarter": f"Q{q[1]} {q[0]}",
            "Total":   qd[q],
            "QoQ":     pct,
            "Type":    "History" if q <= last_h else "Forecast",
        })
    return pd.DataFrame(rows)


def build_mom_df(history, hist_dates, forecast, fc_dates):
    all_v  = list(history) + list(forecast)
    all_d  = list(hist_dates) + list(fc_dates)
    md     = mon_vals(all_v, all_d)
    mons   = sorted(md)
    rows   = []
    for i, dt in enumerate(mons):
        prev = md[mons[i-1]] if i > 0 else None
        pct  = md[dt] / prev - 1 if prev else None
        rows.append({
            "Month": dt.strftime("%b %Y"),
            "Value": md[dt],
            "MoM":   pct,
            "Type":  "History" if dt <= hist_dates[-1] else "Forecast",
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# HTML TABLE RENDERERS
# ══════════════════════════════════════════════════════════════

def _pct_html(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '<span class="neu">—</span>'
    cls  = "pos" if v > 0.005 else ("neg" if v < -0.005 else "neu")
    sign = "+" if v > 0 else ""
    return f'<span class="{cls}">{sign}{v*100:.1f}%</span>'


def _num(v, dec=0):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:,.{dec}f}"


def _badge(t):
    return '<span class="badge b-hist">H</span>' if t == "History" \
           else '<span class="badge b-fc">F</span>'


def render_yoy_table(df):
    rows = ""
    for _, r in df.iterrows():
        rows += f"<tr><td>{_badge(r['Type'])} {int(r['Year'])}</td>" \
                f"<td>{_num(r['Total'])}</td>" \
                f"<td>{_pct_html(r['YoY'])}</td></tr>"
    return f"""<table class="growth-table">
        <thead><tr><th>Year</th><th>Volume</th><th>YoY %</th></tr></thead>
        <tbody>{rows}</tbody></table>"""


def render_qoq_table(df, max_rows=16):
    df = df.tail(max_rows)
    rows = ""
    for _, r in df.iterrows():
        rows += f"<tr><td>{_badge(r['Type'])} {r['Quarter']}</td>" \
                f"<td>{_num(r['Total'])}</td>" \
                f"<td>{_pct_html(r['QoQ'])}</td></tr>"
    return f"""<table class="growth-table">
        <thead><tr><th>Quarter</th><th>Volume</th><th>QoQ %</th></tr></thead>
        <tbody>{rows}</tbody></table>"""


def render_mom_table(df, max_rows=24):
    df = df.tail(max_rows)
    rows = ""
    for _, r in df.iterrows():
        rows += f"<tr><td>{_badge(r['Type'])} {r['Month']}</td>" \
                f"<td>{_num(r['Value'])}</td>" \
                f"<td>{_pct_html(r['MoM'])}</td></tr>"
    return f"""<table class="growth-table">
        <thead><tr><th>Month</th><th>Volume</th><th>MoM %</th></tr></thead>
        <tbody>{rows}</tbody></table>"""


def metric_card_html(label, value, sub="", color=C_TEXT):
    return f"""<div class="metric-card">
        <div class="mc-label">{label}</div>
        <div class="mc-value" style="color:{color}">{value}</div>
        <div class="mc-sub">{sub}</div>
    </div>"""


def section(text):
    return f'<div class="section-title">{text}</div>'