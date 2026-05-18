"""
page_multi.py — Tab: Multi-SKU Run
No SKU selector here — reads selection from sidebar (_multi_sel).
RUN is triggered by sidebar button (_multi_run_trigger counter).
"""

import math
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st

from baseline_forecast_tool import grid_search, select_best
from config import (
    section, metric_card_html,
    C_TEXT, C_MUTED, C_HIST, C_FC, C_WARN, C_TS,
    MODEL_LABELS, MODEL_COLORS,
)
from export_excel import export_multi_sku

MODEL_ICONS = {
    "trend_seasonal": "📈",
    "seasonal":       "🌀",
    "trend":          "📉",
    "constant":       "➡️",
}


def _fc_dates(hist_dates, h_steps):
    return [hist_dates[-1] + relativedelta(months=i + 1) for i in range(h_steps)]


def _annual_vol(values, dates):
    d = defaultdict(float)
    for dt, v in zip(dates, values):
        yr = dt.year if isinstance(dt, datetime) else pd.Timestamp(dt).year
        d[yr] += v
    return dict(sorted(d.items()))


def _mstr(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v*100:.1f}%"


# ══════════════════════════════════════════════════════════════
# MAIN RENDER
# ══════════════════════════════════════════════════════════════

def render():
    st.markdown(
        '<div class="page-header" style="border-left-color:#A78BFA;">'
        '<h1>📦 Multi-SKU Run</h1>'
        '<p>Chọn SKUs ở sidebar → nhấn RUN → xem kết quả và export Excel</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Guard: need file ──────────────────────────────────────
    if "skus" not in st.session_state:
        st.markdown(
            '<div class="warn-box">⚠️ Upload file BBL PPG ở sidebar trước.</div>',
            unsafe_allow_html=True,
        )
        return

    skus       = st.session_state["skus"]
    all_names  = [s["name"] for s in skus]
    sel_names  = st.session_state.get("_multi_sel", [])
    sel_names  = [n for n in sel_names if n in all_names]  # guard stale
    n_sel      = len(sel_names)
    n_total    = len(all_names)

    params     = st.session_state.get("_fc_params", {
        "alpha_min": 0.01, "alpha_max": 0.30,
        "beta_min":  0.01, "beta_max":  0.30,
        "gamma_min": 0.50, "gamma_max": 0.99,
    })
    step       = st.session_state.get("grid_step", 0.03)
    h_steps    = st.session_state.get("h_steps", 24)
    top_n      = st.session_state.get("top_n", 10)

    # ── Selection summary ─────────────────────────────────────
    st.markdown(section("SKUs Selected from Sidebar"), unsafe_allow_html=True)

    if n_sel == 0:
        st.markdown(
            '<div class="warn-box">⚠️ Chưa chọn SKU nào. '
            'Chuyển sang <strong>⊞ Multi</strong> mode ở sidebar và chọn SKUs.</div>',
            unsafe_allow_html=True,
        )
    else:
        # Show pills grid
        pills = ""
        for nm in sel_names:
            short = nm if len(nm) <= 30 else nm[:28] + "…"
            pills += (
                f'<div style="background:#1a1f35;border:1px solid #A78BFA40;'
                f'border-radius:6px;padding:6px 10px;font-size:11px;'
                f'font-family:\'IBM Plex Mono\',monospace;color:#C4B5FD;">'
                f'{short}</div>'
            )
        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));'
            f'gap:6px;margin:4px 0 12px 0;">{pills}</div>',
            unsafe_allow_html=True,
        )

    # ── Params display ────────────────────────────────────────
    st.markdown(
        f'<div class="info-box" style="margin-bottom:8px">'
        f'<strong>Params</strong> &nbsp;·&nbsp; '
        f'α [{params["alpha_min"]:.2f}–{params["alpha_max"]:.2f}] &nbsp;'
        f'β [{params["beta_min"]:.2f}–{params["beta_max"]:.2f}] &nbsp;'
        f'γ [{params["gamma_min"]:.2f}–{params["gamma_max"]:.2f}] &nbsp;'
        f'| Step <strong>{step}</strong> | Horizon <strong>{h_steps}M</strong><br>'
        f'<span style="font-size:10px;color:#64748B">Điều chỉnh params ở sidebar · '
        f'Nhấn <strong>▶ RUN {n_sel} SKUs</strong> ở sidebar để chạy</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Detect RUN trigger from sidebar ──────────────────────
    current_trigger = st.session_state.get("_multi_run_trigger", 0)
    last_trigger    = st.session_state.get("_multi_last_trigger", 0)

    if current_trigger > last_trigger and n_sel > 0:
        st.session_state["_multi_last_trigger"] = current_trigger
        selected_skus = [s for s in skus if s["name"] in sel_names]
        _run_multi(selected_skus, params, step, h_steps, top_n)

    # ── Show results if available ─────────────────────────────
    if "multi_results" in st.session_state:
        _render_results()


# ══════════════════════════════════════════════════════════════
# RUN ENGINE
# ══════════════════════════════════════════════════════════════

def _run_multi(selected_skus, params, step, h_steps, top_n):
    n          = len(selected_skus)
    results_all = []
    t0          = time.time()

    # ── Progress UI ───────────────────────────────────────────
    st.markdown(
        '<div style="font-size:10px;color:#64748B;letter-spacing:1px;'
        'text-transform:uppercase;font-family:\'IBM Plex Mono\',monospace;'
        'margin:8px 0 6px 0">RUNNING FORECAST</div>',
        unsafe_allow_html=True,
    )

    pbar        = st.progress(0)
    status_box  = st.empty()
    done_box    = st.empty()

    for i, sku in enumerate(selected_skus):
        name = sku["name"]
        pct  = i / n

        # Progress bar
        pbar.progress(pct)

        # Status card
        status_box.markdown(
            f'<div style="background:#1E2130;border:1px solid #A78BFA40;border-left:3px solid #A78BFA;'
            f'border-radius:6px;padding:10px 14px;margin:4px 0;">'
            f'<div style="font-size:9px;color:#64748B;letter-spacing:1px;'
            f'font-family:\'IBM Plex Mono\',monospace;text-transform:uppercase;margin-bottom:4px">'
            f'Processing SKU {i+1} / {n}</div>'
            f'<div style="font-size:12px;font-family:\'IBM Plex Mono\',monospace;color:#C4B5FD;">'
            f'{name}</div>'
            f'<div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap">'
            + "".join(
                f'<span style="background:{"#10B981" if j < i else "#1E2130"};'
                f'border:1px solid {"#10B981" if j < i else "#2D3250"};'
                f'border-radius:3px;width:14px;height:6px;display:inline-block;"></span>'
                for j in range(n)
            )
            + f'</div></div>',
            unsafe_allow_html=True,
        )

        try:
            res  = grid_search(
                sku["history"], params, h_steps=h_steps, period=12, step=step,
                verbose=False, hist_dates=sku["dates"],
                yoy_anchor_extra=sku.get("yoy_anchor_extra"),
            )
            top  = select_best(res, sku["history"], top_n=top_n)
            best = top[0]
            fc_d = _fc_dates(sku["dates"], h_steps)
            all_v = sku["history"] + list(best["forecast"])
            all_d = list(sku["dates"]) + fc_d
            fc_yr_vol = {
                yr: v for yr, v in _annual_vol(all_v, all_d).items()
                if yr > sku["dates"][-1].year
            }
            results_all.append({
                "name":       name,
                "ppg":        sku.get("ppg",     name.split("|")[0].strip()),
                "channel":    sku.get("channel", "—"),
                "history":    sku["history"],
                "hist_dates": sku["dates"],
                "fc_dates":   fc_d,
                "best":       best,
                "fc_year_vol":fc_yr_vol,
                "n_combos":   len(res),
                "error":      None,
            })

        except Exception as e:
            results_all.append({
                "name":    name,
                "ppg":     sku.get("ppg", name),
                "channel": sku.get("channel", "—"),
                "best":    None,
                "error":   str(e),
            })

    elapsed = time.time() - t0
    ok      = sum(1 for r in results_all if not r["error"])

    # ── Final state ───────────────────────────────────────────
    pbar.progress(1.0)
    status_box.empty()
    done_box.markdown(
        f'<div style="background:#0d2318;border:1px solid #10B98160;border-left:3px solid #10B981;'
        f'border-radius:6px;padding:10px 14px;">'
        f'<div style="font-size:11px;font-family:\'IBM Plex Mono\',monospace;color:#10B981;">'
        f'✓ {ok}/{n} SKUs completed in {elapsed:.1f}s</div>'
        f'<div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap">'
        + "".join(
            f'<span style="background:#10B981;border:1px solid #10B981;'
            f'border-radius:3px;width:14px;height:6px;display:inline-block;"></span>'
            for _ in range(n)
        )
        + f'</div></div>',
        unsafe_allow_html=True,
    )

    st.session_state["multi_results"]      = results_all
    st.session_state["multi_h_steps"]      = h_steps
    st.session_state.pop("excel_bytes", None)  # clear stale export


# ══════════════════════════════════════════════════════════════
# RESULTS SECTION
# ══════════════════════════════════════════════════════════════

def _render_results():
    results_all = st.session_state["multi_results"]
    ok_results  = [r for r in results_all if not r.get("error")]
    err_results = [r for r in results_all if r.get("error")]

    st.markdown(section("Results Overview"), unsafe_allow_html=True)

    fc_start_yr = (
        min(d.year for r in ok_results for d in r["fc_dates"])
        if ok_results else datetime.now().year
    )

    # KPI row
    model_dist = defaultdict(int)
    for r in ok_results:
        model_dist[r["best"]["model"]] += 1
    top_model = max(model_dist, key=model_dist.get) if model_dist else "—"

    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.markdown(metric_card_html("SKUs Run",    str(len(results_all)), "total",  C_HIST), unsafe_allow_html=True)
    kc2.markdown(metric_card_html("Success",     str(len(ok_results)),  "OK",     C_TS),   unsafe_allow_html=True)
    kc3.markdown(metric_card_html("Errors",      str(len(err_results)), "failed",
                 C_WARN if err_results else C_TS), unsafe_allow_html=True)
    kc4.markdown(metric_card_html(
        "Most Selected",
        MODEL_ICONS.get(top_model, "") + " " + MODEL_LABELS.get(top_model, top_model),
        f"{model_dist.get(top_model, 0)} SKUs",
        MODEL_COLORS.get(top_model, C_MUTED),
    ), unsafe_allow_html=True)

    # Scorecard table
    st.markdown(section("Model Scorecard"), unsafe_allow_html=True)
    rows = []
    for r in ok_results:
        best = r["best"]
        sb   = best.get("score_breakdown", {})
        fc_vol = r.get("fc_year_vol", {}).get(fc_start_yr)
        fc_yoy = sb.get("yoy_forecast", {}).get(fc_start_yr)
        anchor = sb.get("yoy_anchor",   {}).get(fc_start_yr)
        rows.append({
            "SKU":                      r["name"],
            "Model":                    MODEL_ICONS.get(best["model"], "") + " " + MODEL_LABELS.get(best["model"], best["model"]),
            "α":                        f"{best['alpha']:.2f}",
            "β":                        f"{best['beta']:.2f}",
            "γ":                        f"{best['gamma']:.2f}",
            "MAPE":                     _mstr(best["mape"]),
            "Score":                    f"{best['score']:.4f}",
            f"FC {fc_start_yr} (MVND)": f"{fc_vol/1e6:,.2f}" if fc_vol else "—",
            f"YoY {fc_start_yr}%":      f"{fc_yoy:+.1f}%" if fc_yoy is not None else "—",
            "Anchor%":                  f"{anchor:+.1f}%" if anchor is not None else "—",
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True,
                     height=min(600, len(rows) * 36 + 40),
                     key="multi_scorecard_df")

    if err_results:
        with st.expander(f"⚠️ {len(err_results)} error(s)", expanded=True):
            for r in err_results:
                st.markdown(
                    f'<div class="warn-box">❌ <strong>{r["name"]}</strong>: {r["error"]}</div>',
                    unsafe_allow_html=True,
                )

    # Export
    st.markdown(section("Export to Excel"), unsafe_allow_html=True)
    if not ok_results:
        st.markdown('<div class="warn-box">⚠️ Không có kết quả để export.</div>', unsafe_allow_html=True)
        return

    col_btn, col_info = st.columns([2, 3])
    with col_btn:
        if st.button("📥  Generate Excel", key="gen_excel_btn", type="primary"):
            with st.spinner("Building Excel..."):
                try:
                    hist_end    = ok_results[0]["hist_dates"][-1]
                    excel_bytes = export_multi_sku(ok_results, hist_end)
                    ts          = datetime.now().strftime("%Y%m%d_%H%M")
                    st.session_state["excel_bytes"]    = excel_bytes
                    st.session_state["excel_filename"] = f"AFC_Baseline_{ts}.xlsx"
                    st.success("Excel ready ✓")
                except Exception as e:
                    st.error(f"Export error: {e}")

    with col_info:
        st.markdown(
            '<div class="info-box">'
            f'Sheets: <strong>Overview</strong> · <strong>Forecast_Detail</strong> · '
            f'<strong>{len(ok_results)} SKU sheets</strong><br>'
            '<span style="font-size:11px;color:#64748B">'
            'Values: Mil VND · Growth: YoY / QoQ / MoM %</span></div>',
            unsafe_allow_html=True,
        )

    if "excel_bytes" in st.session_state:
        st.download_button(
            label="⬇️  Download Excel",
            data=st.session_state["excel_bytes"],
            file_name=st.session_state.get("excel_filename", "AFC_Forecast.xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_excel_btn",
        )