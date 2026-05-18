"""
sidebar.py — File upload, parameter controls, run button
Mondelez Vietnam / AFC Baseline Forecast App
"""

import os
import tempfile
from datetime import datetime
from dateutil.relativedelta import relativedelta
import streamlit as st
import pandas as pd

from baseline_forecast_tool import (
    read_single_sku_xlsm, read_multi_sku_excel, read_csv, read_bbl_ppg,
    estimate_grid_size, grid_search, select_best,
)
from config import CSS


# ── BBL PPG format detector ───────────────────────────────────
def _is_bbl_ppg(path: str) -> bool:
    try:
        xl = pd.ExcelFile(path)
        return "BBL PPG" in xl.sheet_names
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════
# SIDEBAR BUILDER
# ══════════════════════════════════════════════════════════════

def build_sidebar():
    st.markdown(CSS, unsafe_allow_html=True)

    # Extra CSS for pill badges + mode toggle
    st.markdown("""
    <style>
    /* ── Mode toggle pill ── */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        padding: 0 3px;
    }
    .mode-btn-active {
        background: #1D3461; color: #4A9EFF; border: 1px solid #4A9EFF;
        border-radius: 6px; padding: 6px 10px; text-align: center;
        font-size: 11px; font-weight: 700; cursor: default;
        font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.5px;
    }
    .mode-btn-inactive {
        background: #1E2130; color: #64748B; border: 1px solid #2D3250;
        border-radius: 6px; padding: 6px 10px; text-align: center;
        font-size: 11px; cursor: default;
        font-family: 'IBM Plex Mono', monospace;
    }
    /* ── SKU pill list ── */
    .sku-pills { display: flex; flex-wrap: wrap; gap: 4px; margin: 6px 0 4px 0; }
    .sku-pill {
        background: #1a2640; border: 1px solid #2563EB60; color: #7CB9FF;
        border-radius: 100px; padding: 2px 9px;
        font-size: 10px; font-family: 'IBM Plex Mono', monospace;
        white-space: nowrap; max-width: 140px;
        overflow: hidden; text-overflow: ellipsis;
    }
    .sku-pill-multi {
        background: #1a1f35; border: 1px solid #A78BFA60; color: #C4B5FD;
        border-radius: 100px; padding: 2px 9px;
        font-size: 10px; font-family: 'IBM Plex Mono', monospace;
        white-space: nowrap; max-width: 140px;
        overflow: hidden; text-overflow: ellipsis;
    }
    .sku-count-badge {
        display: inline-block; background: #A78BFA22; border: 1px solid #A78BFA60;
        color: #C4B5FD; border-radius: 4px; padding: 1px 8px;
        font-size: 10px; font-family: 'IBM Plex Mono', monospace; font-weight: 700;
    }
    /* ── Tighten sidebar number inputs ── */
    [data-testid="stSidebar"] div[data-testid="stNumberInput"] input {
        font-size: 12px; padding: 4px 6px;
    }
    /* ── Progress label ── */
    .multi-progress-wrap {
        background: #1E2130; border: 1px solid #2D3250;
        border-radius: 8px; padding: 12px 14px; margin: 8px 0;
    }
    .multi-progress-wrap .mp-label {
        font-size: 10px; font-family: 'IBM Plex Mono', monospace;
        color: #64748B; letter-spacing: 1px; text-transform: uppercase;
        margin-bottom: 6px;
    }
    .multi-progress-wrap .mp-sku {
        font-size: 11px; font-family: 'IBM Plex Mono', monospace;
        color: #A78BFA; font-weight: 600;
    }
    .multi-progress-wrap .mp-done {
        font-size: 11px; color: #10B981; font-family: 'IBM Plex Mono', monospace;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:

        # ── Logo / title ──────────────────────────────────────
        st.markdown("""
        <div style="padding:12px 0 4px 0;">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:13px;
                        font-weight:700; color:#4A9EFF; letter-spacing:1px;">
                AFC BASELINE FORECAST
            </div>
            <div style="font-size:10px; color:#64748B; letter-spacing:1px;">
                MONDELEZ VIETNAM
            </div>
        </div>
        <hr style="border:none; border-top:1px solid #2D3250; margin:10px 0 16px 0;">
        """, unsafe_allow_html=True)

        # ── FILE UPLOAD ──────────────────────────────────────
        st.markdown("### 📂 DATA INPUT")
        uploaded = st.file_uploader(
            "Upload file",
            type=["xlsm", "xlsx", "csv"],
            help=(
                "**xlsx BBL PPG** — file AFC Mondelez (auto-detected)\n\n"
                "**xlsm** — file gốc Baseline_Tool (single SKU)\n\n"
                "**xlsx** — multi-SKU (wide-table hoặc multi-sheet)\n\n"
                "**csv** — cột date (YYYY-MM) + qty [+ sku]"
            ),
            label_visibility="collapsed",
        )

        if uploaded:
            suffix = "." + uploaded.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                if suffix == ".csv":
                    skus, file_params = read_csv(tmp_path)
                    st.session_state.pop("bbl_meta", None)

                elif suffix == ".xlsm":
                    skus, file_params = read_single_sku_xlsm(tmp_path)
                    st.session_state.pop("bbl_meta", None)

                elif _is_bbl_ppg(tmp_path):
                    hist_start = st.session_state.get("bbl_hist_start", datetime(2023, 5, 1))
                    hist_end   = st.session_state.get("bbl_hist_end",   datetime(2026, 4, 1))
                    skus, file_params, bbl_meta = read_bbl_ppg(
                        tmp_path,
                        history_start=hist_start,
                        history_end=hist_end,
                    )
                    st.session_state["bbl_meta"] = bbl_meta

                else:
                    skus, file_params = read_multi_sku_excel(tmp_path)
                    st.session_state.pop("bbl_meta", None)

                st.session_state["skus"]        = skus
                st.session_state["file_params"] = file_params
                # Reset selections on new file
                st.session_state.pop("_multi_sel", None)
                st.session_state.pop("multi_results", None)
                st.success(f"✓  {uploaded.name}  —  {len(skus)} SKU(s)")

            except Exception as e:
                st.error(f"Lỗi đọc file:\n{e}")
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        # ════════════════════════════════════════════════════
        # SKU SELECTOR SECTION
        # ════════════════════════════════════════════════════
        sku_index      = 0
        multi_selected = []
        run_mode       = "Single SKU"

        if "skus" in st.session_state:
            names = [s["name"] for s in st.session_state["skus"]]

            if len(names) == 1:
                # ── Only 1 SKU ────────────────────────────────
                run_mode = "Single SKU"
                st.session_state["run_mode"] = run_mode
                st.markdown(
                    f'<div style="margin:10px 0 4px 0;">'
                    f'<div class="sku-pill">{names[0]}</div></div>',
                    unsafe_allow_html=True,
                )

            else:
                # ── Mode selector ─────────────────────────────
                st.markdown(
                    '<div style="font-size:10px;color:#64748B;letter-spacing:1px;'
                    'text-transform:uppercase;margin:12px 0 8px 0;'
                    'font-family:\'IBM Plex Mono\',monospace;">RUN MODE</div>',
                    unsafe_allow_html=True,
                )

                c_s, c_m = st.columns(2)
                with c_s:
                    if st.button(
                        "① Single",
                        key="mode_btn_single",
                        use_container_width=True,
                        type="primary" if st.session_state.get("run_mode") != "Multi-SKU" else "secondary",
                    ):
                        st.session_state["run_mode"] = "Single SKU"
                        st.rerun()
                with c_m:
                    if st.button(
                        "⊞ Multi",
                        key="mode_btn_multi",
                        use_container_width=True,
                        type="primary" if st.session_state.get("run_mode") == "Multi-SKU" else "secondary",
                    ):
                        st.session_state["run_mode"] = "Multi-SKU"
                        st.rerun()

                run_mode = st.session_state.get("run_mode", "Single SKU")

                # ── SINGLE MODE ───────────────────────────────
                if run_mode == "Single SKU":
                    sel = st.selectbox(
                        "PPG × Channel",
                        names,
                        key="sku_select_single",
                        label_visibility="collapsed",
                    )
                    sku_index = names.index(sel)
                    # Show selected pill
                    short = sel if len(sel) <= 28 else sel[:26] + "…"
                    st.markdown(
                        f'<div class="sku-pills"><span class="sku-pill">✓ {short}</span></div>'
                        f'<div style="font-size:10px;color:#64748B;margin-bottom:4px">'
                        f'Kết quả → tabs Analysis / Model / Summary</div>',
                        unsafe_allow_html=True,
                    )

                # ── MULTI MODE ────────────────────────────────
                else:
                    # Persistent selection stored in session
                    current_sel = st.session_state.get("_multi_sel", names[:min(6, len(names))])
                    # Filter out names that no longer exist in file
                    current_sel = [n for n in current_sel if n in names]

                    # Header row: count badge + All / Clear buttons
                    # Use live widget value for count
                    live_sel    = st.session_state.get("sb_multi_widget", current_sel)
                    n_total     = len(names)
                    n_sel       = len(live_sel)
                    badge_color = "#10B981" if n_sel > 0 else "#EF4444"

                    hc1, hc2, hc3 = st.columns([3, 1, 1])
                    with hc1:
                        st.markdown(
                            f'<div style="margin-top:6px;">'
                            f'<span class="sku-count-badge">{n_sel} / {n_total} selected</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    with hc2:
                        if st.button("All", key="sb_all", use_container_width=True):
                            st.session_state["_multi_sel"] = names[:]
                            st.rerun()
                    with hc3:
                        if st.button("✕", key="sb_clr", use_container_width=True):
                            st.session_state["_multi_sel"] = []
                            st.rerun()

                    # Multiselect — compact, no label
                    # NOTE: do NOT rerun on change — Streamlit keeps widget value in sync.
                    # Read directly from widget key after render.
                    st.multiselect(
                        "multi_sel_widget",
                        options=names,
                        default=current_sel,
                        key="sb_multi_widget",
                        label_visibility="collapsed",
                        placeholder="Chọn PPG × Channel…",
                    )
                    # Always read live value from widget (not from _multi_sel cache)
                    multi_selected = st.session_state.get("sb_multi_widget", current_sel)
                    # Keep _multi_sel in sync for persistence across reruns
                    st.session_state["_multi_sel"] = multi_selected

                    # Pill preview (show first 5, then "+N more")
                    if multi_selected:
                        pills_html = '<div class="sku-pills">'
                        show_max   = 5
                        for nm in multi_selected[:show_max]:
                            short = nm if len(nm) <= 22 else nm[:20] + "…"
                            pills_html += f'<span class="sku-pill-multi">{short}</span>'
                        extra = len(multi_selected) - show_max
                        if extra > 0:
                            pills_html += (
                                f'<span class="sku-pill-multi" '
                                f'style="background:#2D2040;color:#A78BFA;">'
                                f'+{extra} more</span>'
                            )
                        pills_html += "</div>"
                        st.markdown(pills_html, unsafe_allow_html=True)
                        st.markdown(
                            f'<div style="font-size:10px;color:#64748B;margin-bottom:4px">'
                            f'Kết quả → tab 📦 Multi-SKU</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div style="font-size:10px;color:#EF4444;margin:4px 0 6px 0">'
                            '⚠️ Chọn ít nhất 1 SKU</div>',
                            unsafe_allow_html=True,
                        )

        # ── BBL PPG HISTORY WINDOW ────────────────────────────
        if "bbl_meta" in st.session_state:
            m = st.session_state["bbl_meta"]
            st.markdown("### 📅 HISTORY WINDOW")

            st.markdown(
                '<div style="font-size:10px;color:#94A3B8;letter-spacing:1px;'
                'text-transform:uppercase;margin-bottom:4px">History End</div>',
                unsafe_allow_html=True,
            )
            col_ey, col_em = st.columns(2)
            with col_ey:
                he_yr = st.number_input(
                    "End Year", 2020, 2030, m["history_end"].year,
                    key="bbl_he_yr", label_visibility="collapsed",
                )
            with col_em:
                he_mo = st.number_input(
                    "End Month", 1, 12, m["history_end"].month,
                    key="bbl_he_mo", label_visibility="collapsed",
                )
            hist_end = datetime(int(he_yr), int(he_mo), 1)

            suggested_start = hist_end - relativedelta(months=35)

            st.markdown(
                '<div style="font-size:10px;color:#94A3B8;letter-spacing:1px;'
                'text-transform:uppercase;margin:8px 0 4px 0">'
                'History Start '
                '<span style="color:#4A9EFF;font-size:9px;text-transform:none">'
                '(auto 36M)</span></div>',
                unsafe_allow_html=True,
            )
            col_sy, col_sm = st.columns(2)
            with col_sy:
                hs_yr = st.number_input(
                    "Start Year", 2020, 2030, suggested_start.year,
                    key="bbl_hs_yr", label_visibility="collapsed",
                )
            with col_sm:
                hs_mo = st.number_input(
                    "Start Month", 1, 12, suggested_start.month,
                    key="bbl_hs_mo", label_visibility="collapsed",
                )
            hist_start = datetime(int(hs_yr), int(hs_mo), 1)

            n_months = (
                (hist_end.year - hist_start.year) * 12
                + (hist_end.month - hist_start.month) + 1
            )
            if n_months == 36:
                box_cls, note = "success-box", "Đủ 36M ✓"
            elif n_months < 36:
                box_cls, note = "warn-box", f"Thiếu {36 - n_months}M — kéo Start về trước"
            else:
                box_cls, note = "info-box", f"Dư {n_months - 36}M — có thể giảm bớt"

            st.markdown(
                f'<div class="{box_cls}" style="margin-top:6px">'
                f'<strong>{hist_start.strftime("%b %Y")} → '
                f'{hist_end.strftime("%b %Y")} ({n_months}M)</strong>'
                f'<br><span style="font-size:11px">{note}</span></div>',
                unsafe_allow_html=True,
            )

            st.session_state["bbl_hist_start"] = hist_start
            st.session_state["bbl_hist_end"]   = hist_end

        # ── PARAMETERS ───────────────────────────────────────
        st.markdown("### ⚙️ PARAMETERS")

        fp = st.session_state.get("file_params", {
            "alpha_min": 0.01, "alpha_max": 0.30,
            "beta_min":  0.01, "beta_max":  0.30,
            "gamma_min": 0.01, "gamma_max": 0.99,
        })

        use_full = st.toggle(
            "Full range  α,β,γ  →  0.01–0.99",
            value=False,
            help="Bật để thử toàn bộ tham số. Kết hợp step >= 0.05 để tránh quá chậm.",
        )

        if use_full:
            a_min = b_min = g_min = 0.01
            a_max = b_max = g_max = 0.99
        else:
            c1, c2 = st.columns(2)
            with c1:
                a_min = st.number_input("α min", 0.01, 0.99, float(fp.get("alpha_min", 0.01)), 0.01, format="%.2f", key="amin")
                b_min = st.number_input("β min", 0.01, 0.99, float(fp.get("beta_min",  0.01)), 0.01, format="%.2f", key="bmin")
                g_min = st.number_input("γ min", 0.01, 0.99, float(fp.get("gamma_min", 0.01)), 0.01, format="%.2f", key="gmin")
            with c2:
                a_max = st.number_input("α max", 0.01, 0.99, float(fp.get("alpha_max", 0.30)), 0.01, format="%.2f", key="amax")
                b_max = st.number_input("β max", 0.01, 0.99, float(fp.get("beta_max",  0.30)), 0.01, format="%.2f", key="bmax")
                g_max = st.number_input("γ max", 0.01, 0.99, float(fp.get("gamma_max", 0.99)), 0.01, format="%.2f", key="gmax")

        params = {
            "alpha_min": a_min, "alpha_max": a_max,
            "beta_min":  b_min, "beta_max":  b_max,
            "gamma_min": g_min, "gamma_max": g_max,
        }

        # Grid step
        step = st.select_slider(
            "Grid step  (fine ↔ fast)",
            options=[0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.10, 0.20],
            value=0.03,
            help=(
                "**0.005** — cực mịn, rất chậm\n\n"
                "**0.01–0.025** — fine, phù hợp khi cần params chính xác\n\n"
                "**0.03** — balance tốt, recommend ✓\n\n"
                "**0.05** — nhanh, đủ tốt cho daily run\n\n"
                "**0.10+** — explore nhanh, draft run"
            ),
            key="grid_step",
        )

        # Combo counter
        est = estimate_grid_size(params, step)
        est_sec = max(1, int(est["total"] * 0.00042))
        st.markdown(f"""
        <div class="combo-box">
          Combinations: <span>{est['total']:,}</span><br>
          TS={est['trend_seasonal']:,} · S={est['seasonal']:,}<br>
          T={est['trend']:,} · C={est['constant']:,}<br>
          Est. time per SKU: ~{est_sec}s
        </div>
        """, unsafe_allow_html=True)

        # ── FORECAST SETTINGS ─────────────────────────────────
        st.markdown("### 📅 FORECAST")
        h_steps = st.slider("Months ahead", 12, 36, 24, 1, key="h_steps")
        top_n   = st.slider("Top N models", 3, 20, 10, 1, key="top_n")

        # ── DISPLAY ───────────────────────────────────────────
        st.markdown("### 🖥️ DISPLAY")
        show_expost = st.toggle("Show fitted values", value=True, key="show_expost")

        # ── RUN BUTTON ────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)

        _mode_now  = st.session_state.get("run_mode", "Single SKU")
        _no_skus   = "skus" not in st.session_state
        _is_multi  = _mode_now == "Multi-SKU"
        # Read live from widget if in multi mode, else fall back to cached value
        _multi_sel = st.session_state.get("sb_multi_widget") or st.session_state.get("_multi_sel", [])
        _has_multi = len(_multi_sel) > 0

        if not _is_multi:
            # ── Single SKU RUN ────────────────────────────────
            run_clicked = st.button(
                "▶  RUN FORECAST",
                use_container_width=True,
                type="primary",
                disabled=_no_skus,
                key="run_btn",
            )
        else:
            # ── Multi-SKU RUN ─────────────────────────────────
            run_clicked = False
            run_multi_clicked = st.button(
                f"▶  RUN {len(_multi_sel)} SKUs" if _has_multi else "▶  RUN Multi-SKU",
                use_container_width=True,
                type="primary",
                disabled=_no_skus or not _has_multi,
                key="run_multi_sidebar_btn",
            )
            # Store trigger so page_multi can detect it
            if run_multi_clicked:
                st.session_state["_multi_run_trigger"] = st.session_state.get("_multi_run_trigger", 0) + 1

        if est["total"] > 500_000:
            st.markdown(
                '<div class="warn-box">⚠️ >500k combos — có thể mất vài phút. '
                "Tăng step để chạy nhanh hơn.</div>",
                unsafe_allow_html=True,
            )

    run_mode_val = st.session_state.get("run_mode", "Single SKU")

    return dict(
        params=params,
        step=step,
        h_steps=h_steps,
        top_n=top_n,
        show_expost=show_expost,
        sku_index=sku_index,
        run_clicked=run_clicked if not _is_multi else False,
        run_mode=run_mode_val,
        multi_selected=multi_selected,
    )


# ══════════════════════════════════════════════════════════════
# RUN COMPUTATION
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def run_grid_search(history_tuple, params_tuple, h_steps, period, step,
                    hist_dates_tuple=None, yoy_anchor_extra_tuple=None):
    """Cached grid search — re-runs only when inputs change."""
    history  = list(history_tuple)
    keys     = ["alpha_min","alpha_max","beta_min","beta_max","gamma_min","gamma_max"]
    params   = dict(zip(keys, params_tuple))
    hist_dates       = list(hist_dates_tuple) if hist_dates_tuple else None
    yoy_anchor_extra = list(yoy_anchor_extra_tuple) if yoy_anchor_extra_tuple else None
    return grid_search(
        history, params, h_steps, period, step,
        verbose=False,
        hist_dates=hist_dates,
        yoy_anchor_extra=yoy_anchor_extra,
    )


def execute_forecast(sidebar_state):
    """Run grid search for single SKU & store results in st.session_state."""
    if "skus" not in st.session_state:
        return

    sku_data         = st.session_state["skus"][sidebar_state["sku_index"]]
    history          = sku_data["history"]
    hist_dates       = sku_data["dates"]
    params           = sidebar_state["params"]
    step             = sidebar_state["step"]
    h_steps          = sidebar_state["h_steps"]
    top_n            = sidebar_state["top_n"]
    yoy_anchor_extra = sku_data.get("yoy_anchor_extra", None)

    est = estimate_grid_size(params, step)
    with st.spinner(f"Running grid search — {est['total']:,} combinations…"):
        results = run_grid_search(
            tuple(history),
            tuple(params[k] for k in
                  ["alpha_min","alpha_max","beta_min","beta_max","gamma_min","gamma_max"]),
            h_steps, 12, step,
            hist_dates_tuple=tuple(hist_dates),
            yoy_anchor_extra_tuple=tuple(
                (d, v) for d, v in yoy_anchor_extra
            ) if yoy_anchor_extra else None,
        )
        top = select_best(results, history, top_n)

    st.session_state.update({
        "results":     results,
        "top":         top,
        "history":     history,
        "hist_dates":  hist_dates,
        "sku_name":    sku_data["name"],
        "_fc_params":  params,
        "_fc_step":    step,
        "_fc_h_steps": h_steps,
        "_fc_top_n":   top_n,
    })
    st.success(f"✓ Done — {len(results):,} combinations evaluated.")