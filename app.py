"""
S.KOe COOL -- V2G Optimisation  |  Streamlit App (v7)
TU Dortmund IE3 x Schmitz Cargobull AG | 2026
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from io import BytesIO
from pathlib import Path

from v2g_single_day4 import (
    V2GParams, WINTER_M, SUMMER_M, SC_COL, SC_FILL,
    FIXED_PRICE_EUR_KWH,
    _interpolate_to_15min, _load_csv_raw,
    get_tru_15min_trace, tru_avg_kw,
    compute_reefer_costs,
    yearly_extrapolation,
    get_wd_window, build_wd_display,
    run_A_dumb, run_B_smart, run_C_milp, run_D_mpc,
    make_kpi, plot_kpi_multi, plot_price_profiles,
    soc_ramp,
)

CSV_PATH = "2025_Electricity_Price.csv"

st.set_page_config(
    page_title="S.KOe COOL -- V2G Optimisation",
    page_icon="⚡",
    layout="wide",
)


# =============================================================================
#  HELPERS
# =============================================================================

def parse_hhmm(s: str, default: float) -> float:
    try:
        parts = s.strip().split(":")
        h = int(parts[0]); m = int(parts[1]) if len(parts) > 1 else 0
        assert 0 <= h <= 23 and 0 <= m <= 59
        return h + m / 60.0
    except Exception:
        return default


def fmt_hhmm(h: float) -> str:
    return f"{int(h):02d}:{int((h % 1) * 60):02d}"


def fig_to_buf(fig) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# =============================================================================
#  AXIS SETUP HELPERS
# =============================================================================

def _setup_xaxis(ax, is_48h, is_wknd_fullday=False):
    """Configure x-axis ticks for overnight weekday or 48h weekend."""
    if is_48h:
        pos  = np.arange(0, 49, 4)
        lbls = [f"{'Sat' if h < 24 else 'Sun'}\n{int(h%24):02d}:00" for h in pos]
        ax.set_xlim(0, 48)
    elif is_wknd_fullday:
        pos  = np.arange(0, 25, 2)
        lbls = [f"{int(h):02d}:00" for h in pos]
        ax.set_xlim(0, 24)
    else:
        pos  = np.arange(12, 37, 2)
        lbls = [f"{int(h%24):02d}:00" for h in pos]
        ax.set_xlim(12, 36)
    ax.set_xticks(pos)
    ax.set_xticklabels(lbls, fontsize=6, rotation=30, ha="right")
    ax.grid(True, alpha=0.20, zorder=0)


def _vlines(ax, arrival_h, departure_h, is_48h, is_wknd_fullday=False):
    """Add vertical reference lines for arrival, departure, midnight."""
    if is_48h:
        ax.axvline(24, color="#555", lw=1.0, ls="--", alpha=0.55, zorder=6)
        return
    if is_wknd_fullday:
        return
    def dx(h): return h if h >= 12. else h + 24.
    ax.axvline(dx(arrival_h),   color="#1B5E20", lw=1.0, ls=":", alpha=0.80, zorder=6)
    ax.axvline(dx(departure_h), color="#B71C1C", lw=1.0, ls=":", alpha=0.80, zorder=6)
    ax.axvline(dx(0.),          color="#555",    lw=1.0, ls="--", alpha=0.55, zorder=6)


# =============================================================================
#  PAIRWISE POWER CHART
#  Dumb (grey) vs one scenario (coloured)
#  Left Y-axis: kW  |  Right Y-axis: EUR/MWh
# =============================================================================

def make_power_chart(v2g, hours_d, buy_d, plug_d,
                     result_A, result_X,
                     x_label, x_key,
                     arrival_h, departure_h,
                     is_48h, is_wknd_fullday=False,
                     tru_d=None):

    col_a  = SC_COL["A"];   fill_a  = SC_FILL["A"]
    col_x  = SC_COL[x_key]; fill_x  = SC_FILL[x_key]
    lbl_x  = result_X["label"].split("(")[0].strip()

    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#FFFFFF")

    # Plugged-in background shading
    for t in range(len(hours_d)):
        if plug_d[t] > 0.5:
            ax.axvspan(hours_d[t], hours_d[t] + 0.25,
                       color="gold", alpha=0.14, lw=0, zorder=1)

    # Dumb charging — grey fill + line (always positive)
    ax.fill_between(hours_d, result_A["Pc_d"],
                    step="post", color=fill_a, alpha=0.55, zorder=2)
    h_a, = ax.step(hours_d, result_A["Pc_d"], where="post",
                   color=col_a, lw=1.8, zorder=3, label="A - Dumb charge")

    # Scenario X charging — coloured fill + line
    ax.fill_between(hours_d, result_X["Pc_d"],
                    step="post", color=fill_x, alpha=0.48, zorder=4)
    h_x, = ax.step(hours_d, result_X["Pc_d"], where="post",
                   color=col_x, lw=2.0, zorder=5, label=f"{lbl_x} charge")

    handles = [h_a, h_x]

    # V2G discharge — below zero, dashed
    if result_X["v2g_kwh"] > 0.05:
        ax.fill_between(hours_d, -result_X["Pd_d"],
                        step="post", color=fill_x, alpha=0.28, zorder=4)
        h_d, = ax.step(hours_d, -result_X["Pd_d"], where="post",
                       color=col_x, lw=2.0, ls="--", alpha=0.90, zorder=5,
                       label=f"{lbl_x} V2G (−)")
        handles.append(h_d)

    # TRU load — dotted red, below zero
    if tru_d is not None and np.any(tru_d > 0.01):
        h_t, = ax.step(hours_d, -tru_d, where="post",
                       color="#C62828", lw=1.2, ls=":", alpha=0.75, zorder=5,
                       label="TRU (−)")
        handles.append(h_t)

    ax.axhline(0, color="black", lw=0.6)
    _vlines(ax, arrival_h, departure_h, is_48h, is_wknd_fullday)

    # Right Y-axis: electricity price
    ax2 = ax.twinx()
    h_p, = ax2.step(hours_d, buy_d * 1000, where="post",
                    color="#2E7D32", lw=1.4, alpha=0.85, label="Price")
    ax2.fill_between(hours_d, buy_d * 1000,
                     step="post", color="#2E7D32", alpha=0.07)
    ax2.set_ylabel("EUR/MWh", fontsize=6, color="#2E7D32")
    ax2.tick_params(axis="y", labelcolor="#2E7D32", labelsize=5.5)
    ax2.set_ylim(bottom=min(0, (buy_d * 1000).min() - 5))
    handles.append(h_p)

    ax.set_ylabel("Power (kW)", fontsize=7)
    ax.set_title(f"Power — Dumb vs {x_label}",
                 fontsize=7.5, fontweight="bold", loc="left", pad=2)
    _setup_xaxis(ax, is_48h, is_wknd_fullday)

    ax.legend(handles=handles, fontsize=5.5, ncol=len(handles),
              loc="upper center", bbox_to_anchor=(0.5, -0.32),
              framealpha=0.92, edgecolor="#CCC", handlelength=1.0,
              borderpad=0.3, columnspacing=0.8)

    plt.tight_layout(pad=0.3)
    return fig_to_buf(fig)


# =============================================================================
#  PAIRWISE SOC CHART
#  Dumb (grey) vs one scenario (coloured)
# =============================================================================

def make_soc_chart(v2g, hours_d, plug_d,
                   result_A, result_X,
                   x_label, x_key,
                   arrival_h, departure_h,
                   is_48h, is_wknd_fullday=False):

    col_a = SC_COL["A"]
    col_x = SC_COL[x_key]
    lbl_x = result_X["label"].split("(")[0].strip()

    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#FFFFFF")

    # Plugged-in background
    for t in range(len(hours_d)):
        if plug_d[t] > 0.5:
            ax.axvspan(hours_d[t], hours_d[t] + 0.25,
                       color="gold", alpha=0.14, lw=0, zorder=1)

    # SoC ramp curves (linear interpolation, no step jumps)
    xA, yA = soc_ramp(hours_d, result_A["soc_d"], result_A["E_init_pct"])
    xX, yX = soc_ramp(hours_d, result_X["soc_d"], result_X["E_init_pct"])

    h_a, = ax.plot(xA, yA, color=col_a, lw=2.0, label="A - Dumb SoC")
    h_x, = ax.plot(xX, yX, color=col_x, lw=2.3, label=f"{lbl_x} SoC")

    # Cold-chain floor and departure target
    ax.axhline(v2g.soc_min_pct,       color="#C62828", ls=":", lw=1.2, zorder=3)
    ax.axhline(v2g.soc_departure_pct, color="#0D47A1", ls=":", lw=1.2, zorder=3)

    _vlines(ax, arrival_h, departure_h, is_48h, is_wknd_fullday)

    handles = [
        h_a, h_x,
        Line2D([0],[0], color="#C62828", ls=":", lw=1.2,
               label=f"Floor {v2g.soc_min_pct:.0f}%"),
        Line2D([0],[0], color="#0D47A1", ls=":", lw=1.2,
               label=f"Target {v2g.soc_departure_pct:.0f}%"),
        mpatches.Patch(color="gold", alpha=0.40, label="Plugged-in"),
    ]

    ax.set_ylabel("SoC (%)", fontsize=7)
    ax.set_ylim(0, 115)
    ax.set_title(f"SoC — Dumb vs {x_label}",
                 fontsize=7.5, fontweight="bold", loc="left", pad=2)
    _setup_xaxis(ax, is_48h, is_wknd_fullday)

    ax.legend(handles=handles, fontsize=5.5, ncol=5,
              loc="upper center", bbox_to_anchor=(0.5, -0.32),
              framealpha=0.92, edgecolor="#CCC", handlelength=1.0,
              borderpad=0.3, columnspacing=0.8)

    plt.tight_layout(pad=0.3)
    return fig_to_buf(fig)


# =============================================================================
#  RENDER ONE SEASON BLOCK
#  Layout: LEFT column = 3 power charts (vertical)
#          RIGHT column = 3 SoC charts (vertical)
#  Point 5: exactly as requested
# =============================================================================

def render_season_block(v2g, season_title, color_hex,
                         hours_d, buy_d, plug_d, results,
                         arrival_h, departure_h,
                         is_48h, is_wknd_fullday,
                         do_B, do_C, do_D, tru_d=None):
    """
    Renders 6 compact charts for one season:
      Left col  : Power charts — Dumb vs Smart / MILP / MPC (3 rows)
      Right col : SoC   charts — Dumb vs Smart / MILP / MPC (3 rows)
    """
    result_A = results[0]

    # Build list of (result, short_label, key) for each enabled scenario
    comparisons = []
    label_map   = {"B": "B - Smart", "C": "C - MILP", "D": "D - MPC"}
    sc_idx = 1
    for key, do_it in [("B", do_B), ("C", do_C), ("D", do_D)]:
        if do_it and sc_idx < len(results):
            comparisons.append((results[sc_idx], label_map[key], key))
            sc_idx += 1

    if not comparisons:
        st.info(f"{season_title}: enable at least one of B / C / D.")
        return

    # Coloured section header
    st.markdown(
        f"<div style='background:{color_hex};color:white;padding:5px 14px;"
        f"border-radius:5px;font-weight:bold;font-size:14px;margin-bottom:4px;'>"
        f"{season_title}</div>",
        unsafe_allow_html=True
    )

    col_pow, col_soc = st.columns(2)

    # LEFT: 3 power charts stacked
    with col_pow:
        st.caption("⚡ **Charge / Discharge Power**  "
                   "(left axis = kW  |  right axis = EUR/MWh price)")
        for result_X, x_label, x_key in comparisons:
            buf = make_power_chart(
                v2g, hours_d, buy_d, plug_d,
                result_A, result_X, x_label, x_key,
                arrival_h, departure_h, is_48h, is_wknd_fullday, tru_d
            )
            st.image(buf, use_container_width=True)

    # RIGHT: 3 SoC charts stacked
    with col_soc:
        st.caption("🔋 **Battery State of Charge (%)**")
        for result_X, x_label, x_key in comparisons:
            buf = make_soc_chart(
                v2g, hours_d, plug_d,
                result_A, result_X, x_label, x_key,
                arrival_h, departure_h, is_48h, is_wknd_fullday
            )
            st.image(buf, use_container_width=True)


# =============================================================================
#  CACHED DATA LOADERS
# =============================================================================

@st.cache_data(show_spinner=False)
def load_seasonal_profile(months: tuple, is_weekend: bool) -> np.ndarray:
    df   = _load_csv_raw(CSV_PATH)
    mask = df["month"].isin(list(months)) & (df["is_weekend"] == is_weekend)
    sub  = df[mask]
    if len(sub) == 0:
        raise ValueError(f"No data for months={months}, weekend={is_weekend}")
    profile = sub.groupby("slot")["price"].mean().values
    if len(profile) != 96:
        raise ValueError(f"Expected 96 slots, got {len(profile)}")
    return _interpolate_to_15min(profile)


@st.cache_data(show_spinner=False)
def load_date_profile(date_str: str) -> np.ndarray:
    """Load 96-slot price profile for a specific calendar date."""
    df     = _load_csv_raw(CSV_PATH)
    target = pd.Timestamp(date_str).date()
    day_df = df[df["date"] == target]
    if len(day_df) == 0:
        raise ValueError(
            f"No price data found for {date_str}. "
            "Check the date exists in your CSV."
        )
    prices = day_df["price"].values
    if len(prices) == 24:
        return _interpolate_to_15min(prices)
    if len(prices) != 96:
        raise ValueError(
            f"Expected 96 price slots for {date_str}, got {len(prices)}."
        )
    return _interpolate_to_15min(prices)



@st.cache_data(show_spinner=False)
def load_two_day_profile(date_str: str) -> np.ndarray:
    """
    Load 192-slot price array for the overnight window that starts on `date_str`
    and ends on the following calendar day.
    Slots 0–95   = prices for day D  (date_str)
    Slots 96–191 = prices for day D+1
    """
    day1 = load_date_profile(date_str)

    next_date_str = (
        pd.Timestamp(date_str) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    try:
        day2 = load_date_profile(next_date_str)
    except ValueError:
        raise ValueError(
            f"Next-day price data for **{next_date_str}** was not found in the "
            f"CSV file. Please select an earlier date so that the full "
            f"overnight window (arrival on {date_str} → departure on "
            f"{next_date_str}) is available."
        )

    return np.concatenate([day1, day2])

# =============================================================================
#  SCENARIO RUNNER — seasonal average
# =============================================================================

@st.cache_data(show_spinner=False)
def run_seasonal(season_key, arrival_h, departure_h,
                 soc_pct, soc_departure_pct, tru_cycle,
                 do_B, do_C, do_D):
    months_map = {
        "winter_weekday": (WINTER_M, False, False),
        "summer_weekday": (SUMMER_M, False, False),
        "winter_weekend": (WINTER_M, True,  True),
        "summer_weekend": (SUMMER_M, True,  True),
    }
    months, is_wknd, is_48h = months_map[season_key]
    buy  = load_seasonal_profile(tuple(months), is_wknd)
    v2gp = buy.copy()
    v2g  = V2GParams(soc_departure_pct=soc_departure_pct)
    E_init = v2g.usable_capacity_kWh * soc_pct / 100.0

    if is_48h:
        buy48   = np.concatenate([buy, buy])
        v2gp48  = buy48.copy()
        W       = 192
        tru_w   = get_tru_15min_trace(tru_cycle, W, v2g.dt_h)
        hours_d = np.arange(W) * v2g.dt_h
        plug_d  = np.ones(W)
        buy_d   = buy48

        Pc,Pd,soc = run_A_dumb(v2g,buy48,v2gp48,W,E_init,tru_w)
        results = [make_kpi("A - Dumb",v2g,Pc,Pd,soc,buy48,v2gp48,E_init,
                            is_weekend_48=True,tru_w=tru_w)]
        if do_B:
            Pc,Pd,soc = run_B_smart(v2g,buy48,v2gp48,E_init,tru_w)
            results.append(make_kpi("B - Smart (no V2G)",v2g,Pc,Pd,soc,
                                    buy48,v2gp48,E_init,is_weekend_48=True,tru_w=tru_w))
        if do_C:
            Pc,Pd,soc = run_C_milp(v2g,buy48,v2gp48,E_init,tru_w)
            results.append(make_kpi("C - MILP Day-Ahead",v2g,Pc,Pd,soc,
                                    buy48,v2gp48,E_init,is_weekend_48=True,tru_w=tru_w))
        if do_D:
            Pc,Pd,soc = run_D_mpc(v2g,buy48,v2gp48,E_init,tru_w)
            results.append(make_kpi("D - MPC (receding)",v2g,Pc,Pd,soc,
                                    buy48,v2gp48,E_init,is_weekend_48=True,tru_w=tru_w))

        results_kpi = [{**r,
            "net_cost":r["net_cost"]/2,"charge_cost":r["charge_cost"]/2,
            "v2g_rev":r["v2g_rev"]/2,"tru_cost":r["tru_cost"]/2,
            "total_cost":r["total_cost"]/2,"v2g_kwh":r["v2g_kwh"]/2,
            "charge_kwh":r["charge_kwh"]/2,
        } for r in results]
        rc = compute_reefer_costs(tru_w[:96], buy[:96], v2g.dt_h)
        return results, results_kpi, buy_d, plug_d, hours_d, is_wknd, is_48h, tru_w, rc

    else:
        win, arr, dep, W = get_wd_window(v2g, arrival_h, departure_h)
        buy_w  = buy[win]; v2gp_w = v2gp[win]
        tru_w  = get_tru_15min_trace(tru_cycle, W, v2g.dt_h)
        buy_d, plug_d, hours_d = build_wd_display(v2g, buy, arrival_h, departure_h)
        tru_d = np.zeros(v2g.n_slots); tru_d[arr:dep] = tru_w[:dep - arr]

        Pc,Pd,soc = run_A_dumb(v2g,buy_w,v2gp_w,W,E_init,tru_w)
        results = [make_kpi("A - Dumb",v2g,Pc,Pd,soc,buy_w,v2gp_w,E_init,arr,dep,tru_w=tru_w)]
        if do_B:
            Pc,Pd,soc = run_B_smart(v2g,buy_w,v2gp_w,E_init,tru_w)
            results.append(make_kpi("B - Smart (no V2G)",v2g,Pc,Pd,soc,
                                    buy_w,v2gp_w,E_init,arr,dep,tru_w=tru_w))
        if do_C:
            Pc,Pd,soc = run_C_milp(v2g,buy_w,v2gp_w,E_init,tru_w)
            results.append(make_kpi("C - MILP Day-Ahead",v2g,Pc,Pd,soc,
                                    buy_w,v2gp_w,E_init,arr,dep,tru_w=tru_w))
        if do_D:
            Pc,Pd,soc = run_D_mpc(v2g,buy_w,v2gp_w,E_init,tru_w)
            results.append(make_kpi("D - MPC (receding)",v2g,Pc,Pd,soc,
                                    buy_w,v2gp_w,E_init,arr,dep,tru_w=tru_w))

        rc = compute_reefer_costs(tru_w, buy_w, v2g.dt_h)
        return results, results, buy_d, plug_d, hours_d, is_wknd, is_48h, tru_d, rc


# =============================================================================
#  SCENARIO RUNNER — specific date
# =============================================================================

@st.cache_data(show_spinner=False)
def run_specific_date(date_str, arrival_h, departure_h,
                      soc_pct, soc_departure_pct, tru_cycle,
                      do_B, do_C, do_D):
    ts      = pd.Timestamp(date_str)
    is_wknd = ts.dayofweek >= 5
    v2g     = V2GParams(soc_departure_pct=soc_departure_pct)
    E_init  = v2g.usable_capacity_kWh * soc_pct / 100.0

    # ── WEEKEND — single day, full 24 h (unchanged) ───────────────────────────
    if is_wknd:
        buy             = load_date_profile(date_str)
        v2gp            = buy.copy()
        W               = 96
        buy_w           = buy
        v2gp_w          = v2gp
        tru_w           = get_tru_15min_trace(tru_cycle, W, v2g.dt_h)
        buy_d           = buy
        plug_d          = np.ones(96)
        hours_d         = np.arange(96) * v2g.dt_h
        tru_d           = tru_w
        arr, dep        = 0, 96
        is_48h          = False
        is_wknd_fullday = True

    # ── WEEKDAY — overnight window spanning two calendar days ─────────────────
    else:
        # 192-slot array: [day D (slots 0–95)] + [day D+1 (slots 96–191)]
        # Raises ValueError with user-readable message if D+1 is missing
        buy_192  = load_two_day_profile(date_str)
        v2gp_192 = buy_192.copy()

        # ROLL = 48 slots = 12 h → display window starts at 12:00 on day D
        ROLL     = round(12.0 / v2g.dt_h)          # 48

        # Slot indices within the 192-slot array
        arr_slot = round(arrival_h   / v2g.dt_h) % 96   # e.g. 16:00 → slot 64
        dep_slot = round(departure_h / v2g.dt_h) % 96   # e.g. 06:00 → slot 24

        # Optimisation window — correct prices for both days
        buy_w   = buy_192[arr_slot : 96 + dep_slot]
        v2gp_w  = buy_w.copy()
        W       = len(buy_w)

        # Display array — 96 slots, 12:00 day D → 12:00 day D+1 (same x-axis style)
        buy_d   = buy_192[ROLL : ROLL + 96]
        hours_d = np.arange(96) * v2g.dt_h + 12.0      # 12.00 … 35.75

        # Plugged-in mask on the 12–36 h chart
        dep_on_chart = (departure_h + 24.0) if departure_h < 12.0 else departure_h
        plug_d = (
            (hours_d >= arrival_h) & (hours_d < dep_on_chart)
        ).astype(float)

        # Display positions for to_display_wd / make_kpi (0-based into 96-slot display)
        arr_disp = arr_slot - ROLL          # e.g. 64 - 48 = 16
        dep_disp = ROLL     + dep_slot      # e.g. 48 + 24 = 72
        arr, dep = arr_disp, dep_disp

        # TRU display trace
        tru_w = get_tru_15min_trace(tru_cycle, W, v2g.dt_h)
        tru_d = np.zeros(96)
        d_s   = max(0, arr_disp)
        d_e   = min(96, dep_disp)
        w_s   = d_s - arr_disp
        w_e   = w_s + (d_e - d_s)
        if w_e > w_s:
            tru_d[d_s:d_e] = tru_w[w_s:w_e]

        is_48h          = False
        is_wknd_fullday = False

    # ── Run scenarios (identical for both paths) ──────────────────────────────
    Pc, Pd, soc = run_A_dumb(v2g, buy_w, v2gp_w, W, E_init, tru_w)
    results = [make_kpi("A - Dumb", v2g, Pc, Pd, soc,
                        buy_w, v2gp_w, E_init, arr, dep, tru_w=tru_w)]

    if do_B:
        Pc, Pd, soc = run_B_smart(v2g, buy_w, v2gp_w, E_init, tru_w)
        results.append(make_kpi("B - Smart (no V2G)", v2g, Pc, Pd, soc,
                                buy_w, v2gp_w, E_init, arr, dep, tru_w=tru_w))
    if do_C:
        Pc, Pd, soc = run_C_milp(v2g, buy_w, v2gp_w, E_init, tru_w)
        results.append(make_kpi("C - MILP Day-Ahead", v2g, Pc, Pd, soc,
                                buy_w, v2gp_w, E_init, arr, dep, tru_w=tru_w))
    if do_D:
        Pc, Pd, soc = run_D_mpc(v2g, buy_w, v2gp_w, E_init, tru_w)
        results.append(make_kpi("D - MPC (receding)", v2g, Pc, Pd, soc,
                                buy_w, v2gp_w, E_init, arr, dep, tru_w=tru_w))

    rc = compute_reefer_costs(tru_w, buy_w, v2g.dt_h)
    return (results, buy_d, plug_d, hours_d,
            is_wknd, is_48h, is_wknd_fullday, tru_d, rc)


# =============================================================================
#  AUTHENTICATION
# =============================================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("S.KOe COOL -- Access Restricted")
    with st.form("login_form"):
        user = st.text_input("Username")
        pwd  = st.text_input("Password", type="password")
        if st.form_submit_button("Login", type="primary"):
            if user == "admin" and pwd == "dontshare":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials.")
    st.stop()


# =============================================================================
#  SESSION STATE DEFAULTS
# =============================================================================

DEFAULTS = {
    "arrival_str":   "16:00",
    "departure_str": "06:00",
    "soc_winter":    80,
    "soc_summer":    40,
    "soc_departure": 100,
    "winter_months": 6,
    "tru_cycle":     "OFF",
    "do_B":          True,
    "do_C":          True,
    "do_D":          False,
    "do_wwe":        False,
    "do_swe":        False,
    "do_price":      False,
    "fixed_price":   FIXED_PRICE_EUR_KWH,
    "wd_per_month":  22.0,
    "mode":          "Seasonal Average",
    "specific_date": "2025-01-15",
}

if "cfg" not in st.session_state:
    st.session_state.cfg = DEFAULTS.copy()
if "show_output" not in st.session_state:
    st.session_state.show_output = False


# =============================================================================
#  INPUT PANEL  — shown first when app opens (Point 1)
# =============================================================================

def render_input_panel():
    st.title("S.KOe COOL -- V2G Optimisation")
    st.caption(
        "TU Dortmund IE3 x Schmitz Cargobull AG  |  "
        "Master's Thesis 2026  |  Kuldip Bhadreshvara"
    )
    st.markdown("---")

    if not Path(CSV_PATH).exists():
        st.error(
            f"Price CSV '{CSV_PATH}' not found. "
            "Commit '2025_Electricity_Price.csv' to the repo root."
        )
        st.stop()

    cfg = st.session_state.cfg

    with st.form("input_form", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns([1.15, 1.15, 1.0, 0.9])

        # ── Column 1: Schedule + Analysis Mode (Point 3) ──────────────────────
        with c1:
            st.subheader("Weekday Schedule")
            cfg["arrival_str"]   = st.text_input(
                "Arrival time (HH:MM)", cfg["arrival_str"],
                help="Time trailer returns to depot")
            cfg["departure_str"] = st.text_input(
                "Departure time (HH:MM)", cfg["departure_str"],
                help="Next-morning departure time")

            st.markdown("##### Analysis Mode")
            mode = st.radio(
                "Price data source",
                ["Seasonal Average", "Specific Date"],
                index=0 if cfg.get("mode", "Seasonal Average") == "Seasonal Average" else 1,
                help=(
                    "**Seasonal Average:** Average winter / summer weekday prices "
                    "computed from the full 2025 CSV.\n\n"
                    "**Specific Date:** Pick any single day in 2025 — uses the "
                    "actual SMARD prices for that exact date."
                )
            )
            cfg["mode"] = mode

            if mode == "Specific Date":
                date_val = st.date_input(
                    "Select date (2025)",
                    value=pd.Timestamp(cfg.get("specific_date", "2025-01-15")),
                    min_value=pd.Timestamp("2025-01-01"),
                    max_value=pd.Timestamp("2025-12-31"),
                    help="The date must exist in your price CSV"
                )
                cfg["specific_date"] = str(date_val)
                ts_sel  = pd.Timestamp(date_val)
                dow_sel = ts_sel.dayofweek
                mth_sel = ts_sel.month
                st.caption(
                    f"**{date_val.strftime('%A, %d %B %Y')}**  |  "
                    f"{'Weekend — full 24h' if dow_sel >= 5 else 'Weekday — overnight window'}  |  "
                    f"{'Winter' if mth_sel in WINTER_M else 'Summer'}"
                )

            st.markdown("##### Season Split")
            cfg["winter_months"] = st.slider(
                "Winter months", 1, 11, int(cfg["winter_months"]))
            st.caption(f"Summer months auto: **{12 - int(cfg['winter_months'])}**")

            st.markdown("##### Fixed-Tariff Benchmark")
            cfg["fixed_price"] = st.number_input(
                "Fixed price (EUR/kWh)",
                value=float(cfg["fixed_price"]),
                min_value=0.05, max_value=1.0, step=0.01,
                help="Comparison baseline only. Optimisation uses raw SMARD spot prices."
            )

        # ── Column 2: SoC + scenarios ──────────────────────────────────────────
        with c2:
            st.subheader("State of Charge")
            cfg["soc_winter"]    = st.slider(
                "Winter arrival SoC (%)", 20, 100, int(cfg["soc_winter"]),
                help="Battery % when trailer arrives in winter")
            cfg["soc_summer"]    = st.slider(
                "Summer arrival SoC (%)", 20, 100, int(cfg["soc_summer"]),
                help="Battery % when trailer arrives in summer")
            cfg["soc_departure"] = st.slider(
                "Departure target SoC (%)", 50, 100, int(cfg["soc_departure"]),
                help="Minimum battery % required when leaving depot")

            st.markdown("##### Yearly Extrapolation")
            cfg["wd_per_month"] = st.number_input(
                "Working days per month",
                value=float(cfg["wd_per_month"]),
                min_value=10.0, max_value=31.0, step=0.5
            )

            st.markdown("##### Scenarios to Run")
            cfg["do_B"] = st.checkbox(
                "B -- Smart charging (no V2G)", bool(cfg["do_B"]))
            cfg["do_C"] = st.checkbox(
                "C -- MILP Day-Ahead + V2G",   bool(cfg["do_C"]))
            cfg["do_D"] = st.checkbox(
                "D -- MPC receding horizon",   bool(cfg["do_D"]),
                help="Adds ~2 min compute — cached after first run")

        # ── Column 3: TRU ─────────────────────────────────────────────────────
        with c3:
            st.subheader("Reefer (TRU) at Depot")
            cycle_choice = st.radio(
                "TRU cycle while plugged in",
                ["Continuous", "Start-Stop", "OFF"],
                index={"Continuous": 0, "Start-Stop": 1, "OFF": 2}.get(
                    cfg["tru_cycle"], 2),
                help=(
                    "**Continuous:** 7.6 / 0.7 kW cycle (avg ~6.6 kW)\n\n"
                    "**Start-Stop:** 9.7 / 0.65 / 0 kW (avg ~3.9 kW)\n\n"
                    "**OFF:** TRU powered by diesel genset; no grid draw"
                )
            )
            cfg["tru_cycle"] = cycle_choice
            if cycle_choice != "OFF":
                avg_kw = tru_avg_kw(cycle_choice)
                st.info(
                    f"Avg TRU load: **{avg_kw:.1f} kW**\n\n"
                    f"Charging headroom: **{max(0, 22 - avg_kw):.1f} -- 22 kW**"
                )

        # ── Column 4: Extras + submit ──────────────────────────────────────────
        with c4:
            st.subheader("Extras")
            cfg["do_wwe"]   = st.checkbox(
                "Winter weekend (48h)", bool(cfg["do_wwe"]),
                help="Adds a 48h Sat+Sun block below the weekday charts")
            cfg["do_swe"]   = st.checkbox(
                "Summer weekend (48h)", bool(cfg["do_swe"]))
            cfg["do_price"] = st.checkbox(
                "Price profile analysis", bool(cfg["do_price"]))

            st.markdown("---")
            st.markdown("**S.KOe COOL specs**")
            st.caption("70 kWh total / 60 kWh usable")
            st.caption("22 kW AC bidirectional (ISO 15118-20)")
            st.caption("Cold-chain floor: SoC >= 20%")
            st.caption("CCS2 / OCPP 2.1")

            st.markdown("")
            submitted = st.form_submit_button(
                "Calculate", type="primary", use_container_width=True)

        if submitted:
            arr_h = parse_hhmm(cfg["arrival_str"], 16.0)
            dep_h = parse_hhmm(cfg["departure_str"], 6.0)
            if arr_h == dep_h:
                st.error("Arrival and departure times cannot be equal.")
                return
            if not any([cfg["do_B"], cfg["do_C"], cfg["do_D"]]):
                st.error("Enable at least one scenario (B / C / D).")
                return
            st.session_state.cfg         = cfg
            st.session_state.show_output = True
            st.rerun()


# =============================================================================
#  ROUTING — show input panel until Calculate is clicked
# =============================================================================

if not st.session_state.show_output:
    render_input_panel()
    st.stop()


# =============================================================================
#  RESULTS PAGE
# =============================================================================

st.title("S.KOe COOL -- V2G Optimisation Results")
st.caption(
    "TU Dortmund IE3 x Schmitz Cargobull AG  |  "
    "Master's Thesis 2026  |  Kuldip Bhadreshvara"
)

cfg           = st.session_state.cfg
arr_h         = parse_hhmm(cfg["arrival_str"],   16.0)
dep_h         = parse_hhmm(cfg["departure_str"],  6.0)
soc_w         = int(cfg["soc_winter"])
soc_s         = int(cfg["soc_summer"])
soc_dep       = int(cfg["soc_departure"])
tru_cycle     = cfg["tru_cycle"]
w_months      = int(cfg["winter_months"])
s_months      = 12 - w_months
do_B          = bool(cfg["do_B"])
do_C          = bool(cfg["do_C"])
do_D          = bool(cfg["do_D"])
fixed_price   = float(cfg["fixed_price"])
mode          = cfg.get("mode", "Seasonal Average")
specific_date = cfg.get("specific_date", "2025-01-15")

# ── Sidebar quick-edit ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Quick Edit")
    st.caption("Changes apply immediately")
    cfg["arrival_str"]   = st.text_input("Arrival (HH:MM)",   cfg["arrival_str"])
    cfg["departure_str"] = st.text_input("Departure (HH:MM)", cfg["departure_str"])
    cfg["soc_winter"]    = st.slider("Winter arrival SoC (%)",   20, 100, soc_w)
    cfg["soc_summer"]    = st.slider("Summer arrival SoC (%)",   20, 100, soc_s)
    cfg["soc_departure"] = st.slider("Departure target SoC (%)", 50, 100, soc_dep)
    cfg["winter_months"] = st.slider("Winter months", 1, 11, w_months)
    cfg["tru_cycle"]     = st.radio(
        "TRU cycle", ["Continuous", "Start-Stop", "OFF"],
        index=["Continuous", "Start-Stop", "OFF"].index(tru_cycle))
    st.markdown("---")
    cfg["mode"] = st.radio(
        "Analysis mode",
        ["Seasonal Average", "Specific Date"],
        index=0 if mode == "Seasonal Average" else 1)
    if cfg["mode"] == "Specific Date":
        dv = st.date_input(
            "Date (2025)",
            value=pd.Timestamp(specific_date),
            min_value=pd.Timestamp("2025-01-01"),
            max_value=pd.Timestamp("2025-12-31"))
        cfg["specific_date"] = str(dv)
    st.markdown("---")
    if st.button("Back to Input", use_container_width=True):
        st.session_state.show_output = False
        st.rerun()
    st.session_state.cfg = cfg
    # Refresh local vars after sidebar edits
    arr_h         = parse_hhmm(cfg["arrival_str"],   16.0)
    dep_h         = parse_hhmm(cfg["departure_str"],  6.0)
    soc_w         = int(cfg["soc_winter"])
    soc_s         = int(cfg["soc_summer"])
    soc_dep       = int(cfg["soc_departure"])
    tru_cycle     = cfg["tru_cycle"]
    w_months      = int(cfg["winter_months"])
    s_months      = 12 - w_months
    mode          = cfg["mode"]
    specific_date = cfg.get("specific_date", "2025-01-15")

# ── Verify CSV ────────────────────────────────────────────────────────────────
if not Path(CSV_PATH).exists():
    st.error(f"'{CSV_PATH}' not found -- commit it to the GitHub repo.")
    st.stop()

with st.spinner("Loading price data..."):
    try:
        df_info = _load_csv_raw(CSV_PATH)
        n_days  = len(df_info) // 96
        st.success(
            f"2025 SMARD DE/LU spot prices  |  {n_days} days  |  "
            f"{df_info.index[0].date()} to {df_info.index[-1].date()}  |  "
            f"Range: {df_info['price'].min()*1000:.0f} -- "
            f"{df_info['price'].max()*1000:.0f} EUR/MWh  |  "
            f"Mode: **{mode}**"
            + (f"  |  Date: **{specific_date}**" if mode == "Specific Date" else "")
        )
    except Exception as e:
        st.error(f"Could not load CSV: {e}"); st.stop()

v2g     = V2GParams(soc_departure_pct=float(soc_dep))
tru_avg = tru_avg_kw(tru_cycle)

# Summary metrics banner
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Arrival / Departure",  f"{fmt_hhmm(arr_h)} / {fmt_hhmm(dep_h)}")
m2.metric("Winter / Summer SoC",  f"{soc_w}% / {soc_s}%")
m3.metric("Departure Target",     f"{soc_dep}%")
m4.metric("TRU Reefer",
          f"{tru_cycle} ({tru_avg:.1f} kW)" if tru_cycle != "OFF" else "OFF")
m5.metric("Mode", "Seasonal avg" if mode == "Seasonal Average"
                  else f"Date: {specific_date}")
st.markdown("---")


# =============================================================================
#  KPI TABLE HELPER
# =============================================================================

def show_kpi_table(results, fixed_price, tru_cycle, rc, label=""):
    ref  = results[0]["net_cost"]
    rows = []
    for r in results:
        sav      = ref - r["net_cost"]
        fixed_ev = r["charge_kwh"] * fixed_price
        rows.append({
            "Scenario"                               : r["label"],
            "EV charge cost (EUR/d)"                 : round(r["charge_cost"],  4),
            "V2G revenue (EUR/d)"                    : round(r["v2g_rev"],       4),
            "Net EV cost (EUR/d)"                    : round(r["net_cost"],      4),
            f"Fixed @{fixed_price:.2f}/kWh (EUR/d)"  : round(fixed_ev,           4),
            "V2G export (kWh/d)"                     : round(r["v2g_kwh"],       2),
            "Daily savings vs Dumb"                  : "--" if sav == 0 else f"EUR {sav:+.4f}",
            "Annual savings (x365)"                  : "--" if sav == 0 else f"EUR {sav*365:+,.0f}",
        })
    hdr = f"**EV Charging KPI{' — ' + label if label else ''}**"
    st.markdown(hdr)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if tru_cycle != "OFF" and rc["E_kWh"] > 0.01:
        st.markdown("**Reefer (TRU) Energy Cost**")
        st.dataframe(pd.DataFrame([
            ["TRU energy (kWh/d)",                         f"{rc['E_kWh']:.2f}"],
            ["Grid spot price (EUR/d)",                    f"EUR {rc['cost_dynamic']:.3f}"],
            [f"Fixed @EUR{fixed_price:.2f}/kWh (EUR/d)",  f"EUR {rc['E_kWh']*fixed_price:.3f}"],
            ["Diesel genset (EUR/d)",                      f"EUR {rc['cost_diesel']:.3f}"],
            ["Diesel (L/d)",                               f"{rc['diesel_liters']:.2f} L"],
            ["Grid vs diesel saving (EUR/d)",
             f"EUR {rc['cost_diesel'] - rc['cost_dynamic']:+.3f}"],
        ], columns=["Metric","Value"]),
        use_container_width=True, hide_index=True)


# =============================================================================
#  MAIN DISPLAY
# =============================================================================

all_season_res_kpi = {}
all_reefer_costs   = {}

# ─────────────────────────────────────────────────────────────────────────────
#  SPECIFIC DATE MODE  (Point 3)
# ─────────────────────────────────────────────────────────────────────────────
if mode == "Specific Date":
    ts_date   = pd.Timestamp(specific_date)
    day_label = ts_date.strftime("%A, %d %B %Y")
    is_winter = ts_date.month in WINTER_M
    is_wknd   = ts_date.dayofweek >= 5
    soc_init  = soc_w if is_winter else soc_s
    color_hex = "#1565C0" if is_winter else "#E65100"

    st.subheader(f"Specific Date: {day_label}")
    st.caption(
        f"{'Weekend — full 24h plugged-in' if is_wknd else 'Weekday — overnight window'}  |  "
        f"{'Winter' if is_winter else 'Summer'} pricing  |  "
        f"Arrival SoC: {soc_init}%"
    )

    with st.spinner(f"Computing {day_label}..."):
        try:
            (results, buy_d, plug_d, hours_d,
             is_wknd_r, is_48h, is_wknd_fullday, tru_d, rc) = run_specific_date(
                specific_date, arr_h, dep_h,
                float(soc_init), float(soc_dep),
                tru_cycle, do_B, do_C, do_D
            )
        except Exception as e:
            st.error(f"Error: {e}"); st.stop()

    # ── 6 charts: 3 power (left) | 3 SoC (right) ──────────────────────────
    render_season_block(
        v2g, day_label, color_hex,
        hours_d, buy_d, plug_d, results,
        arr_h, dep_h, is_48h, is_wknd_fullday,
        do_B, do_C, do_D, tru_d
    )
    st.markdown("---")
    show_kpi_table(results, fixed_price, tru_cycle, rc)

# ─────────────────────────────────────────────────────────────────────────────
#  SEASONAL AVERAGE MODE
#  Points 2, 4, 5: Winter 6 charts on top, Summer 6 charts below
#  Each block: LEFT col = 3 power charts | RIGHT col = 3 SoC charts
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown(
        "**Chart layout:** Each row below = one season.  "
        "**Left half** of each row = ⚡ Power charts (Dumb vs Smart / MILP / MPC).  "
        "**Right half** = 🔋 SoC charts.  "
        "Each power chart has **dual Y-axis** (kW left | EUR/MWh right)."
    )
    st.markdown("---")

    # ── WINTER WEEKDAY ────────────────────────────────────────────────────────
    res_w = None
    with st.spinner("Computing Winter Weekday..."):
        try:
            (res_w, res_w_kpi, buy_d_w, plug_d_w, hours_d_w,
             is_wknd_w, is_48h_w, tru_d_w, rc_w) = run_seasonal(
                "winter_weekday", arr_h, dep_h,
                float(soc_w), float(soc_dep),
                tru_cycle, do_B, do_C, do_D
            )
            all_season_res_kpi["winter_weekday"] = res_w_kpi
            all_reefer_costs["winter_weekday"]    = rc_w
        except Exception as e:
            st.error(f"Winter weekday error: {e}")

    if res_w is not None:
        render_season_block(
            v2g,
            "Winter Weekday  (Oct – Mar average)",
            "#1565C0",
            hours_d_w, buy_d_w, plug_d_w, res_w,
            arr_h, dep_h, False, False,
            do_B, do_C, do_D, tru_d_w
        )

    # Thin divider between winter and summer
    st.markdown(
        "<hr style='border:1px solid #BBBBBB;margin:8px 0 8px 0;'>",
        unsafe_allow_html=True
    )

    # ── SUMMER WEEKDAY (directly below winter — Point 5) ─────────────────────
    res_s = None
    with st.spinner("Computing Summer Weekday..."):
        try:
            (res_s, res_s_kpi, buy_d_s, plug_d_s, hours_d_s,
             is_wknd_s, is_48h_s, tru_d_s, rc_s) = run_seasonal(
                "summer_weekday", arr_h, dep_h,
                float(soc_s), float(soc_dep),
                tru_cycle, do_B, do_C, do_D
            )
            all_season_res_kpi["summer_weekday"] = res_s_kpi
            all_reefer_costs["summer_weekday"]    = rc_s
        except Exception as e:
            st.error(f"Summer weekday error: {e}")

    if res_s is not None:
        render_season_block(
            v2g,
            "Summer Weekday  (Apr – Sep average)",
            "#E65100",
            hours_d_s, buy_d_s, plug_d_s, res_s,
            arr_h, dep_h, False, False,
            do_B, do_C, do_D, tru_d_s
        )

    st.markdown("---")

    # ── OPTIONAL WEEKEND BLOCKS ───────────────────────────────────────────────
    weekend_cfgs = []
    if cfg["do_wwe"]:
        weekend_cfgs.append(("winter_weekend","Winter Weekend (48h Sat+Sun)","#6A1B9A",soc_w))
    if cfg["do_swe"]:
        weekend_cfgs.append(("summer_weekend","Summer Weekend (48h Sat+Sun)","#2E7D32",soc_s))

    for (sk, lbl, col_hex, soc_init) in weekend_cfgs:
        with st.spinner(f"Computing {lbl}..."):
            try:
                (res_we, res_we_kpi, buy_d_we, plug_d_we, hours_d_we,
                 _, is_48h_we, tru_d_we, rc_we) = run_seasonal(
                    sk, arr_h, dep_h, float(soc_init), float(soc_dep),
                    tru_cycle, do_B, do_C, do_D
                )
                all_season_res_kpi[sk] = res_we_kpi
                all_reefer_costs[sk]   = rc_we
            except Exception as e:
                st.error(f"{lbl} error: {e}"); continue

        render_season_block(
            v2g, lbl, col_hex,
            hours_d_we, buy_d_we, plug_d_we, res_we,
            arr_h, dep_h, is_48h_we, False,
            do_B, do_C, do_D, tru_d_we
        )
        st.markdown("---")
        show_kpi_table(res_we, fixed_price, tru_cycle, rc_we, lbl)
        st.markdown("---")

    # ── KPI TABLES (winter + summer in tabs) ──────────────────────────────────
    st.subheader("KPI Tables")
    tab_specs = []
    if res_w is not None: tab_specs.append(("Winter Weekday", res_w, rc_w))
    if res_s is not None: tab_specs.append(("Summer Weekday", res_s, rc_s))

    if tab_specs:
        tab_objs = st.tabs([t[0] for t in tab_specs])
        for tab_obj, (lbl, res, rc) in zip(tab_objs, tab_specs):
            with tab_obj:
                show_kpi_table(res, fixed_price, tru_cycle, rc, lbl)

    # ── YEARLY EXTRAPOLATION ──────────────────────────────────────────────────
    if all_season_res_kpi:
        st.markdown("---")
        st.subheader("Yearly Cost Extrapolation")
        we_per_month = max(0.0, (365 - float(cfg["wd_per_month"]) * 12) / 12)
        yr_df = yearly_extrapolation(
            all_season_res_kpi,
            winter_months=w_months,
            wd_per_month=float(cfg["wd_per_month"]),
            we_days_per_month=we_per_month,
        )
        if not yr_df.empty:
            st.dataframe(yr_df, use_container_width=True)

        if tru_cycle != "OFF" and all_reefer_costs:
            st.markdown("**Annual TRU Grid Cost (EUR/year)**")
            tru_yearly = {}
            for dt_key, rc in all_reefer_costs.items():
                mults = {
                    "winter_weekday": w_months  * float(cfg["wd_per_month"]),
                    "summer_weekday": s_months  * float(cfg["wd_per_month"]),
                    "winter_weekend": w_months  * we_per_month,
                    "summer_weekend": s_months  * we_per_month,
                }
                mult = mults.get(dt_key, 0)
                lbl  = dt_key.replace("_"," ").title()
                tru_yearly[lbl] = {
                    "TRU energy (kWh)":                    round(rc["E_kWh"]        * mult, 0),
                    "Spot price cost (EUR)":               round(rc["cost_dynamic"]  * mult, 0),
                    f"Fixed @{fixed_price:.2f}/kWh (EUR)": round(rc["E_kWh"]*fixed_price*mult, 0),
                    "Diesel equiv (EUR)":                  round(rc["cost_diesel"]   * mult, 0),
                }
            tru_yr_df = pd.DataFrame(tru_yearly).T
            tru_yr_df.loc["TOTAL"] = tru_yr_df.sum()
            st.dataframe(tru_yr_df, use_container_width=True)

    # ── KPI MULTI-CHART ───────────────────────────────────────────────────────
    if len(all_season_res_kpi) > 1:
        st.markdown("---")
        st.subheader("KPI Comparison Chart -- All Day Types")
        try:
            buf = BytesIO()
            plot_kpi_multi(all_season_res_kpi, v2g, arr_h, dep_h,
                           run_mpc=do_D, out=buf)
            buf.seek(0)
            st.image(buf, use_container_width=True)
            buf2 = BytesIO()
            plot_kpi_multi(all_season_res_kpi, v2g, arr_h, dep_h,
                           run_mpc=do_D, out=buf2)
            buf2.seek(0)
            st.download_button(
                "Download KPI comparison (PNG)",
                data=buf2, file_name="v2g_KPI_multi.png",
                mime="image/png", key="dl_kpi"
            )
        except Exception as e:
            st.error(f"KPI chart error: {e}")

    # ── PRICE ANALYSIS ────────────────────────────────────────────────────────
    if cfg["do_price"]:
        st.markdown("---")
        st.subheader("Electricity Price Analysis")
        with st.spinner("Building price charts..."):
            try:
                buf = BytesIO()
                plot_price_profiles(CSV_PATH, buf)
                buf.seek(0)
                st.image(buf, use_container_width=True)
                buf2 = BytesIO()
                plot_price_profiles(CSV_PATH, buf2)
                buf2.seek(0)
                st.download_button(
                    "Download price analysis (PNG)",
                    data=buf2, file_name="v2g_price_profiles.png",
                    mime="image/png", key="dl_price"
                )
            except Exception as e:
                st.error(f"Price analysis error: {e}")


# =============================================================================
#  METHODOLOGY
# =============================================================================

with st.expander("Methodology & Assumptions", expanded=False):
    st.markdown(f"""
**Price data:** Raw SMARD DE/LU 15-min day-ahead spot prices.
No tariff surcharges or VAT added. Both charge cost and V2G revenue use the raw spot price.

**Chart layout (Seasonal Average mode):**
- Winter block on top: LEFT = 3 Power charts vertical | RIGHT = 3 SoC charts vertical
- Summer block directly below with identical structure
- Power charts: dual Y-axis — kW (left), EUR/MWh price (right)
- Each pairwise: Dumb (grey) vs Smart / MILP / MPC

**Specific Date mode:** Uses actual SMARD prices for the chosen date.
Weekday shows the overnight depot window. Weekend shows full 24h.

**Optimisation:** MILP with binary charge/discharge mutex.

**TRU modelling ({tru_cycle}):** Competes with charger on 22 kW grid connection.
Effective charging = max(0, 22 kW minus TRU load). TRU does NOT drain the battery.
- Continuous: 7.6 / 0.7 kW (1717 s / 292 s), avg **{tru_avg_kw("Continuous"):.1f} kW**
- Start-Stop: 9.7 / 0.65 / 0 kW, avg **{tru_avg_kw("Start-Stop"):.1f} kW**

**Cold-chain floor:** SoC >= 20% hard MILP constraint.
**Departure target:** SoC >= {soc_dep}%.
**Battery:** 70 kWh total / 60 kWh usable | eta_c = 0.92 | eta_d = 0.92
**Yearly extrapolation:** {int(cfg['wd_per_month'])} working days/month x seasonal split.
**Fixed-tariff benchmark:** EUR {fixed_price:.2f}/kWh (comparison only).
    """)

st.caption(
    "S.KOe COOL V2G Optimisation  -  "
    "TU Dortmund IE3 x Schmitz Cargobull AG  -  "
    "Thesis 2026  -  Confidential"
)