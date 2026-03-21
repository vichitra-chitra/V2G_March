"""
S.KOe COOL -- V2G Optimisation  |  Streamlit App (v7)
TU Dortmund IE3 x Schmitz Cargobull AG | 2026

Layout:
  - Input panel first (always shown on open)
  - Output: 6 charts per season (3 power left | 3 SoC right)
  - Winter 6 + Summer 6 visible together
  - Option: specific date from 2025 CSV instead of seasonal avg
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from io import BytesIO
from pathlib import Path

from v2g_single_day4 import (
    V2GParams, WINTER_M, SUMMER_M, SC_COL, SC_FILL,
    GERMAN_TARIFF, FIXED_PRICE_EUR_KWH,
    _interpolate_to_15min, _load_csv_raw,
    get_tru_15min_trace, tru_avg_kw,
    compose_all_in_price, compute_reefer_costs,
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

def buf_from_fig(fig) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf

def _ax_setup(ax, hours_d, is_48h, ylabel, title):
    if is_48h:
        pos  = np.arange(0, 49, 4)
        lbls = [f"{'Sat' if h<24 else 'Sun'}\n{int(h%24):02d}:00" for h in pos]
        ax.set_xlim(0, 48)
    else:
        pos  = np.arange(12, 37, 2)
        lbls = [f"{int(h%24):02d}:00" for h in pos]
        ax.set_xlim(12, 36)
    ax.set_xticks(pos)
    ax.set_xticklabels(lbls, fontsize=6.5, rotation=30, ha="right")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8.5, fontweight="bold", loc="left", pad=3)
    ax.grid(True, alpha=0.20, zorder=0)

def _vert(ax, arrival_h, departure_h, is_48h):
    if is_48h:
        ax.axvline(24, color="#555", lw=1.0, ls="--", alpha=0.55, zorder=5)
    else:
        def dx(h): return h if h >= 12 else h + 24
        ax.axvline(dx(arrival_h),   color="#1B5E20", lw=1.0, ls=":", alpha=0.75, zorder=5)
        ax.axvline(dx(departure_h), color="#B71C1C", lw=1.0, ls=":", alpha=0.75, zorder=5)
        ax.axvline(dx(0),           color="#555",    lw=1.0, ls="--", alpha=0.55, zorder=5)


# =============================================================================
#  CHART FUNCTIONS  (pairwise: Dumb vs one scenario)
# =============================================================================

def make_power_chart(v2g, hours_d, buy_d, plug_d,
                     result_A, result_X, x_label, x_key,
                     arrival_h, departure_h, is_48h, tru_d=None):
    """
    Power comparison chart: Dumb (grey) vs scenario X (coloured).
    Dual Y-axis: left = kW, right = EUR/MWh (price).
    """
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#FFFFFF")

    col_a = SC_COL["A"]; fill_a = SC_FILL["A"]
    col_x = SC_COL[x_key]; fill_x = SC_FILL[x_key]

    # Plugged-in shading (subtle)
    for t in range(len(hours_d)):
        if plug_d[t] > 0.5:
            ax.axvspan(hours_d[t], hours_d[t] + 0.25, color="gold", alpha=0.14, lw=0, zorder=1)

    # Dumb charge
    ax.fill_between(hours_d, result_A["Pc_d"], step="post",
                    color=fill_a, alpha=0.55, zorder=2)
    h_a, = ax.step(hours_d, result_A["Pc_d"], where="post",
                   color=col_a, lw=1.6, zorder=3, label="A Dumb charge")

    # Scenario X charge
    ax.fill_between(hours_d, result_X["Pc_d"], step="post",
                    color=fill_x, alpha=0.50, zorder=4)
    h_x, = ax.step(hours_d, result_X["Pc_d"], where="post",
                   color=col_x, lw=2.0, zorder=5, label=f"{x_key} charge")

    handles = [h_a, h_x]

    # V2G discharge (below 0)
    if result_X["v2g_kwh"] > 0.05:
        ax.fill_between(hours_d, -result_X["Pd_d"], step="post",
                        color=fill_x, alpha=0.28, zorder=4)
        h_d, = ax.step(hours_d, -result_X["Pd_d"], where="post",
                       color=col_x, lw=2.0, ls="--", alpha=0.90, zorder=5,
                       label=f"{x_key} V2G")
        handles.append(h_d)

    # TRU line
    if tru_d is not None and np.any(tru_d > 0.01):
        h_t, = ax.step(hours_d, -tru_d, where="post",
                       color="#C62828", lw=1.2, ls=":", alpha=0.75, zorder=5,
                       label="TRU")
        handles.append(h_t)

    ax.axhline(0, color="black", lw=0.6)
    _vert(ax, arrival_h, departure_h, is_48h)

    # Price on twin right axis
    ax2 = ax.twinx()
    h_p, = ax2.step(hours_d, buy_d * 1000, where="post",
                    color="#2E7D32", lw=1.4, alpha=0.80, label="Price")
    ax2.fill_between(hours_d, buy_d * 1000, step="post",
                     color="#2E7D32", alpha=0.06)
    ax2.set_ylabel("EUR/MWh", fontsize=7, color="#2E7D32")
    ax2.tick_params(axis="y", labelcolor="#2E7D32", labelsize=6.5)
    ax2.set_ylim(bottom=min(0, (buy_d * 1000).min() - 5))
    handles.append(h_p)

    _ax_setup(ax, hours_d, is_48h, "Power (kW)",
              f"Power — Dumb vs {x_label}")
    ax.legend(handles=handles, fontsize=6.5, loc="upper left", ncol=len(handles),
              framealpha=0.90, edgecolor="#CCCCCC", handlelength=1.2)

    plt.tight_layout(pad=0.35)
    return buf_from_fig(fig)


def make_soc_chart(v2g, hours_d, plug_d,
                   result_A, result_X, x_label, x_key,
                   arrival_h, departure_h, is_48h):
    """
    SoC comparison chart: Dumb (grey) vs scenario X (coloured).
    Single Y-axis in %.
    """
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#FFFFFF")

    col_a = SC_COL["A"]; col_x = SC_COL[x_key]

    # Plugged-in shading
    for t in range(len(hours_d)):
        if plug_d[t] > 0.5:
            ax.axvspan(hours_d[t], hours_d[t] + 0.25, color="gold", alpha=0.14, lw=0, zorder=1)

    # SoC ramps
    xA, yA = soc_ramp(hours_d, result_A["soc_d"], result_A["E_init_pct"])
    xX, yX = soc_ramp(hours_d, result_X["soc_d"], result_X["E_init_pct"])

    h_a, = ax.plot(xA, yA, color=col_a, lw=2.0, label="A Dumb SoC")
    h_x, = ax.plot(xX, yX, color=col_x, lw=2.3, label=f"{x_key} SoC")

    ax.axhline(v2g.soc_min_pct,       color="#C62828", ls=":", lw=1.2, zorder=3)
    ax.axhline(v2g.soc_departure_pct, color="#0D47A1", ls=":", lw=1.2, zorder=3)

    _vert(ax, arrival_h, departure_h, is_48h)

    handles = [
        h_a, h_x,
        mlines.Line2D([], [], color="#C62828", ls=":", lw=1.2,
                      label=f"Floor {v2g.soc_min_pct:.0f}%"),
        mlines.Line2D([], [], color="#0D47A1", ls=":", lw=1.2,
                      label=f"Target {v2g.soc_departure_pct:.0f}%"),
        mpatches.Patch(color="gold", alpha=0.35, label="Plugged-in"),
    ]

    _ax_setup(ax, hours_d, is_48h, "SoC (%)",
              f"SoC — Dumb vs {x_label}")
    ax.set_ylim(0, 115)
    ax.legend(handles=handles, fontsize=6.5, loc="lower right", ncol=3,
              framealpha=0.90, edgecolor="#CCCCCC", handlelength=1.2)

    plt.tight_layout(pad=0.35)
    return buf_from_fig(fig)


def render_6_charts(v2g, hours_d, buy_d, plug_d, results,
                    arrival_h, departure_h, is_48h,
                    section_title, do_B, do_C, do_D, tru_d=None):
    """
    Render 6 charts in a 2-column layout:
      Left  col: 3 Power charts (Dumb vs Smart / MILP / MPC)
      Right col: 3 SoC   charts (Dumb vs Smart / MILP / MPC)
    """
    result_A = results[0]

    # Build comparison list in order B → C → D
    comparisons = []
    label_map = {"B": "B - Smart", "C": "C - MILP", "D": "D - MPC"}
    idx = 1
    for key, do_it in [("B", do_B), ("C", do_C), ("D", do_D)]:
        if do_it and idx < len(results):
            comparisons.append((results[idx], label_map[key], key))
            idx += 1

    if not comparisons:
        st.info(f"{section_title}: enable at least one of B / C / D in settings.")
        return

    st.markdown(f"##### {section_title}")

    col_pow, col_soc = st.columns(2)

    with col_pow:
        st.caption("⚡ Charge / Discharge Power  (dual axis: kW left | EUR/MWh right)")
        for result_X, x_label, x_key in comparisons:
            buf = make_power_chart(v2g, hours_d, buy_d, plug_d,
                                   result_A, result_X, x_label, x_key,
                                   arrival_h, departure_h, is_48h, tru_d)
            st.image(buf, use_container_width=True)

    with col_soc:
        st.caption("🔋 Battery State of Charge (%)")
        for result_X, x_label, x_key in comparisons:
            buf = make_soc_chart(v2g, hours_d, plug_d,
                                 result_A, result_X, x_label, x_key,
                                 arrival_h, departure_h, is_48h)
            st.image(buf, use_container_width=True)


# =============================================================================
#  CACHED SCENARIO RUNNERS
# =============================================================================

@st.cache_data(show_spinner=False)
def load_profile_cached(months: tuple, is_weekend: bool) -> np.ndarray:
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
def run_seasonal(season_key, arrival_h, departure_h, soc_pct,
                 soc_departure_pct, tru_cycle,
                 do_B, do_C, do_D, use_allin):
    """Run scenarios for a seasonal average profile."""
    months_map = {
        "winter_weekday": (WINTER_M, False),
        "summer_weekday": (SUMMER_M, False),
    }
    months, is_wknd = months_map[season_key]

    buy_spot = load_profile_cached(tuple(months), is_wknd)
    buy      = compose_all_in_price(buy_spot) if use_allin else buy_spot
    v2gp     = buy_spot.copy()

    v2g    = V2GParams(soc_departure_pct=soc_departure_pct)
    E_init = v2g.usable_capacity_kWh * soc_pct / 100.0

    win, arr, dep, W = get_wd_window(v2g, arrival_h, departure_h)
    buy_w  = buy[win]; v2gp_w = v2gp[win]; spot_w = buy_spot[win]
    tru_w  = get_tru_15min_trace(tru_cycle, W, v2g.dt_h)
    buy_d, plug_d, hours_d = build_wd_display(v2g, buy, arrival_h, departure_h)
    tru_d  = np.zeros(v2g.n_slots); tru_d[arr:dep] = tru_w[:dep-arr]

    Pc,Pd,soc = run_A_dumb(v2g,buy_w,v2gp_w,W,E_init,tru_w)
    results = [make_kpi("A - Dumb",v2g,Pc,Pd,soc,buy_w,v2gp_w,E_init,arr,dep,tru_w=tru_w)]
    if do_B:
        Pc,Pd,soc = run_B_smart(v2g,buy_w,v2gp_w,E_init,tru_w)
        results.append(make_kpi("B - Smart (no V2G)",v2g,Pc,Pd,soc,buy_w,v2gp_w,E_init,arr,dep,tru_w=tru_w))
    if do_C:
        Pc,Pd,soc = run_C_milp(v2g,buy_w,v2gp_w,E_init,tru_w)
        results.append(make_kpi("C - MILP Day-Ahead",v2g,Pc,Pd,soc,buy_w,v2gp_w,E_init,arr,dep,tru_w=tru_w))
    if do_D:
        Pc,Pd,soc = run_D_mpc(v2g,buy_w,v2gp_w,E_init,tru_w)
        results.append(make_kpi("D - MPC (receding)",v2g,Pc,Pd,soc,buy_w,v2gp_w,E_init,arr,dep,tru_w=tru_w))

    rc = compute_reefer_costs(tru_w, spot_w, v2g.dt_h)
    return results, results, buy_d, plug_d, hours_d, tru_d, rc


@st.cache_data(show_spinner=False)
def run_specific_date(date_str, arrival_h, departure_h, soc_pct,
                      soc_departure_pct, tru_cycle,
                      do_B, do_C, do_D, use_allin):
    """Run scenarios for a specific calendar date."""
    df     = _load_csv_raw(CSV_PATH)
    target = pd.Timestamp(date_str).date()
    day_df = df[df["date"] == target]
    if len(day_df) == 0:
        raise ValueError(f"No price data found for {date_str}. Check your CSV covers this date.")
    if len(day_df) != 96:
        # Try interpolating if we have 24 hourly slots
        if len(day_df) == 24:
            prices = _interpolate_to_15min(day_df["price"].values)
        else:
            raise ValueError(f"Expected 96 slots for {date_str}, got {len(day_df)}.")
    else:
        prices = _interpolate_to_15min(day_df["price"].values)

    buy_spot = prices
    buy      = compose_all_in_price(buy_spot) if use_allin else buy_spot
    v2gp     = buy_spot.copy()

    ts      = pd.Timestamp(date_str)
    is_wknd = ts.dayofweek >= 5

    v2g    = V2GParams(soc_departure_pct=soc_departure_pct)
    E_init = v2g.usable_capacity_kWh * soc_pct / 100.0

    if is_wknd:
        # Full 24h plugged, display 00:00-24:00
        W      = 96
        buy_w  = buy; v2gp_w = v2gp; spot_w = buy_spot
        tru_w  = get_tru_15min_trace(tru_cycle, W, v2g.dt_h)
        buy_d  = buy
        plug_d = np.ones(96)
        hours_d = np.arange(96) * v2g.dt_h  # 0 to 23.75
        tru_d  = tru_w
        arr    = 0; dep = 96
        is_48h = False
    else:
        win, arr, dep, W = get_wd_window(v2g, arrival_h, departure_h)
        buy_w  = buy[win]; v2gp_w = v2gp[win]; spot_w = buy_spot[win]
        tru_w  = get_tru_15min_trace(tru_cycle, W, v2g.dt_h)
        buy_d, plug_d, hours_d = build_wd_display(v2g, buy, arrival_h, departure_h)
        tru_d  = np.zeros(v2g.n_slots); tru_d[arr:dep] = tru_w[:dep-arr]
        is_48h = False

    Pc,Pd,soc = run_A_dumb(v2g,buy_w,v2gp_w,W,E_init,tru_w)
    results = [make_kpi("A - Dumb",v2g,Pc,Pd,soc,buy_w,v2gp_w,E_init,arr,dep,tru_w=tru_w)]
    if do_B:
        Pc,Pd,soc = run_B_smart(v2g,buy_w,v2gp_w,E_init,tru_w)
        results.append(make_kpi("B - Smart (no V2G)",v2g,Pc,Pd,soc,buy_w,v2gp_w,E_init,arr,dep,tru_w=tru_w))
    if do_C:
        Pc,Pd,soc = run_C_milp(v2g,buy_w,v2gp_w,E_init,tru_w)
        results.append(make_kpi("C - MILP Day-Ahead",v2g,Pc,Pd,soc,buy_w,v2gp_w,E_init,arr,dep,tru_w=tru_w))
    if do_D:
        Pc,Pd,soc = run_D_mpc(v2g,buy_w,v2gp_w,E_init,tru_w)
        results.append(make_kpi("D - MPC (receding)",v2g,Pc,Pd,soc,buy_w,v2gp_w,E_init,arr,dep,tru_w=tru_w))

    rc = compute_reefer_costs(tru_w, spot_w, v2g.dt_h)
    return results, buy_d, plug_d, hours_d, is_wknd, is_48h, tru_d, rc


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
    "arrival_str":    "16:00",
    "departure_str":  "06:00",
    "soc_winter":     80,
    "soc_summer":     40,
    "soc_departure":  100,
    "winter_months":  6,
    "tru_cycle":      "OFF",
    "do_B":           True,
    "do_C":           True,
    "do_D":           False,
    "do_price":       False,
    "use_allin":      False,
    "fixed_price":    FIXED_PRICE_EUR_KWH,
    "wd_per_month":   22.0,
    "analysis_mode":  "Seasonal Average",
    "specific_date":  "2025-01-15",
}

if "cfg" not in st.session_state:
    st.session_state.cfg = DEFAULTS.copy()
if "show_output" not in st.session_state:
    st.session_state.show_output = False


# =============================================================================
#  INPUT PANEL  (shown when app is first opened / after login)
# =============================================================================

def render_input_panel():
    st.title("S.KOe COOL -- V2G Optimisation")
    st.caption("TU Dortmund IE3 x Schmitz Cargobull AG  |  Master's Thesis 2026  |  Kuldip Bhadreshvara")

    # Verify CSV exists before showing the form
    if not Path(CSV_PATH).exists():
        st.error(f"Price CSV '{CSV_PATH}' not found in the repository. "
                 "Please commit '2025_Electricity_Price.csv' to the same folder as app.py.")
        st.stop()

    st.markdown("---")
    st.markdown("Configure all parameters below, then click **Calculate** to run the optimisation.")

    cfg = st.session_state.cfg

    with st.form("input_form", clear_on_submit=False):

        # ── Row 1: Schedule | SoC | TRU | Analysis ────────────────────────────
        c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.0, 1.0])

        # Col 1: Schedule + season + price basis
        with c1:
            st.subheader("Weekday Schedule")
            cfg["arrival_str"]   = st.text_input("Arrival time (HH:MM)",   cfg["arrival_str"],
                                                  help="Time trailer returns to depot")
            cfg["departure_str"] = st.text_input("Departure time (HH:MM)", cfg["departure_str"],
                                                  help="Next-morning departure time")

            st.markdown("##### Season Split (yearly extrapolation)")
            cfg["winter_months"] = st.slider("Winter months (Oct-Mar)", 1, 11, int(cfg["winter_months"]))
            st.caption(f"Summer months auto: **{12 - int(cfg['winter_months'])}**")
            cfg["wd_per_month"]  = st.number_input("Working days / month", value=float(cfg["wd_per_month"]),
                                                    min_value=10.0, max_value=31.0, step=0.5)

            st.markdown("##### Price Basis")
            cfg["use_allin"] = st.checkbox(
                "Use all-in German tariff",
                value=bool(cfg["use_allin"]),
                help="Adds Netzentgelt 6.63 ct + levies + 19% VAT to SMARD spot. V2G revenue stays at spot."
            )
            cfg["fixed_price"] = st.number_input(
                "Fixed-tariff benchmark (EUR/kWh)", value=float(cfg["fixed_price"]),
                min_value=0.05, max_value=1.0, step=0.01,
                help="Flat rate used as cost comparison baseline"
            )

        # Col 2: SoC settings
        with c2:
            st.subheader("State of Charge")
            cfg["soc_winter"]    = st.slider("Winter arrival SoC (%)", 20, 100, int(cfg["soc_winter"]),
                                              help="Battery % on arrival in winter (shorter routes / heating)")
            cfg["soc_summer"]    = st.slider("Summer arrival SoC (%)", 20, 100, int(cfg["soc_summer"]),
                                              help="Battery % on arrival in summer (longer routes / AC load)")
            cfg["soc_departure"] = st.slider("Departure target SoC (%)", 50, 100, int(cfg["soc_departure"]),
                                              help="Minimum battery % required when leaving depot")

            st.markdown("##### Scenarios to Optimise")
            cfg["do_B"] = st.checkbox("B -- Smart charging (no V2G)", bool(cfg["do_B"]))
            cfg["do_C"] = st.checkbox("C -- MILP Day-Ahead + V2G",   bool(cfg["do_C"]))
            cfg["do_D"] = st.checkbox("D -- MPC receding horizon",   bool(cfg["do_D"]),
                                       help="~2 min first run -- cached after that")

        # Col 3: TRU reefer
        with c3:
            st.subheader("Reefer (TRU) at Depot")
            cycle_choice = st.radio(
                "TRU cycle while plugged in",
                ["Continuous", "Start-Stop", "OFF"],
                index={"Continuous": 0, "Start-Stop": 1, "OFF": 2}.get(cfg["tru_cycle"], 2),
                help=(
                    "**Continuous:** 7.6 kW high / 0.7 kW low  (avg ~6.6 kW)\n\n"
                    "**Start-Stop:** 9.7 / 0.65 / 0 kW  (avg ~3.9 kW)\n\n"
                    "**OFF:** TRU powered by diesel genset; no grid draw"
                )
            )
            cfg["tru_cycle"] = cycle_choice
            if cycle_choice != "OFF":
                avg_kw = tru_avg_kw(cycle_choice)
                st.info(
                    f"Avg TRU load: **{avg_kw:.1f} kW**\n\n"
                    f"Effective charging headroom: **{max(0, 22-avg_kw):.1f} – 22 kW**"
                )

            st.markdown("##### Extras")
            cfg["do_price"] = st.checkbox("Show price profile analysis", bool(cfg["do_price"]))

        # Col 4: Analysis mode
        with c4:
            st.subheader("Analysis Mode")
            analysis_mode = st.radio(
                "Data source for charts",
                ["Seasonal Average", "Specific Date"],
                index=0 if cfg.get("analysis_mode","Seasonal Average") == "Seasonal Average" else 1,
                help=(
                    "**Seasonal Average:** Uses mean winter/summer weekday price from CSV.\n\n"
                    "**Specific Date:** Pick any single day in 2025 — uses actual prices."
                )
            )
            cfg["analysis_mode"] = analysis_mode

            if analysis_mode == "Specific Date":
                date_val = st.date_input(
                    "Select date (2025)",
                    value=pd.Timestamp(cfg.get("specific_date", "2025-01-15")),
                    min_value=pd.Timestamp("2025-01-01"),
                    max_value=pd.Timestamp("2025-12-31"),
                    help="Pick any date that exists in your price CSV"
                )
                cfg["specific_date"] = str(date_val)
                st.caption(
                    f"Selected: **{date_val.strftime('%A, %d %B %Y')}**  "
                    f"({'Weekend' if date_val.weekday() >= 5 else 'Weekday'})"
                )
                st.info(
                    "Weekday: shows noon-to-noon window.\n"
                    "Weekend: shows full 24h plugged-in day."
                )

            st.markdown("---")
            st.markdown("**S.KOe COOL specs**")
            st.caption("70 kWh total / 60 kWh usable")
            st.caption("22 kW AC bidirectional (ISO 15118-20)")
            st.caption("Cold-chain floor: SoC >= 20%")
            st.caption("CCS2 / OCPP 2.1")

        # ── Submit ─────────────────────────────────────────────────────────────
        st.markdown("")
        _, btn_col, _ = st.columns([2, 1, 2])
        with btn_col:
            submitted = st.form_submit_button(
                "Calculate", type="primary", use_container_width=True
            )

        if submitted:
            arr_h = parse_hhmm(cfg["arrival_str"], 16.0)
            dep_h = parse_hhmm(cfg["departure_str"], 6.0)
            if arr_h == dep_h:
                st.error("Arrival and departure times cannot be equal.")
                return
            if not cfg["do_B"] and not cfg["do_C"] and not cfg["do_D"]:
                st.error("Enable at least one of B / C / D to compare against Dumb charging.")
                return
            st.session_state.cfg         = cfg
            st.session_state.show_output = True
            st.rerun()


# =============================================================================
#  ROUTING: show input panel until Calculate clicked
# =============================================================================

if not st.session_state.show_output:
    render_input_panel()
    st.stop()


# =============================================================================
#  RESULTS PAGE
# =============================================================================

cfg = st.session_state.cfg

arr_h         = parse_hhmm(cfg["arrival_str"],   16.0)
dep_h         = parse_hhmm(cfg["departure_str"],  6.0)
soc_w         = int(cfg["soc_winter"])
soc_s         = int(cfg["soc_summer"])
soc_dep       = int(cfg["soc_departure"])
tru_cycle     = cfg["tru_cycle"]
w_months      = int(cfg["winter_months"])
s_months      = 12 - w_months
use_allin     = bool(cfg["use_allin"])
do_B          = bool(cfg["do_B"])
do_C          = bool(cfg["do_C"])
do_D          = bool(cfg["do_D"])
analysis_mode = cfg.get("analysis_mode", "Seasonal Average")
specific_date = cfg.get("specific_date", "2025-01-15")
fixed_price   = float(cfg["fixed_price"])

st.title("S.KOe COOL -- V2G Optimisation Results")
st.caption("TU Dortmund IE3 x Schmitz Cargobull AG  |  Master's Thesis 2026  |  Kuldip Bhadreshvara")

# ── Sidebar quick-edit ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Quick Edit")
    st.caption("Changes apply immediately")
    cfg["arrival_str"]   = st.text_input("Arrival (HH:MM)",   cfg["arrival_str"])
    cfg["departure_str"] = st.text_input("Departure (HH:MM)", cfg["departure_str"])
    cfg["soc_winter"]    = st.slider("Winter arrival SoC (%)",    20, 100, soc_w)
    cfg["soc_summer"]    = st.slider("Summer arrival SoC (%)",    20, 100, soc_s)
    cfg["soc_departure"] = st.slider("Departure target SoC (%)",  50, 100, soc_dep)
    cfg["winter_months"] = st.slider("Winter months", 1, 11, w_months)
    cfg["tru_cycle"]     = st.radio("TRU cycle", ["Continuous","Start-Stop","OFF"],
                                     index=["Continuous","Start-Stop","OFF"].index(tru_cycle))
    cfg["use_allin"]     = st.checkbox("All-in tariff", use_allin)

    st.markdown("---")
    cfg["analysis_mode"] = st.radio("Analysis mode",
                                     ["Seasonal Average","Specific Date"],
                                     index=0 if analysis_mode=="Seasonal Average" else 1)
    if cfg["analysis_mode"] == "Specific Date":
        dv = st.date_input("Date", value=pd.Timestamp(specific_date),
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
    use_allin     = bool(cfg["use_allin"])
    analysis_mode = cfg["analysis_mode"]
    specific_date = cfg.get("specific_date","2025-01-15")

# ── Verify CSV ────────────────────────────────────────────────────────────────
if not Path(CSV_PATH).exists():
    st.error(f"'{CSV_PATH}' not found. Commit it to the GitHub repo."); st.stop()

with st.spinner("Loading price data..."):
    try:
        df_info = _load_csv_raw(CSV_PATH)
        n_days  = len(df_info) // 96
        st.success(
            f"2025 SMARD DE/LU  |  {n_days} days  |  "
            f"{df_info.index[0].date()} to {df_info.index[-1].date()}  |  "
            f"Spot: {df_info['price'].min()*1000:.0f}-{df_info['price'].max()*1000:.0f} EUR/MWh  |  "
            f"Price basis: {'all-in tariff+VAT' if use_allin else 'SMARD spot'}"
        )
    except Exception as e:
        st.error(f"Could not load CSV: {e}"); st.stop()

v2g      = V2GParams(soc_departure_pct=float(soc_dep))
tru_avg  = tru_avg_kw(tru_cycle)

# ── Summary metrics ───────────────────────────────────────────────────────────
m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("Arrival / Departure", f"{fmt_hhmm(arr_h)} / {fmt_hhmm(dep_h)}")
m2.metric("Winter / Summer SoC", f"{soc_w}% / {soc_s}%")
m3.metric("Departure Target",    f"{soc_dep}%")
m4.metric("TRU Reefer", f"{tru_cycle} ({tru_avg:.1f} kW)" if tru_cycle != "OFF" else "OFF")
m5.metric("Mode", analysis_mode.replace("Seasonal Average","Seasonal avg"))
st.markdown("---")


# =============================================================================
#  MAIN DISPLAY LOGIC
# =============================================================================

all_season_res_kpi = {}
all_reefer_costs   = {}

if analysis_mode == "Specific Date":
    # ── SPECIFIC DATE MODE ────────────────────────────────────────────────────
    ts_date = pd.Timestamp(specific_date)
    day_label = ts_date.strftime("%A, %d %B %Y")
    is_weekend_flag = ts_date.dayofweek >= 5
    is_winter_flag  = ts_date.month in WINTER_M
    soc_init = soc_w if is_winter_flag else soc_s

    st.subheader(f"Specific Date: {day_label}")
    st.caption(
        f"{'Weekend — full 24h plugged-in' if is_weekend_flag else 'Weekday — overnight window'}  |  "
        f"{'Winter' if is_winter_flag else 'Summer'} pricing  |  "
        f"Arrival SoC used: {soc_init}%"
    )

    with st.spinner(f"Computing {day_label}..."):
        try:
            (results, buy_d, plug_d, hours_d,
             is_wknd, is_48h, tru_d, rc) = run_specific_date(
                specific_date, arr_h, dep_h, float(soc_init), float(soc_dep),
                tru_cycle, do_B, do_C, do_D, use_allin
            )
        except Exception as e:
            st.error(f"Error: {e}"); st.stop()

    render_6_charts(v2g, hours_d, buy_d, plug_d, results,
                    arr_h, dep_h, is_48h,
                    f"{day_label}",
                    do_B, do_C, do_D, tru_d)

    # KPI table for this date
    st.markdown("---")
    st.markdown("**Daily KPI Table**")
    ref = results[0]["net_cost"]
    table = []
    for r in results:
        sav = ref - r["net_cost"]
        table.append({
            "Scenario"                 : r["label"],
            "EV charge cost (EUR)"     : round(r["charge_cost"],  4),
            "V2G revenue (EUR)"        : round(r["v2g_rev"],       4),
            "Net EV cost (EUR)"        : round(r["net_cost"],      4),
            f"Fixed @{fixed_price:.2f}/kWh" : round(r["charge_kwh"] * fixed_price, 4),
            "V2G export (kWh)"         : round(r["v2g_kwh"],       2),
            "Savings vs Dumb"          : "--" if sav == 0 else f"EUR {sav:+.4f}",
        })
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

    if tru_cycle != "OFF" and rc["E_kWh"] > 0.01:
        st.markdown("**Reefer TRU Energy Cost**")
        st.dataframe(pd.DataFrame([
            ["TRU energy (kWh)",              f"{rc['E_kWh']:.2f}"],
            ["Grid dynamic price (EUR)",      f"{rc['cost_dynamic']:.3f}"],
            [f"Grid fixed @{fixed_price:.2f}/kWh (EUR)", f"{rc['E_kWh']*fixed_price:.3f}"],
            ["Diesel genset (EUR)",           f"{rc['cost_diesel']:.3f}"],
            ["Diesel (L)",                    f"{rc['diesel_liters']:.2f}"],
        ], columns=["Metric","Value"]), use_container_width=True, hide_index=True)

else:
    # ── SEASONAL AVERAGE MODE ─────────────────────────────────────────────────
    # Run winter weekday
    with st.spinner("Computing Winter Weekday..."):
        try:
            (res_w, res_w_kpi, buy_d_w, plug_d_w, hours_d_w,
             tru_d_w, rc_w) = run_seasonal(
                "winter_weekday", arr_h, dep_h, float(soc_w), float(soc_dep),
                tru_cycle, do_B, do_C, do_D, use_allin
            )
            all_season_res_kpi["winter_weekday"] = res_w_kpi
            all_reefer_costs["winter_weekday"]    = rc_w
        except Exception as e:
            st.error(f"Winter weekday error: {e}"); res_w = None

    # Run summer weekday
    with st.spinner("Computing Summer Weekday..."):
        try:
            (res_s, res_s_kpi, buy_d_s, plug_d_s, hours_d_s,
             tru_d_s, rc_s) = run_seasonal(
                "summer_weekday", arr_h, dep_h, float(soc_s), float(soc_dep),
                tru_cycle, do_B, do_C, do_D, use_allin
            )
            all_season_res_kpi["summer_weekday"] = res_s_kpi
            all_reefer_costs["summer_weekday"]    = rc_s
        except Exception as e:
            st.error(f"Summer weekday error: {e}"); res_s = None

    # ── 12-CHART LAYOUT: Winter (top) | Summer (bottom) ───────────────────────
    st.markdown(
        "**Left column:** Charge / Discharge Power  (dual axis: kW + EUR/MWh)  "
        "**|**  **Right column:** Battery SoC (%)"
    )
    st.markdown("---")

    if res_w is not None:
        render_6_charts(v2g, hours_d_w, buy_d_w, plug_d_w, res_w,
                        arr_h, dep_h, False,
                        "Winter Weekday  (Oct – Mar average)",
                        do_B, do_C, do_D, tru_d_w)

    st.markdown("---")

    if res_s is not None:
        render_6_charts(v2g, hours_d_s, buy_d_s, plug_d_s, res_s,
                        arr_h, dep_h, False,
                        "Summer Weekday  (Apr – Sep average)",
                        do_B, do_C, do_D, tru_d_s)

    st.markdown("---")

    # ── KPI TABLES ────────────────────────────────────────────────────────────
    st.subheader("KPI Tables")
    tab_w, tab_s = st.tabs(["Winter Weekday", "Summer Weekday"])

    for tab, res, season_label in [(tab_w, res_w, "Winter"), (tab_s, res_s, "Summer")]:
        if res is None:
            continue
        with tab:
            ref = res[0]["net_cost"]
            table = []
            for r in res:
                sav = ref - r["net_cost"]
                table.append({
                    "Scenario"                      : r["label"],
                    "EV charge cost (EUR/d)"        : round(r["charge_cost"],  4),
                    "V2G revenue (EUR/d)"           : round(r["v2g_rev"],       4),
                    "Net EV cost (EUR/d)"           : round(r["net_cost"],      4),
                    f"Fixed @{fixed_price:.2f}/kWh" : round(r["charge_kwh"] * fixed_price, 4),
                    "V2G export (kWh/d)"            : round(r["v2g_kwh"],       2),
                    "Daily savings vs Dumb"         : "--" if sav==0 else f"EUR {sav:+.4f}",
                    "Annual savings (x365)"         : "--" if sav==0 else f"EUR {sav*365:+,.0f}",
                })
            st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

            rc = rc_w if season_label == "Winter" else rc_s
            if tru_cycle != "OFF" and rc["E_kWh"] > 0.01:
                st.markdown("**Reefer TRU Energy Cost**")
                st.dataframe(pd.DataFrame([
                    ["TRU energy (kWh/d)",                        f"{rc['E_kWh']:.2f}"],
                    ["Grid dynamic price (EUR/d)",                 f"{rc['cost_dynamic']:.3f}"],
                    [f"Grid fixed @{fixed_price:.2f}/kWh (EUR/d)", f"{rc['E_kWh']*fixed_price:.3f}"],
                    ["Diesel genset equiv (EUR/d)",                f"{rc['cost_diesel']:.3f}"],
                    ["Grid vs diesel saving (EUR/d)",
                     f"{rc['cost_diesel']-rc['cost_dynamic']:+.3f}"],
                ], columns=["Metric","Value"]), use_container_width=True, hide_index=True)

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
            st.markdown("**Annual TRU Grid Cost**")
            tru_yearly = {}
            for dt_key, rc in all_reefer_costs.items():
                mults = {
                    "winter_weekday": w_months * float(cfg["wd_per_month"]),
                    "summer_weekday": s_months * float(cfg["wd_per_month"]),
                }
                mult = mults.get(dt_key, 0)
                lbl  = dt_key.replace("_", " ").title()
                tru_yearly[lbl] = {
                    "TRU energy (kWh)":     round(rc["E_kWh"]        * mult, 0),
                    "Dynamic (EUR)":        round(rc["cost_dynamic"]  * mult, 0),
                    f"Fixed @{fixed_price:.2f} (EUR)":
                                            round(rc["E_kWh"] * fixed_price * mult, 0),
                    "Diesel equiv (EUR)":   round(rc["cost_diesel"]   * mult, 0),
                }
            tru_yr_df = pd.DataFrame(tru_yearly).T
            tru_yr_df.loc["TOTAL"] = tru_yr_df.sum()
            st.dataframe(tru_yr_df, use_container_width=True)

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
                st.download_button("Download price analysis (PNG)",
                                   data=buf2, file_name="v2g_price_profiles.png",
                                   mime="image/png", key="dl_price")
            except Exception as e:
                st.error(f"Price analysis error: {e}")

# ── METHODOLOGY ───────────────────────────────────────────────────────────────
with st.expander("Methodology & Assumptions", expanded=False):
    st.markdown(f"""
**Optimisation:** MILP with binary charge/discharge mutex.
V2G revenue at SMARD spot; charging cost at {'all-in German tariff' if use_allin else 'SMARD spot'}.

**TRU modelling ({tru_cycle}):** Competes with charger on 22 kW grid connection.
Effective charging = max(0, 22 kW - TRU load). TRU does NOT drain the battery.
- Continuous: 7.6 / 0.7 kW (1717 s / 292 s), avg **{tru_avg_kw("Continuous"):.1f} kW**
- Start-Stop: 9.7 / 0.65 / 0 kW, avg **{tru_avg_kw("Start-Stop"):.1f} kW**

**German all-in tariff:** Netzentgelt 6.63 ct + concession 1.99 ct + offshore 0.82 ct
+ CHP 0.28 ct + electricity tax 2.05 ct + NEV19 1.56 ct + 19% VAT on net.

**Cold-chain floor:** SoC >= 20% enforced as hard MILP constraint.
**Departure target:** SoC >= {soc_dep}%.
**Battery:** 70 kWh total / 60 kWh usable | eta_c = 0.92 | eta_d = 0.92
**Yearly extrapolation:** {int(cfg['wd_per_month'])} working days/month x seasonal month split.
**Fixed-tariff benchmark:** EUR {fixed_price:.2f}/kWh flat rate.
    """)

st.caption(
    "S.KOe COOL V2G Optimisation  -  TU Dortmund IE3 x Schmitz Cargobull AG  -  "
    "Thesis 2026  -  Confidential"
)