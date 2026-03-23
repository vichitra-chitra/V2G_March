import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "lines.linewidth":  2.0,
})
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from io import BytesIO
from pathlib import Path

# ── German all-in tariff components (BNetzA 2024) ─────────────────────────
TARIFF = {
    "network_fee_ct":     6.63,
    "concession_ct":      1.992,
    "offshore_levy_ct":   0.941,
    "chp_levy_ct":        0.446,
    "electricity_tax_ct": 2.05,
    "nev19_levy_ct":      1.559,
}
FIXED_NET_CT = sum(TARIFF.values())
VAT_RATE     = 0.19

V2G_RECOVERABLE_CURRENT_CT = 0.0

V2G_RECOVERABLE_FUTURE = {
    "network_fee_ct":     6.63,
    "electricity_tax_ct": 2.05,
}
V2G_RECOVERABLE_FUTURE_CT = sum(V2G_RECOVERABLE_FUTURE.values())

def to_allin_ct(spot_eur_kwh: np.ndarray,
                fixed_net_ct: float = None,
                vat: float = None) -> np.ndarray:
    """Convert raw SMARD spot price (EUR/kWh) to all-in depot price (ct/kWh)."""
    fn  = fixed_net_ct if fixed_net_ct is not None else FIXED_NET_CT
    vr  = vat          if vat          is not None else VAT_RATE
    return (spot_eur_kwh * 100.0 + fn) * (1.0 + vr)

from v2g_single_day4 import (
    V2GParams, WINTER_M, SUMMER_M, SC_COL, SC_FILL,
    FIXED_PRICE_EUR_KWH,
    _interpolate_to_15min, _load_csv_raw,
    get_tru_15min_trace, tru_avg_kw,
    compute_reefer_costs,
    get_wd_window, build_wd_display,
    run_A_dumb, run_B_smart, run_C_milp, run_D_mpc,
    make_kpi, plot_price_profiles,
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
    fig.savefig(buf, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# =============================================================================
#  AXIS SETUP HELPERS
# =============================================================================

def _setup_xaxis(ax, is_48h, is_wknd_fullday=False):
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
    ax.set_xticklabels(lbls, fontsize=8, rotation=30, ha="right")
    ax.grid(True, alpha=0.20, zorder=0)


def _vlines(ax, arrival_h, departure_h, is_48h, is_wknd_fullday=False):
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
#  POWER CHART
# =============================================================================

def make_power_chart(v2g, hours_d, buy_d, plug_d,
                     result_A, result_X,
                     x_label, x_key,
                     arrival_h, departure_h,
                     is_48h, is_wknd_fullday=False,
                     tru_d=None,
                     fixed_net_ct=None, vat_rate=None):

    col_a  = SC_COL["A"];   fill_a  = SC_FILL["A"]
    col_x  = SC_COL[x_key]; fill_x  = SC_FILL[x_key]
    lbl_x  = result_X["label"].split("(")[0].strip()

    fig, ax = plt.subplots(figsize=(9.0, 3.5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#FFFFFF")

    for t in range(len(hours_d)):
        if plug_d[t] > 0.5:
            ax.axvspan(hours_d[t], hours_d[t] + 0.25,
                       color="gold", alpha=0.14, lw=0, zorder=1)

    ax.fill_between(hours_d, result_A["Pc_d"],
                    step="post", color=fill_a, alpha=0.55, zorder=2)
    h_a, = ax.step(hours_d, result_A["Pc_d"], where="post",
                   color=col_a, lw=1.8, zorder=3, label="A - Dumb charge")

    ax.fill_between(hours_d, result_X["Pc_d"],
                    step="post", color=fill_x, alpha=0.48, zorder=4)
    h_x, = ax.step(hours_d, result_X["Pc_d"], where="post",
                   color=col_x, lw=2.0, zorder=5, label=f"{lbl_x} charge")

    handles = [h_a, h_x]

    if result_X["v2g_kwh"] > 0.05:
        ax.fill_between(hours_d, -result_X["Pd_d"],
                        step="post", color=fill_x, alpha=0.28, zorder=4)
        h_d, = ax.step(hours_d, -result_X["Pd_d"], where="post",
                       color=col_x, lw=2.0, ls="--", alpha=0.90, zorder=5,
                       label=f"{lbl_x} V2G (−)")
        handles.append(h_d)

    if tru_d is not None and np.any(tru_d > 0.01):
        h_t, = ax.step(hours_d, -tru_d, where="post",
                       color="#C62828", lw=1.2, ls=":", alpha=0.75, zorder=5,
                       label="TRU (−)")
        handles.append(h_t)

    ax.axhline(0, color="black", lw=0.6)
    _vlines(ax, arrival_h, departure_h, is_48h, is_wknd_fullday)

    buy_d_ct = to_allin_ct(buy_d, fixed_net_ct, vat_rate)
    ax2 = ax.twinx()
    h_p, = ax2.step(hours_d, buy_d_ct, where="post",
                    color="#2E7D32", lw=1.4, alpha=0.85, label="Price (all-in)")
    ax2.fill_between(hours_d, buy_d_ct,
                     step="post", color="#2E7D32", alpha=0.07)
    ax2.set_ylabel("ct/kWh (all-in)", fontsize=8, color="#2E7D32")
    ax2.tick_params(axis="y", labelcolor="#2E7D32", labelsize=8)
    ax2.set_ylim(bottom=min(0, buy_d_ct.min() - 1))
    handles.append(h_p)

    ax.set_ylabel("Power (kW)", fontsize=9)
    ax.set_title(f"Power — Dumb vs {x_label}",
                 fontsize=10, fontweight="bold", loc="left", pad=4)
    _setup_xaxis(ax, is_48h, is_wknd_fullday)

    ax.legend(handles=handles, fontsize=8, ncol=len(handles),
              loc="upper center", bbox_to_anchor=(0.5, -0.28),
              framealpha=0.92, edgecolor="#CCC", handlelength=1.2,
              borderpad=0.4, columnspacing=1.0)

    plt.tight_layout(pad=0.3)
    return fig_to_buf(fig)


# =============================================================================
#  SOC CHART
# =============================================================================

def make_soc_chart(v2g, hours_d, plug_d,
                   result_A, result_X,
                   x_label, x_key,
                   arrival_h, departure_h,
                   is_48h, is_wknd_fullday=False):

    col_a = SC_COL["A"]
    col_x = SC_COL[x_key]
    lbl_x = result_X["label"].split("(")[0].strip()

    fig, ax = plt.subplots(figsize=(9.0, 3.5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#FFFFFF")

    for t in range(len(hours_d)):
        if plug_d[t] > 0.5:
            ax.axvspan(hours_d[t], hours_d[t] + 0.25,
                       color="gold", alpha=0.14, lw=0, zorder=1)

    xA, yA = soc_ramp(hours_d, result_A["soc_d"], result_A["E_init_pct"])
    xX, yX = soc_ramp(hours_d, result_X["soc_d"], result_X["E_init_pct"])

    h_a, = ax.plot(xA, yA, color=col_a, lw=2.0, label="A - Dumb SoC")
    h_x, = ax.plot(xX, yX, color=col_x, lw=2.3, label=f"{lbl_x} SoC")

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

    ax.set_ylabel("SoC (%)", fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_title(f"SoC — Dumb vs {x_label}",
                 fontsize=10, fontweight="bold", loc="left", pad=4)
    _setup_xaxis(ax, is_48h, is_wknd_fullday)

    ax.legend(handles=handles, fontsize=8, ncol=5,
              loc="upper center", bbox_to_anchor=(0.5, -0.28),
              framealpha=0.92, edgecolor="#CCC", handlelength=1.2,
              borderpad=0.4, columnspacing=1.0)

    plt.tight_layout(pad=0.3)
    return fig_to_buf(fig)


# =============================================================================
#  RENDER ONE SEASON BLOCK
# =============================================================================

def render_season_block(v2g, season_title, color_hex,
                         hours_d, buy_d, plug_d, results,
                         arrival_h, departure_h,
                         is_48h, is_wknd_fullday,
                         do_B, do_C, do_D, tru_d=None,
                         fixed_net_ct=None, vat_rate=None):

    result_A = results[0]
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

    st.markdown(
        f"<div style='background:{color_hex};color:white;padding:5px 14px;"
        f"border-radius:5px;font-weight:bold;font-size:14px;margin-bottom:4px;'>"
        f"{season_title}</div>",
        unsafe_allow_html=True
    )

    col_pow, col_soc = st.columns(2)

    with col_pow:
        st.caption("⚡ **Charge / Discharge Power**  "
                   "(left axis = kW  |  right axis = ct/kWh all-in incl. taxes & grid fees)")
        for result_X, x_label, x_key in comparisons:
            buf = make_power_chart(
                v2g, hours_d, buy_d, plug_d,
                result_A, result_X, x_label, x_key,
                arrival_h, departure_h, is_48h, is_wknd_fullday, tru_d,
                fixed_net_ct=fixed_net_ct, vat_rate=vat_rate
            )
            st.image(buf, use_container_width=True)

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
    df     = _load_csv_raw(CSV_PATH)
    target = pd.Timestamp(date_str).date()
    day_df = df[df["date"] == target]
    if len(day_df) == 0:
        raise ValueError(f"No price data found for {date_str}.")
    prices = day_df["price"].values
    if len(prices) == 24:
        return _interpolate_to_15min(prices)
    if len(prices) != 96:
        raise ValueError(f"Expected 96 price slots for {date_str}, got {len(prices)}.")
    return _interpolate_to_15min(prices)


@st.cache_data(show_spinner=False)
def load_two_day_profile(date_str: str) -> np.ndarray:
    day1 = load_date_profile(date_str)
    next_ts       = pd.Timestamp(date_str) + pd.Timedelta(days=1)
    next_date_str = next_ts.strftime("%Y-%m-%d")
    try:
        day2 = load_date_profile(next_date_str)
    except ValueError:
        fallback_ts       = next_ts.replace(year=next_ts.year - 1)
        fallback_date_str = fallback_ts.strftime("%Y-%m-%d")
        try:
            day2 = load_date_profile(fallback_date_str)
            st.info(
                f"ℹ️ Post-midnight prices for **{next_date_str}** are not in the CSV. "
                f"Using **{fallback_date_str}** (same day, prior year) as a proxy."
            )
        except ValueError:
            raise ValueError(
                f"Next-day data for **{next_date_str}** not found, and fallback "
                f"**{fallback_date_str}** is also missing. Check your CSV coverage."
            )
    return np.concatenate([day1, day2])


# =============================================================================
#  SCENARIO RUNNERS
# =============================================================================

@st.cache_data(show_spinner=False)
def run_seasonal(season_key, arrival_h, departure_h,
                 soc_pct, soc_departure_pct, tru_cycle,
                 do_B, do_C, do_D, mpc_noise_std=0.0):
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
            Pc,Pd,soc = run_D_mpc(v2g,buy48,v2gp48,E_init,tru_w,noise_std=mpc_noise_std)
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
            Pc,Pd,soc = run_D_mpc(v2g,buy_w,v2gp_w,E_init,tru_w,noise_std=mpc_noise_std)
            results.append(make_kpi("D - MPC (receding)",v2g,Pc,Pd,soc,
                                    buy_w,v2gp_w,E_init,arr,dep,tru_w=tru_w))

        rc = compute_reefer_costs(tru_w, buy_w, v2g.dt_h)
        return results, results, buy_d, plug_d, hours_d, is_wknd, is_48h, tru_d, rc


@st.cache_data(show_spinner=False)
def run_specific_date(date_str, arrival_h, departure_h,
                      soc_pct, soc_departure_pct, tru_cycle,
                      do_B, do_C, do_D, mpc_noise_std=0.0):
    ts      = pd.Timestamp(date_str)
    is_wknd = ts.dayofweek >= 5
    v2g     = V2GParams(soc_departure_pct=soc_departure_pct)
    E_init  = v2g.usable_capacity_kWh * soc_pct / 100.0

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
    else:
        buy_192  = load_two_day_profile(date_str)
        v2gp_192 = buy_192.copy()
        ROLL     = round(12.0 / v2g.dt_h)
        arr_slot = round(arrival_h   / v2g.dt_h) % 96
        dep_slot = round(departure_h / v2g.dt_h) % 96
        buy_w    = buy_192[arr_slot : 96 + dep_slot]
        v2gp_w   = buy_w.copy()
        W        = len(buy_w)
        buy_d    = buy_192[ROLL : ROLL + 96]
        hours_d  = np.arange(96) * v2g.dt_h + 12.0
        dep_on_chart = (departure_h + 24.0) if departure_h < 12.0 else departure_h
        plug_d   = ((hours_d >= arrival_h) & (hours_d < dep_on_chart)).astype(float)
        arr_disp = arr_slot - ROLL
        dep_disp = ROLL     + dep_slot
        arr, dep = arr_disp, dep_disp
        tru_w    = get_tru_15min_trace(tru_cycle, W, v2g.dt_h)
        tru_d    = np.zeros(96)
        d_s = max(0, arr_disp); d_e = min(96, dep_disp)
        w_s = d_s - arr_disp;   w_e = w_s + (d_e - d_s)
        if w_e > w_s:
            tru_d[d_s:d_e] = tru_w[w_s:w_e]
        is_48h          = False
        is_wknd_fullday = False

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
        Pc, Pd, soc = run_D_mpc(v2g, buy_w, v2gp_w, E_init, tru_w, noise_std=mpc_noise_std)
        results.append(make_kpi("D - MPC (receding)", v2g, Pc, Pd, soc,
                                buy_w, v2gp_w, E_init, arr, dep, tru_w=tru_w))

    rc = compute_reefer_costs(tru_w, buy_w, v2g.dt_h)
    return (results, buy_d, plug_d, hours_d,
            is_wknd, is_48h, is_wknd_fullday, tru_d, rc)


# =============================================================================
#  ANNUAL COMPUTATION — all days in CSV individually
# =============================================================================

@st.cache_data(show_spinner=False)
def run_annual_all_days(
    arrival_h, departure_h,
    soc_w, soc_s, soc_dep,
    tru_cycle, do_B, do_C, do_D, mpc_noise_std,
    fixed_net_ct, vat_rate,
    v2g_double_ct, v2g_exempt_ct, vat_fut_rate,
    fixed_price_eur,
):
    """
    Runs optimisation for every available date in the 2025 CSV individually.
    Winter = Oct-Mar (months 10,11,12,1,2,3).
    Summer = Apr-Sep (months 4,5,6,7,8,9).
    Weekdays : overnight window arrival_h -> departure_h next morning.
    Weekends : full 24h plugged-in.
    Missing dates silently skipped.
    F benchmark uses Scenario A charge_kwh x fixed_price_eur (no V2G).
    Charge cost uses all-in price (spot + fixed fees + VAT).
    V2G rev current : max(0, v2g_kwh x avg_spot - v2g_kwh x double_tax x (1+VAT)).
    V2G rev future  : v2g_kwh x avg_spot + v2g_kwh x exempt_ct x (1+VAT_fut).
    Annual = raw sum of all valid days (no averaging).
    """
    df_all    = _load_csv_raw(CSV_PATH)
    all_dates = sorted(df_all["date"].unique())

    sc_keys = ["A"]
    if do_B: sc_keys.append("B")
    if do_C: sc_keys.append("C")
    if do_D: sc_keys.append("D")

    double_tax_eur = v2g_double_ct / 100.0
    exempt_eur     = v2g_exempt_ct / 100.0

    acc = {sc: {"charge_cost": 0.0, "v2g_rev_cur": 0.0, "v2g_rev_fut": 0.0}
           for sc in sc_keys}
    acc["F"] = {"charge_cost": 0.0, "v2g_rev_cur": 0.0, "v2g_rev_fut": 0.0}

    n_valid = 0

    for date in all_dates:
        date_str = str(date)
        ts       = pd.Timestamp(date_str)
        is_win   = ts.month in WINTER_M
        soc_init = float(soc_w) if is_win else float(soc_s)

        try:
            (results, buy_d, _plug, _hours,
             _is_wknd, _is_48h, _is_wknd_fd, _tru_d, _rc) = run_specific_date(
                date_str, arrival_h, departure_h,
                soc_init, float(soc_dep),
                tru_cycle, do_B, do_C, do_D, mpc_noise_std,
            )
        except Exception:
            continue  # skip missing / broken dates silently

        avg_spot_eur  = float(np.mean(buy_d))
        avg_allin_eur = float(np.mean(
            to_allin_ct(buy_d, fixed_net_ct, vat_rate))) / 100.0

        result_A = results[0]

        # F benchmark: A's charge_kwh at fixed tariff, no V2G
        acc["F"]["charge_cost"] += result_A["charge_kwh"] * fixed_price_eur

        # Scenarios A ... D
        for i, sc in enumerate(sc_keys):
            r           = results[i]
            charge_cost = r["charge_kwh"] * avg_allin_eur
            v2g_rev_cur = max(0.0,
                r["v2g_kwh"] * avg_spot_eur
                - r["v2g_kwh"] * double_tax_eur * (1.0 + vat_rate))
            v2g_rev_fut = (
                r["v2g_kwh"] * avg_spot_eur
                + r["v2g_kwh"] * exempt_eur * (1.0 + vat_fut_rate))

            acc[sc]["charge_cost"] += charge_cost
            acc[sc]["v2g_rev_cur"] += v2g_rev_cur
            acc[sc]["v2g_rev_fut"] += v2g_rev_fut

        n_valid += 1

    out_scenarios = ["F"] + sc_keys
    out = {"n_days": n_valid, "scenarios": out_scenarios}
    for sc in out_scenarios:
        cc     = acc[sc]["charge_cost"]
        vr_cur = acc[sc]["v2g_rev_cur"]
        vr_fut = acc[sc]["v2g_rev_fut"]
        out[sc] = {
            "charge_cost": cc,
            "v2g_rev_cur": vr_cur,
            "v2g_rev_fut": vr_fut,
            "net_cur":     cc - vr_cur,
            "net_fut":     cc - vr_fut,
        }
    return out

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
    "fixed_price":   FIXED_PRICE_EUR_KWH,
    "mode":          "Seasonal Average",
    "specific_date": "2025-01-15",
    "mpc_noise_std": 0.0,
    # Tariff fields — current regulation
    "t_network_fee":    6.63,
    "t_concession":     1.992,
    "t_offshore":       0.941,
    "t_chp":            0.446,
    "t_elec_tax":       2.05,
    "t_nev19":          1.559,
    "t_vat":            19.0,
    # Future — exempt fees (MiSpeL)
    "t_fut_network":    6.63,
    "t_fut_concession": 1.992,
    "t_fut_offshore":   0.941,
    "t_fut_chp":        0.446,
    "t_fut_elec_tax":   2.05,
    "t_fut_nev19":      1.559,
    "t_fut_vat":        19.0,
}

if "cfg" not in st.session_state:
    st.session_state.cfg = DEFAULTS.copy()
if "show_output" not in st.session_state:
    st.session_state.show_output = False


# =============================================================================
#  INPUT PANEL
# =============================================================================

def render_input_panel():
    st.title("S.KOe COOL 2.0 - V2G Optimisation")
    st.caption("Master's Thesis 2026  |  Kuldip Bhadreshvara")
    st.markdown("---")

    if not Path(CSV_PATH).exists():
        st.error(f"Price CSV '{CSV_PATH}' not found.")
        st.stop()

    cfg = st.session_state.cfg

    with st.form("input_form", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns([1.15, 1.15, 1.0, 0.9])

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
                    "**Seasonal Average:** Average prices of full 2025.\n\n"
                    "**Specific Date:** Actual prices for that exact date."
                )
            )
            cfg["mode"] = mode

            if mode == "Specific Date":
                date_val = st.date_input(
                    "Select date (2025)",
                    value=pd.Timestamp(cfg.get("specific_date", "2025-01-15")),
                    min_value=pd.Timestamp("2025-01-01"),
                    max_value=pd.Timestamp("2025-12-31"),
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

            st.markdown("##### Fixed-Tariff Benchmark")
            cfg["fixed_price"] = st.number_input(
                "Fixed price (EUR/kWh)",
                value=float(cfg["fixed_price"]),
                min_value=0.05, max_value=1.0, step=0.01,
                help="Comparison baseline only.")

            st.markdown("##### Tariff Configuration")

            # ── Current Regulation ─────────────────────────────────────────
            st.markdown(
                "<div style='background:#1565C0;color:white;padding:2px 8px;"
                "border-radius:4px;font-size:11px;font-weight:bold;margin-bottom:4px;'>"
                "📋 Current — All-in Charges (ct/kWh)</div>",
                unsafe_allow_html=True
            )
            _ta, _tb = st.columns(2)
            with _ta:
                cfg["t_network_fee"] = st.number_input(
                    "Netzentgelt", value=float(cfg["t_network_fee"]),
                    min_value=0.0, max_value=20.0, step=0.01, format="%.3f",
                    key="ti_nf")
                cfg["t_concession"]  = st.number_input(
                    "Konzession", value=float(cfg["t_concession"]),
                    min_value=0.0, max_value=10.0, step=0.001, format="%.3f",
                    key="ti_con")
                cfg["t_offshore"]    = st.number_input(
                    "Offshore", value=float(cfg["t_offshore"]),
                    min_value=0.0, max_value=10.0, step=0.001, format="%.3f",
                    key="ti_off")
            with _tb:
                cfg["t_chp"]         = st.number_input(
                    "KWKG", value=float(cfg["t_chp"]),
                    min_value=0.0, max_value=5.0, step=0.001, format="%.3f",
                    key="ti_chp")
                cfg["t_elec_tax"]    = st.number_input(
                    "Stromsteuer", value=float(cfg["t_elec_tax"]),
                    min_value=0.0, max_value=10.0, step=0.01, format="%.3f",
                    key="ti_et")
                cfg["t_nev19"]       = st.number_input(
                    "NEV-19", value=float(cfg["t_nev19"]),
                    min_value=0.0, max_value=10.0, step=0.001, format="%.3f",
                    key="ti_nev")
            cfg["t_vat"] = st.number_input(
                "VAT (%)", value=float(cfg["t_vat"]),
                min_value=0.0, max_value=30.0, step=0.1, format="%.1f",
                key="ti_vat")
            _fn  = (cfg["t_network_fee"] + cfg["t_concession"] + cfg["t_offshore"]
                    + cfg["t_chp"] + cfg["t_elec_tax"] + cfg["t_nev19"])
            _vat = cfg["t_vat"] / 100.0
            st.caption(
                f"Fixed net: **{_fn:.3f} ct/kWh** | VAT: **{cfg['t_vat']:.1f}%** | "
                f"@ 10 ct spot → **{(10.0 + _fn) * (1 + _vat):.2f} ct all-in**"
            )

            # ── Future Regulation (MiSpeL) ─────────────────────────────────
            # ── Future Regulation (MiSpeL) ─────────────────────────────────
            st.markdown(
                "<div style='background:#2E7D32;color:white;padding:2px 8px;"
                "border-radius:4px;font-size:11px;font-weight:bold;"
                "margin-bottom:4px;margin-top:6px;'>"
                "🔮 Future (MiSpeL) — Exempt Fees on V2G Export (ct/kWh)</div>",
                unsafe_allow_html=True
            )
            _fe, _ff = st.columns(2)
            with _fe:
                cfg["t_fut_network"] = st.number_input(
                    "Netzentgelt", value=float(cfg.get("t_fut_network", 6.63)),
                    min_value=0.0, max_value=20.0, step=0.01, format="%.3f",
                    key="tf_nf", help="Set to 0 to remove exemption")
                cfg["t_fut_concession"] = st.number_input(
                    "Konzession", value=float(cfg.get("t_fut_concession", 1.992)),
                    min_value=0.0, max_value=10.0, step=0.001, format="%.3f",
                    key="tf_con", help="Set to 0 to remove exemption")
                cfg["t_fut_offshore"] = st.number_input(
                    "Offshore", value=float(cfg.get("t_fut_offshore", 0.941)),
                    min_value=0.0, max_value=10.0, step=0.001, format="%.3f",
                    key="tf_off", help="Set to 0 to remove exemption")
            with _ff:
                cfg["t_fut_chp"] = st.number_input(
                    "KWKG", value=float(cfg.get("t_fut_chp", 0.446)),
                    min_value=0.0, max_value=5.0, step=0.001, format="%.3f",
                    key="tf_chp", help="Set to 0 to remove exemption")
                cfg["t_fut_elec_tax"] = st.number_input(
                    "Stromsteuer", value=float(cfg.get("t_fut_elec_tax", 2.05)),
                    min_value=0.0, max_value=10.0, step=0.01, format="%.3f",
                    key="tf_et", help="Set to 0 to remove exemption")
                cfg["t_fut_nev19"] = st.number_input(
                    "NEV-19", value=float(cfg.get("t_fut_nev19", 1.559)),
                    min_value=0.0, max_value=10.0, step=0.001, format="%.3f",
                    key="tf_nev", help="Set to 0 to remove exemption")
            cfg["t_fut_vat"] = st.number_input(
                "VAT on exempt (%)", value=float(cfg.get("t_fut_vat", 19.0)),
                min_value=0.0, max_value=30.0, step=0.1, format="%.1f",
                key="tf_vat")
            _fut_total = (cfg["t_fut_network"] + cfg["t_fut_concession"]
                          + cfg["t_fut_offshore"] + cfg["t_fut_chp"]
                          + cfg["t_fut_elec_tax"] + cfg["t_fut_nev19"])
            _vat_fut = cfg["t_fut_vat"] / 100.0
            st.caption(
                f"Total exempt: **{_fut_total:.3f} ct/kWh** "
                f"(+VAT: **{_fut_total * (1 + _vat_fut):.3f} ct**) added to V2G rev | "
                f"Pending EU state aid approval (BNetzA 2025)"
            )

        with c2:
            st.subheader("State of Charge")
            cfg["soc_winter"]    = st.slider(
                "Winter arrival SoC (%)", 20, 100, int(cfg["soc_winter"]))
            cfg["soc_summer"]    = st.slider(
                "Summer arrival SoC (%)", 20, 100, int(cfg["soc_summer"]))
            cfg["soc_departure"] = st.slider(
                "Departure target SoC (%)", 50, 100, int(cfg["soc_departure"]))

            st.markdown("##### Scenarios to Run")
            cfg["do_B"] = st.checkbox(
                "B -- Smart charging (no V2G)", bool(cfg["do_B"]), key="form_do_B")
            cfg["do_C"] = st.checkbox(
                "C -- MILP Day-Ahead + V2G",   bool(cfg["do_C"]), key="form_do_C")
            cfg["do_D"] = st.checkbox(
                "D -- MPC receding horizon", bool(cfg["do_D"]), key="form_do_D")
            if st.session_state.get("form_do_D", bool(cfg["do_D"])):
                cfg["mpc_noise_std"] = st.slider(
                    "MPC forecast noise σ (EUR/kWh)",
                    min_value=0.000, max_value=0.050,
                    value=float(cfg.get("mpc_noise_std", 0.0)),
                    step=0.001, format="%.3f",
                    help=(
                        "0.000 = perfect foresight (MPC = MILP).\n\n"
                        "0.012 = realistic intraday uncertainty (Liu 2023).\n\n"
                        "Higher = MPC makes worse decisions due to forecast error."
                    ),
                    key="form_mpc_noise"
                )
            else:
                cfg["mpc_noise_std"] = 0.0
                st.caption("↑ Enable MPC to set forecast noise")

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

        with c4:
            st.subheader("Extras")
            cfg["do_wwe"] = st.checkbox(
                "Winter weekend (48h)", bool(cfg["do_wwe"]),
                help="Adds a 48h Sat+Sun block below the weekday charts",
                key="form_do_wwe")
            cfg["do_swe"] = st.checkbox(
                "Summer weekend (48h)", bool(cfg["do_swe"]),
                key="form_do_swe")

            st.markdown("---")
            st.markdown("**S.KOe COOL specs**")
            st.caption("70 kWh total / 60 kWh usable")
            st.caption("22 kW AC bidirectional OBC")
            st.caption("Cold-chain floor: SoC >= 20%")

        # ── Submit ─────────────────────────────────────────────────────────────
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
#  ROUTING
# =============================================================================

if not st.session_state.show_output:
    render_input_panel()
    st.stop()


# =============================================================================
#  RESULTS PAGE
# =============================================================================

st.title("S.KOe COOL 2.0 - V2G Optimisation Results")
st.caption("Master's Thesis 2026  |  Kuldip Bhadreshvara")

cfg           = st.session_state.cfg
arr_h         = parse_hhmm(cfg["arrival_str"],   16.0)
dep_h         = parse_hhmm(cfg["departure_str"],  6.0)
soc_w         = int(cfg["soc_winter"])
soc_s         = int(cfg["soc_summer"])
soc_dep       = int(cfg["soc_departure"])
tru_cycle     = cfg["tru_cycle"]
mpc_noise_std = float(cfg.get("mpc_noise_std", 0.0))
w_months      = 6
s_months      = 6
do_B          = bool(cfg["do_B"])
do_C          = bool(cfg["do_C"])
do_D          = bool(cfg["do_D"])
fixed_price   = float(cfg["fixed_price"])
mode          = cfg.get("mode", "Seasonal Average")
specific_date = cfg.get("specific_date", "2025-01-15")

# Derive tariff values from cfg
_fixed_net_ct = (cfg["t_network_fee"] + cfg["t_concession"] + cfg["t_offshore"]
                 + cfg["t_chp"] + cfg["t_elec_tax"] + cfg["t_nev19"])
_vat_rate     = cfg["t_vat"] / 100.0
_v2g_double_ct = cfg["t_network_fee"] + cfg["t_elec_tax"]   # current: re-applied on V2G
_v2g_exempt_ct = (cfg.get("t_fut_network", 6.63) + cfg.get("t_fut_concession", 1.992)
                  + cfg.get("t_fut_offshore", 0.941) + cfg.get("t_fut_chp", 0.446)
                  + cfg.get("t_fut_elec_tax", 2.05) + cfg.get("t_fut_nev19", 1.559))
_vat_fut_rate  = cfg.get("t_fut_vat", 19.0) / 100.0

# ── Sidebar quick-edit ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Quick Edit")
    st.caption("Changes apply immediately")
    cfg["arrival_str"]   = st.text_input("Arrival (HH:MM)",   cfg["arrival_str"])
    cfg["departure_str"] = st.text_input("Departure (HH:MM)", cfg["departure_str"])
    cfg["soc_winter"]    = st.slider("Winter arrival SoC (%)",   20, 100, soc_w)
    cfg["soc_summer"]    = st.slider("Summer arrival SoC (%)",   20, 100, soc_s)
    cfg["soc_departure"] = st.slider("Departure target SoC (%)", 50, 100, soc_dep)
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
    arr_h         = parse_hhmm(cfg["arrival_str"],   16.0)
    dep_h         = parse_hhmm(cfg["departure_str"],  6.0)
    soc_w         = int(cfg["soc_winter"])
    soc_s         = int(cfg["soc_summer"])
    soc_dep       = int(cfg["soc_departure"])
    tru_cycle     = cfg["tru_cycle"]
    w_months      = 6
    s_months      = 6
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
            f"Mode: **{mode}**"
            + (f"  |  Date: **{specific_date}**" if mode == "Specific Date" else "")
        )
    except Exception as e:
        st.error(f"Could not load CSV: {e}"); st.stop()

v2g     = V2GParams(soc_departure_pct=float(soc_dep))
tru_avg = tru_avg_kw(tru_cycle)

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
#  ANNUAL GRAPHS
# =============================================================================

def make_annual_graphs(annual_data):
    """3 side-by-side bar charts: Charge Cost | V2G Revenue | Net Cost."""
    sc_list  = annual_data["scenarios"]   # ["F","A", ...]
    n_days   = annual_data["n_days"]

    SC_COLOR_MAP = {"F": "#78909C", "A": SC_COL["A"],
                    "B": SC_COL.get("B","#F57C00"), "C": SC_COL.get("C","#6A1B9A"),
                    "D": SC_COL.get("D","#00838F")}
    sc_labels = {"F": "F\nFixed", "A": "A\nDumb",
                 "B": "B\nSmart", "C": "C\nMILP", "D": "D\nMPC"}
    x_labels  = [sc_labels.get(sc, sc) for sc in sc_list]
    colors    = [SC_COLOR_MAP.get(sc, "#888888") for sc in sc_list]

    st.markdown(
        "<div style='background:#37474F;color:white;padding:5px 14px;"
        "border-radius:5px;font-weight:bold;font-size:14px;margin-bottom:6px;'>"
        f"📊 Annual Summary — {n_days} days computed (2025, day-by-day optimisation)"
        "</div>",
        unsafe_allow_html=True,
    )

    tab_cur, tab_fut = st.tabs([
        "📋 Current Regulation (2025)",
        "🔮 Future Regulation (MiSpeL)",
    ])

    for tab, reg, reg_lbl in [
        (tab_cur, "cur", "Current Regulation"),
        (tab_fut, "fut", "Future Regulation (MiSpeL)"),
    ]:
        with tab:
            charge_vals = [annual_data[sc]["charge_cost"]       for sc in sc_list]
            v2g_vals    = [annual_data[sc][f"v2g_rev_{reg}"]    for sc in sc_list]
            net_vals    = [annual_data[sc][f"net_{reg}"]        for sc in sc_list]

            graph_specs = [
                ("Annual Charge Cost (€/year)",   charge_vals),
                ("Annual V2G Revenue (€/year)",   v2g_vals),
                ("Annual Net Cost (€/year)",      net_vals),
            ]

            fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
            fig.patch.set_facecolor("#F8F9FA")

            for ax, (title, vals) in zip(axes, graph_specs):
                ax.set_facecolor("#FFFFFF")
                bars = ax.bar(
                    x_labels, vals, color=colors,
                    edgecolor="white", linewidth=0.8,
                    zorder=3, width=0.55,
                )
                ax.set_title(f"{title}\n{reg_lbl}",
                             fontsize=9, fontweight="bold", pad=5)
                ax.set_ylabel("EUR / year", fontsize=8)
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
                ax.grid(True, axis="y", alpha=0.22, zorder=0)
                ax.axhline(0, color="black", lw=0.6)
                ax.tick_params(axis="x", labelsize=8)
                ax.tick_params(axis="y", labelsize=7)

                for bar, val in zip(bars, vals):
                    h = bar.get_height()
                    offset = max(abs(h) * 0.01, 8)
                    if h >= 0:
                        ypos, va = h + offset, "bottom"
                    else:
                        ypos, va = h - offset, "top"
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            ypos, f"€{val:,.0f}",
                            ha="center", va=va, fontsize=7, fontweight="bold")

            plt.tight_layout(pad=0.8)
            st.image(fig_to_buf(fig), use_container_width=True)


# =============================================================================
#  KPI TABLE HELPER
# =============================================================================

def show_kpi_table(results, fixed_price, tru_cycle, rc, label="", buy_d=None):
    if buy_d is not None and len(buy_d) > 0:
        avg_spot_eur  = float(np.mean(buy_d))
        avg_allin_eur = float(np.mean(to_allin_ct(buy_d, _fixed_net_ct, _vat_rate))) / 100.0
    else:
        ref = results[0]
        avg_spot_eur  = (ref["charge_cost"] / ref["charge_kwh"]
                         if ref["charge_kwh"] > 0.01 else 0.10)
        avg_allin_eur = float(np.mean(
            to_allin_ct(np.array([avg_spot_eur]), _fixed_net_ct, _vat_rate))) / 100.0

    double_tax_eur = _v2g_double_ct / 100.0
    exempt_eur     = _v2g_exempt_ct / 100.0

    def _build_rows(reg):
        ref_A         = results[0]
        F_charge_cost = ref_A["charge_kwh"] * fixed_price
        F_net_cost    = F_charge_cost

        rows = [{
            "Scenario"                  : f"F  Fixed@{fixed_price:.2f}€/kWh",
            "Charge (€/d)"              : f"{F_charge_cost:.3f}",
            "V2G Rev (€/d)"             : "0.000",
            "Net (€/d)"                 : f"{F_charge_cost:.3f}"
        }]

        for r in results:
            charge_cost = r["charge_kwh"] * avg_allin_eur
            if reg == "current":
                v2g_rev = max(0.0,
                    r["v2g_kwh"] * avg_spot_eur
                    - r["v2g_kwh"] * double_tax_eur * (1.0 + _vat_rate))
            else:
                v2g_rev = (r["v2g_kwh"] * avg_spot_eur
                           + r["v2g_kwh"] * exempt_eur * (1.0 + _vat_rate))
            net = charge_cost - v2g_rev
            rows.append({
                "Scenario"     : (r["label"]
                                  .replace(" (no V2G)","")
                                  .replace(" Day-Ahead","")
                                  .replace(" (receding)","")),
                "Charge (€/d)" : f"{charge_cost:.3f}",
                "V2G Rev (€/d)": f"{v2g_rev:.3f}",
                "Net (€/d)"    : f"{net:.3f}"
            })
        return rows

    hdr = f"**EV Charging KPI{' — ' + label if label else ''}**"
    st.markdown(hdr)

    tab_cur, tab_fut = st.tabs([
        "📋 Current Regulation (2025)",
        "🔮 Future Regulation (MiSpeL)"
    ])

    with tab_cur:
        st.caption(
            f"Charge: all-in ≈{avg_allin_eur*100:.1f} ct/kWh  |  "
            f"V2G: spot minus re-applied Netzentgelt+Stromsteuer "
            f"({_v2g_double_ct:.2f} ct × 1+VAT = {_v2g_double_ct*(1+_vat_rate):.2f} ct penalty)"
        )
        st.dataframe(pd.DataFrame(_build_rows("current")),
                     use_container_width=True, hide_index=True)

    with tab_fut:
        st.caption(
            f"Charge: same all-in  |  "
            f"V2G: spot + exempt fees "
            f"({_v2g_exempt_ct:.2f} ct × 1+VAT = {_v2g_exempt_ct*(1+_vat_rate):.2f} ct bonus)  |  "
            f"MiSpeL Pauschaloption — pending EU state aid approval (BNetzA 2025)"
        )
        st.dataframe(pd.DataFrame(_build_rows("future")),
                     use_container_width=True, hide_index=True)

    if tru_cycle != "OFF" and rc["E_kWh"] > 0.01:
        st.markdown("**Reefer (TRU) Energy Cost**")
        st.dataframe(pd.DataFrame([
            ["TRU energy (kWh/d)",                        f"{rc['E_kWh']:.2f}"],
            ["Grid spot cost (EUR/d)",                    f"EUR {rc['cost_dynamic']:.3f}"],
            [f"Fixed @EUR{fixed_price:.2f}/kWh (EUR/d)", f"EUR {rc['E_kWh']*fixed_price:.3f}"],
            ["Diesel genset (EUR/d)",                     f"EUR {rc['cost_diesel']:.3f}"],
            ["Diesel (L/d)",                              f"{rc['diesel_liters']:.2f} L"],
            ["Grid vs diesel saving (EUR/d)",
             f"EUR {rc['cost_diesel'] - rc['cost_dynamic']:+.3f}"],
        ], columns=["Metric", "Value"]),
        use_container_width=True, hide_index=True)


# =============================================================================
#  MAIN DISPLAY
# =============================================================================

all_season_res_kpi = {}
all_reefer_costs   = {}

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

    render_season_block(
        v2g, day_label, color_hex,
        hours_d, buy_d, plug_d, results,
        arr_h, dep_h, is_48h, is_wknd_fullday,
        do_B, do_C, do_D, tru_d,
        fixed_net_ct=_fixed_net_ct, vat_rate=_vat_rate
    )
    st.markdown("---")
    show_kpi_table(results, fixed_price, tru_cycle, rc, buy_d=buy_d)

else:
    res_w = None
    with st.spinner("Computing Winter Weekday..."):
        try:
            (res_w, res_w_kpi, buy_d_w, plug_d_w, hours_d_w,
             is_wknd_w, is_48h_w, tru_d_w, rc_w) = run_seasonal(
                "winter_weekday", arr_h, dep_h,
                float(soc_w), float(soc_dep),
                tru_cycle, do_B, do_C, do_D, mpc_noise_std
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
            do_B, do_C, do_D, tru_d_w,
            fixed_net_ct=_fixed_net_ct, vat_rate=_vat_rate
        )

    st.markdown(
        "<hr style='border:1px solid #BBBBBB;margin:8px 0 8px 0;'>",
        unsafe_allow_html=True
    )

    res_s = None
    with st.spinner("Computing Summer Weekday..."):
        try:
            (res_s, res_s_kpi, buy_d_s, plug_d_s, hours_d_s,
             is_wknd_s, is_48h_s, tru_d_s, rc_s) = run_seasonal(
                "summer_weekday", arr_h, dep_h,
                float(soc_s), float(soc_dep),
                tru_cycle, do_B, do_C, do_D, mpc_noise_std
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
            do_B, do_C, do_D, tru_d_s,
            fixed_net_ct=_fixed_net_ct, vat_rate=_vat_rate
        )

    st.markdown("---")

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
            do_B, do_C, do_D, tru_d_we,
            fixed_net_ct=_fixed_net_ct, vat_rate=_vat_rate
        )
        st.markdown("---")
        show_kpi_table(res_we, fixed_price, tru_cycle, rc_we, lbl, buy_d=buy_d_we)
        st.markdown("---")
# ── Annual graphs ──────────────────────────────────────────────────────────
        with st.spinner("Running annual day-by-day optimisation for all 365 days..."):
            try:
                annual_data = run_annual_all_days(
                    arr_h, dep_h,
                    soc_w, soc_s, soc_dep,
                    tru_cycle, do_B, do_C, do_D, mpc_noise_std,
                    _fixed_net_ct, _vat_rate,
                    _v2g_double_ct, _v2g_exempt_ct, _vat_fut_rate,
                    fixed_price,
                )
                make_annual_graphs(annual_data)
            except Exception as e:
                st.warning(f"Annual computation error: {e}")
        st.markdown("---")

    st.subheader("KPI Tables")
    tab_specs = []
    if res_w is not None: tab_specs.append(("Winter Weekday", res_w, rc_w, buy_d_w))
    if res_s is not None: tab_specs.append(("Summer Weekday", res_s, rc_s, buy_d_s))

    if tab_specs:
        tab_objs = st.tabs([t[0] for t in tab_specs])
        for tab_obj, (lbl, res, rc, buy_d_tab) in zip(tab_objs, tab_specs):
            with tab_obj:
                show_kpi_table(res, fixed_price, tru_cycle, rc, lbl, buy_d=buy_d_tab)


# =============================================================================
#  METHODOLOGY
# =============================================================================

with st.expander("Methodology & Assumptions", expanded=False):
    st.markdown(f"""
**Price data:** SMARD DE/LU 15-min day-ahead spot prices.

**Charts:** All-in price = (spot + {_fixed_net_ct:.3f} ct/kWh fixed) × {1+_vat_rate:.2f} VAT.

**Current regulation:** V2G export kWh penalised by re-applied Netzentgelt + Stromsteuer ({_v2g_double_ct:.2f} ct × 1+VAT).

**Future regulation (MiSpeL):** V2G export kWh receives exemption of {_v2g_exempt_ct:.2f} ct/kWh (× 1+VAT). Pending EU state aid approval.

**Cold-chain floor:** SoC >= 20% hard MILP constraint.

**Departure target:** SoC >= {soc_dep}%.

**Battery:** 70 kWh total / 60 kWh usable | eta_c = 0.92 | eta_d = 0.92

**Fixed-tariff benchmark:** EUR {fixed_price:.2f}/kWh (comparison only).
    """)