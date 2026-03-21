"""
S.KOe COOL -- V2G Optimisation  |  Streamlit App (v6)
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


# =============================================================================
#  CHART — season overview (3-panel: price | power | SoC)
#  Built directly in app.py so no dependency on plot_season_chart
# =============================================================================

def plot_season_chart_inline(v2g, label, buy_d, plug_d, hours_d,
                              results, arrival_h, departure_h,
                              is_48h, tru_d=None, tru_cycle="OFF"):
    """
    3-panel season chart returned as BytesIO PNG.
    Panel 1 : Day-ahead price + plugged-in shading
    Panel 2 : Charge / discharge power (fill + line, all scenarios)
    Panel 3 : Battery SoC % (gradual ramp, all scenarios)
    """
    if is_48h:
        pos  = np.arange(0, 49, 4)
        xlbl = [f"{'Sat' if h < 24 else 'Sun'}\n{int(h % 24):02d}:00" for h in pos]
        xmin, xmax = 0, 48
    else:
        pos  = np.arange(12, 37, 2)
        xlbl = [f"{int(h % 24):02d}:00" for h in pos]
        xmin, xmax = 12, 36

    def dx(h):
        return h if (is_48h or h >= 12.) else h + 24.

    tru_label = (f"TRU: {tru_cycle} ({tru_avg_kw(tru_cycle):.1f} kW avg)"
                 if tru_cycle.strip().lower() not in ("off", "noreeferstationary")
                 else "TRU: OFF")

    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(15, 13),
        gridspec_kw={"height_ratios": [1.0, 1.8, 1.8], "hspace": 0.78})
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        f"S.KOe COOL  --  {label}\n"
        f"Arrival {fmt_hhmm(arrival_h)} | Departure {fmt_hhmm(departure_h)} | "
        f"Battery {v2g.usable_capacity_kWh:.0f} kWh usable | "
        f"{tru_label} | Dep. target {v2g.soc_departure_pct:.0f}%",
        fontsize=10, fontweight="bold", y=0.99)

    def _fmt(ax, ylabel, title):
        ax.set_xlim(xmin, xmax)
        ax.set_xticks(pos)
        ax.set_xticklabels(xlbl, fontsize=7.5, rotation=35, ha="right")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left", pad=4)
        ax.grid(True, alpha=0.22, zorder=0)

    def _vlines(ax):
        if is_48h:
            ax.axvline(24., color="#555555", lw=1.2, ls="--", alpha=0.65, zorder=5)
        else:
            ax.axvline(dx(arrival_h),   color="#1B5E20", lw=1.1, ls=":", alpha=0.80, zorder=5)
            ax.axvline(dx(departure_h), color="#B71C1C", lw=1.1, ls=":", alpha=0.80, zorder=5)
            ax.axvline(dx(0.),          color="#555555", lw=1.2, ls="--", alpha=0.65, zorder=5)

    def _leg_below(ax, handles, ncol=4):
        ax.legend(handles=handles, fontsize=8, ncol=ncol,
                  loc="upper center", bbox_to_anchor=(0.5, -0.40),
                  framealpha=0.95, edgecolor="#CCCCCC")

    # ── Panel 1: Price ────────────────────────────────────────────────────────
    for t in range(len(hours_d)):
        if plug_d[t] > 0.5:
            ax0.axvspan(hours_d[t], hours_d[t] + 0.25,
                        color="gold", alpha=0.22, lw=0, zorder=1)
    p_line, = ax0.step(hours_d, buy_d * 1000, where="post",
                       color="#2E7D32", lw=2.0, zorder=3,
                       label="Day-ahead price (EUR/MWh)")
    ax0.fill_between(hours_d, buy_d * 1000, step="post",
                     color="#2E7D32", alpha=0.10, zorder=2)
    _vlines(ax0)
    leg0 = [p_line, mpatches.Patch(color="gold", alpha=0.5, label="Plugged-in")]
    if is_48h:
        leg0.append(Line2D([0],[0], color="#555555", ls="--", lw=1.3, label="Sat midnight"))
    else:
        leg0 += [
            Line2D([0],[0], color="#1B5E20", ls=":", lw=1.3,
                   label=f"Arrival {fmt_hhmm(arrival_h)}"),
            Line2D([0],[0], color="#B71C1C", ls=":", lw=1.3,
                   label=f"Departure {fmt_hhmm(departure_h)}"),
            Line2D([0],[0], color="#555555", ls="--", lw=1.3, label="Midnight"),
        ]
    _leg_below(ax0, leg0, ncol=5)
    _fmt(ax0, "EUR / MWh", "(1) Day-Ahead Electricity Price + Plugged-In Availability")

    # ── Panel 2: Power ────────────────────────────────────────────────────────
    leg1 = []
    for key, r in zip(["A", "B", "C", "D"], results):
        col  = SC_COL[key]; fill = SC_FILL[key]
        lbl  = r["label"].split("(")[0].strip()
        Pc   = r["Pc_d"]; Pd = r["Pd_d"]
        ax1.fill_between(hours_d, Pc, step="post", color=fill, alpha=0.38, zorder=2)
        hc, = ax1.step(hours_d, Pc, where="post", color=col, lw=1.5, alpha=0.90,
                       zorder=3, label=f"{lbl} charge")
        leg1.append(hc)
        if r["v2g_kwh"] > 0.05:
            ax1.fill_between(hours_d, -Pd, step="post", color=fill, alpha=0.28, zorder=2)
            hd, = ax1.step(hours_d, -Pd, where="post", color=col, lw=1.5, ls="--",
                           alpha=0.90, zorder=3, label=f"{lbl} V2G (shown -)")
            leg1.append(hd)
    if tru_d is not None and np.any(tru_d > 0.01):
        ht, = ax1.step(hours_d, -tru_d, where="post",
                       color="#C62828", lw=1.4, ls=":", alpha=0.80, zorder=4,
                       label=f"TRU load (shown -)")
        leg1.append(ht)
    ax1.axhline(0, color="black", lw=0.7)
    _vlines(ax1)
    _leg_below(ax1, leg1, ncol=3)
    _fmt(ax1, "Power (kW)",
         "(2) Charge / Discharge Power  [solid=charge | dashed below 0=V2G | dotted=TRU]")

    # ── Panel 3: SoC ──────────────────────────────────────────────────────────
    for t in range(len(hours_d)):
        if plug_d[t] > 0.5:
            ax2.axvspan(hours_d[t], hours_d[t] + 0.25,
                        color="gold", alpha=0.12, lw=0, zorder=1)
    leg2 = []
    for key, r, ls in zip(["A", "B", "C", "D"], results, ["-", "-", "-", "--"]):
        col = SC_COL[key]; lbl = r["label"].split("(")[0].strip()
        xr, yr = soc_ramp(hours_d, r["soc_d"], r["E_init_pct"])
        h, = ax2.plot(xr, yr, color=col, lw=2.2, ls=ls, label=f"{lbl} SoC (%)")
        leg2.append(h)
    ax2.axhline(v2g.soc_min_pct,       color="#C62828", ls=":", lw=1.5, zorder=3)
    ax2.axhline(v2g.soc_departure_pct, color="#0D47A1", ls=":", lw=1.5, zorder=3)
    if is_48h:
        ax2.axvline(24., color="#555555", lw=1.2, ls="--", alpha=0.65, zorder=5)
    else:
        _vlines(ax2)
    leg2 += [
        Line2D([0],[0], color="#C62828", ls=":", lw=1.5,
               label=f"Cold-chain floor {v2g.soc_min_pct:.0f}%"),
        Line2D([0],[0], color="#0D47A1", ls=":", lw=1.5,
               label=f"Departure target {v2g.soc_departure_pct:.0f}%"),
    ]
    _leg_below(ax2, leg2, ncol=3)
    _fmt(ax2, "State of Charge (%)",
         "(3) Battery SoC Trajectory  [gradual ramp = real energy flow]")
    ax2.set_ylim(0, 112)
    ax2.set_xlabel("Time of Day", fontsize=9)

    if is_48h:
        tr = ax2.get_xaxis_transform()
        ax2.text(12., 0.97, "Saturday", transform=tr, ha="center", va="top",
                 fontsize=9, color="#333333", fontweight="bold")
        ax2.text(36., 0.97, "Sunday",   transform=tr, ha="center", va="top",
                 fontsize=9, color="#333333", fontweight="bold")

    buf = BytesIO()
    fig.savefig(buf, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# =============================================================================
#  CACHED SCENARIO RUNNER
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
def run_scenarios_cached(season_key, arrival_h, departure_h, soc_pct,
                          soc_departure_pct, tru_cycle,
                          do_B, do_C, do_D, use_allin):
    months_map = {
        "winter_weekday": (WINTER_M, False, False),
        "summer_weekday": (SUMMER_M, False, False),
        "winter_weekend": (WINTER_M, True,  True),
        "summer_weekend": (SUMMER_M, True,  True),
    }
    months, is_wknd, is_48h = months_map[season_key]

    buy_spot = load_profile_cached(tuple(months), is_wknd)
    buy      = compose_all_in_price(buy_spot) if use_allin else buy_spot
    v2gp     = buy_spot.copy()

    v2g    = V2GParams(soc_departure_pct=soc_departure_pct)
    E_init = v2g.usable_capacity_kWh * soc_pct / 100.0

    if is_48h:
        buy48   = np.concatenate([buy,  buy])
        v2gp48  = np.concatenate([v2gp, v2gp])
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
            "net_cost":    r["net_cost"]    / 2,
            "charge_cost": r["charge_cost"] / 2,
            "v2g_rev":     r["v2g_rev"]     / 2,
            "tru_cost":    r["tru_cost"]    / 2,
            "total_cost":  r["total_cost"]  / 2,
            "v2g_kwh":     r["v2g_kwh"]     / 2,
            "charge_kwh":  r["charge_kwh"]  / 2,
        } for r in results]

        rc = compute_reefer_costs(tru_w[:96], buy_spot[:96], v2g.dt_h)
        return results, results_kpi, buy_d, plug_d, hours_d, is_wknd, is_48h, tru_w, rc

    else:
        win, arr, dep, W = get_wd_window(v2g, arrival_h, departure_h)
        buy_w  = buy[win]; v2gp_w = v2gp[win]; spot_w = buy_spot[win]
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

        rc = compute_reefer_costs(tru_w, spot_w, v2g.dt_h)
        return results, results, buy_d, plug_d, hours_d, is_wknd, is_48h, tru_d, rc


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
    "do_wwd":        True,
    "do_swd":        True,
    "do_wwe":        False,
    "do_swe":        False,
    "do_price":      False,
    "use_allin":     False,
    "fixed_price":   FIXED_PRICE_EUR_KWH,
    "wd_per_month":  22.0,
}

if "cfg" not in st.session_state:
    st.session_state.cfg = DEFAULTS.copy()
if "show_output" not in st.session_state:
    st.session_state.show_output = False


# =============================================================================
#  INPUT PANEL
# =============================================================================

def render_input_panel():
    st.title("S.KOe COOL -- V2G Optimisation Configuration")
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
        c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.0, 0.9])

        # ── Column 1: Schedule + season + price ───────────────────────────────
        with c1:
            st.subheader("Weekday Schedule")
            cfg["arrival_str"]   = st.text_input(
                "Arrival time (HH:MM)", cfg["arrival_str"],
                help="Time trailer returns to depot")
            cfg["departure_str"] = st.text_input(
                "Departure time (HH:MM)", cfg["departure_str"],
                help="Next-morning departure time")

            st.markdown("##### Season Split")
            cfg["winter_months"] = st.slider(
                "Winter months (Oct-Mar)", 1, 11, int(cfg["winter_months"]))
            st.caption(f"Summer months auto: **{12 - int(cfg['winter_months'])}**")

            st.markdown("##### Price Basis")
            cfg["use_allin"] = st.checkbox(
                "Use all-in German tariff",
                value=bool(cfg["use_allin"]),
                help=(
                    "Adds Netzentgelt 6.63 ct + concession 1.99 ct + "
                    "offshore 0.82 ct + CHP 0.28 ct + electricity tax 2.05 ct "
                    "+ NEV19 1.56 ct + 19% VAT to SMARD spot. "
                    "V2G revenue stays at spot."
                )
            )
            cfg["fixed_price"] = st.number_input(
                "Fixed-tariff benchmark (EUR/kWh)",
                value=float(cfg["fixed_price"]),
                min_value=0.05, max_value=1.0, step=0.01,
                help="Flat electricity rate used as cost benchmark"
            )

        # ── Column 2: SoC ─────────────────────────────────────────────────────
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
                help="~2 min first run -- cached after that")

        # ── Column 3: TRU ─────────────────────────────────────────────────────
        with c3:
            st.subheader("Reefer (TRU) at Depot")
            cycle_choice = st.radio(
                "TRU cycle while plugged in",
                ["Continuous", "Start-Stop", "OFF"],
                index={"Continuous": 0, "Start-Stop": 1, "OFF": 2}.get(
                    cfg["tru_cycle"], 2),
                help=(
                    "**Continuous:** 7.6 kW / 0.7 kW cycle (avg ~6.6 kW)\n\n"
                    "**Start-Stop:** 9.7 / 0.65 / 0 kW (avg ~3.9 kW)\n\n"
                    "**OFF:** TRU powered by diesel genset; no grid draw"
                )
            )
            cfg["tru_cycle"] = cycle_choice
            if cycle_choice != "OFF":
                avg_kw = tru_avg_kw(cycle_choice)
                st.info(
                    f"Avg TRU load: **{avg_kw:.1f} kW**\n\n"
                    f"Effective charging headroom: "
                    f"**{max(0, 22 - avg_kw):.1f} -- 22 kW**"
                )

        # ── Column 4: Day types + submit ──────────────────────────────────────
        with c4:
            st.subheader("Analysis")
            cfg["do_wwd"]   = st.checkbox(
                "Winter weekday",         bool(cfg["do_wwd"]))
            cfg["do_swd"]   = st.checkbox(
                "Summer weekday",         bool(cfg["do_swd"]))
            cfg["do_wwe"]   = st.checkbox(
                "Winter weekend (48h)",   bool(cfg["do_wwe"]))
            cfg["do_swe"]   = st.checkbox(
                "Summer weekend (48h)",   bool(cfg["do_swe"]))
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
            if not any([cfg["do_wwd"], cfg["do_swd"], cfg["do_wwe"],
                        cfg["do_swe"], cfg["do_price"]]):
                st.error("Select at least one day type or price analysis.")
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

st.title("S.KOe COOL -- V2G Optimisation Results")
st.caption(
    "TU Dortmund IE3 x Schmitz Cargobull AG  |  "
    "Master's Thesis 2026  |  Kuldip Bhadreshvara"
)

cfg = st.session_state.cfg
arr_h     = parse_hhmm(cfg["arrival_str"],   16.0)
dep_h     = parse_hhmm(cfg["departure_str"],  6.0)
soc_w     = int(cfg["soc_winter"])
soc_s     = int(cfg["soc_summer"])
soc_dep   = int(cfg["soc_departure"])
tru_cycle = cfg["tru_cycle"]
w_months  = int(cfg["winter_months"])
s_months  = 12 - w_months
use_allin = bool(cfg["use_allin"])
do_B      = bool(cfg["do_B"])
do_C      = bool(cfg["do_C"])
do_D      = bool(cfg["do_D"])
fixed_price = float(cfg["fixed_price"])

# ── Sidebar quick-edit ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Quick Edit")
    st.caption("Changes take effect immediately")
    cfg["arrival_str"]   = st.text_input("Arrival (HH:MM)",   cfg["arrival_str"])
    cfg["departure_str"] = st.text_input("Departure (HH:MM)", cfg["departure_str"])
    cfg["soc_winter"]    = st.slider("Winter arrival SoC (%)",   20, 100, soc_w)
    cfg["soc_summer"]    = st.slider("Summer arrival SoC (%)",   20, 100, soc_s)
    cfg["soc_departure"] = st.slider("Departure target SoC (%)", 50, 100, soc_dep)
    cfg["winter_months"] = st.slider("Winter months",  1, 11, w_months)
    cfg["tru_cycle"]     = st.radio(
        "TRU cycle", ["Continuous", "Start-Stop", "OFF"],
        index=["Continuous","Start-Stop","OFF"].index(tru_cycle))
    cfg["use_allin"]     = st.checkbox("All-in tariff price", use_allin)
    st.divider()
    if st.button("Back to Input", use_container_width=True):
        st.session_state.show_output = False
        st.rerun()

    st.session_state.cfg = cfg
    # Refresh local vars after sidebar edits
    arr_h     = parse_hhmm(cfg["arrival_str"],   16.0)
    dep_h     = parse_hhmm(cfg["departure_str"],  6.0)
    soc_w     = int(cfg["soc_winter"])
    soc_s     = int(cfg["soc_summer"])
    soc_dep   = int(cfg["soc_departure"])
    tru_cycle = cfg["tru_cycle"]
    w_months  = int(cfg["winter_months"])
    s_months  = 12 - w_months
    use_allin = bool(cfg["use_allin"])

# ── Verify CSV ────────────────────────────────────────────────────────────────
if not Path(CSV_PATH).exists():
    st.error(f"'{CSV_PATH}' not found -- commit it to the GitHub repo.")
    st.stop()

with st.spinner("Loading 2025 SMARD price data..."):
    try:
        df_info = _load_csv_raw(CSV_PATH)
        n_days  = len(df_info) // 96
        st.success(
            f"2025 DE/LU prices loaded  |  {n_days} days  |  "
            f"{df_info.index[0].date()} to {df_info.index[-1].date()}  |  "
            f"Spot: {df_info['price'].min()*1000:.0f}-"
            f"{df_info['price'].max()*1000:.0f} EUR/MWh  |  "
            f"Price basis: {'all-in tariff+VAT' if use_allin else 'SMARD spot'}"
        )
    except Exception as e:
        st.error(f"Could not load CSV: {e}"); st.stop()

v2g     = V2GParams(soc_departure_pct=float(soc_dep))
tru_avg = tru_avg_kw(tru_cycle)

# ── Summary banner ────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Arrival / Departure",  f"{fmt_hhmm(arr_h)} / {fmt_hhmm(dep_h)}")
m2.metric("Winter / Summer SoC",  f"{soc_w}% / {soc_s}%")
m3.metric("Departure Target",     f"{soc_dep}%")
m4.metric("TRU Reefer",
          f"{tru_cycle} ({tru_avg:.1f} kW)" if tru_cycle != "OFF" else "OFF")
m5.metric("Season Split",         f"{w_months} winter / {s_months} summer months")
st.markdown("---")

# ── Build configs ─────────────────────────────────────────────────────────────
CONFIGS = []
if cfg["do_wwd"]: CONFIGS.append(("Winter Weekday",               "winter_weekday", soc_w))
if cfg["do_swd"]: CONFIGS.append(("Summer Weekday",               "summer_weekday", soc_s))
if cfg["do_wwe"]: CONFIGS.append(("Winter Weekend (48h Sat+Sun)", "winter_weekend", soc_w))
if cfg["do_swe"]: CONFIGS.append(("Summer Weekend (48h Sat+Sun)", "summer_weekend", soc_s))

if not CONFIGS and not cfg["do_price"]:
    st.warning("Select at least one day type in the sidebar."); st.stop()

all_season_res   = {}
all_reefer_costs = {}


# =============================================================================
#  RUN EACH DAY TYPE
# =============================================================================

for (label, season_key, soc_init) in CONFIGS:
    st.subheader(label)

    with st.spinner(f"Optimising {label} -- first run ~30 s, cached after..."):
        try:
            (results, results_kpi,
             buy_d, plug_d, hours_d,
             is_wknd, is_48h, tru_d, rc) = run_scenarios_cached(
                season_key, arr_h, dep_h,
                float(soc_init), float(soc_dep),
                tru_cycle, do_B, do_C, do_D, use_allin
            )
            all_season_res[season_key]   = results_kpi
            all_reefer_costs[season_key] = rc
        except Exception as e:
            st.error(f"Error in {label}: {e}"); continue

    # ── Season chart (built inline) ───────────────────────────────────────────
    buf = plot_season_chart_inline(
        v2g, label, buy_d, plug_d, hours_d,
        results, arr_h, dep_h, is_48h,
        tru_d=tru_d, tru_cycle=tru_cycle
    )
    st.image(buf, use_container_width=True)

    # Download button for the same chart
    buf2 = plot_season_chart_inline(
        v2g, label, buy_d, plug_d, hours_d,
        results, arr_h, dep_h, is_48h,
        tru_d=tru_d, tru_cycle=tru_cycle
    )
    st.download_button(
        f"Download {label} chart (PNG)",
        data=buf2,
        file_name=f"v2g_{season_key}.png",
        mime="image/png",
        key=f"dl_{season_key}"
    )

    # ── EV KPI table ──────────────────────────────────────────────────────────
    st.markdown("**EV Charging KPI**")
    ref = results[0]["net_cost"]
    table = []
    for r in results:
        sav      = ref - r["net_cost"]
        fixed_ev = r["charge_kwh"] * fixed_price
        table.append({
            "Scenario"                          : r["label"],
            "EV charge cost (EUR/d)"            : round(r["charge_cost"],  4),
            "V2G revenue (EUR/d)"               : round(r["v2g_rev"],       4),
            "Net EV cost (EUR/d)"               : round(r["net_cost"],      4),
            f"Fixed @{fixed_price:.2f}/kWh (EUR/d)": round(fixed_ev,       4),
            "V2G export (kWh/d)"                : round(r["v2g_kwh"],       2),
            "Daily savings vs Dumb"             : "--" if sav == 0 else f"EUR {sav:+.4f}",
            "Annual savings (x365)"             : "--" if sav == 0 else f"EUR {sav*365:+,.0f}",
        })
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

    # ── Reefer cost table ─────────────────────────────────────────────────────
    if tru_cycle != "OFF" and rc["E_kWh"] > 0.01:
        st.markdown("**Reefer (TRU) Energy Cost -- Supply Scenario Comparison**")
        st.dataframe(pd.DataFrame([
            ["TRU energy consumed (kWh/d)",
             f"{rc['E_kWh']:.2f}"],
            ["Cost -- grid dynamic pricing (EUR/d)",
             f"EUR {rc['cost_dynamic']:.3f}"],
            [f"Cost -- fixed grid @EUR{fixed_price:.2f}/kWh (EUR/d)",
             f"EUR {rc['E_kWh'] * fixed_price:.3f}"],
            ["Cost -- diesel genset (EUR/d)",
             f"EUR {rc['cost_diesel']:.3f}"],
            ["Diesel consumption (L/d)",
             f"{rc['diesel_liters']:.2f} L"],
            ["Grid vs diesel saving (EUR/d)",
             f"EUR {rc['cost_diesel'] - rc['cost_dynamic']:+.3f}"],
        ], columns=["Metric", "Value"]),
        use_container_width=True, hide_index=True)

    st.markdown("---")


# =============================================================================
#  YEARLY EXTRAPOLATION
# =============================================================================

if all_season_res:
    st.subheader("Yearly Cost Extrapolation")
    we_per_month = max(0.0, (365 - float(cfg["wd_per_month"]) * 12) / 12)
    yr_df = yearly_extrapolation(
        all_season_res,
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
                "winter_weekday": w_months * float(cfg["wd_per_month"]),
                "summer_weekday": s_months * float(cfg["wd_per_month"]),
                "winter_weekend": w_months * we_per_month,
                "summer_weekend": s_months * we_per_month,
            }
            mult = mults.get(dt_key, 0)
            lbl  = dt_key.replace("_", " ").title()
            tru_yearly[lbl] = {
                "TRU energy (kWh)":                   round(rc["E_kWh"]        * mult, 0),
                "Dynamic (EUR)":                      round(rc["cost_dynamic"]  * mult, 0),
                f"Fixed @{fixed_price:.2f}/kWh (EUR)":round(rc["E_kWh"] * fixed_price * mult, 0),
                "Diesel equiv (EUR)":                 round(rc["cost_diesel"]   * mult, 0),
            }
        tru_yr_df = pd.DataFrame(tru_yearly).T
        tru_yr_df.loc["TOTAL"] = tru_yr_df.sum()
        st.dataframe(tru_yr_df, use_container_width=True)

    st.markdown("---")


# =============================================================================
#  KPI MULTI-TABLE
# =============================================================================

if len(all_season_res) > 1:
    st.subheader("KPI Comparison -- All Day Types")
    try:
        buf = BytesIO()
        plot_kpi_multi(all_season_res, v2g, arr_h, dep_h, run_mpc=do_D, out=buf)
        buf.seek(0)
        st.image(buf, use_container_width=True)
        buf2 = BytesIO()
        plot_kpi_multi(all_season_res, v2g, arr_h, dep_h, run_mpc=do_D, out=buf2)
        buf2.seek(0)
        st.download_button(
            "Download KPI comparison (PNG)",
            data=buf2, file_name="v2g_KPI_multi.png",
            mime="image/png", key="dl_kpi"
        )
    except Exception as e:
        st.error(f"KPI table error: {e}")
    st.markdown("---")


# =============================================================================
#  PRICE ANALYSIS
# =============================================================================

if cfg["do_price"]:
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
**Optimisation:** MILP with binary charge/discharge mutex.
V2G revenue at SMARD spot price; charging cost at
{'all-in German tariff' if use_allin else 'SMARD spot'}.

**TRU modelling ({tru_cycle}):** TRU competes with charger on the 22 kW grid connection.
Effective charging capacity = max(0, 22 kW minus TRU load). TRU does NOT drain the battery.
- Continuous: 7.6 / 0.7 kW cycle (1717 s / 292 s), avg **{tru_avg_kw("Continuous"):.1f} kW**
- Start-Stop: 9.7 / 0.65 / 0 kW (975 / 295 / 1207 s), avg **{tru_avg_kw("Start-Stop"):.1f} kW**

**German all-in tariff:** Netzentgelt 6.63 ct + concession 1.99 ct + offshore 0.82 ct
+ CHP 0.28 ct + electricity tax 2.05 ct + NEV19 1.56 ct + 19% VAT on net total.

**Cold-chain floor:** SoC >= 20% enforced as hard MILP constraint at all times.
**Departure target:** SoC >= {soc_dep}% at end of plugged-in window.
**Battery:** 70 kWh total / 60 kWh usable | eta_charge = 0.92 | eta_discharge = 0.92
**Yearly extrapolation:** {int(cfg['wd_per_month'])} working days/month x seasonal month split.
**Fixed-tariff benchmark:** EUR {fixed_price:.2f}/kWh flat rate.
    """)

st.caption(
    "S.KOe COOL V2G Optimisation  -  "
    "TU Dortmund IE3 x Schmitz Cargobull AG  -  "
    "Thesis 2026  -  Confidential"
)