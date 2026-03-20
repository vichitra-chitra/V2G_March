"""
S.KOe COOL -- V2G Optimisation  |  Streamlit App (v6)
TU Dortmund IE3 x Schmitz Cargobull AG | 2026
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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
    make_kpi, plot_season_chart, plot_kpi_multi, plot_price_profiles,
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
            "net_cost":r["net_cost"]/2,"charge_cost":r["charge_cost"]/2,
            "v2g_rev":r["v2g_rev"]/2,"tru_cost":r["tru_cost"]/2,
            "total_cost":r["total_cost"]/2,"v2g_kwh":r["v2g_kwh"]/2,
            "charge_kwh":r["charge_kwh"]/2,
        } for r in results]

        rc = compute_reefer_costs(tru_w[:96], buy_spot[:96], v2g.dt_h)
        return results, results_kpi, buy_d, plug_d, hours_d, is_wknd, is_48h, tru_w, rc

    else:
        win, arr, dep, W = get_wd_window(v2g, arrival_h, departure_h)
        buy_w  = buy[win]; v2gp_w = v2gp[win]; spot_w = buy_spot[win]
        tru_w  = get_tru_15min_trace(tru_cycle, W, v2g.dt_h)
        buy_d, plug_d, hours_d = build_wd_display(v2g, buy, arrival_h, departure_h)
        tru_d = np.zeros(v2g.n_slots); tru_d[arr:dep] = tru_w[:dep-arr]

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
    st.caption("TU Dortmund IE3 x Schmitz Cargobull AG  |  Master's Thesis 2026  |  Kuldip Bhadreshvara")
    st.markdown("---")

    cfg = st.session_state.cfg

    with st.form("input_form", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.0, 0.9])

        with c1:
            st.subheader("Weekday Schedule")
            cfg["arrival_str"]   = st.text_input("Arrival time (HH:MM)",   cfg["arrival_str"])
            cfg["departure_str"] = st.text_input("Departure time (HH:MM)", cfg["departure_str"])
            st.markdown("##### Season Split")
            cfg["winter_months"] = st.slider("Winter months (Oct-Mar)", 1, 11, int(cfg["winter_months"]))
            st.caption(f"Summer months auto: **{12 - int(cfg['winter_months'])}**")
            st.markdown("##### Price Basis")
            cfg["use_allin"] = st.checkbox(
                "Use all-in German tariff price", value=bool(cfg["use_allin"]),
                help="Adds Netzentgelt + concession + CHP levy + electricity tax + offshore levy + NEV19 + 19% VAT"
            )
            cfg["fixed_price"] = st.number_input(
                "Fixed-tariff benchmark (EUR/kWh)", value=float(cfg["fixed_price"]),
                min_value=0.05, max_value=1.0, step=0.01
            )

        with c2:
            st.subheader("State of Charge")
            cfg["soc_winter"]    = st.slider("Winter arrival SoC (%)",     20, 100, int(cfg["soc_winter"]))
            cfg["soc_summer"]    = st.slider("Summer arrival SoC (%)",     20, 100, int(cfg["soc_summer"]))
            cfg["soc_departure"] = st.slider("Departure target SoC (%)",   50, 100, int(cfg["soc_departure"]))
            st.markdown("##### Yearly Extrapolation Basis")
            cfg["wd_per_month"] = st.number_input(
                "Working days per month", value=float(cfg["wd_per_month"]),
                min_value=10.0, max_value=31.0, step=0.5
            )

        with c3:
            st.subheader("Reefer (TRU) at Depot")
            cycle_choice = st.radio(
                "TRU cycle while plugged in",
                ["Continuous", "Start-Stop", "OFF"],
                index={"Continuous": 0, "Start-Stop": 1, "OFF": 2}.get(cfg["tru_cycle"], 2),
                help="Continuous: 7.6/0.7 kW | Start-Stop: 9.7/0.65/0 kW | OFF: diesel genset"
            )
            cfg["tru_cycle"] = cycle_choice
            if cycle_choice != "OFF":
                avg_kw = tru_avg_kw(cycle_choice)
                st.info(f"Avg TRU load: **{avg_kw:.1f} kW**\nEffective max charging: **{max(0,22-avg_kw):.1f}-22 kW**")

            st.markdown("##### Scenarios to Run")
            cfg["do_B"] = st.checkbox("B -- Smart charging (no V2G)", bool(cfg["do_B"]))
            cfg["do_C"] = st.checkbox("C -- MILP Day-Ahead",          bool(cfg["do_C"]))
            cfg["do_D"] = st.checkbox("D -- MPC receding horizon",    bool(cfg["do_D"]),
                                      help="~2 min first run -- cached after that")

        with c4:
            st.subheader("Analysis")
            cfg["do_wwd"]   = st.checkbox("Winter weekday",         bool(cfg["do_wwd"]))
            cfg["do_swd"]   = st.checkbox("Summer weekday",         bool(cfg["do_swd"]))
            cfg["do_wwe"]   = st.checkbox("Winter weekend (48h)",   bool(cfg["do_wwe"]))
            cfg["do_swe"]   = st.checkbox("Summer weekend (48h)",   bool(cfg["do_swe"]))
            cfg["do_price"] = st.checkbox("Price profile analysis", bool(cfg["do_price"]))
            st.markdown("---")
            st.markdown("**S.KOe COOL specs**")
            st.caption("70 kWh total / 60 kWh usable")
            st.caption("22 kW AC bidirectional - ISO 15118-20")
            st.caption("Cold-chain floor: SoC >= 20%")
            st.markdown("")
            submitted = st.form_submit_button("Calculate", type="primary", use_container_width=True)

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
            st.session_state.cfg        = cfg
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
st.caption("TU Dortmund IE3 x Schmitz Cargobull AG  |  Master's Thesis 2026  |  Kuldip Bhadreshvara")

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
do_B = bool(cfg["do_B"]); do_C = bool(cfg["do_C"]); do_D = bool(cfg["do_D"])


# Sidebar quick-edit
with st.sidebar:
    st.header("Quick Edit")
    st.caption("Changes take effect immediately")
    cfg["arrival_str"]   = st.text_input("Arrival (HH:MM)",   cfg["arrival_str"])
    cfg["departure_str"] = st.text_input("Departure (HH:MM)", cfg["departure_str"])
    cfg["soc_winter"]    = st.slider("Winter SoC (%)",    20, 100, int(cfg["soc_winter"]))
    cfg["soc_summer"]    = st.slider("Summer SoC (%)",    20, 100, int(cfg["soc_summer"]))
    cfg["soc_departure"] = st.slider("Departure SoC (%)", 50, 100, int(cfg["soc_departure"]))
    cfg["winter_months"] = st.slider("Winter months",      1,  11, int(cfg["winter_months"]))
    cfg["tru_cycle"]     = st.radio("TRU cycle", ["Continuous","Start-Stop","OFF"],
                                     index=["Continuous","Start-Stop","OFF"].index(cfg["tru_cycle"]))
    cfg["use_allin"]     = st.checkbox("All-in tariff price", bool(cfg["use_allin"]))
    st.divider()
    if st.button("Back to Input", use_container_width=True):
        st.session_state.show_output = False
        st.rerun()
    st.session_state.cfg = cfg
    arr_h     = parse_hhmm(cfg["arrival_str"],   16.0)
    dep_h     = parse_hhmm(cfg["departure_str"],  6.0)
    soc_w     = int(cfg["soc_winter"])
    soc_s     = int(cfg["soc_summer"])
    soc_dep   = int(cfg["soc_departure"])
    tru_cycle = cfg["tru_cycle"]
    w_months  = int(cfg["winter_months"])
    s_months  = 12 - w_months
    use_allin = bool(cfg["use_allin"])


# Verify CSV
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
            f"Spot range: {df_info['price'].min()*1000:.0f}-"
            f"{df_info['price'].max()*1000:.0f} EUR/MWh  |  "
            f"Price basis: {'all-in (tariff+VAT)' if use_allin else 'SMARD spot'}"
        )
    except Exception as e:
        st.error(f"Could not load CSV: {e}"); st.stop()

v2g = V2GParams(soc_departure_pct=float(soc_dep))
tru_avg = tru_avg_kw(tru_cycle)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Arrival / Departure", f"{fmt_hhmm(arr_h)} / {fmt_hhmm(dep_h)}")
m2.metric("Winter SoC | Summer SoC", f"{soc_w}% | {soc_s}%")
m3.metric("Departure Target", f"{soc_dep}%")
m4.metric("TRU Reefer", f"{tru_cycle} ({tru_avg:.1f} kW)" if tru_cycle != "OFF" else "OFF")
m5.metric("Season Split", f"{w_months} winter / {s_months} summer months")
st.markdown("---")


# Build configs
CONFIGS = []
if cfg["do_wwd"]: CONFIGS.append(("Winter Weekday",               "winter_weekday", soc_w))
if cfg["do_swd"]: CONFIGS.append(("Summer Weekday",               "summer_weekday", soc_s))
if cfg["do_wwe"]: CONFIGS.append(("Winter Weekend (48h Sat+Sun)", "winter_weekend", soc_w))
if cfg["do_swe"]: CONFIGS.append(("Summer Weekend (48h Sat+Sun)", "summer_weekend", soc_s))

if not CONFIGS and not cfg["do_price"]:
    st.warning("Select at least one day type in the sidebar."); st.stop()

all_season_res   = {}
all_reefer_costs = {}


# Run each day type
for (label, season_key, soc_init) in CONFIGS:
    st.subheader(f"{label}")

    with st.spinner(f"Optimising {label} -- first run ~30s, cached after..."):
        try:
            (results, results_kpi,
             buy_d, plug_d, hours_d,
             is_wknd, is_48h, tru_d, rc) = run_scenarios_cached(
                season_key, arr_h, dep_h, float(soc_init), float(soc_dep),
                tru_cycle, do_B, do_C, do_D, use_allin
            )
            all_season_res[season_key]   = results_kpi
            all_reefer_costs[season_key] = rc
        except Exception as e:
            st.error(f"Error in {label}: {e}"); continue

    buf = BytesIO()
    plot_season_chart(v2g, label, buy_d, plug_d, hours_d,
                      results, is_wknd, arr_h, dep_h,
                      is_48h=is_48h, out=buf, tru_d=tru_d, tru_cycle=tru_cycle)
    buf.seek(0)
    st.image(buf, use_container_width=True)

    buf2 = BytesIO()
    plot_season_chart(v2g, label, buy_d, plug_d, hours_d,
                      results, is_wknd, arr_h, dep_h,
                      is_48h=is_48h, out=buf2, tru_d=tru_d, tru_cycle=tru_cycle)
    buf2.seek(0)
    st.download_button(f"Download {label} chart (PNG)",
                       data=buf2, file_name=f"v2g_{season_key}.png",
                       mime="image/png", key=f"dl_{season_key}")

    # EV KPI table
    st.markdown("**EV Charging KPI**")
    ref = results[0]["net_cost"]
    fixed_ev_rate = float(cfg["fixed_price"])
    table = []
    for r in results:
        sav = ref - r["net_cost"]
        fixed_ev = r["charge_kwh"] * fixed_ev_rate
        table.append({
            "Scenario"                   : r["label"],
            "EV charge cost (EUR/d)"     : round(r["charge_cost"],  4),
            "V2G revenue (EUR/d)"        : round(r["v2g_rev"],       4),
            "Net EV cost (EUR/d)"        : round(r["net_cost"],      4),
            f"Fixed @{fixed_ev_rate:.2f}/kWh (EUR/d)": round(fixed_ev, 4),
            "V2G export (kWh/d)"         : round(r["v2g_kwh"],       2),
            "Daily savings vs Dumb"      : "--" if sav == 0 else f"EUR {sav:+.4f}",
            "Annual savings (x365)"      : "--" if sav == 0 else f"EUR {sav*365:+,.0f}",
        })
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

    # Reefer cost table
    if tru_cycle != "OFF" and rc["E_kWh"] > 0.01:
        st.markdown("**Reefer (TRU) Energy Cost -- Supply Scenario Comparison**")
        reefer_table = pd.DataFrame([
            ["TRU energy consumed (kWh/d)",                  f"{rc['E_kWh']:.2f}"],
            ["Cost -- grid dynamic pricing (EUR/d)",          f"EUR {rc['cost_dynamic']:.3f}"],
            [f"Cost -- fixed grid @EUR{fixed_ev_rate:.2f}/kWh (EUR/d)",
             f"EUR {rc['E_kWh'] * fixed_ev_rate:.3f}"],
            ["Cost -- diesel genset (EUR/d)",                 f"EUR {rc['cost_diesel']:.3f}"],
            ["Diesel consumption (L/d)",                      f"{rc['diesel_liters']:.2f} L"],
            ["Grid vs diesel saving (EUR/d)",
             f"EUR {rc['cost_diesel'] - rc['cost_dynamic']:+.3f}"],
        ], columns=["Metric", "Value"])
        st.dataframe(reefer_table, use_container_width=True, hide_index=True)

    st.markdown("---")


# Yearly extrapolation
if all_season_res:
    st.subheader("Yearly Cost Extrapolation")
    we_per_month = (365 - float(cfg["wd_per_month"]) * 12) / 12
    yr_df = yearly_extrapolation(
        all_season_res, winter_months=w_months,
        wd_per_month=float(cfg["wd_per_month"]),
        we_days_per_month=max(0.0, we_per_month),
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
                    "winter_weekend": w_months * max(0, we_per_month),
                    "summer_weekend": s_months * max(0, we_per_month),
                }
                mult = mults.get(dt_key, 0)
                lbl  = dt_key.replace("_", " ").title()
                tru_yearly[lbl] = {
                    "TRU energy (kWh)":    round(rc["E_kWh"]       * mult, 0),
                    "Dynamic (EUR)":       round(rc["cost_dynamic"] * mult, 0),
                    f"Fixed @{float(cfg['fixed_price']):.2f}/kWh": round(rc["E_kWh"] * float(cfg["fixed_price"]) * mult, 0),
                    "Diesel equiv (EUR)":  round(rc["cost_diesel"]  * mult, 0),
                }
            tru_yr_df = pd.DataFrame(tru_yearly).T
            tru_yr_df.loc["TOTAL"] = tru_yr_df.sum()
            st.dataframe(tru_yr_df, use_container_width=True)
    st.markdown("---")


# KPI multi-table
if len(all_season_res) > 1:
    st.subheader("KPI Comparison -- All Day Types")
    try:
        buf = BytesIO()
        plot_kpi_multi(all_season_res, v2g, arr_h, dep_h, run_mpc=do_D, out=buf)
        buf.seek(0); st.image(buf, use_container_width=True)
        buf2 = BytesIO()
        plot_kpi_multi(all_season_res, v2g, arr_h, dep_h, run_mpc=do_D, out=buf2)
        buf2.seek(0)
        st.download_button("Download KPI comparison (PNG)",
                           data=buf2, file_name="v2g_KPI_multi.png",
                           mime="image/png", key="dl_kpi")
    except Exception as e:
        st.error(f"KPI table error: {e}")
    st.markdown("---")


# Price analysis
if cfg["do_price"]:
    st.subheader("Electricity Price Analysis")
    with st.spinner("Building price charts..."):
        try:
            buf = BytesIO()
            plot_price_profiles(CSV_PATH, buf)
            buf.seek(0); st.image(buf, use_container_width=True)
            buf2 = BytesIO()
            plot_price_profiles(CSV_PATH, buf2)
            buf2.seek(0)
            st.download_button("Download price analysis (PNG)",
                               data=buf2, file_name="v2g_price_profiles.png",
                               mime="image/png", key="dl_price")
        except Exception as e:
            st.error(f"Price analysis error: {e}")


# Methodology
with st.expander("Methodology & Assumptions", expanded=False):
    st.markdown(f"""
**Optimisation:** MILP with binary charge/discharge mutex.
V2G revenue at SMARD spot price; charging cost at {'all-in German tariff' if use_allin else 'SMARD spot'}.

**TRU modelling ({tru_cycle}):** TRU competes with charger for the 22 kW grid connection.
Effective charging capacity = max(0, 22 kW minus TRU load). TRU does NOT drain the battery.
- Continuous: 7.6/0.7 kW cycle (1717s/292s), avg **{tru_avg_kw("Continuous"):.1f} kW**
- Start-Stop: 9.7/0.65/0 kW (975/295/1207s), avg **{tru_avg_kw("Start-Stop"):.1f} kW**

**German all-in tariff:** Netzentgelt 6.63 ct + concession 1.99 ct + offshore 0.82 ct
+ CHP 0.28 ct + electricity tax 2.05 ct + NEV19 1.56 ct + 19% VAT.

**Cold-chain floor:** SoC >= 20% enforced as hard MILP constraint.
**Departure target:** SoC >= {soc_dep}%.
**Battery:** 70 kWh total / 60 kWh usable | eta_charge = 0.92 | eta_discharge = 0.92
**Yearly extrapolation:** {int(cfg['wd_per_month'])} working days/month x seasonal split.
**Fixed-tariff benchmark:** EUR{float(cfg['fixed_price']):.2f}/kWh flat rate.
    """)

st.caption("S.KOe COOL V2G Optimisation  -  TU Dortmund IE3 x Schmitz Cargobull AG  -  Thesis 2026  -  Confidential")