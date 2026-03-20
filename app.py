import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from io import BytesIO
from pathlib import Path

from v2g_single_day4 import (
    V2GParams, WINTER_M, SUMMER_M, SC_COL, SC_FILL,
    _interpolate_to_15min, _load_csv_raw,
    get_wd_window, build_wd_display, to_display_wd, soc_ramp,
    run_A_dumb, run_B_smart, run_C_milp, run_D_mpc,
    make_kpi, plot_season_chart, plot_kpi_multi,
    plot_price_profiles
)

CSV_PATH = "2025_Electricity_Price.csv"

st.set_page_config(
    page_title="S.KOe COOL — V2G Optimisation",
    page_icon="⚡",
    layout="wide"
)

st.title("S.KOe COOL — V2G Charging Optimisation")
st.caption(
    "TU Dortmund IE³ × Schmitz Cargobull AG  |  "
    "Master's Thesis 2026  |  Kuldip Bhadreshvara"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    st.subheader("Weekday schedule")
    arrival_h    = st.slider("Arrival hour",    10, 23, 16, 1)
    departure_h  = st.slider("Departure hour",   1,  9,  6, 1)
    soc_init_pct = st.slider("Arrival SoC (%)", 20, 90, 45, 5)

    st.divider()
    st.subheader("Scenarios")
    do_B = st.checkbox("B — Smart charging (no V2G)", True)
    do_C = st.checkbox("C — MILP Day-Ahead",          True)
    do_D = st.checkbox("D — MPC receding horizon",    False,
                        help="Adds ~2 min compute per day type")

    st.divider()
    st.subheader("Day types")
    do_wwd   = st.checkbox("Winter weekday",           True)
    do_swd   = st.checkbox("Summer weekday",           True)
    do_wwe   = st.checkbox("Winter weekend (48h)",     False)
    do_swe   = st.checkbox("Summer weekend (48h)",     False)
    do_price = st.checkbox("Price profile analysis",   False)

    st.divider()
    run_btn = st.button(
        "Run optimisation", type="primary", use_container_width=True
    )

# ── About section before run ──────────────────────────────────────────────────
if not run_btn:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**About this tool**

Optimises charging and V2G discharge for the
**Schmitz Cargobull S.KOe COOL** electric reefer trailer
using 2025 SMARD DE/LU day-ahead electricity prices.

| Scenario | Strategy |
|---|---|
| A — Dumb   | Full power on arrival, no price awareness |
| B — Smart  | MILP shifts charge to cheapest slots |
| C — MILP   | Full day-ahead optimal schedule + V2G |
| D — MPC    | Receding-horizon real-time control |
        """)
    with col2:
        st.markdown("""
**Trailer specifications**

| Parameter | Value |
|---|---|
| Battery total | 70 kWh |
| Battery usable | 60 kWh (SoC 20–100%) |
| Charging power | 22 kW AC bidirectional |
| Standard | ISO 15118-20 |
| Cold-chain floor | SoC ≥ 20% always |
| Departure target | SoC = 100% |
| Protocol | CCS2 + OCPP 2.1 |
        """)
    st.info("Set parameters in the sidebar and click **Run optimisation**.")
    st.stop()

# ── Load CSV from repo ────────────────────────────────────────────────────────
with st.spinner("Loading 2025 price data..."):
    try:
        if not Path(CSV_PATH).exists():
            st.error(
                f"CSV file '{CSV_PATH}' not found in the repo. "
                "Make sure it is committed to GitHub."
            )
            st.stop()
        df_prices = _load_csv_raw(CSV_PATH)
        n_days    = len(df_prices) // 96
        st.success(
            f"2025 SMARD DE/LU prices loaded  |  "
            f"{n_days} days  |  "
            f"{df_prices.index[0].date()} → {df_prices.index[-1].date()}  |  "
            f"Range: {df_prices['price'].min()*1000:.0f}–"
            f"{df_prices['price'].max()*1000:.0f} EUR/MWh"
        )
    except Exception as e:
        st.error(f"Could not load CSV: {e}")
        st.stop()


def load_profile(months, is_weekend):
    mask    = df_prices["month"].isin(months) & (df_prices["is_weekend"] == is_weekend)
    sub     = df_prices[mask]
    if len(sub) == 0:
        raise ValueError(f"No data for months={months}, weekend={is_weekend}")
    profile = sub.groupby("slot")["price"].mean().values
    if len(profile) != 96:
        raise ValueError(f"Expected 96 slots, got {len(profile)}")
    return _interpolate_to_15min(profile)


v2g    = V2GParams()
E_init = v2g.usable_capacity_kWh * soc_init_pct / 100.0

CONFIGS = []
if do_wwd: CONFIGS.append(("Winter Weekday",            WINTER_M, False, False))
if do_swd: CONFIGS.append(("Summer Weekday",            SUMMER_M, False, False))
if do_wwe: CONFIGS.append(("Winter Weekend (48h Sat+Sun)", WINTER_M, True,  True))
if do_swe: CONFIGS.append(("Summer Weekend (48h Sat+Sun)", SUMMER_M, True,  True))

if not CONFIGS and not do_price:
    st.warning("Select at least one day type in the sidebar.")
    st.stop()

all_season_res = {}

# ── Run each day type ─────────────────────────────────────────────────────────
for (label, months, is_wknd, is_48h) in CONFIGS:
    st.subheader(label)

    with st.spinner(f"Optimising {label}..."):
        try:
            buy  = load_profile(months, is_wknd)
            v2gp = buy.copy()

            if is_48h:
                buy48   = np.concatenate([buy, buy])
                v2gp48  = buy48.copy()
                W       = 192
                hours_d = np.arange(W) * v2g.dt_h
                plug_d  = np.ones(W)
                buy_d   = buy48

                Pc, Pd, soc = run_A_dumb(v2g, buy48, v2gp48, W, E_init)
                results = [make_kpi("A - Dumb", v2g, Pc, Pd, soc,
                                    buy48, v2gp48, E_init, is_weekend_48=True)]
                if do_B:
                    Pc, Pd, soc = run_B_smart(v2g, buy48, v2gp48, E_init)
                    results.append(make_kpi("B - Smart (no V2G)", v2g, Pc, Pd,
                                            soc, buy48, v2gp48, E_init,
                                            is_weekend_48=True))
                if do_C:
                    Pc, Pd, soc = run_C_milp(v2g, buy48, v2gp48, E_init)
                    results.append(make_kpi("C - MILP Day-Ahead", v2g, Pc, Pd,
                                            soc, buy48, v2gp48, E_init,
                                            is_weekend_48=True))
                if do_D:
                    with st.spinner(f"MPC: solving {W} sub-problems (~2 min)..."):
                        Pc, Pd, soc = run_D_mpc(v2g, buy48, v2gp48, E_init)
                    results.append(make_kpi("D - MPC (receding)", v2g, Pc, Pd,
                                            soc, buy48, v2gp48, E_init,
                                            is_weekend_48=True))

                results_kpi = [{**r,
                    "net_cost"   : r["net_cost"]    / 2,
                    "charge_cost": r["charge_cost"] / 2,
                    "v2g_rev"    : r["v2g_rev"]     / 2,
                    "v2g_kwh"    : r["v2g_kwh"]     / 2,
                } for r in results]
                key = "winter_weekend" if "Winter" in label else "summer_weekend"
                all_season_res[key] = results_kpi

            else:
                win, arr, dep, W = get_wd_window(v2g, arrival_h, departure_h)
                buy_w  = buy[win]; v2gp_w = v2gp[win]
                buy_d, plug_d, hours_d = build_wd_display(
                    v2g, buy, arrival_h, departure_h
                )

                Pc, Pd, soc = run_A_dumb(v2g, buy_w, v2gp_w, W, E_init)
                results = [make_kpi("A - Dumb", v2g, Pc, Pd, soc,
                                    buy_w, v2gp_w, E_init, arr, dep)]
                if do_B:
                    Pc, Pd, soc = run_B_smart(v2g, buy_w, v2gp_w, E_init)
                    results.append(make_kpi("B - Smart (no V2G)", v2g, Pc, Pd,
                                            soc, buy_w, v2gp_w, E_init, arr, dep))
                if do_C:
                    Pc, Pd, soc = run_C_milp(v2g, buy_w, v2gp_w, E_init)
                    results.append(make_kpi("C - MILP Day-Ahead", v2g, Pc, Pd,
                                            soc, buy_w, v2gp_w, E_init, arr, dep))
                if do_D:
                    with st.spinner(f"MPC: solving {W} sub-problems (~2 min)..."):
                        Pc, Pd, soc = run_D_mpc(v2g, buy_w, v2gp_w, E_init)
                    results.append(make_kpi("D - MPC (receding)", v2g, Pc, Pd,
                                            soc, buy_w, v2gp_w, E_init, arr, dep))

                key = "winter_weekday" if "Winter" in label else "summer_weekday"
                all_season_res[key] = results

        except Exception as e:
            st.error(f"Error in {label}: {e}")
            continue

    # Chart
    buf = BytesIO()
    plot_season_chart(
        v2g, label, buy_d, plug_d, hours_d,
        results, is_wknd, arrival_h, departure_h,
        is_48h=is_48h, out=buf
    )
    buf.seek(0)
    st.image(buf, use_container_width=True)

    # Download
    buf2 = BytesIO()
    plot_season_chart(
        v2g, label, buy_d, plug_d, hours_d,
        results, is_wknd, arrival_h, departure_h,
        is_48h=is_48h, out=buf2
    )
    buf2.seek(0)
    safe = label.lower().replace(" ","_").replace("(","").replace(")","")
    st.download_button(
        f"Download {label} chart (PNG)",
        data=buf2, file_name=f"v2g_{safe}.png",
        mime="image/png", key=f"dl_{label}"
    )

    # KPI table
    ref   = results[0]["net_cost"]
    table = []
    for r in results:
        sav = ref - r["net_cost"]
        table.append({
            "Scenario"              : r["label"],
            "Net cost (EUR/day)"    : round(r["net_cost"],    4),
            "Charge cost (EUR/day)" : round(r["charge_cost"], 4),
            "V2G revenue (EUR/day)" : round(r["v2g_rev"],     4),
            "V2G export (kWh/day)"  : round(r["v2g_kwh"],     2),
            "Daily savings vs Dumb" : "—" if sav == 0 else f"EUR {sav:+.4f}",
            "Annual savings (x365)" : "—" if sav == 0 else f"EUR {sav*365:+,.0f}",
        })
    st.dataframe(
        pd.DataFrame(table), use_container_width=True, hide_index=True
    )
    st.divider()

# ── KPI multi-table ───────────────────────────────────────────────────────────
if len(all_season_res) > 1:
    st.subheader("KPI Comparison — All Day Types")
    with st.spinner("Building KPI comparison table..."):
        buf = BytesIO()
        plot_kpi_multi(
            all_season_res, v2g, arrival_h, departure_h,
            run_mpc=do_D, out=buf
        )
        buf.seek(0)
    st.image(buf, use_container_width=True)
    buf2 = BytesIO()
    plot_kpi_multi(
        all_season_res, v2g, arrival_h, departure_h,
        run_mpc=do_D, out=buf2
    )
    buf2.seek(0)
    st.download_button(
        "Download KPI comparison (PNG)",
        data=buf2, file_name="v2g_KPI_multi.png",
        mime="image/png", key="dl_kpi"
    )
    st.divider()

# ── Price profile analysis ────────────────────────────────────────────────────
if do_price:
    st.subheader("Electricity Price Analysis")
    with st.spinner("Building price profile charts..."):
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

st.caption(
    "S.KOe COOL V2G Optimisation  ·  "
    "TU Dortmund IE³ × Schmitz Cargobull AG  ·  "
    "Thesis 2026  ·  Confidential"
)