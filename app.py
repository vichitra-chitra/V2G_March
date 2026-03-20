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
    get_wd_window, build_wd_display,
    run_A_dumb, run_B_smart, run_C_milp, run_D_mpc,
    make_kpi, plot_season_chart, plot_kpi_multi,
    plot_price_profiles
)

CSV_PATH = "2025_Electricity_Price.csv"


# =============================================================================
#  HELPERS
# =============================================================================

def parse_hhmm(s: str, default: float) -> float:
    """Parse HH:MM string to decimal hours. Returns default on error."""
    try:
        parts = s.strip().split(":")
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        assert 0 <= h <= 23 and 0 <= m <= 59
        return h + m / 60.0
    except Exception:
        return default


@st.cache_data(show_spinner=False)
def run_day_type_cached(season_key: str, arrival_h: float, departure_h: float,
                        soc_init_pct: float, do_B: bool, do_C: bool, do_D: bool):
    """
    Run selected scenarios for one day type.
    Cached by all parameters — reruns only when something changes.
    First run: ~30s per day type. Subsequent runs with same params: instant.
    """
    months_map = {
        "winter_weekday": (WINTER_M, False, False),
        "summer_weekday": (SUMMER_M, False, False),
        "winter_weekend": (WINTER_M, True,  True),
        "summer_weekend": (SUMMER_M, True,  True),
    }
    months, is_wknd, is_48h = months_map[season_key]

    df      = _load_csv_raw(CSV_PATH)
    mask    = df["month"].isin(months) & (df["is_weekend"] == is_wknd)
    sub     = df[mask]
    profile = sub.groupby("slot")["price"].mean().values
    buy     = _interpolate_to_15min(profile)
    v2gp    = buy.copy()

    v2g    = V2GParams()
    E_init = v2g.usable_capacity_kWh * soc_init_pct / 100.0

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
            results.append(make_kpi("B - Smart (no V2G)", v2g, Pc, Pd, soc,
                                    buy48, v2gp48, E_init, is_weekend_48=True))
        if do_C:
            Pc, Pd, soc = run_C_milp(v2g, buy48, v2gp48, E_init)
            results.append(make_kpi("C - MILP Day-Ahead", v2g, Pc, Pd, soc,
                                    buy48, v2gp48, E_init, is_weekend_48=True))
        if do_D:
            Pc, Pd, soc = run_D_mpc(v2g, buy48, v2gp48, E_init)
            results.append(make_kpi("D - MPC (receding)", v2g, Pc, Pd, soc,
                                    buy48, v2gp48, E_init, is_weekend_48=True))

        results_kpi = [{**r,
            "net_cost"   : r["net_cost"]    / 2,
            "charge_cost": r["charge_cost"] / 2,
            "v2g_rev"    : r["v2g_rev"]     / 2,
            "v2g_kwh"    : r["v2g_kwh"]     / 2,
        } for r in results]

        return results, results_kpi, buy_d, plug_d, hours_d, is_wknd, is_48h

    else:
        win, arr, dep, W = get_wd_window(v2g, arrival_h, departure_h)
        buy_w  = buy[win]; v2gp_w = v2gp[win]
        buy_d, plug_d, hours_d = build_wd_display(v2g, buy, arrival_h, departure_h)

        Pc, Pd, soc = run_A_dumb(v2g, buy_w, v2gp_w, W, E_init)
        results = [make_kpi("A - Dumb", v2g, Pc, Pd, soc,
                            buy_w, v2gp_w, E_init, arr, dep)]
        if do_B:
            Pc, Pd, soc = run_B_smart(v2g, buy_w, v2gp_w, E_init)
            results.append(make_kpi("B - Smart (no V2G)", v2g, Pc, Pd, soc,
                                    buy_w, v2gp_w, E_init, arr, dep))
        if do_C:
            Pc, Pd, soc = run_C_milp(v2g, buy_w, v2gp_w, E_init)
            results.append(make_kpi("C - MILP Day-Ahead", v2g, Pc, Pd, soc,
                                    buy_w, v2gp_w, E_init, arr, dep))
        if do_D:
            Pc, Pd, soc = run_D_mpc(v2g, buy_w, v2gp_w, E_init)
            results.append(make_kpi("D - MPC (receding)", v2g, Pc, Pd, soc,
                                    buy_w, v2gp_w, E_init, arr, dep))

        return results, results, buy_d, plug_d, hours_d, is_wknd, is_48h


# =============================================================================
#  PAGE SETUP
# =============================================================================

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


# =============================================================================
#  SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("Configuration")
    st.caption("Results update automatically when parameters change.")

    st.subheader("Weekday schedule")
    arrival_str   = st.text_input(
        "Arrival time (HH:MM)", "16:00",
        help="Time trailer returns to depot"
    )
    departure_str = st.text_input(
        "Departure time (HH:MM)", "06:00",
        help="Time trailer leaves next morning"
    )
    soc_str = st.text_input(
        "Arrival battery SoC (%)", "45",
        help="Battery charge level on arrival — enter a number between 20 and 100"
    )

    # Parse all inputs
    arrival_h   = parse_hhmm(arrival_str,   16.0)
    departure_h = parse_hhmm(departure_str,  6.0)
    try:
        soc_init_pct = float(soc_str)
        soc_init_pct = max(20.0, min(100.0, soc_init_pct))
    except Exception:
        soc_init_pct = 45.0

    # Confirm parsed values
    st.caption(
        f"Parsed → arrival {int(arrival_h):02d}:{int((arrival_h % 1)*60):02d}  |  "
        f"departure {int(departure_h):02d}:{int((departure_h % 1)*60):02d}  |  "
        f"SoC {soc_init_pct:.0f}%"
    )

    st.divider()
    st.subheader("Scenarios to run")
    do_B = st.checkbox("B — Smart charging (no V2G)", True)
    do_C = st.checkbox("C — MILP Day-Ahead",          True)
    do_D = st.checkbox("D — MPC receding horizon",    False,
                        help="~2 min first run — cached after that")

    st.divider()
    st.subheader("Day types")
    do_wwd   = st.checkbox("Winter weekday",         True)
    do_swd   = st.checkbox("Summer weekday",         True)
    do_wwe   = st.checkbox("Winter weekend (48h)",   False)
    do_swe   = st.checkbox("Summer weekend (48h)",   False)
    do_price = st.checkbox("Price profile analysis", False)

    st.divider()
    st.markdown("**S.KOe COOL specs**")
    st.caption("70 kWh total / 60 kWh usable")
    st.caption("22 kW AC bidirectional (ISO 15118-20)")
    st.caption("Cold-chain floor: SoC ≥ 20%")
    st.caption("Departure target: SoC = 100%")


# =============================================================================
#  LOAD CSV
# =============================================================================

if not Path(CSV_PATH).exists():
    st.error(
        f"CSV file '{CSV_PATH}' not found. "
        "Make sure it is committed to the GitHub repo."
    )
    st.stop()

with st.spinner("Loading 2025 SMARD price data..."):
    try:
        df_info = _load_csv_raw(CSV_PATH)
        n_days  = len(df_info) // 96
        st.success(
            f"2025 SMARD DE/LU prices loaded  |  {n_days} days  |  "
            f"{df_info.index[0].date()} → {df_info.index[-1].date()}  |  "
            f"Price range: {df_info['price'].min()*1000:.0f}–"
            f"{df_info['price'].max()*1000:.0f} EUR/MWh"
        )
    except Exception as e:
        st.error(f"Could not load CSV: {e}")
        st.stop()


# =============================================================================
#  RUN AND DISPLAY
# =============================================================================

CONFIGS = []
if do_wwd: CONFIGS.append(("Winter Weekday",               "winter_weekday"))
if do_swd: CONFIGS.append(("Summer Weekday",               "summer_weekday"))
if do_wwe: CONFIGS.append(("Winter Weekend (48h Sat+Sun)", "winter_weekend"))
if do_swe: CONFIGS.append(("Summer Weekend (48h Sat+Sun)", "summer_weekend"))

if not CONFIGS and not do_price:
    st.warning("Select at least one day type in the sidebar.")
    st.stop()

v2g            = V2GParams()
all_season_res = {}

for (label, season_key) in CONFIGS:
    st.subheader(label)

    with st.spinner(f"Computing {label} — first run ~30s, cached after..."):
        try:
            results, results_kpi, buy_d, plug_d, hours_d, is_wknd, is_48h = \
                run_day_type_cached(
                    season_key, arrival_h, departure_h, soc_init_pct,
                    do_B, do_C, do_D
                )
            all_season_res[season_key] = results_kpi
        except Exception as e:
            st.error(f"Optimisation error in {label}: {e}")
            continue

    # ── Chart ─────────────────────────────────────────────────────────────────
    buf = BytesIO()
    plot_season_chart(
        v2g, label, buy_d, plug_d, hours_d,
        results, is_wknd, arrival_h, departure_h,
        is_48h=is_48h, out=buf
    )
    buf.seek(0)
    st.image(buf, use_container_width=True)

    # ── Download chart ────────────────────────────────────────────────────────
    buf2 = BytesIO()
    plot_season_chart(
        v2g, label, buy_d, plug_d, hours_d,
        results, is_wknd, arrival_h, departure_h,
        is_48h=is_48h, out=buf2
    )
    buf2.seek(0)
    st.download_button(
        f"Download {label} chart (PNG)",
        data=buf2, file_name=f"v2g_{season_key}.png",
        mime="image/png", key=f"dl_{season_key}"
    )

    # ── KPI table ─────────────────────────────────────────────────────────────
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

# ── KPI multi-table ────────────────────────────────────────────────────────────
if len(all_season_res) > 1:
    st.subheader("KPI Comparison — All Day Types")
    try:
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
    except Exception as e:
        st.error(f"KPI comparison error: {e}")
    st.divider()

# ── Price analysis ─────────────────────────────────────────────────────────────
if do_price:
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

st.caption(
    "S.KOe COOL V2G Optimisation  ·  "
    "TU Dortmund IE³ × Schmitz Cargobull AG  ·  "
    "Thesis 2026  ·  Confidential"
)