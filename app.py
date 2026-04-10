import streamlit as st
import altair as alt

from v2g import (
    V2GParams, WINTER_M, SUMMER_M, SC_COL,
    FIXED_PRICE_EUR_KWH,
    _passthrough_profile, _load_csv_raw,
    get_tru_1h_trace, tru_avg_kw,
    compute_reefer_costs,
    get_wd_window, build_wd_display,
    run_A_dumb, run_B_smart, run_C_milp, run_D_mpc,
    make_kpi,
    realize_planned_window_under_actual_times,
    expand_to_minutes,compose_v2gp_price
)

from pathlib import Path
import numpy as np
import pandas as pd

# ── German all-in tariff components ───────────────────────
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
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        assert 0 <= h <= 23 and 0 <= m <= 59
        return h + m / 60.0
    except Exception:
        return default


def fmt_hhmm(h: float) -> str:
    return f"{int(h):02d}:{int((h % 1) * 60):02d}"

# =============================================================================
#  ALTAIR HELPERS
# =============================================================================

def _alt_x(hours_d, is_48h, is_wknd_fullday):
    """Shared X encoding + domain bounds for all charts."""
    dt   = float(hours_d[1] - hours_d[0]) if len(hours_d) > 1 else 1.0
    xmin = float(hours_d[0])
    xmax = float(hours_d[-1]) + dt
    if is_48h:
        ticks = list(range(0, 49, 4))
        expr  = ("(datum.value < 24 ? 'Sat ' : 'Sun ') +"
                 "((datum.value%24)<10?'0':'')+floor(datum.value%24)+':00'")
    elif is_wknd_fullday:
        ticks = list(range(0, 25, 2))
        expr  = "(datum.value<10?'0':'')+datum.value+':00'"
    else:
        ticks = list(range(12, 37, 2))
        expr  = "((datum.value%24)<10?'0':'')+floor(datum.value%24)+':00'"
    enc = alt.X('hour:Q',
                scale=alt.Scale(domain=[xmin, xmax]),
                axis=alt.Axis(values=ticks, labelExpr=expr, labelAngle=-35, title=''))
    return enc, xmin, xmax, dt


def _plug_rects(hours_d, plug_d, dt):
    """Returns list of {x, x2} dicts for plug-in shading."""
    rects, in_p, s = [], False, None
    for t, h in enumerate(hours_d):
        if plug_d[t] > 0.5 and not in_p:
            s, in_p = float(h), True
        elif plug_d[t] < 0.5 and in_p:
            rects.append({'x': s, 'x2': float(h)})
            in_p = False
    if in_p and s is not None:
        rects.append({'x': s, 'x2': float(hours_d[-1]) + dt})
    return rects


def _vrow_data(arrival_h, departure_h, is_48h, is_wknd_fullday,
               arrival_act_h=None, departure_act_h=None):
    """Returns list of {x, c, lbl} for vertical rule marks."""
    if is_wknd_fullday:
        return []
    def dx(h): return h if h >= 12. else h + 24.
    rows = [{'x': dx(0.), 'c': '#888888', 'lbl': 'Midnight'}]
    if is_48h:
        rows.append({'x': 24., 'c': '#555555', 'lbl': 'Day boundary'})
    else:
        rows += [
            {'x': dx(arrival_h),   'c': '#1B5E20', 'lbl': f'Plan arr {int(arrival_h):02d}:00'},
            {'x': dx(departure_h), 'c': '#B71C1C', 'lbl': f'Plan dep {int(departure_h):02d}:00'},
        ]
        if arrival_act_h is not None and abs(arrival_act_h - arrival_h) > 1e-9:
            rows.append({'x': dx(arrival_act_h), 'c': '#00C853',
                         'lbl': f'Actual arr {int(arrival_act_h):02d}:00'})
        if departure_act_h is not None and abs(departure_act_h - departure_h) > 1e-9:
            rows.append({'x': dx(departure_act_h), 'c': '#FF5252',
                         'lbl': f'Actual dep {int(departure_act_h):02d}:00'})
    return rows


# =============================================================================
#  POWER CHART  (Altair — interactive, minute-level resolution)
# =============================================================================

def make_power_chart(v2g, hours_d, buy_d, plug_d,
                     result_A, result_X,
                     x_label, x_key,
                     arrival_h, departure_h,
                     is_48h, is_wknd_fullday=False,
                     tru_d=None,
                     fixed_net_ct=None, vat_rate=None,
                     arrival_act_h=None, departure_act_h=None):

    lbl_x  = result_X["label"].split("(")[0].strip()
    buy_ct = to_allin_ct(buy_d, fixed_net_ct, vat_rate)
    x_enc, xmin, xmax, dt = _alt_x(hours_d, is_48h, is_wknd_fullday)

    # Helper to expand hourly display arrays to compressed minute pulses.
    # Consecutive charging (or discharging) slots are treated as one run and
    # front-loaded across the entire run, so the power trace is one continuous
    # block with no drops at slot boundaries.
    def expand_arr(P_c, P_d, tru_arr):
        N = len(hours_d) * 60
        h_min = np.zeros(N)
        pc_min = np.zeros(N)
        pd_min = np.zeros(N)
        tru_min = np.zeros(N)
        dt_h = float(hours_d[1] - hours_d[0]) if len(hours_d) > 1 else 1.0
        dt_m = dt_h / 60.0
        n = len(hours_d)

        def _pc(k): return float(P_c[k]) if k < len(P_c) else 0.0
        def _pd(k): return float(P_d[k]) if k < len(P_d) else 0.0
        def _tru(k): return float(tru_arr[k]) if (tru_arr is not None and k < len(tru_arr)) else 0.0

        # Pre-fill h_min and tru_min
        for i in range(n):
            tru_val = _tru(i)
            for m in range(60):
                idx = i * 60 + m
                h_min[idx] = float(hours_d[i]) + m * dt_m
                tru_min[idx] = tru_val

        # Charging runs: collect total energy, front-load across the whole run
        i = 0
        while i < n:
            if _pc(i) > 1e-6:
                j = i
                while j < n and _pc(j) > 1e-6:
                    j += 1
                E_run = sum(_pc(k) * dt_h for k in range(i, j))
                for slot in range(i, j):
                    p_max_c = max(0.0, v2g.charge_power_kW - _tru(slot))
                    for m in range(60):
                        if E_run > 1e-6 and p_max_c > 1e-9:
                            charge_e = min(E_run, p_max_c * dt_m)
                            pc_min[slot * 60 + m] = charge_e / dt_m
                            E_run -= charge_e
                i = j
            else:
                i += 1

        # Discharging runs: same approach
        i = 0
        while i < n:
            if _pd(i) > 1e-6:
                j = i
                while j < n and _pd(j) > 1e-6:
                    j += 1
                E_run = sum(_pd(k) * dt_h for k in range(i, j))
                p_max_d = v2g.discharge_power_kW
                for slot in range(i, j):
                    for m in range(60):
                        if E_run > 1e-6:
                            discharge_e = min(E_run, p_max_d * dt_m)
                            pd_min[slot * 60 + m] = discharge_e / dt_m
                            E_run -= discharge_e
                i = j
            else:
                i += 1

        return h_min, pc_min, pd_min, tru_min

    h_min, pcA_m, pdA_m, _ = expand_arr(result_A['Pc_d'], result_A['Pd_d'], tru_d)
    _, pcX_m, pdX_m, tru_m = expand_arr(result_X['Pc_d'], result_X['Pd_d'], tru_d)

    # Series styling map
    series_props = {
        'A - Dumb':       {'color': SC_COL['A'],   'dash': [1, 0]},
        f'{lbl_x}':       {'color': SC_COL[x_key], 'dash': [1, 0]},
        f'{lbl_x} V2G':   {'color': SC_COL[x_key], 'dash': [8, 3]},
        'TRU':            {'color': '#C62828',     'dash': [4, 2]}
    }

    # Build DataFrame with minute-level points
    STEP = 1
    rows =[]
    
    # 1. Dumb charge
    for i in range(0, len(h_min), STEP):
        rows.append({'hour': float(h_min[i]), 'kW': float(pcA_m[i]), 'series': 'A - Dumb'})
        
    # 2. X Smart/MILP/MPC charge
    for i in range(0, len(h_min), STEP):
        rows.append({'hour': float(h_min[i]), 'kW': float(pcX_m[i]), 'series': f'{lbl_x}'})
        
    # 3. X Smart/MILP/MPC discharge (V2G)
    if result_X['v2g_kwh'] > 0.05:
        for i in range(0, len(h_min), STEP):
            rows.append({'hour': float(h_min[i]), 'kW': float(-pdX_m[i]), 'series': f'{lbl_x} V2G'})
            
    # 4. TRU
    if tru_d is not None and np.any(np.array(tru_d) > 0.01):
        for i in range(0, len(h_min), STEP):
            rows.append({'hour': float(h_min[i]), 'kW': float(-tru_m[i]), 'series': 'TRU'})

    df_p = pd.DataFrame(rows)

    hours_ext = list(hours_d) + [float(hours_d[-1]) + dt]
    price_ext = list(buy_ct)  + [float(buy_ct[-1])]
    df_price  = pd.DataFrame({'hour': hours_ext, 'ct_kwh': price_ext})

    layers =[]

    # 1. Gold plug-in shading
    plug_r = _plug_rects(hours_d, plug_d, dt)
    if plug_r:
        layers.append(
            alt.Chart(pd.DataFrame(plug_r))
            .mark_rect(color='gold', opacity=0.18)
            .encode(x=alt.X('x:Q', scale=alt.Scale(domain=[xmin, xmax])), x2='x2:Q')
        )

    # 2. Vertical rules
    vrows = _vrow_data(arrival_h, departure_h, is_48h, is_wknd_fullday,
                       arrival_act_h, departure_act_h)
    if vrows:
        layers.append(
            alt.Chart(pd.DataFrame(vrows))
            .mark_rule(strokeDash=[4, 2], opacity=0.65)
            .encode(x='x:Q', color=alt.Color('c:N', scale=None, legend=None), tooltip='lbl:N')
        )

    # 3. Solid lines
    solid_series =['A - Dumb', f'{lbl_x}']
    df_solid = df_p[df_p['series'].isin(solid_series)]
    if not df_solid.empty:
        dom_s = df_solid['series'].unique().tolist()
        rng_s = [series_props[s]['color'] for s in dom_s]
        layers.append(
            alt.Chart(df_solid).mark_line(interpolate='step-after').encode(
                x=x_enc,
                y=alt.Y('kW:Q', axis=alt.Axis(title='Power (kW)')),
                color=alt.Color('series:N',
                    scale=alt.Scale(domain=dom_s, range=rng_s),
                    legend=alt.Legend(orient='bottom', title=None, columns=len(dom_s), symbolStrokeWidth=2)),
                tooltip=['series:N', alt.Tooltip('kW:Q', format='.2f'), alt.Tooltip('hour:Q', format='.2f', title='Hour')]
            )
        )

    # 4. Dashed lines
    dash_series = [s for s in df_p['series'].unique() if s not in solid_series]
    if dash_series:
        for sname in dash_series:
            col_val  = series_props[sname]['color']
            dash_pat = series_props[sname]['dash']
            layers.append(
                alt.Chart(df_p[df_p['series'] == sname])
                .mark_line(interpolate='step-after', strokeDash=dash_pat)
                .encode(
                    x=x_enc,
                    y=alt.Y('kW:Q'),
                    color=alt.Color('series:N',
                        scale=alt.Scale(domain=[sname], range=[col_val]),
                        legend=alt.Legend(orient='bottom', title=None, columns=1, symbolStrokeWidth=2, symbolDash=dash_pat)),
                    tooltip=['series:N', alt.Tooltip('kW:Q', format='.2f'), alt.Tooltip('hour:Q', format='.2f', title='Hour')]
                )
            )

    # 5. Price line
    price_layer = (
        alt.Chart(df_price)
        .mark_line(interpolate='step-after', color='#2E7D32', opacity=0.80, strokeDash=[3, 1])
        .encode(
            x=x_enc,
            y=alt.Y('ct_kwh:Q', axis=alt.Axis(title='ct/kWh (all-in)', titleColor='#2E7D32', orient='right'),
                    scale=alt.Scale(domain=[max(0., float(buy_ct.min()) - 2.), float(buy_ct.max()) + 2.])),
            tooltip=[alt.Tooltip('ct_kwh:Q', format='.1f', title='Price ct/kWh')]
        )
    )

    left  = alt.layer(*layers)
    chart = (
        alt.layer(left, price_layer)
        .resolve_scale(y='independent')
        .properties(title=f'Power — Dumb vs {x_label}', height=300, background='#FAFAFA')
        .interactive()
    )
    return chart


# =============================================================================
#  SOC CHART  (Altair — minute resolution)
# =============================================================================

def make_soc_chart(v2g, hours_d, plug_d,
                   result_A, result_X,
                   x_label, x_key,
                   arrival_h, departure_h,
                   is_48h, is_wknd_fullday=False,
                   arrival_act_h=None, departure_act_h=None):

    lbl_x = result_X["label"].split("(")[0].strip()
    x_enc, xmin, xmax, dt = _alt_x(hours_d, is_48h, is_wknd_fullday)

    if is_48h or is_wknd_fullday:
        arr_h0 = 0.0
    else:
        idx = np.where(np.array(plug_d) > 0.5)[0]
        arr_h0 = float(hours_d[idx[0]]) if len(idx) > 0 else float(arrival_h)

    def _minsoc(result):
        E0  = result['E_init_pct'] * v2g.usable_capacity_kWh / 100.0
        Pc, Pd = result['Pc_w'], result['Pd_w']
        tru = result.get('tru_w', None)
        # Guard: if tru is stored as full 24-slot display array, slice to window length
        if tru is not None and len(tru) != len(Pc):
            tru = tru[:len(Pc)]
        if len(Pc) == 0:
            return np.array([arr_h0]), np.array([result['E_init_pct']])
        t_m, _, _, soc_p = expand_to_minutes(v2g, np.asarray(Pc), np.asarray(Pd), E0, tru)
        return arr_h0 + t_m, soc_p

    t_A, s_A = _minsoc(result_A)
    t_X, s_X = _minsoc(result_X)

    STEP = 2
    rows =[]
    for i in range(0, len(t_A), STEP):
        rows.append({'hour': float(t_A[i]), 'SoC': float(s_A[i]), 'series': 'A - Dumb'})
    for i in range(0, len(t_X), STEP):
        rows.append({'hour': float(t_X[i]), 'SoC': float(s_X[i]), 'series': lbl_x})
    df_soc = pd.DataFrame(rows)

    layers =[]

    # 1. Gold plug-in shading
    plug_r = _plug_rects(hours_d, plug_d, dt)
    if plug_r:
        layers.append(
            alt.Chart(pd.DataFrame(plug_r))
            .mark_rect(color='gold', opacity=0.18)
            .encode(x=alt.X('x:Q', scale=alt.Scale(domain=[xmin, xmax])), x2='x2:Q')
        )

    # 2. Vertical rules
    vrows = _vrow_data(arrival_h, departure_h, is_48h, is_wknd_fullday,
                       arrival_act_h, departure_act_h)
    if vrows:
        layers.append(
            alt.Chart(pd.DataFrame(vrows))
            .mark_rule(strokeDash=[4, 2], opacity=0.65)
            .encode(x='x:Q', color=alt.Color('c:N', scale=None, legend=None), tooltip='lbl:N')
        )

    # 3. SoC lines
    layers.append(
        alt.Chart(df_soc).mark_line().encode(
            x=x_enc,
            y=alt.Y('SoC:Q', scale=alt.Scale(domain=[0, 115]), axis=alt.Axis(title='SoC (%)')),
            color=alt.Color('series:N',
                scale=alt.Scale(domain=['A - Dumb', lbl_x], range=[SC_COL['A'], SC_COL[x_key]]),
                legend=alt.Legend(orient='bottom', title=None, columns=3, symbolStrokeWidth=2)),
            tooltip=['series:N', alt.Tooltip('SoC:Q', format='.1f'), alt.Tooltip('hour:Q', format='.2f', title='Hour')]
        )
    )

    # 4. Reference lines
    ref_df = pd.DataFrame([
        {'y': v2g.soc_min_pct,       'c': '#C62828', 'lbl': f'SoC floor {v2g.soc_min_pct:.0f}%'},
        {'y': v2g.soc_departure_pct, 'c': '#0D47A1', 'lbl': f'Departure target {v2g.soc_departure_pct:.0f}%'},
    ])
    layers.append(
        alt.Chart(ref_df)
        .mark_rule(strokeDash=[4, 2], opacity=0.80)
        .encode(y='y:Q', color=alt.Color('c:N', scale=None, legend=None), tooltip='lbl:N')
    )

    chart = (
        alt.layer(*layers)
        .properties(title=f'SoC (1-min resolution) — Dumb vs {x_label}', height=300, background='#FAFAFA')
        .interactive()
    )
    return chart

# =============================================================================
#  RENDER ONE SEASON BLOCK
# =============================================================================

def render_season_block(v2g, season_title, color_hex,
                         hours_d, buy_d, plug_d, results,
                         arrival_h, departure_h,
                         is_48h, is_wknd_fullday,
                         do_B, do_C, do_D, tru_d=None,
                         fixed_net_ct=None, vat_rate=None,
                         arrival_act_h=None, departure_act_h=None):

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

    for result_X, x_label, x_key in comparisons:
        col_pow, col_soc = st.columns(2)
        with col_pow:
            chart = make_power_chart(
                v2g, hours_d, buy_d, plug_d,
                result_A, result_X, x_label, x_key,
                arrival_h, departure_h, is_48h, is_wknd_fullday, tru_d,
                fixed_net_ct=fixed_net_ct, vat_rate=vat_rate,
                arrival_act_h=arrival_act_h, departure_act_h=departure_act_h,
            )
            st.altair_chart(chart, use_container_width=True)
        with col_soc:
            chart = make_soc_chart(
                v2g, hours_d, plug_d,
                result_A, result_X, x_label, x_key,
                arrival_h, departure_h, is_48h, is_wknd_fullday,
                arrival_act_h=arrival_act_h, departure_act_h=departure_act_h,
            )
            st.altair_chart(chart, use_container_width=True)

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
    if len(profile) != 24:
        raise ValueError(f"Expected 24 slots, got {len(profile)}")
    return _passthrough_profile(profile)


@st.cache_data(show_spinner=False)
def load_date_profile(date_str: str) -> np.ndarray:
    df     = _load_csv_raw(CSV_PATH)
    target = pd.Timestamp(date_str).date()
    day_df = df[df["date"] == target]
    if len(day_df) == 0:
        raise ValueError(f"No price data found for {date_str}.")
    prices = day_df["price"].values
    if len(prices) != 24:
        raise ValueError(f"Expected 24 price slots for {date_str}, got {len(prices)}.")
    return _passthrough_profile(prices)


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

def _apply_mpc_distortions(buy_arr, reduce_night, reduce_pct,
                            increase_eve, increase_pct, spikes):
    p = buy_arr.copy().astype(float)
    for i in range(len(p)):
        h = i % 24
        if reduce_night and h <= 5:
            p[i] *= (1.0 - reduce_pct / 100.0)
        if increase_eve and 16 <= h <= 21:
            p[i] *= (1.0 + increase_pct / 100.0)
        for sp in spikes:
            if h == int(sp[0]) % 24:
                p[i] *= float(sp[1])
    return p
# =============================================================================
#  SCENARIO RUNNERS
# =============================================================================
@st.cache_data(show_spinner=False)
def run_seasonal(season_key, arrival_h, departure_h,
                 soc_pct, soc_departure_pct, tru_cycle,
                 do_B, do_C, do_D,
                 arrival_dev_h=0.0, departure_dev_h=0.0,
                 aux_power_w=400, bat_heat_w=100,
                 reduce_night=False, reduce_night_pct=0,
                 increase_eve=False, increase_eve_pct=0,
                 price_spikes=()):
    
    months_map = {
        "winter_weekday": (WINTER_M, False, False),
        "summer_weekday": (SUMMER_M, False, False),
        "winter_weekend": (WINTER_M, True, True),
        "summer_weekend": (SUMMER_M, True, True),
    }
    months, is_wknd, is_48h = months_map[season_key]
    buy  = load_seasonal_profile(tuple(months), is_wknd)
    v2gp = compose_v2gp_price(buy, exempt_ct=0.0)   # current reg: spot only, no exemption
    v2g  = V2GParams(soc_departure_pct=soc_departure_pct)
    E_init = v2g.usable_capacity_kWh * soc_pct / 100.0

    # Parasitic constant loads while plugged in
    is_winter_season = any(m in WINTER_M for m in months)
    _aux_kw  = aux_power_w / 1000.0
    _heat_kw = (bat_heat_w / 1000.0) if is_winter_season else 0.0
    _parasitic_kw = _aux_kw + _heat_kw

    # ---------------------------------------------------------
    # WEEKEND 48H CASE (unchanged)
    # ---------------------------------------------------------
    if is_48h:
        buy48  = np.concatenate([buy, buy])
        v2gp48 = compose_v2gp_price(buy48, exempt_ct=0.0)
        buy48_mpc = _apply_mpc_distortions(buy48, reduce_night, reduce_night_pct,
                                            increase_eve, increase_eve_pct, price_spikes)
        W      = 48
        tru_w  = get_tru_1h_trace(tru_cycle, W, v2g.dt_h) + _parasitic_kw

        hours_d = np.arange(W) * v2g.dt_h
        plug_d  = np.ones(W)
        buy_d   = buy48_mpc

        results = []

        Pc, Pd, soc = run_A_dumb(v2g, buy48, v2gp48, W, E_init, tru_w)
        results.append(make_kpi("A - Dumb", v2g, Pc, Pd, soc,
                                buy48, v2gp48, E_init,
                                is_weekend_48=True, tru_w=tru_w))

        if do_B:
            Pc, Pd, soc = run_B_smart(v2g, buy48, v2gp48, E_init, tru_w)
            results.append(make_kpi("B - Smart (no V2G)", v2g, Pc, Pd, soc,
                                    buy48, v2gp48, E_init,
                                    is_weekend_48=True, tru_w=tru_w))

        if do_C:
            Pc, Pd, soc = run_C_milp(v2g, buy48, v2gp48, E_init, tru_w)
            results.append(make_kpi("C - MILP Day-Ahead", v2g, Pc, Pd, soc,
                                    buy48, v2gp48, E_init,
                                    is_weekend_48=True, tru_w=tru_w))

        if do_D:
            Pc, Pd, soc = run_D_mpc(v2g, buy48_mpc, v2gp48, E_init, tru_w)
            results.append(make_kpi("D - MPC (receding)", v2g, Pc, Pd, soc,
                                    buy48, v2gp48, E_init,
                                    is_weekend_48=True, tru_w=tru_w))

        results_kpi = [{
            **r,
            "net_cost":    r["net_cost"]/2,
            "charge_cost": r["charge_cost"]/2,
            "v2g_rev":     r["v2g_rev"]/2,
            "tru_cost":    r["tru_cost"]/2,
            "total_cost":  r["total_cost"]/2,
            "v2g_kwh":     r["v2g_kwh"]/2,
            "charge_kwh":  r["charge_kwh"]/2
        } for r in results]

        rc = compute_reefer_costs(tru_w[:24], buy[:24], v2g.dt_h)
        return results, results_kpi, buy_d, plug_d, hours_d, is_wknd, is_48h, tru_w, rc

# ---------------------------------------------------------

    abnormal = (abs(arrival_dev_h) > 1e-12) or (abs(departure_dev_h) > 1e-12)

    # Actual arrival/departure after applying real-world deviation
    arrival_act_h   = (arrival_h   + arrival_dev_h)   % 24.0
    departure_act_h = (departure_h + departure_dev_h) % 24.0

    if not (12.0 <= arrival_act_h < 24.0):
        raise ValueError("Actual weekday arrival must be 12:00–23:59.")
    if not (0.0 <= departure_act_h < 12.0):
        raise ValueError("Actual weekday departure must be 00:00–11:59.")
    if abs(arrival_act_h - departure_act_h) < 1e-12:
        raise ValueError("Actual arrival and departure cannot be equal.")

    # ── PLANNED window — what B and C optimise against (day-ahead commitment)
    win_plan, _, _, W_plan = get_wd_window(v2g, arrival_h, departure_h)
    buy_w_plan  = buy[win_plan]
    v2gp_w_plan = v2gp[win_plan]
    tru_w_plan  = get_tru_1h_trace(tru_cycle, W_plan, v2g.dt_h) + _parasitic_kw

    # ── ACTUAL window — the real plug-in period; used by A, D, and for KPI eval
    win_act, arr_act, dep_act, W_act = get_wd_window(v2g, arrival_act_h, departure_act_h)
    buy_w_act  = buy[win_act]
    v2gp_w_act = v2gp[win_act]
    tru_w_act  = get_tru_1h_trace(tru_cycle, W_act, v2g.dt_h) + _parasitic_kw
    buy_dist = _apply_mpc_distortions(buy, reduce_night, reduce_night_pct,
                                      increase_eve, increase_eve_pct, price_spikes)
    buy_w_act_mpc = buy_dist[win_act]

    # ── Display: distorted prices for chart price line + KPI cost evaluation
    buy_d, plug_d, hours_d = build_wd_display(v2g, buy_dist, arrival_act_h, departure_act_h)

    # TRU display: reefer only draws grid power while physically plugged in
    tru_d = np.zeros(v2g.n_slots)
    tru_d[arr_act:dep_act] = tru_w_act[:dep_act - arr_act]

    results = []

    # ── A — DUMB ──────────────────────────────────────────────────────────────
    # No price awareness, no pre-committed schedule.
    # Always runs on ACTUAL window — deviation affects it immediately and fully.
    Pc, Pd, soc = run_A_dumb(v2g, buy_w_act, v2gp_w_act, W_act, E_init, tru_w_act)
    results.append(make_kpi("A - Dumb", v2g, Pc, Pd, soc,
                            buy_w_act, v2gp_w_act, E_init,
                            arr_act, dep_act, tru_w=tru_w_act))

    # ── B — SMART (no V2G) ────────────────────────────────────────────────────
    # Optimised day-ahead on PLANNED window (cheapest hours, charge-only).
    # When abnormal: plan is fixed; realized on ACTUAL window.
    # Slots that fall outside the actual plug-in period are skipped.
    if do_B:
        Pc, Pd, soc = run_B_smart(v2g, buy_w_plan, v2gp_w_plan, E_init, tru_w_plan)
        if abnormal:
            Pc, Pd, soc = realize_planned_window_under_actual_times(
                v2g, Pc, Pd, E_init,
                arrival_h,     departure_h,
                arrival_act_h, departure_act_h,
            )
        results.append(make_kpi("B - Smart (no V2G)", v2g, Pc, Pd, soc,
                                buy_w_act, v2gp_w_act, E_init,
                                arr_act, dep_act, tru_w=tru_w_act))

    # ── C — MILP Day-Ahead ────────────────────────────────────────────────────
    # Full MILP with V2G, optimised day-ahead on PLANNED window.
    # Same plan-then-realize logic as B. If the trailer arrives late and misses
    # the cheapest charging slots (e.g. 02:00–04:00), those slots are simply
    # lost — the plan cannot be retroactively changed. This exposes the
    # key weakness of open-loop day-ahead MILP under schedule uncertainty.
    if do_C:
        Pc, Pd, soc = run_C_milp(v2g, buy_w_plan, v2gp_w_plan, E_init, tru_w_plan)
        if abnormal:
            Pc, Pd, soc = realize_planned_window_under_actual_times(
                v2g, Pc, Pd, E_init,
                arrival_h,     departure_h,
                arrival_act_h, departure_act_h,
            )
        results.append(make_kpi("C - MILP Day-Ahead", v2g, Pc, Pd, soc,
                                buy_w_act, v2gp_w_act, E_init,
                                arr_act, dep_act, tru_w=tru_w_act))

    # ── D — MPC Receding Horizon ──────────────────────────────────────────────
    # Re-solves MILP at every hour using only the remaining actual horizon.
    # No pre-committed plan → adapts automatically to late or early arrival.
    # Always runs on ACTUAL window; no realize step is needed.
    if do_D:
        Pc, Pd, soc = run_D_mpc(v2g, buy_w_act_mpc, v2gp_w_act, E_init, tru_w_act)
        results.append(make_kpi("D - MPC (receding)", v2g, Pc, Pd, soc,
                                buy_w_act, v2gp_w_act, E_init,
                                arr_act, dep_act, tru_w=tru_w_act))

    rc = compute_reefer_costs(tru_w_act, buy_w_act, v2g.dt_h)

    return results, results, buy_d, plug_d, hours_d, is_wknd, is_48h, tru_d, rc

@st.cache_data(show_spinner=False)
def run_specific_date(date_str, arrival_h, departure_h,
                      soc_pct, soc_departure_pct, tru_cycle,
                      do_B, do_C, do_D,
                      arrival_dev_h=0.0, departure_dev_h=0.0,
                      aux_power_w=400, bat_heat_w=100,
                      reduce_night=False, reduce_night_pct=0,
                      increase_eve=False, increase_eve_pct=0,
                      price_spikes=()):

    ts = pd.Timestamp(date_str)
    is_wknd  = ts.dayofweek >= 5
    is_winter_date = ts.month in WINTER_M
    v2g = V2GParams(soc_departure_pct=soc_departure_pct)
    E_init = v2g.usable_capacity_kWh * soc_pct / 100.0

    # Parasitic loads
    _aux_kw      = aux_power_w / 1000.0
    _heat_kw     = (bat_heat_w / 1000.0) if is_winter_date else 0.0
    _parasitic_kw = _aux_kw + _heat_kw

    # -------------------------------------------------------
    # Weekend = simple 24h plug-in, abnormality NOT applied
    # -------------------------------------------------------
    if is_wknd:
        buy = load_date_profile(date_str)
        v2gp = compose_v2gp_price(buy, exempt_ct=0.0)
        W = 24

        buy_w = buy
        v2gp_w = v2gp
        tru_w = get_tru_1h_trace(tru_cycle, W, v2g.dt_h) + _parasitic_kw

        buy_d = _apply_mpc_distortions(buy, reduce_night, reduce_night_pct,
                                       increase_eve, increase_eve_pct, price_spikes)
        plug_d = np.ones(24)
        hours_d = np.arange(24) * v2g.dt_h
        tru_d = tru_w

        arr, dep = 0, 24
        is_48h = False
        is_wknd_fullday = True

        results = []

        Pc, Pd, soc = run_A_dumb(v2g, buy_w, v2gp_w, W, E_init, tru_w)
        results.append(make_kpi("A - Dumb", v2g, Pc, Pd, soc,
                                buy_w, v2gp_w, E_init, arr, dep, tru_w=tru_w))

        if do_B:
            Pc, Pd, soc = run_B_smart(v2g, buy_w, v2gp_w, E_init, tru_w)
            results.append(make_kpi("B - Smart (no V2G)", v2g, Pc, Pd, soc,
                                    buy_w, v2gp_w, E_init, arr, dep, tru_w=tru_w))

        if do_C:
            Pc, Pd, soc = run_C_milp(v2g, buy_w, v2gp_w, E_init, tru_w)
            results.append(make_kpi("C - MILP Day-Ahead", v2g, Pc, Pd, soc,
                                    buy_w, v2gp_w, E_init, arr, dep, tru_w=tru_w))

        if do_D:
            Pc, Pd, soc = run_D_mpc(v2g, buy_d, v2gp_w, E_init, tru_w)
            results.append(make_kpi("D - MPC (receding)", v2g, Pc, Pd, soc,
                                    buy_w, v2gp_w, E_init, arr, dep, tru_w=tru_w))

        rc = compute_reefer_costs(tru_w, buy_w, v2g.dt_h)

        return (results, buy_d, plug_d, hours_d,
                is_wknd, is_48h, is_wknd_fullday, tru_d, rc)

    # -------------------------------------------------------
    # Weekday abnormality handling (arrival/departure deviation)
    # -------------------------------------------------------
    arrival_act_h = (arrival_h + arrival_dev_h) % 24.0
    departure_act_h = (departure_h + departure_dev_h) % 24.0

    abnormal = (abs(arrival_dev_h) > 1e-12) or (abs(departure_dev_h) > 1e-12)

    # Validate actual weekday times
    if not (12.0 <= arrival_act_h < 24.0):
        raise ValueError("Actual weekday arrival must remain between 12:00 and 23:59.")
    if not (0.0 <= departure_act_h < 12.0):
        raise ValueError("Actual weekday departure must remain between 00:00 and 11:59.")
    if abs(arrival_act_h - departure_act_h) < 1e-12:
        raise ValueError("Actual arrival and departure times cannot be equal.")

    # Load 48h demand
    buy_48 = load_two_day_profile(date_str)
    v2gp_48 = compose_v2gp_price(buy_48, exempt_ct=0.0)

    # Slot builder helper (planned vs actual)
    def _slots_48(arr_h, dep_h):
        a = round(arr_h / v2g.dt_h) % 24
        d = round(dep_h / v2g.dt_h) % 24
        if a == d:
            raise ValueError("Arrival and departure cannot be equal.")
        if d <= a:
            return list(range(a, 24)) + list(range(24, 24 + d))
        return list(range(a, d))

    # Planned window (day-ahead)
    slots_plan = _slots_48(arrival_h, departure_h)
    buy_w_plan = buy_48[slots_plan]
    v2gp_w_plan = v2gp_48[slots_plan]
    tru_w_plan = get_tru_1h_trace(tru_cycle, len(slots_plan), v2g.dt_h) + _parasitic_kw

    # Actual window (real execution + MPC)
    slots_act = _slots_48(arrival_act_h, departure_act_h)
    buy_w_act = buy_48[slots_act]
    v2gp_w_act = v2gp_48[slots_act]
    tru_w_act = get_tru_1h_trace(tru_cycle, len(slots_act), v2g.dt_h) + _parasitic_kw
    buy_48_dist = _apply_mpc_distortions(buy_48, reduce_night, reduce_night_pct,
                                         increase_eve, increase_eve_pct, price_spikes)
    buy_w_act_mpc = buy_48_dist[slots_act]

    # 24h display window: 12:00 -> next 12:00
    ROLL = round(12.0 / v2g.dt_h)
    buy_d = buy_48_dist[ROLL:ROLL + 24]
    hours_d = np.arange(24) * v2g.dt_h + 12.0

    dep_on_chart = (departure_act_h + 24.0) if departure_act_h < 12.0 else departure_act_h
    plug_d = ((hours_d >= arrival_act_h) & (hours_d < dep_on_chart)).astype(float)

    # Display indices for KPI
    arr_slot_act = round(arrival_act_h / v2g.dt_h) % 24
    arr_disp = arr_slot_act - ROLL
    dep_disp = arr_disp + len(slots_act)

    arr, dep = arr_disp, dep_disp

    # Build display TRU
    tru_d = np.zeros(24)
    d_s = max(0, arr_disp)
    d_e = min(24, dep_disp)
    w_s = d_s - arr_disp
    w_e = w_s + (d_e - d_s)
    if w_e > w_s:
        tru_d[d_s:d_e] = tru_w_act[w_s:w_e]

    is_48h = False
    is_wknd_fullday = False

    # -------------------------------------------------------
    # Run scenarios
    # -------------------------------------------------------
    results = []

    # ── A — DUMB
    # No schedule, no price awareness. Charges greedily from actual arrival.
    # Runs directly on ACTUAL window — no plan to realize.
    Pc, Pd, soc = run_A_dumb(v2g, buy_w_act, v2gp_w_act, len(slots_act), E_init, tru_w_act)
    results.append(make_kpi("A - Dumb", v2g, Pc, Pd, soc,
                            buy_w_act, v2gp_w_act, E_init,
                            arr, dep, tru_w=tru_w_act))

    # B - SMART
    if do_B:
        Pc, Pd, soc = run_B_smart(v2g, buy_w_plan, v2gp_w_plan, E_init, tru_w_plan)
        if abnormal:
            Pc, Pd, soc = realize_planned_window_under_actual_times(
                v2g, Pc, Pd, E_init,
                arrival_h, departure_h, arrival_act_h, departure_act_h
            )
        results.append(make_kpi("B - Smart (no V2G)", v2g, Pc, Pd, soc,
                                buy_w_act, v2gp_w_act, E_init,
                                arr, dep, tru_w=tru_w_act))

    # C - MILP
    if do_C:
        Pc, Pd, soc = run_C_milp(v2g, buy_w_plan, v2gp_w_plan, E_init, tru_w_plan)
        if abnormal:
            Pc, Pd, soc = realize_planned_window_under_actual_times(
                v2g, Pc, Pd, E_init,
                arrival_h, departure_h, arrival_act_h, departure_act_h
            )
        results.append(make_kpi("C - MILP Day-Ahead", v2g, Pc, Pd, soc,
                                buy_w_act, v2gp_w_act, E_init,
                                arr, dep, tru_w=tru_w_act))

    # D - MPC (adapts to actual)
    if do_D:
        Pc, Pd, soc = run_D_mpc(v2g, buy_w_act_mpc, v2gp_w_act, E_init, tru_w_act)
        results.append(make_kpi("D - MPC (receding)", v2g, Pc, Pd, soc,
                                buy_w_act, v2gp_w_act, E_init,
                                arr, dep, tru_w=tru_w_act))

    # TRU cost on actual window
    rc = compute_reefer_costs(tru_w_act, buy_w_act, v2g.dt_h)

    # -------------------------------------------------------
    # Return everything
    # -------------------------------------------------------
    return (results, buy_d, plug_d, hours_d,
            is_wknd, is_48h, is_wknd_fullday, tru_d, rc)


# =============================================================================
#  ANNUAL COMPUTATION — all days in CSV individually
# =============================================================================

@st.cache_data(show_spinner=False)
def run_annual_all_days(
    arrival_h, departure_h,
    soc_w, soc_s, soc_dep,
    tru_cycle, do_B, do_C, do_D,
    fixed_net_ct, vat_rate,
    v2g_exempt_ct, vat_fut_rate,
    fixed_price_eur,
):
    df_all    = _load_csv_raw(CSV_PATH)
    all_dates = sorted(df_all["date"].unique())

    sc_keys = ["A"]
    if do_B: sc_keys.append("B")
    if do_C: sc_keys.append("C")
    if do_D: sc_keys.append("D")

    # Current reg: double-tax = 0 (V2G_RECOVERABLE_CURRENT_CT)
    double_tax_eur = V2G_RECOVERABLE_CURRENT_CT / 100.0
    exempt_eur     = v2g_exempt_ct / 100.0
    # Future buy price: fixed levies reduced by exempt amount
    fixed_net_fut_ct = fixed_net_ct - v2g_exempt_ct

    acc = {sc: {"charge_cost": 0.0, "charge_cost_fut": 0.0,
                "v2g_rev_cur": 0.0, "v2g_rev_fut": 0.0}
           for sc in sc_keys}
    acc["F"] = {"charge_cost": 0.0, "charge_cost_fut": 0.0,
                "v2g_rev_cur": 0.0, "v2g_rev_fut": 0.0}

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
                tru_cycle, do_B, do_C, do_D,
            )
        except Exception:
            continue

        avg_spot_eur      = float(np.mean(buy_d))
        avg_allin_eur     = float(np.mean(to_allin_ct(buy_d, fixed_net_ct, vat_rate))) / 100.0
        avg_allin_fut_eur = float(np.mean(to_allin_ct(buy_d, fixed_net_fut_ct, vat_rate))) / 100.0

        result_A = results[0]
        acc["F"]["charge_cost"]     += result_A["charge_kwh"] * fixed_price_eur
        acc["F"]["charge_cost_fut"] += result_A["charge_kwh"] * fixed_price_eur

        for i, sc in enumerate(sc_keys):
            r = results[i]
            if sc == "A":
                charge_cost     = r["charge_kwh"] * fixed_price_eur
                charge_cost_fut = r["charge_kwh"] * fixed_price_eur
            else:
                charge_cost     = r["charge_kwh"] * avg_allin_eur
                charge_cost_fut = r["charge_kwh"] * avg_allin_fut_eur

            v2g_rev_cur = max(0.0,
                r["v2g_kwh"] * avg_spot_eur
                - r["v2g_kwh"] * double_tax_eur)

            v2g_rev_fut = max(0.0,
                r["v2g_kwh"] * avg_spot_eur
                + r["v2g_kwh"] * exempt_eur)

            acc[sc]["charge_cost"]     += charge_cost
            acc[sc]["charge_cost_fut"] += charge_cost_fut
            acc[sc]["v2g_rev_cur"]     += v2g_rev_cur
            acc[sc]["v2g_rev_fut"]     += v2g_rev_fut

        n_valid += 1

    out_scenarios = ["F"] + sc_keys
    out = {"n_days": n_valid, "scenarios": out_scenarios}
    for sc in out_scenarios:
        out[sc] = {
            "charge_cost":     acc[sc]["charge_cost"],
            "charge_cost_fut": acc[sc]["charge_cost_fut"],
            "v2g_rev_cur":     acc[sc]["v2g_rev_cur"],
            "v2g_rev_fut":     acc[sc]["v2g_rev_fut"],
            "net_cur":         acc[sc]["charge_cost"]     - acc[sc]["v2g_rev_cur"],
            "net_fut":         acc[sc]["charge_cost_fut"] - acc[sc]["v2g_rev_fut"],
        }
    return out

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
    "price_reduce_night": False,
    "price_reduce_night_pct": 20,
    "price_increase_eve": False,
    "price_increase_eve_pct": 20,
    "arrival_dev_h": 0.0,
    "departure_dev_h": 0.0,
    "aux_power_w": 400,
    "bat_heat_w":  100,
    "fut_network_fee_ct":  6.63,
    "fut_elec_tax_ct":     2.05,
}

if "cfg" not in st.session_state:
    st.session_state.cfg = DEFAULTS.copy()
if "show_output" not in st.session_state:
    st.session_state.show_output = False
if "price_spikes" not in st.session_state:
    st.session_state.price_spikes = []


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
            
            st.markdown("##### Abnormality / schedule deviation")
            cfg["arrival_dev_h"] = st.slider(
            "Actual arrival deviation (h)",
            min_value=-4.0, max_value=4.0,
            value=float(cfg.get("arrival_dev_h", 0.0)),
            step=1.0,
            help="Positive = later arrival, negative = earlier arrival. Applied to weekday abnormality analysis."
            )
            cfg["departure_dev_h"] = st.slider(
            "Actual departure deviation (h)",
            min_value=-4.0, max_value=4.0,
            value=float(cfg.get("departure_dev_h", 0.0)),
            step=1.0,
            help="Positive = later departure, negative = earlier departure. Applied to weekday abnormality analysis."
            )
            st.caption(
            "MILP / Smart are optimized on the planned schedule and then realized on the actual schedule. "
            "MPC adapts to the actual schedule."
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
            st.markdown("##### MPC Price Distortions")
            cfg["price_reduce_night"] = st.checkbox(
                "Reduce 00:00–05:00 prices", bool(cfg.get("price_reduce_night", False)), key="form_rn")
            if cfg["price_reduce_night"]:
                cfg["price_reduce_night_pct"] = st.number_input(
                    "Reduction %", -100, 100, int(cfg.get("price_reduce_night_pct", 20)), key="form_rn_pct")
            cfg["price_increase_eve"] = st.checkbox(
                "Increase 16:00–22:00 prices", bool(cfg.get("price_increase_eve", False)), key="form_ie")
            if cfg["price_increase_eve"]:
                cfg["price_increase_eve_pct"] = st.number_input(
                    "Increase %", -100, 200, int(cfg.get("price_increase_eve_pct", 20)), key="form_ie_pct")
            st.caption("Distortions apply only to MPC (D). B and C use original day-ahead prices.")

        with c3:
            st.subheader("Reefer (TRU) at Depot")
            cfg["aux_power_w"] = st.number_input(
                "Auxiliary power loss (W)",
                min_value=0, max_value=2000,
                value=int(cfg.get("aux_power_w", 400)), step=10,
                help="Constant grid draw while trailer is plugged in "
                     "(charger standby, lighting, control systems, etc.)")
            cfg["bat_heat_w"] = st.number_input(
                "Battery heating (W) — Winter only",
                min_value=0, max_value=1000,
                value=int(cfg.get("bat_heat_w", 100)), step=10,
                help="Constant power to keep battery warm in winter months (Oct–Mar). "
                     "Added only when is_winter=True.")
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

            cfg["fixed_price"] = st.number_input(
                "Fixed price (EUR/kWh)",
                value=float(cfg["fixed_price"]),
                min_value=0.05, max_value=1.0, step=0.01,
                help="Fixed-tariff benchmark baseline.")

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
            st.markdown("##### Future V2G charges")
            cfg["fut_network_fee_ct"] = st.number_input(
                "Network fee exempt (ct/kWh)",
                value=float(cfg.get("fut_network_fee_ct", 6.63)),
                min_value=0.0, max_value=15.0, step=0.01, format="%.3f")
            cfg["fut_elec_tax_ct"] = st.number_input(
                "Elec. tax exempt (ct/kWh)",
                value=float(cfg.get("fut_elec_tax_ct", 2.05)),
                min_value=0.0, max_value=10.0, step=0.01, format="%.3f")
            
        

            # ── ADD THIS block inside the st.form, just before the Submit button ─────
        # ── System Parameters (read-only reference) ──────────────────────
        with st.expander("📋 System Parameters & Pre-defined Values", expanded=False):
            st.markdown("These values are fixed in the model. They are shown here "
                        "for transparency — no input needed.")
            col_sp1, col_sp2, col_sp3 = st.columns(3)

            with col_sp1:
                st.markdown("**🔋 Battery (S.KOe COOL)**")
                st.caption("Total capacity: **70 kWh**")
                st.caption("Usable capacity: **60 kWh**")
                st.caption("Charge power: **22 kW**")
                st.caption("Discharge power: **22 kW**")
                st.caption("Charge efficiency η: **0.92**")
                st.caption("Discharge efficiency η: **0.92**")
                st.caption("Cold-chain SoC floor: **20 %**")
                st.caption("Degradation cost: **€0.00/kWh** (not yet active)")

            with col_sp2:
                st.markdown("**❄️ TRU Reefer Cycle Power**")
                st.caption("Continuous — High: **7.6 kW** for 1,717 s")
                st.caption("Continuous — Low:  **0.7 kW** for 292 s")
                st.caption("Continuous avg: **~6.6 kW**")
                st.caption("Start-Stop — High: **9.7 kW** for 975 s")
                st.caption("Start-Stop — Mid:  **0.65 kW** for 295 s")
                st.caption("Start-Stop — Off:  **0.0 kW** for 1,207 s")
                st.caption("Start-Stop avg: **~3.9 kW**")
                st.caption("Hi-res simulation: **10-second** intervals → averaged to 1h")

            with col_sp3:
                st.markdown("**⚡ Fixed Diesel Benchmark**")
                st.caption("Diesel price: **€1.80/L**")
                st.caption("Diesel energy: **9.8 kWh/L**")
                st.caption("Genset efficiency: **30%**")
                st.markdown("**📅 Seasonality**")
                st.caption("Winter: **Oct–Mar** (months 1,2,3,10,11,12)")
                st.caption("Summer: **Apr–Sep** (months 4,5,6,7,8,9)")
                st.markdown("**🔧 Optimisation**")
                st.caption("MILP solver: **scipy.optimize.milp** (HiGHS)")
                st.caption("Time limit per solve: **60 s**")
                st.caption("Price resolution: **hourly** (SMARD DE/LU 2025)")

        # ── Methodology (mirrored from bottom of results page) ────────────
        with st.expander("📐 Methodology & Assumptions", expanded=False):
            _fn_disp  = FIXED_NET_CT
            _vat_disp = VAT_RATE
            _fut_disp = V2G_RECOVERABLE_FUTURE_CT
            _dep_disp = int(cfg.get("soc_departure", 100))
            _fp_disp  = float(cfg.get("fixed_price", 0.35))
            st.markdown(f"""
**Price data:** SMARD DE/LU hourly day-ahead spot prices (2025).

**All-in depot price:** `(spot + {_fn_disp:.3f} ct/kWh taxes & levies) × {1+_vat_disp:.2f} VAT`

**Scenario A — Dumb:** Greedy charge from arrival until target SoC reached. No price awareness.

**Scenario B — Smart:** MILP, charge-only (`allow_discharge=False`). Schedules charging at cheapest hours while meeting departure SoC.

**Scenario C — MILP Day-Ahead:** Full MILP with V2G. Optimises both charge and discharge over the full overnight window using perfect price foresight.

**Scenario D — MPC Receding Horizon:** Re-solves MILP at every hour using remaining horizon.

**Scenario F — Fixed Tariff Benchmark:** Applies a flat rate of €{_fp_disp:.2f}/kWh to Scenario A's charge volume. No V2G.

**Current regulation V2G cost:** Re-applied Netzentgelt + Stromsteuer on each exported kWh (double taxation).

**Future regulation (MiSpeL):** Exported kWh receives fee exemption of {_fut_disp:.3f} ct/kWh (×VAT). **Pending EU state aid approval** — BNetzA 2025.

**Cold-chain floor:** SoC ≥ 20 % hard MILP constraint (cold-chain integrity — highest priority).

**Departure SoC target:** ≥ {_dep_disp} % at end of overnight window.

**Battery:** 70 kWh total / 60 kWh usable. Charge/discharge: 22 kW AC bidirectional.

**MILP slot-use penalty:** 1×10⁻⁴ EUR per active charging slot (consolidates charging into contiguous blocks; negligible effect on costs).
            """)
        
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
arrival_dev_h = float(cfg.get("arrival_dev_h", 0.0))
departure_dev_h = float(cfg.get("departure_dev_h", 0.0))
aux_power_w     = int(cfg.get("aux_power_w", 400))
bat_heat_w      = int(cfg.get("bat_heat_w",  100))
do_B          = bool(cfg["do_B"])
do_C          = bool(cfg["do_C"])
do_D          = bool(cfg["do_D"])
fixed_price   = float(cfg["fixed_price"])
mode          = cfg.get("mode", "Seasonal Average")
specific_date = cfg.get("specific_date", "2025-01-15")

# Derive tariff values from cfg
_fixed_net_ct  = FIXED_NET_CT
_vat_rate      = VAT_RATE
_v2g_double_ct = V2G_RECOVERABLE_CURRENT_CT   # currently 0.0 per your module-level constant
_v2g_exempt_ct = cfg.get("fut_network_fee_ct", 6.63) + cfg.get("fut_elec_tax_ct", 2.05)
_vat_fut_rate  = VAT_RATE                      # future VAT same as current unless you add a separate constant

# ── Sidebar quick-edit ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Quick Edit")
    st.caption("Changes apply immediately")
    cfg["arrival_str"]   = st.text_input("Arrival (HH:MM)",   cfg["arrival_str"])
    cfg["departure_str"] = st.text_input("Departure (HH:MM)", cfg["departure_str"])
    cfg["arrival_dev_h"] = st.slider(
        "Arrival deviation (h)", -4.0, 4.0,
        float(cfg.get("arrival_dev_h", 0.0)), 1.0
    )
    cfg["departure_dev_h"] = st.slider(
        "Departure deviation (h)", -4.0, 4.0,
        float(cfg.get("departure_dev_h", 0.0)), 1.0
    )
    cfg["aux_power_w"] = st.number_input(
        "Aux power loss (W)", min_value=0, max_value=2000,
        value=int(cfg.get("aux_power_w", 400)), step=10, key="sb_aux")
    cfg["bat_heat_w"] = st.number_input(
        "Bat heating (W, winter)", min_value=0, max_value=1000,
        value=int(cfg.get("bat_heat_w", 100)), step=10, key="sb_heat")

    # ── Auto-clear cache when deviation or parasitic loads change ──────────
    _watch = (cfg["arrival_dev_h"], cfg["departure_dev_h"],
              cfg["aux_power_w"], cfg["bat_heat_w"])
    if st.session_state.get("_watch_prev") != _watch:
        st.session_state["_watch_prev"] = _watch
        st.cache_data.clear()
        st.rerun()

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
    if st.button("🗑️ Clear Cache & Recompute", use_container_width=True,
                 help="Forces fresh computation — use after changing deviation sliders"):
        st.cache_data.clear()
        st.rerun()
    st.session_state.cfg = cfg
    arr_h         = parse_hhmm(cfg["arrival_str"],   16.0)
    dep_h         = parse_hhmm(cfg["departure_str"],  6.0)
    soc_w         = int(cfg["soc_winter"])
    soc_s         = int(cfg["soc_summer"])
    soc_dep       = int(cfg["soc_departure"])
    tru_cycle     = cfg["tru_cycle"]
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
        st.error(f"Could not load CSV: {e}")
        st.stop()

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

if abs(arrival_dev_h) > 1e-12 or abs(departure_dev_h) > 1e-12:
    st.info(
        "Abnormality active for weekday runs: "
        f"planned arrival/departure = {fmt_hhmm(arr_h)} / {fmt_hhmm(dep_h)}, "
        f"actual arrival/departure = {fmt_hhmm((arr_h + arrival_dev_h) % 24)} / "
        f"{fmt_hhmm((dep_h + departure_dev_h) % 24)}. "
        "Smart/MILP are planned on the nominal window and realized on the actual window; "
        "MPC adapts to the actual window."
    )

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

        rows = [{
            "Scenario"                  : f"F - Fixed Price",
            "Charge (€/d)"              : f"{F_charge_cost:.3f}",
            "V2G Rev (€/d)"             : "0.000",
            "Net (€/d)"                 : f"{F_charge_cost:.3f}"
        }]

        for r in results:
            if r["label"].startswith("A"):
                charge_cost = r["charge_kwh"] * fixed_price
            else:
                if reg == "future":
                    exempt_buy_eur = (_v2g_exempt_ct / 100.0)
                    reduced_fixed_ct = _fixed_net_ct - exempt_buy_eur * 100.0
                    avg_allin_fut_eur = float(np.mean(
                        to_allin_ct(buy_d, reduced_fixed_ct, _vat_rate))) / 100.0 if buy_d is not None and len(buy_d) > 0 else avg_allin_eur
                    charge_cost = r["charge_kwh"] * avg_allin_fut_eur
                else:
                    charge_cost = r["charge_kwh"] * avg_allin_eur

            if reg == "current":
                v2g_rev = max(0.0,
                    r["v2g_kwh"] * avg_spot_eur
                    - r["v2g_kwh"] * double_tax_eur)
            else:
                v2g_rev = max(0.0,
                    r["v2g_kwh"] * avg_spot_eur
                    + r["v2g_kwh"] * exempt_eur)
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

    hdr = f"**Charging KPI{' — ' + label if label else ''}**"
    st.markdown(hdr)

    col_cur, col_fut = st.columns(2)

    with col_cur:
        st.markdown("📋 **Current Regulation (2025)**")
        st.dataframe(pd.DataFrame(_build_rows("current")),
                     use_container_width=True, hide_index=True)

    with col_fut:
        st.markdown("🔮 **Future Regulation (MiSpeL)**")
        st.dataframe(pd.DataFrame(_build_rows("future")),
                     use_container_width=True, hide_index=True)

    if tru_cycle != "OFF" and rc["E_kWh"] > 0.01:
        st.markdown("**Reefer + Aux + Heating Energy Cost**")
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
                tru_cycle, do_B, do_C, do_D,
                aux_power_w=aux_power_w,
                bat_heat_w=bat_heat_w,
                reduce_night=bool(cfg.get("price_reduce_night", False)),
                reduce_night_pct=int(cfg.get("price_reduce_night_pct", 20)),
                increase_eve=bool(cfg.get("price_increase_eve", False)),
                increase_eve_pct=int(cfg.get("price_increase_eve_pct", 20)),
                price_spikes=tuple((sp["hour"], sp["mult"]) for sp in st.session_state.price_spikes),
            )
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

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
                tru_cycle, do_B, do_C, do_D,
                arrival_dev_h=arrival_dev_h,
                departure_dev_h=departure_dev_h,
                aux_power_w=aux_power_w,
                bat_heat_w=bat_heat_w,
                reduce_night=bool(cfg.get("price_reduce_night", False)),
                reduce_night_pct=int(cfg.get("price_reduce_night_pct", 20)),
                increase_eve=bool(cfg.get("price_increase_eve", False)),
                increase_eve_pct=int(cfg.get("price_increase_eve_pct", 20)),
                price_spikes=tuple((sp["hour"], sp["mult"]) for sp in st.session_state.price_spikes),
            )
        except Exception as e:
            st.error(f"Winter weekday error: {e}")

    if res_w is not None:
        # Compute actual times for chart annotation (None = no deviation)
        _arr_act_w = (arr_h + arrival_dev_h) % 24 if abs(arrival_dev_h) > 1e-9 else None
        _dep_act_w = (dep_h + departure_dev_h) % 24 if abs(departure_dev_h) > 1e-9 else None
        render_season_block(
            v2g,
            "Winter Weekday  (Oct – Mar average)",
            "#1565C0",
            hours_d_w, buy_d_w, plug_d_w, res_w,
            arr_h, dep_h, False, False,
            do_B, do_C, do_D, tru_d_w,
            fixed_net_ct=_fixed_net_ct, vat_rate=_vat_rate,
            arrival_act_h=_arr_act_w, departure_act_h=_dep_act_w,
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
                tru_cycle, do_B, do_C, do_D,
                arrival_dev_h=arrival_dev_h,
                departure_dev_h=departure_dev_h,
                aux_power_w=aux_power_w,
                bat_heat_w=bat_heat_w,
                reduce_night=bool(cfg.get("price_reduce_night", False)),
                reduce_night_pct=int(cfg.get("price_reduce_night_pct", 20)),
                increase_eve=bool(cfg.get("price_increase_eve", False)),
                increase_eve_pct=int(cfg.get("price_increase_eve_pct", 20)),
                price_spikes=tuple((sp["hour"], sp["mult"]) for sp in st.session_state.price_spikes),
            )
        except Exception as e:
            st.error(f"Summer weekday error: {e}")

    if res_s is not None:
        _arr_act_s = (arr_h + arrival_dev_h) % 24 if abs(arrival_dev_h) > 1e-9 else None
        _dep_act_s = (dep_h + departure_dev_h) % 24 if abs(departure_dev_h) > 1e-9 else None
        render_season_block(
            v2g,
            "Summer Weekday  (Apr – Sep average)",
            "#E65100",
            hours_d_s, buy_d_s, plug_d_s, res_s,
            arr_h, dep_h, False, False,
            do_B, do_C, do_D, tru_d_s,
            fixed_net_ct=_fixed_net_ct, vat_rate=_vat_rate,
            arrival_act_h=_arr_act_s, departure_act_h=_dep_act_s,
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
                    tru_cycle, do_B, do_C, do_D,
                    aux_power_w=aux_power_w,
                    bat_heat_w=bat_heat_w,
                    reduce_night=bool(cfg.get("price_reduce_night", False)),
                    reduce_night_pct=int(cfg.get("price_reduce_night_pct", 20)),
                    increase_eve=bool(cfg.get("price_increase_eve", False)),
                    increase_eve_pct=int(cfg.get("price_increase_eve_pct", 20)),
                    price_spikes=tuple((sp["hour"], sp["mult"]) for sp in st.session_state.price_spikes),
                )
            except Exception as e:
                st.error(f"{lbl} error: {e}")
                continue

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

    # ── Annual computation removed because make_annual_graphs was removed ──
    # If you ever want to restore the annual calculation, it must be indented 
    # OUTSIDE the weekend `for` loop, aligned with `st.subheader("KPI Tables")` below.

if mode != "Specific Date":
    st.subheader("KPI Tables")
    res_w_exists = "res_w" in dir() and res_w is not None
    res_s_exists = "res_s" in dir() and res_s is not None
    if res_w_exists and res_s_exists:
        col_w, col_s = st.columns(2)
        with col_w:
            show_kpi_table(res_w, fixed_price, tru_cycle, rc_w, "Winter Weekday", buy_d=buy_d_w)
        with col_s:
            show_kpi_table(res_s, fixed_price, tru_cycle, rc_s, "Summer Weekday", buy_d=buy_d_s)
    elif res_w_exists:
        show_kpi_table(res_w, fixed_price, tru_cycle, rc_w, "Winter Weekday", buy_d=buy_d_w)
    elif res_s_exists:
        show_kpi_table(res_s, fixed_price, tru_cycle, rc_s, "Summer Weekday", buy_d=buy_d_s)