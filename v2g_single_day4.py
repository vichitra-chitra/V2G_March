#!/usr/bin/env python3
"""
S.KOe COOL — V2G Optimisation Suite (v6)
TU Dortmund IE3 x Schmitz Cargobull AG | 2026

New in v6 vs v5:
  1. TRU reefer modelling — grid-capacity competition (Continuous / Start-Stop / OFF)
  2. German all-in tariff composition (spot + Netzentgelt + levies + VAT)
  3. Reefer energy cost table (dynamic / fixed / diesel comparison)
  4. Departure SoC target configurable (soc_departure_pct, default 100 %)
  5. Yearly extrapolation helper using seasonal month split
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from dataclasses import dataclass, field
from pathlib import Path
from datetime import timedelta

SC_COL   = {"A": "#999999", "B": "#2196F3", "C": "#00ACC1", "D": "#FF7700", "price": "#2E7D32"}
SC_FILL  = {"A": "#CCCCCC", "B": "#A5D6A7", "C": "#80DEEA", "D": "#FFCC80"}
WINTER_M = [1, 2, 3, 10, 11, 12]
SUMMER_M = [4, 5, 6, 7, 8, 9]

# Reefer cycle constants (10-second resolution, from measured data)
REEFER_DEFAULTS = {
    "P_HIGH_C":  7.6,   "T_HIGH_C":  1717,
    "P_LOW_C":   0.7,   "T_LOW_C":    292,
    "P_HIGH_SS": 9.7,   "T_HIGH_SS":  975,
    "P_MID_SS":  0.65,  "T_MID_SS":   295,
    "P_LOW_SS":  0.0,   "T_LOW_SS":  1207,
}

# German grid tariff components (ct/kWh unless noted)
GERMAN_TARIFF: dict = {
    "ConcessionFee_ct":    1.992,
    "OffshoreGridLevy_ct": 0.816,
    "CHPLevy_ct":          0.277,
    "ElectricityTax_ct":   2.05,
    "NEV19Levy_ct":        1.558,
    "NetworkUsageFees_ct": 6.63,
    "VAT_pc":             19.0,
}

FIXED_PRICE_EUR_KWH = 0.35
DIESEL_PRICE_EUR_L  = 1.80
DIESEL_KWH_PER_L    = 9.8
GENSET_EFF          = 0.30


# =============================================================================
#  1. PARAMETERS
# =============================================================================

@dataclass
class V2GParams:
    battery_capacity_kWh : float = 70.0
    usable_capacity_kWh  : float = 60.0
    soc_min_pct          : float = 20.0
    soc_max_pct          : float = 100.0
    soc_departure_pct    : float = 100.0
    charge_power_kW      : float = 22.0
    discharge_power_kW   : float = 22.0
    eta_charge           : float = 0.92
    eta_discharge        : float = 0.92
    deg_cost_eur_kwh     : float = 0.0
    dt_h                 : float = 0.25
    n_slots              : int   = 96

    @property
    def E_min(self) -> float:
        return self.usable_capacity_kWh * self.soc_min_pct / 100.0

    @property
    def E_max(self) -> float:
        return self.usable_capacity_kWh * self.soc_max_pct / 100.0

    @property
    def E_fin(self) -> float:
        return self.usable_capacity_kWh * self.soc_departure_pct / 100.0


# =============================================================================
#  2. TRU REEFER CYCLE FUNCTIONS
# =============================================================================

def _reefer_hi_res(cycle_type: str, N: int, dt_sec: int = 10) -> np.ndarray:
    D  = REEFER_DEFAULTS
    ct = cycle_type.strip().lower().replace("-", "").replace(" ", "")
    if ct == "continuous":
        pattern = [(D["P_HIGH_C"], D["T_HIGH_C"]), (D["P_LOW_C"], D["T_LOW_C"])]
    elif ct == "startstop":
        pattern = [(D["P_HIGH_SS"], D["T_HIGH_SS"]),
                   (D["P_MID_SS"],  D["T_MID_SS"]),
                   (D["P_LOW_SS"],  D["T_LOW_SS"])]
    else:
        return np.zeros(N)
    steps = [(p, max(1, round(t / dt_sec))) for p, t in pattern]
    pw    = np.concatenate([np.full(s, p) for p, s in steps])
    reps  = int(np.ceil(N / len(pw)))
    return np.tile(pw, reps)[:N]


def get_tru_15min_trace(cycle_type: str, W: int, dt_h: float = 0.25) -> np.ndarray:
    if cycle_type.strip().lower() in ("off", "noreeferstationary"):
        return np.zeros(W)
    dt_sec_hi       = 10
    slots_per_15min = int(dt_h * 3600 / dt_sec_hi)
    N_hi            = W * slots_per_15min
    P_hi            = _reefer_hi_res(cycle_type, N_hi, dt_sec_hi)
    return np.array([
        float(np.mean(P_hi[i * slots_per_15min:(i + 1) * slots_per_15min]))
        for i in range(W)
    ])


def tru_avg_kw(cycle_type: str) -> float:
    D  = REEFER_DEFAULTS
    ct = cycle_type.strip().lower().replace("-", "").replace(" ", "")
    if ct == "continuous":
        t = D["T_HIGH_C"] + D["T_LOW_C"]
        return (D["P_HIGH_C"] * D["T_HIGH_C"] + D["P_LOW_C"] * D["T_LOW_C"]) / t
    elif ct == "startstop":
        t = D["T_HIGH_SS"] + D["T_MID_SS"] + D["T_LOW_SS"]
        return (D["P_HIGH_SS"] * D["T_HIGH_SS"] + D["P_MID_SS"] * D["T_MID_SS"]) / t
    return 0.0


# =============================================================================
#  3. TARIFF & REEFER COST FUNCTIONS
# =============================================================================

def compose_all_in_price(spot_eur_kwh: np.ndarray,
                          tariff: dict = None) -> np.ndarray:
    if tariff is None:
        tariff = GERMAN_TARIFF
    fixed_ct = sum(v for k, v in tariff.items() if k != "VAT_pc")
    net      = np.asarray(spot_eur_kwh, dtype=float) + fixed_ct / 100.0
    return net * (1.0 + tariff["VAT_pc"] / 100.0)


def compute_reefer_costs(tru_w: np.ndarray,
                          buy_w: np.ndarray,
                          dt_h: float,
                          fixed_price: float = FIXED_PRICE_EUR_KWH,
                          diesel_price: float = DIESEL_PRICE_EUR_L) -> dict:
    E_kWh        = float(np.sum(tru_w) * dt_h)
    cost_dynamic = float(np.sum(tru_w * buy_w) * dt_h)
    cost_fixed   = E_kWh * fixed_price
    liters        = E_kWh / max(1e-12, DIESEL_KWH_PER_L * GENSET_EFF)
    cost_diesel  = liters * diesel_price
    return {
        "E_kWh":        E_kWh,
        "cost_dynamic": cost_dynamic,
        "cost_fixed":   cost_fixed,
        "cost_diesel":  cost_diesel,
        "diesel_liters": liters,
    }


# =============================================================================
#  4. YEARLY EXTRAPOLATION
# =============================================================================

def yearly_extrapolation(all_season_results: dict,
                          winter_months: int = 6,
                          wd_per_month: float = 22.0,
                          we_days_per_month: float = 8.7) -> pd.DataFrame:
    summer_months = 12 - winter_months
    multipliers = {
        "winter_weekday": winter_months * wd_per_month,
        "summer_weekday": summer_months * wd_per_month,
        "winter_weekend": winter_months * we_days_per_month,
        "summer_weekend": summer_months * we_days_per_month,
    }
    sc_labels = []
    for v in all_season_results.values():
        if v:
            sc_labels = [r["label"].split("(")[0].strip() for r in v]
            break
    if not sc_labels:
        return pd.DataFrame()

    rows = []
    for dt_key, mult in multipliers.items():
        if dt_key not in all_season_results or not all_season_results[dt_key]:
            continue
        res   = all_season_results[dt_key]
        label = dt_key.replace("_", " ").title()
        row   = {"Day Type": label}
        for r in res:
            sc      = r["label"].split("(")[0].strip()
            row[sc] = round(r["net_cost"] * mult, 0)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("Day Type")
    total = {sc: df[sc].sum() for sc in sc_labels if sc in df.columns}
    df.loc["TOTAL (EUR/yr)"] = total

    base_col = [c for c in df.columns if "Dumb" in c or c.startswith("A")]
    if base_col:
        bc = base_col[0]
        for sc in df.columns:
            if sc != bc:
                df.loc["Savings vs Dumb (EUR/yr)", sc] = round(
                    df.loc["TOTAL (EUR/yr)", bc] - df.loc["TOTAL (EUR/yr)", sc], 0
                )
        df.loc["Savings vs Dumb (EUR/yr)", bc] = 0.0

    return df


# =============================================================================
#  5. PRICE LOADING + 15-MIN INTERPOLATION
# =============================================================================

_CSV_CACHE: dict = {}


def _load_csv_raw(csv_path: str) -> pd.DataFrame:
    if csv_path in _CSV_CACHE:
        return _CSV_CACHE[csv_path]
    df = None
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(csv_path, sep=";", encoding=enc)
            if len(df.columns) > 1:
                break
            df = pd.read_csv(csv_path, sep=",", encoding=enc)
            if len(df.columns) > 1:
                break
        except Exception:
            continue
    if df is None or df.empty:
        raise ValueError(f"Could not read CSV: {csv_path}")
    pc = next((c for c in df.columns if "Germany" in c and "MWh" in c), None)
    if not pc:
        raise ValueError(f"Germany price column not found. Cols: {list(df.columns)}")
    df = df[["Start date", pc]].copy()
    df.columns = ["dt_str", "price_eur_mwh"]
    df["dt"] = pd.to_datetime(df["dt_str"], format="%b %d, %Y %I:%M %p", errors="coerce")
    df = df.dropna(subset=["dt", "price_eur_mwh"])
    df["price"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce") / 1000.0
    df = df.dropna(subset=["price"]).set_index("dt").sort_index()
    df["slot"]       = df.index.hour * 4 + df.index.minute // 15
    df["is_weekend"] = df.index.dayofweek >= 5
    df["month"]      = df.index.month
    df["date"]       = df.index.date
    _CSV_CACHE[csv_path] = df
    print(f"  CSV: {len(df):,} rows  {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def _interpolate_to_15min(profile: np.ndarray) -> np.ndarray:
    n    = len(profile)
    flat = sum(1 for i in range(0, n, 4) if np.ptp(profile[i:i + 4]) < 1e-9)
    if (n // 4) > 0 and flat / (n // 4) > 0.55:
        hourly  = profile[::4]
        x_h     = np.arange(24) + 0.5
        x_q     = np.arange(96) / 4.0
        profile = np.interp(x_q, x_h, hourly, left=hourly[0], right=hourly[-1])
        print("    [price fix] hourly -> 15-min interpolation applied")
    return profile


def load_avg_profile(csv_path: str, months: list, is_weekend: bool) -> np.ndarray:
    df      = _load_csv_raw(csv_path)
    mask    = df["month"].isin(months) & (df["is_weekend"] == is_weekend)
    sub     = df[mask]
    if len(sub) == 0:
        raise ValueError(f"No data for months={months}, weekend={is_weekend}")
    profile = sub.groupby("slot")["price"].mean().values
    if len(profile) != 96:
        raise ValueError(f"Expected 96 slots, got {len(profile)}")
    profile = _interpolate_to_15min(profile)
    n_days  = int(len(sub) / 96)
    lbl     = "WE" if is_weekend else "WD"
    print(f"    [{lbl} {months[0]}-{months[-1]}] avg {n_days} days | "
          f"{profile.min()*1000:.0f}-{profile.max()*1000:.0f} EUR/MWh")
    return profile


# =============================================================================
#  6. WINDOW + DISPLAY HELPERS
# =============================================================================

def get_wd_window(v2g, arrival_h, departure_h):
    DS = 12.0
    n  = v2g.n_slots; dt = v2g.dt_h
    a  = round(arrival_h   / dt) % n
    d  = round(departure_h / dt) % n
    window_slots = list(range(a, n)) + list(range(0, d))
    W = len(window_slots)
    def to_d(h): return round((h - DS) / dt) if h >= DS else round((h + 24. - DS) / dt)
    return window_slots, to_d(arrival_h), to_d(departure_h), W


def build_wd_display(v2g, buy, arrival_h, departure_h):
    n     = v2g.n_slots
    ROLL  = round(12.0 / v2g.dt_h)
    buy_d = np.roll(buy, -ROLL)
    h     = np.arange(n) * v2g.dt_h
    plug  = np.roll(((h >= arrival_h) | (h < departure_h)).astype(float), -ROLL)
    hours = np.arange(n) * v2g.dt_h + 12.0
    return buy_d, plug, hours


def to_display_wd(v2g, Pc_w, Pd_w, soc_w, arr, dep, E_init):
    n   = v2g.n_slots; pct = 100.0 / v2g.usable_capacity_kWh
    W   = dep - arr
    Pc  = np.zeros(n); Pd = np.zeros(n)
    soc = np.full(n, E_init * pct)
    Pc[arr:dep]  = Pc_w[:W]; Pd[arr:dep]  = Pd_w[:W]
    soc[arr:dep] = soc_w[:W] * pct
    if dep < n:
        soc[dep:] = soc_w[W - 1] * pct
    return Pc, Pd, soc


def soc_ramp(hours, soc_pct, init_pct):
    n  = len(hours)
    dt = (hours[1] - hours[0]) if n > 1 else 0.25
    s0 = np.concatenate([[init_pct], soc_pct[:-1]])
    x  = np.empty(2 * n); y = np.empty(2 * n)
    for i in range(n):
        x[2*i] = hours[i];       y[2*i] = s0[i]
        x[2*i+1] = hours[i] + dt; y[2*i+1] = soc_pct[i]
    return x, y


# =============================================================================
#  7. MILP SOLVER
# =============================================================================

def solve_milp(v2g, buy_w, v2gp_w, E_init, E_fin,
               allow_discharge=True, tru_w=None):
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix, csc_matrix

    W  = len(buy_w); dt = v2g.dt_h
    ic  = np.arange(W);        id_ = np.arange(W,   2*W)
    ie  = np.arange(2*W, 3*W); izc = np.arange(3*W, 4*W)
    izd = np.arange(4*W, 5*W); nv  = 5 * W

    if tru_w is not None and len(tru_w) == W:
        p_c_eff = np.maximum(0.0, v2g.charge_power_kW - np.asarray(tru_w))
    else:
        p_c_eff = np.full(W, v2g.charge_power_kW)

    c = np.zeros(nv)
    c[ic]  = buy_w * dt
    if allow_discharge:
        c[id_] = -v2gp_w * dt

    lb = np.zeros(nv); ub = np.full(nv, np.inf)
    ub[ic]  = p_c_eff
    ub[id_] = v2g.discharge_power_kW if allow_discharge else 0.
    lb[ie]  = v2g.E_min; ub[ie] = v2g.E_max
    lb[izc] = 0.; ub[izc] = 1.
    lb[izd] = 0.; ub[izd] = 1.
    integ = np.zeros(nv); integ[izc] = 1; integ[izd] = 1

    nr = 4*W + 1
    A  = lil_matrix((nr, nv))
    lo = np.full(nr, -np.inf); hi = np.zeros(nr)

    for t in range(W):
        A[t, ie[t]]  =  1.; A[t, ic[t]]  = -v2g.eta_charge * dt
        A[t, id_[t]] =  1. / v2g.eta_discharge * dt
        rhs = E_init if t == 0 else 0.
        if t > 0:
            A[t, ie[t-1]] = -1.
        lo[t] = hi[t] = rhs

    for t in range(W):
        A[W+t,   ic[t]]  =  1.; A[W+t,   izc[t]] = -v2g.charge_power_kW
        A[2*W+t, id_[t]] =  1.; A[2*W+t, izd[t]] = -v2g.discharge_power_kW
        A[3*W+t, izc[t]] =  1.; A[3*W+t, izd[t]] =  1.
        hi[W+t] = hi[2*W+t] = 0.; hi[3*W+t] = 1.

    A[4*W, ie[W-1]] = 1.; lo[4*W] = E_fin; hi[4*W] = v2g.E_max

    res = milp(c,
               constraints=LinearConstraint(csc_matrix(A), lo, hi),
               integrality=integ,
               bounds=Bounds(lb, ub),
               options={"disp": False, "time_limit": 60})
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.status!r} {res.message!r}")
    return np.clip(res.x[ic], 0, None), np.clip(res.x[id_], 0, None), res.x[ie]


# =============================================================================
#  8. SCENARIO RUNNERS
# =============================================================================

def run_A_dumb(v2g, buy_w, v2gp_w, W, E_init, tru_w=None):
    dt     = v2g.dt_h
    E_fin  = v2g.E_fin
    p_c_eff = (np.maximum(0.0, v2g.charge_power_kW - np.asarray(tru_w))
               if tru_w is not None else np.full(W, v2g.charge_power_kW))
    Pc  = np.zeros(W); Pd = np.zeros(W); soc = np.zeros(W); s = E_init
    for t in range(W):
        if s < E_fin:
            p = min(float(p_c_eff[t]), (E_fin - s) / (v2g.eta_charge * dt))
            Pc[t] = p; s = min(E_fin, s + p * v2g.eta_charge * dt)
        soc[t] = s
    return Pc, Pd, soc


def run_B_smart(v2g, buy_w, v2gp_w, E_init, tru_w=None):
    return solve_milp(v2g, buy_w, v2gp_w, E_init, v2g.E_fin,
                      allow_discharge=False, tru_w=tru_w)


def run_C_milp(v2g, buy_w, v2gp_w, E_init, tru_w=None):
    return solve_milp(v2g, buy_w, v2gp_w, E_init, v2g.E_fin,
                      allow_discharge=True, tru_w=tru_w)


def run_D_mpc(v2g, buy_w, v2gp_w, E_init, tru_w=None):
    W = len(buy_w); dt = v2g.dt_h; s = E_init
    Pc_all = np.zeros(W); Pd_all = np.zeros(W); soc_all = np.zeros(W)
    for t in range(W):
        tw_t = tru_w[t:] if tru_w is not None else None
        Pcw, Pdw, _ = solve_milp(v2g, buy_w[t:], v2gp_w[t:],
                                  s, v2g.E_fin, allow_discharge=True, tru_w=tw_t)
        pc = float(np.clip(Pcw[0], 0, v2g.charge_power_kW))
        pd = float(np.clip(Pdw[0], 0, v2g.discharge_power_kW))
        if pc > 1e-6 and pd > 1e-6:
            pc, pd = (0., pd) if v2gp_w[t] > buy_w[t] else (pc, 0.)
        s = float(np.clip(s + pc*v2g.eta_charge*dt - pd/v2g.eta_discharge*dt,
                          v2g.E_min, v2g.E_max))
        Pc_all[t] = pc; Pd_all[t] = pd; soc_all[t] = s
    return Pc_all, Pd_all, soc_all


# =============================================================================
#  9. KPI DICT BUILDER
# =============================================================================

def make_kpi(label, v2g, Pc_w, Pd_w, soc_w, buy_w, v2gp_w, E_init_kwh,
             arr_disp=None, dep_disp=None, is_weekend_48=False, tru_w=None):
    dt       = v2g.dt_h
    chg      = float(np.sum(Pc_w * buy_w)  * dt)
    rev      = float(np.sum(Pd_w * v2gp_w) * dt)
    tru_cost = float(np.sum(tru_w * buy_w) * dt) if tru_w is not None else 0.0

    if is_weekend_48:
        pct   = 100.0 / v2g.usable_capacity_kWh
        Pc_d  = Pc_w; Pd_d = Pd_w
        soc_d = soc_w * pct
    else:
        Pc_d, Pd_d, soc_d = to_display_wd(v2g, Pc_w, Pd_w, soc_w,
                                           arr_disp, dep_disp, E_init_kwh)
    return {
        "label"       : label,
        "Pc_d"        : Pc_d, "Pd_d": Pd_d, "soc_d": soc_d,
        "Pc_w"        : Pc_w, "Pd_w": Pd_w, "soc_w_kwh": soc_w,
        "net_cost"    : chg - rev,
        "charge_cost" : chg,
        "v2g_rev"     : rev,
        "tru_cost"    : tru_cost,
        "total_cost"  : chg - rev + tru_cost,
        "v2g_kwh"     : float(np.sum(Pd_w) * dt),
        "charge_kwh"  : float(np.sum(Pc_w) * dt),
        "E_init_pct"  : E_init_kwh * 100.0 / v2g.usable_capacity_kWh,
    }


# =============================================================================
#  10. PLOT HELPERS
# =============================================================================

def _ticks_wd():
    pos  = np.arange(12., 37., 2.)
    lbls = [f"{int(h%24):02d}:00" for h in pos]
    return pos, lbls

def _ticks_48h():
    pos  = np.arange(0., 49., 4.)
    lbls = [f"{'Sat' if h<24 else 'Sun'}\n{int(h%24):02d}:00" for h in pos]
    return pos, lbls

def _format_ax(ax, ylabel, title, is_48h=False, ylim=None):
    pos, lbls = _ticks_48h() if is_48h else _ticks_wd()
    xmax = 48. if is_48h else 36.
    xmin =  0. if is_48h else 12.
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(pos); ax.set_xticklabels(lbls, fontsize=7.5, rotation=35, ha="right")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left", pad=4)
    ax.grid(True, alpha=0.22, zorder=0)
    if ylim:
        ax.set_ylim(*ylim)

def _legend_below(ax, handles, ncol=4):
    ax.legend(handles=handles, fontsize=8, ncol=ncol,
              loc="upper center", bbox_to_anchor=(0.5, -0.42),
              framealpha=0.95, edgecolor="#CCCCCC")

def _vert_lines(ax, arrival_h, departure_h, is_48h=False):
    if is_48h:
        ax.axvline(24., color="#555555", lw=1.2, ls="--", alpha=0.65, zorder=5)
        return
    def dx(h): return h if h >= 12. else h + 24.
    ax.axvline(dx(arrival_h),   color="#1B5E20", lw=1.1, ls=":", alpha=0.80, zorder=5)
    ax.axvline(dx(departure_h), color="#B71C1C", lw=1.1, ls=":", alpha=0.80, zorder=5)
    ax.axvline(dx(0.),          color="#555555", lw=1.2, ls="--", alpha=0.65, zorder=5)


# =============================================================================
#  11. SEASON CHART
# =============================================================================

def plot_season_chart(v2g, season_label, buy_d, plug_d, hours_d,
                      results, is_weekend, arrival_h, departure_h,
                      is_48h, out, tru_d=None, tru_cycle="OFF"):

    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(16, 15),
        gridspec_kw={"height_ratios": [1.0, 1.9, 1.9], "hspace": 0.82})
    fig.patch.set_facecolor("#F8F9FA")

    tru_label = (f"TRU: {tru_cycle} (avg {tru_avg_kw(tru_cycle):.1f} kW)"
                 if tru_cycle.strip().lower() not in ("off", "noreeferstationary")
                 else "TRU: OFF")

    if is_48h:
        dwell_str = (f"Arrives Friday {int(arrival_h):02d}:00 -> Sat+Sun "
                     f"-> Monday {int(departure_h):02d}:00")
    elif is_weekend:
        dwell_str = "Full 24h depot dwell"
    else:
        dwell_str = (f"Arrival: {int(arrival_h):02d}:00  |  "
                     f"Departure: {int(departure_h):02d}:00")

    fig.suptitle(
        f"S.KOe COOL  --  {season_label}  |  All Scenarios\n"
        f"{dwell_str}  |  Battery: {v2g.usable_capacity_kWh:.0f} kWh usable  |  "
        f"{tru_label}  |  Dep. target: {v2g.soc_departure_pct:.0f}%",
        fontsize=11, fontweight="bold", y=0.99)

    # Panel 1: Price
    for t in range(len(hours_d)):
        if plug_d[t] > 0.5:
            ax0.axvspan(hours_d[t], hours_d[t] + v2g.dt_h,
                        color="gold", alpha=0.22, lw=0, zorder=1)
    p_line, = ax0.step(hours_d, buy_d * 1000, where="post",
                       color=SC_COL["price"], lw=2.0, zorder=3,
                       label="Day-ahead price (EUR/MWh)")
    ax0.fill_between(hours_d, buy_d * 1000, step="post",
                     color=SC_COL["price"], alpha=0.10, zorder=2)
    _vert_lines(ax0, arrival_h, departure_h, is_48h)
    leg0 = [p_line, mpatches.Patch(color="gold", alpha=0.5, label="Plugged-in")]
    if is_48h:
        leg0.append(Line2D([0],[0], color="#555555", ls="--", lw=1.3, label="Sat midnight"))
    elif not is_weekend:
        leg0 += [
            Line2D([0],[0], color="#1B5E20", ls=":", lw=1.3, label=f"Arrival {int(arrival_h):02d}:00"),
            Line2D([0],[0], color="#B71C1C", ls=":", lw=1.3, label=f"Departure {int(departure_h):02d}:00"),
            Line2D([0],[0], color="#555555", ls="--", lw=1.3, label="Midnight"),
        ]
    _legend_below(ax0, leg0, ncol=5)
    _format_ax(ax0, "EUR / MWh", "(1) Day-Ahead Electricity Price + Plugged-In Availability", is_48h)

    # Panel 2: Power with fills + TRU
    leg1 = []
    for key, r in zip(["A", "B", "C", "D"], results):
        col  = SC_COL[key]; fill = SC_FILL[key]
        lbl  = r["label"].split("(")[0].strip()
        Pc   = r["Pc_d"]; Pd = r["Pd_d"]
        ax1.fill_between(hours_d, Pc, step="post", color=fill, alpha=0.38, zorder=2)
        h_c, = ax1.step(hours_d, Pc, where="post", color=col, lw=1.4, alpha=0.90,
                        zorder=3, label=f"{lbl}  charge")
        leg1.append(h_c)
        if r["v2g_kwh"] > 0.05:
            ax1.fill_between(hours_d, -Pd, step="post", color=fill, alpha=0.30, zorder=2)
            h_d, = ax1.step(hours_d, -Pd, where="post", color=col, lw=1.4, ls="--",
                            alpha=0.90, zorder=3, label=f"{lbl}  V2G discharge (-)")
            leg1.append(h_d)

    if tru_d is not None and np.any(tru_d > 0.01):
        h_t, = ax1.step(hours_d, -tru_d, where="post",
                        color="#C62828", lw=1.6, ls=":", alpha=0.85, zorder=4,
                        label=f"TRU load (-{tru_avg_kw(tru_cycle):.1f} kW avg)")
        leg1.append(h_t)

    ax1.axhline(0, color="black", lw=0.7)
    _vert_lines(ax1, arrival_h, departure_h, is_48h)
    _legend_below(ax1, leg1, ncol=3)
    _format_ax(ax1, "Power (kW)",
               "(2) Charge / Discharge Power  [solid=charging | dashed below 0=V2G/TRU]",
               is_48h)

    # Panel 3: SoC
    for t in range(len(hours_d)):
        if plug_d[t] > 0.5:
            ax2.axvspan(hours_d[t], hours_d[t] + v2g.dt_h, color="gold", alpha=0.12, lw=0, zorder=1)
    leg2 = []
    for key, r, ls in zip(["A", "B", "C", "D"], results, ["-", "-", "-", "--"]):
        col = SC_COL[key]; lbl = r["label"].split("(")[0].strip()
        xr, yr = soc_ramp(hours_d, r["soc_d"], r["E_init_pct"])
        h, = ax2.plot(xr, yr, color=col, lw=2.2, ls=ls, label=f"{lbl}  SoC (%)")
        leg2.append(h)
    ax2.axhline(v2g.soc_min_pct,       color="#C62828", ls=":", lw=1.5, zorder=3)
    ax2.axhline(v2g.soc_departure_pct, color="#0D47A1", ls=":", lw=1.5, zorder=3)
    if is_48h:
        ax2.axvline(24., color="#555555", lw=1.2, ls="--", alpha=0.65, zorder=5)
    else:
        _vert_lines(ax2, arrival_h, departure_h, False)
    leg2 += [
        Line2D([0],[0], color="#C62828", ls=":", lw=1.5,
               label=f"Cold-chain floor {v2g.soc_min_pct:.0f}%"),
        Line2D([0],[0], color="#0D47A1", ls=":", lw=1.5,
               label=f"Departure target {v2g.soc_departure_pct:.0f}%"),
    ]
    _legend_below(ax2, leg2, ncol=3)
    _format_ax(ax2, "State of Charge (%)",
               "(3) Battery SoC Trajectory  [gradual ramp = real energy flow]",
               is_48h, ylim=(0, 112))
    ax2.set_xlabel("Time of Day", fontsize=9)
    if is_48h:
        tr = ax2.get_xaxis_transform()
        ax2.text(12., 0.97, "Saturday", transform=tr, ha="center", va="top",
                 fontsize=9, color="#333333", fontweight="bold")
        ax2.text(36., 0.97, "Sunday",   transform=tr, ha="center", va="top",
                 fontsize=9, color="#333333", fontweight="bold")

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    if isinstance(out, str):
        print(f"  Saved -> {out}")


# =============================================================================
#  12. KPI MULTI-TABLE
# =============================================================================

def plot_kpi_multi(all_res, v2g, arrival_h, departure_h, run_mpc, out):
    day_cfg = [
        ("winter_weekday", "Winter Weekday",               "#1565C0"),
        ("summer_weekday", "Summer Weekday",               "#E65100"),
        ("winter_weekend", "Winter Weekend (48h Sat+Sun)", "#6A1B9A"),
        ("summer_weekend", "Summer Weekend (48h Sat+Sun)", "#2E7D32"),
    ]
    metrics = [
        ("Net EV cost (EUR/day)",   "net_cost",    "{:.4f}"),
        ("Charge cost (EUR/day)",   "charge_cost", "{:.4f}"),
        ("V2G revenue (EUR/day)",   "v2g_rev",     "{:.4f}"),
        ("TRU grid cost (EUR/day)", "tru_cost",    "{:.4f}"),
        ("Total grid (EUR/day)",    "total_cost",  "{:.4f}"),
        ("V2G export (kWh/day)",    "v2g_kwh",     "{:.2f}"),
        ("Daily savings vs Dumb",   "savings_day", "{:+.4f}"),
        ("Annual savings (x365)",   "savings_ann", "EUR {:+,.0f}"),
    ]

    def _key(lbl):
        for k in ("Dumb","Smart","MILP","MPC"):
            if k in lbl: return {"Dumb":"A","Smart":"B","MILP":"C","MPC":"D"}[k]
        return "A"

    fig, axes = plt.subplots(2, 2, figsize=(22, 14),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.25})
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        "S.KOe COOL -- KPI Summary: All Scenarios x All Day Types\n"
        f"Weekday arrival {int(arrival_h):02d}:00 | departure {int(departure_h):02d}:00 | "
        f"Dep. target {v2g.soc_departure_pct:.0f}% | Battery {v2g.usable_capacity_kWh:.0f} kWh",
        fontsize=13, fontweight="bold", y=1.01)

    col_bgs = ["#F5F5F5", "#F5F5F5", "#E3F2FD", "#E0F7FA", "#FFF3E0"]

    for ai, (dt_key, dt_lbl, hdr_col) in enumerate(day_cfg):
        ax = axes[ai // 2][ai % 2]
        ax.axis("off")
        ax.set_title(f"  {dt_lbl}  ", fontsize=11, fontweight="bold",
                     color="white", pad=10,
                     bbox=dict(facecolor=hdr_col, edgecolor="none", boxstyle="round,pad=0.4"))
        if dt_key not in all_res or not all_res[dt_key]:
            continue
        res       = all_res[dt_key]
        res_keys  = [_key(r["label"]) for r in res]
        res_short = [r["label"].split("(")[0].strip() for r in res]
        n_sc      = len(res)
        ref       = res[0]["net_cost"]

        cell_data = []
        for mname, mkey, mfmt in metrics:
            row = [mname]
            for i, r in enumerate(res):
                if mkey == "savings_day":
                    v = ref - r["net_cost"]
                    row.append("--" if i == 0 else mfmt.format(v))
                elif mkey == "savings_ann":
                    v = (ref - r["net_cost"]) * 365
                    row.append("--" if i == 0 else mfmt.format(v))
                else:
                    row.append(mfmt.format(r.get(mkey, 0.0)))
            cell_data.append(row)

        tbl = ax.table(cellText=cell_data,
                       colLabels=["Metric"] + res_short,
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 2.1)

        hdr_bgs = ["#263238"] + [SC_COL[k] for k in res_keys]
        for ci, hc in enumerate(hdr_bgs):
            cell = tbl[0, ci]; cell.set_facecolor(hc)
            cell.set_text_props(color="white", fontweight="bold")

        for ri in range(1, len(cell_data) + 1):
            for ci in range(n_sc + 1):
                cell = tbl[ri, ci]
                cell.set_facecolor(col_bgs[min(ci, len(col_bgs) - 1)])
                if ci == 0:
                    cell.set_text_props(fontweight="bold")
                txt = cell.get_text().get_text()
                if "+" in txt and ri >= len(metrics) - 1:
                    cell.set_text_props(color="#1B5E20", fontweight="bold")

        metric_w   = 0.38
        scenario_w = (1.0 - metric_w) / n_sc * 0.95
        for ri in range(len(cell_data) + 1):
            tbl[ri, 0].set_width(metric_w)
            for ci in range(1, n_sc + 1):
                tbl[ri, ci].set_width(scenario_w)

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    if isinstance(out, str):
        print(f"  Saved -> {out}")


# =============================================================================
#  13. FULL-YEAR SIMULATION
# =============================================================================

def run_full_year(v2g, csv_path, arrival_h, departure_h,
                  soc_winter_pct=80.0, soc_summer_pct=40.0,
                  run_mpc=True, tru_cycle="OFF"):
    df       = _load_csv_raw(csv_path)
    dates    = sorted(set(df["date"].unique()))
    records  = []
    carry_soc = {}

    print(f"\n  [Annual] {len(dates)} days | TRU={tru_cycle} | "
          f"{'A+B+C+D' if run_mpc else 'A+B+C'}...")

    for i, d in enumerate(dates):
        day_df = df[df["date"] == d]
        if len(day_df) != 96:
            continue
        buy     = _interpolate_to_15min(day_df["price"].values)
        v2gp    = buy.copy()
        ts      = pd.Timestamp(d)
        dow     = ts.dayofweek; month = int(ts.month)
        is_wknd = dow >= 5
        is_winter = month in WINTER_M

        soc_pct = soc_winter_pct if is_winter else soc_summer_pct
        E_std   = v2g.usable_capacity_kWh * soc_pct / 100.0

        if is_wknd and dow == 5:
            fri    = (ts - timedelta(days=1)).date()
            E_init = carry_soc.get(fri, E_std)
        elif is_wknd and dow == 6:
            sat    = (ts - timedelta(days=1)).date()
            E_init = carry_soc.get(sat, E_std)
        else:
            E_init = E_std

        if is_wknd:
            win = list(range(96)); buy_w = buy[win]; v2gp_w = v2gp[win]; W = 96
        else:
            a     = round(arrival_h   / v2g.dt_h) % 96
            dslot = round(departure_h / v2g.dt_h) % 96
            win   = list(range(a, 96)) + list(range(0, dslot))
            buy_w = buy[win]; v2gp_w = v2gp[win]; W = len(win)

        tru_w = get_tru_15min_trace(tru_cycle, W, v2g.dt_h)

        try:
            Pc, Pd, soc = run_A_dumb(v2g, buy_w, v2gp_w, W, E_init, tru_w)
            A_cost = float(np.sum(Pc * buy_w - Pd * v2gp_w) * v2g.dt_h)
            A_tru  = float(np.sum(tru_w * buy_w) * v2g.dt_h)

            Pc, Pd, soc = run_B_smart(v2g, buy_w, v2gp_w, E_init, tru_w)
            B_cost = float(np.sum(Pc * buy_w - Pd * v2gp_w) * v2g.dt_h)

            Pc, Pd, soc_c = run_C_milp(v2g, buy_w, v2gp_w, E_init, tru_w)
            C_chg  = float(np.sum(Pc * buy_w)  * v2g.dt_h)
            C_rev  = float(np.sum(Pd * v2gp_w) * v2g.dt_h)
            C_cost = C_chg - C_rev
            C_kwh  = float(np.sum(Pd) * v2g.dt_h)
            carry_soc[d] = float(soc_c[-1])

            if run_mpc:
                Pc, Pd, soc = run_D_mpc(v2g, buy_w, v2gp_w, E_init, tru_w)
                D_cost = float(np.sum(Pc * buy_w - Pd * v2gp_w) * v2g.dt_h)
                D_rev  = float(np.sum(Pd * v2gp_w) * v2g.dt_h)
            else:
                D_cost, D_rev = np.nan, np.nan

        except RuntimeError as e:
            print(f"    SKIP {d}: {e}"); continue

        records.append({
            "date": ts, "month": month, "is_wknd": is_wknd, "is_winter": is_winter,
            "cost_A": A_cost, "cost_B": B_cost, "cost_C": C_cost, "cost_D": D_cost,
            "tru_cost": A_tru, "rev_C": C_rev, "kwh_C": C_kwh,
        })
        if (i + 1) % 60 == 0 or i == len(dates) - 1:
            print(f"    {i+1}/{len(dates)} processed ...")

    return pd.DataFrame(records)


# =============================================================================
#  14. ANNUAL RESULTS CHART
# =============================================================================

def plot_annual_results(df, run_mpc, out):
    df = df.copy()
    df["sav_B"] = df["cost_A"] - df["cost_B"]
    df["sav_C"] = df["cost_A"] - df["cost_C"]
    if run_mpc:
        df["sav_D"] = df["cost_A"] - df["cost_D"]
    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12),
                             gridspec_kw={"hspace": 0.44, "wspace": 0.32})
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        "S.KOe COOL -- Full-Year Simulation  |  2025 SMARD DE/LU Day-Ahead Prices\n"
        "Weekdays: overnight dwell  |  Weekends: 48h Sat+Sun, SoC carry-forward",
        fontsize=13, fontweight="bold", y=1.00)

    ax1 = axes[0, 0]
    for key, ls in [("A","-"),("B","-"),("C","-"),("D","--")]:
        ck = f"cost_{key}"
        if key == "D" and not run_mpc: continue
        if ck not in df: continue
        lbl = f"{key}-{'Dumb' if key=='A' else 'Smart' if key=='B' else 'MILP' if key=='C' else 'MPC'}"
        ax1.plot(df["date"], df[ck].cumsum(), color=SC_COL[key], lw=2.5, ls=ls, label=lbl)
        tot = df[ck].sum()
        ax1.annotate(f"  EUR {tot:,.0f}", xy=(df["date"].iloc[-1], tot),
                     xytext=(5, 0), textcoords="offset points",
                     fontsize=8, color=SC_COL[key], fontweight="bold", va="center")
    ax1.set_title("(1) Cumulative Net EV Charging Cost", fontsize=10, fontweight="bold", loc="left")
    ax1.set_ylabel("Cumulative Cost (EUR)"); ax1.set_xlabel("Date")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.22)

    ax2 = axes[0, 1]; x = np.arange(12); w = 0.25
    for offset, key, show in [(-w,"B",True),(0,"C",True),(w,"D",run_mpc)]:
        if not show: continue
        vals = [df[df["month"] == m][f"sav_{key}"].sum() for m in range(1, 13)]
        lbl  = f"{key}-{'Smart' if key=='B' else 'MILP' if key=='C' else 'MPC'}"
        ax2.bar(x + offset, vals, width=w, color=SC_COL[key], alpha=0.85, label=lbl, zorder=3)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(MONTHS, fontsize=9)
    ax2.set_title("(2) Monthly Savings vs Dumb", fontsize=10, fontweight="bold", loc="left")
    ax2.set_ylabel("Savings (EUR)"); ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.22, axis="y", zorder=0)

    ax3 = axes[1, 0]
    sc_list = [("A","Dumb"),("B","Smart"),("C","MILP")] + ([("D","MPC")] if run_mpc else [])
    costs   = [df[f"cost_{k}"].sum() for k, _ in sc_list]
    sc_lbl  = [f"{k}\n{n}" for k, n in sc_list]
    sc_col  = [SC_COL[k] for k, _ in sc_list]
    bars = ax3.bar(range(len(sc_list)), costs, color=sc_col, alpha=0.85, width=0.55, zorder=3)
    for bar, v in zip(bars, costs):
        ax3.text(bar.get_x() + bar.get_width()/2, v + (20 if v >= 0 else -20),
                 f"EUR {v:,.0f}", ha="center", va="bottom" if v >= 0 else "top",
                 fontsize=9, fontweight="bold", color=bar.get_facecolor())
    ref = costs[0]
    for i, (cost, (key, _)) in enumerate(zip(costs[1:], sc_list[1:]), 1):
        sav = ref - cost
        if abs(sav) > 5:
            ax3.annotate(f"saves EUR {sav:+,.0f}/yr",
                         xy=(i, cost), xytext=(i, cost - 80), ha="center",
                         fontsize=8, color="#1B5E20", fontweight="bold",
                         arrowprops=dict(arrowstyle="-", color="#1B5E20", lw=0.5))
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_xticks(range(len(sc_list))); ax3.set_xticklabels(sc_lbl, fontsize=10)
    ax3.set_title("(3) Annual Net EV Charging Cost", fontsize=10, fontweight="bold", loc="left")
    ax3.set_ylabel("Annual Net Cost (EUR/year)"); ax3.grid(True, alpha=0.22, axis="y", zorder=0)

    ax4 = axes[1, 1]
    box_keys = ["cost_A","cost_B","cost_C"] + (["cost_D"] if run_mpc else [])
    box_data = [df[k].dropna().values for k in box_keys]
    box_lbls = ["A - Dumb","B - Smart","C - MILP"] + (["D - MPC"] if run_mpc else [])
    box_cols = [SC_COL[k[5]] for k in box_keys]
    bp = ax4.boxplot(box_data, patch_artist=True, labels=box_lbls,
                     medianprops=dict(color="black", lw=2.0),
                     flierprops=dict(marker=".", markersize=3, alpha=0.5))
    for patch, col in zip(bp["boxes"], box_cols):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    ax4.axhline(0, color="#B71C1C", lw=1.5, ls="--", alpha=0.7,
                label="Zero -- trailer earns below this")
    ax4.set_title("(4) Daily Cost Distribution", fontsize=10, fontweight="bold", loc="left")
    ax4.set_ylabel("Daily Net Cost (EUR/day)"); ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.22, axis="y")

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    if isinstance(out, str):
        print(f"  Saved -> {out}")


# =============================================================================
#  15. PRICE PROFILES CHART
# =============================================================================

def plot_price_profiles(csv_path, out):
    df   = _load_csv_raw(csv_path)
    h_ax = np.arange(96) * 0.25
    print("  Loading 4 season profiles for price analysis ...")
    w_wd = load_avg_profile(csv_path, WINTER_M, False)
    s_wd = load_avg_profile(csv_path, SUMMER_M, False)
    w_we = load_avg_profile(csv_path, WINTER_M, True)
    s_we = load_avg_profile(csv_path, SUMMER_M, True)

    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, axes = plt.subplots(2, 2, figsize=(18, 11),
                             gridspec_kw={"hspace": 0.48, "wspace": 0.30})
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle("S.KOe COOL -- Electricity Price Analysis  |  2025 SMARD DE/LU Day-Ahead",
                 fontsize=13, fontweight="bold", y=1.01)

    HT = np.arange(0, 25, 2); HL = [f"{int(h):02d}:00" for h in HT]
    def sax(ax, title):
        ax.set_xticks(HT); ax.set_xticklabels(HL, fontsize=8, rotation=35, ha="right")
        ax.set_xlim(0, 24); ax.set_ylabel("Price (EUR/MWh)"); ax.set_xlabel("Hour of Day")
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left"); ax.grid(True, alpha=0.22)

    ax1 = axes[0, 0]
    ax1.step(h_ax, w_wd*1000, where="post", color="#1565C0", lw=2.3, label="Winter weekday")
    ax1.step(h_ax, s_wd*1000, where="post", color="#E65100", lw=2.3, label="Summer weekday")
    ax1.fill_between(h_ax, w_wd*1000, step="post", color="#1565C0", alpha=0.10)
    ax1.fill_between(h_ax, s_wd*1000, step="post", color="#E65100", alpha=0.10)
    win_allin = compose_all_in_price(w_wd) * 1000
    sum_allin = compose_all_in_price(s_wd) * 1000
    ax1.step(h_ax, win_allin, where="post", color="#1565C0", lw=1.5, ls="--", alpha=0.6,
             label="Winter all-in (tariff+VAT)")
    ax1.step(h_ax, sum_allin, where="post", color="#E65100", lw=1.5, ls="--", alpha=0.6,
             label="Summer all-in (tariff+VAT)")
    spread = (w_wd - s_wd) * 1000; pk = int(np.argmax(np.abs(spread)))
    ax1.annotate(f"Max delta:\n{spread[pk]:+.0f} EUR/MWh",
                 xy=(h_ax[pk], max(w_wd[pk], s_wd[pk])*1000),
                 xytext=(h_ax[pk]+1.5, max(w_wd[pk], s_wd[pk])*1000+5),
                 fontsize=7.5, color="#333", fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color="#333", lw=0.7))
    ax1.legend(fontsize=8)
    sax(ax1, "(1) Winter vs Summer -- Spot vs All-In Price")

    ax2 = axes[0, 1]
    ax2.step(h_ax, w_wd*1000, where="post", color="#1565C0", lw=2.3, label="Winter weekday")
    ax2.step(h_ax, w_we*1000, where="post", color="#6A1B9A", lw=2.3, ls="--", label="Winter weekend")
    ax2.step(h_ax, s_we*1000, where="post", color="#C62828", lw=1.8, ls=":", alpha=0.75,
             label="Summer weekend")
    ax2.fill_between(h_ax, w_wd*1000, step="post", color="#1565C0", alpha=0.08)
    ax2.fill_between(h_ax, w_we*1000, step="post", color="#6A1B9A", alpha=0.08)
    ax2.legend(fontsize=8)
    sax(ax2, "(2) Weekday vs Weekend -- Winter Price Profile")

    ax3 = axes[1, 0]
    m_avg = df.groupby("month")["price"].mean()*1000
    m_std = df.groupby("month")["price"].std() *1000
    bar_c = ["#1565C0" if m in WINTER_M else "#E65100" for m in range(1, 13)]
    ax3.bar(range(12), [m_avg.get(m, 0) for m in range(1, 13)], color=bar_c,
            alpha=0.82, width=0.65, zorder=3,
            yerr=[m_std.get(m, 0) for m in range(1, 13)],
            error_kw=dict(lw=1.2, capsize=3, capthick=1.2, ecolor="#333"))
    ax3.set_xticks(range(12)); ax3.set_xticklabels(MONTHS, fontsize=9)
    ax3.set_ylabel("Avg Price (EUR/MWh)")
    ax3.set_title("(3) Monthly Average Spot Price (+-1 sigma)", fontsize=10, fontweight="bold", loc="left")
    ax3.legend(handles=[
        mpatches.Patch(color="#1565C0", alpha=0.82, label="Winter (Oct-Mar)"),
        mpatches.Patch(color="#E65100", alpha=0.82, label="Summer (Apr-Sep)"),
    ], fontsize=9)
    ax3.grid(True, alpha=0.22, axis="y", zorder=0)
    for i, v in enumerate([m_avg.get(m, 0) for m in range(1, 13)]):
        ax3.text(i, v+1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax4 = axes[1, 1]
    d_max = df.groupby("date")["price"].max()
    d_min = df.groupby("date")["price"].min()
    d_spr = (d_max - d_min)*1000
    d_mon = df.groupby("date")["month"].first()
    sprd  = [d_spr[d_mon == m].values for m in range(1, 13)]
    bp = ax4.boxplot(sprd, patch_artist=True,
                     medianprops=dict(color="black", lw=2.0),
                     flierprops=dict(marker=".", markersize=3, alpha=0.4))
    for patch, m in zip(bp["boxes"], range(1, 13)):
        patch.set_facecolor("#1565C0" if m in WINTER_M else "#E65100"); patch.set_alpha(0.72)
    ax4.set_xticks(range(1, 13)); ax4.set_xticklabels(MONTHS, fontsize=9)
    ax4.set_ylabel("Daily Price Spread (EUR/MWh)")
    ax4.set_title("(4) Daily Price Spread -- V2G Arbitrage Potential",
                  fontsize=10, fontweight="bold", loc="left")
    ax4.grid(True, alpha=0.22, axis="y")

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    if isinstance(out, str):
        print(f"  Saved -> {out}")


# =============================================================================
#  16. MAIN
# =============================================================================

def main():
    print("\n" + "="*65)
    print("  S.KOe COOL -- V2G Optimisation Suite (v6)")
    print("  TU Dortmund IE3 x Schmitz Cargobull AG | 2026")
    print("="*65)

    csv_path = next(
        (str(p) for p in [
            Path(__file__).parent / "2025_Electricity_Price.csv",
            Path("2025_Electricity_Price.csv"),
        ] if p.exists()), None)
    if not csv_path:
        print("\n  ERROR: 2025_Electricity_Price.csv not found.\n"); sys.exit(1)
    print(f"\n  Price CSV: {csv_path}")

    v2g = V2GParams()

    def ask(prompt, default, lo, hi):
        raw = input(f"  {prompt} [default {default}]: ").strip()
        try:
            val = float(raw); assert lo <= val <= hi; return val
        except Exception:
            print(f"  -> Using default {default}"); return float(default)

    def ask_str(prompt, choices, default):
        raw = input(f"  {prompt} {choices} [default {default}]: ").strip()
        return raw if raw in choices else default

    print("\n" + "-"*55)
    departure_h           = ask("Weekday DEPARTURE hour",     6,   0, 23)
    arrival_h             = ask("Weekday ARRIVAL hour",      16,   0, 23)
    soc_winter_pct        = ask("Winter arrival SoC %",      80,  20, 100)
    soc_summer_pct        = ask("Summer arrival SoC %",      40,  20, 100)
    v2g.soc_departure_pct = ask("Departure target SoC %",   100,  20, 100)
    tru_cycle = ask_str("TRU reefer cycle",
                         ["Continuous", "Start-Stop", "OFF"], "OFF")
    raw = input("  Run MPC (D)? adds ~2 min [Y/n]: ").strip().lower()
    run_mpc_avg = raw not in ("n", "no")
    raw = input("  Run full annual simulation? [Y/n]: ").strip().lower()
    run_annual  = raw not in ("n", "no")
    run_mpc_ann = False
    if run_annual:
        raw = input("  Include MPC in annual? SLOW [y/N]: ").strip().lower()
        run_mpc_ann = raw in ("y", "yes")
    print("-"*55)

    print(f"\n  Weekday {int(arrival_h):02d}:00 -> {int(departure_h):02d}:00 | "
          f"Winter SoC {soc_winter_pct:.0f}% | Summer SoC {soc_summer_pct:.0f}% | "
          f"Dep. {v2g.soc_departure_pct:.0f}% | TRU: {tru_cycle}")

    print("\n  Loading prices ...")
    _load_csv_raw(csv_path)

    WD_CONFIGS = [
        ("winter_weekday","Winter Weekday",WINTER_M,soc_winter_pct,"v2g_winter_weekday.png"),
        ("summer_weekday","Summer Weekday",SUMMER_M,soc_summer_pct,"v2g_summer_weekday.png"),
    ]
    WE_CONFIGS = [
        ("winter_weekend","Winter Weekend (Sat+Sun)",WINTER_M,soc_winter_pct,"v2g_winter_weekend.png"),
        ("summer_weekend","Summer Weekend (Sat+Sun)",SUMMER_M,soc_summer_pct,"v2g_summer_weekend.png"),
    ]

    all_season_res = {}

    for (dt_key, dt_lbl, months, soc_pct, out_f) in WD_CONFIGS:
        print(f"\n  -- {dt_lbl} --")
        buy   = load_avg_profile(csv_path, months, False)
        v2gp  = buy.copy()
        win, arr, dep, W = get_wd_window(v2g, arrival_h, departure_h)
        buy_w = buy[win]; v2gp_w = v2gp[win]
        tru_w = get_tru_15min_trace(tru_cycle, W, v2g.dt_h)
        E_init = v2g.usable_capacity_kWh * soc_pct / 100.0
        buy_d, plug_d, hours_d = build_wd_display(v2g, buy, arrival_h, departure_h)
        tru_d = np.zeros(v2g.n_slots); tru_d[arr:dep] = tru_w[:dep-arr]

        print("    A...", end=" ", flush=True)
        Pc,Pd,soc = run_A_dumb(v2g,buy_w,v2gp_w,W,E_init,tru_w)
        A = make_kpi("A - Dumb",v2g,Pc,Pd,soc,buy_w,v2gp_w,E_init,arr,dep,tru_w=tru_w)
        print("B...", end=" ", flush=True)
        Pc,Pd,soc = run_B_smart(v2g,buy_w,v2gp_w,E_init,tru_w)
        B = make_kpi("B - Smart (no V2G)",v2g,Pc,Pd,soc,buy_w,v2gp_w,E_init,arr,dep,tru_w=tru_w)
        print("C...", end=" ", flush=True)
        Pc,Pd,soc = run_C_milp(v2g,buy_w,v2gp_w,E_init,tru_w)
        C = make_kpi("C - MILP Day-Ahead",v2g,Pc,Pd,soc,buy_w,v2gp_w,E_init,arr,dep,tru_w=tru_w)
        if run_mpc_avg:
            print(f"D (MPC {W})...", flush=True)
            Pc,Pd,soc = run_D_mpc(v2g,buy_w,v2gp_w,E_init,tru_w)
            D = make_kpi("D - MPC (receding)",v2g,Pc,Pd,soc,buy_w,v2gp_w,E_init,arr,dep,tru_w=tru_w)
        else:
            print("D skipped."); D = {**C, "label":"D - MPC (not run)"}

        results = [A,B,C,D]; all_season_res[dt_key] = results
        plot_season_chart(v2g,dt_lbl,buy_d,plug_d,hours_d,results,False,
                          arrival_h,departure_h,is_48h=False,out=out_f,
                          tru_d=tru_d,tru_cycle=tru_cycle)
        rc = compute_reefer_costs(tru_w, buy_w, v2g.dt_h)
        print(f"    TRU: {rc['E_kWh']:.2f} kWh | "
              f"Dynamic EUR{rc['cost_dynamic']:.3f} | "
              f"Fixed EUR{rc['cost_fixed']:.3f} | "
              f"Diesel EUR{rc['cost_diesel']:.3f}")

    for (dt_key, dt_lbl, months, soc_pct, out_f) in WE_CONFIGS:
        print(f"\n  -- {dt_lbl} (48h) --")
        buy_24  = load_avg_profile(csv_path, months, True)
        buy_48  = np.concatenate([buy_24, buy_24])
        v2gp_48 = buy_48.copy(); W_48 = 192
        tru_48  = get_tru_15min_trace(tru_cycle, W_48, v2g.dt_h)
        E_init  = v2g.usable_capacity_kWh * soc_pct / 100.0

        print("    A...", end=" ", flush=True)
        Pc,Pd,soc = run_A_dumb(v2g,buy_48,v2gp_48,W_48,E_init,tru_48)
        A = make_kpi("A - Dumb",v2g,Pc,Pd,soc,buy_48,v2gp_48,E_init,is_weekend_48=True,tru_w=tru_48)
        print("B...", end=" ", flush=True)
        Pc,Pd,soc = run_B_smart(v2g,buy_48,v2gp_48,E_init,tru_48)
        B = make_kpi("B - Smart (no V2G)",v2g,Pc,Pd,soc,buy_48,v2gp_48,E_init,is_weekend_48=True,tru_w=tru_48)
        print("C...", end=" ", flush=True)
        Pc,Pd,soc = run_C_milp(v2g,buy_48,v2gp_48,E_init,tru_48)
        C = make_kpi("C - MILP Day-Ahead",v2g,Pc,Pd,soc,buy_48,v2gp_48,E_init,is_weekend_48=True,tru_w=tru_48)
        if run_mpc_avg:
            print(f"D (MPC {W_48})...", flush=True)
            Pc,Pd,soc = run_D_mpc(v2g,buy_48,v2gp_48,E_init,tru_48)
            D = make_kpi("D - MPC (receding)",v2g,Pc,Pd,soc,buy_48,v2gp_48,E_init,is_weekend_48=True,tru_w=tru_48)
        else:
            print("D skipped."); D = {**C, "label":"D - MPC (not run)"}

        results = [A,B,C,D]
        results_kpi = [{**r,"net_cost":r["net_cost"]/2,"charge_cost":r["charge_cost"]/2,
                        "v2g_rev":r["v2g_rev"]/2,"v2g_kwh":r["v2g_kwh"]/2,
                        "tru_cost":r["tru_cost"]/2,"total_cost":r["total_cost"]/2} for r in results]
        all_season_res[dt_key] = results_kpi
        hours_d = np.arange(W_48) * v2g.dt_h
        plot_season_chart(v2g,dt_lbl,buy_48,np.ones(W_48),hours_d,results,True,
                          arrival_h,departure_h,is_48h=True,out=out_f,
                          tru_d=tru_48,tru_cycle=tru_cycle)

    plot_kpi_multi(all_season_res,v2g,arrival_h,departure_h,run_mpc_avg,"v2g_KPI_multi.png")
    plot_price_profiles(csv_path,"v2g_price_profiles.png")

    yr_df = yearly_extrapolation(all_season_res)
    if not yr_df.empty:
        print("\n" + "="*65)
        print("  YEARLY EXTRAPOLATION (EUR/year)")
        print(yr_df.to_string())
        print("="*65)

    if run_annual:
        ann_df = run_full_year(v2g,csv_path,arrival_h,departure_h,
                               soc_winter_pct,soc_summer_pct,run_mpc_ann,tru_cycle)
        plot_annual_results(ann_df,run_mpc_ann,"v2g_annual_results.png")
        ref = ann_df["cost_A"].sum(); c_tot = ann_df["cost_C"].sum()
        print(f"\n  ANNUAL (C MILP): Dumb EUR{ref:,.0f} | MILP EUR{c_tot:,.0f} | "
              f"Savings EUR{ref-c_tot:+,.0f} | TRU EUR{ann_df['tru_cost'].sum():,.0f}")


if __name__ == "__main__":
    main()