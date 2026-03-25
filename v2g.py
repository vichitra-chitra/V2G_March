#!/usr/bin/env python3

from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from dataclasses import dataclass
from pathlib import Path
from datetime import timedelta

SC_COL   = {"A": "#999999", "B": "#2196F3", "C": "#00ACC1", "D": "#FF7700", "price": "#2E7D32"}
SC_FILL  = {"A": "#CCCCCC", "B": "#A5D6A7", "C": "#80DEEA", "D": "#FFCC80"}
WINTER_M = [1, 2, 3, 10, 11, 12]
SUMMER_M = [4, 5, 6, 7, 8, 9]

REEFER_DEFAULTS = {
    "P_HIGH_C":  7.6,   "T_HIGH_C":  1717,
    "P_LOW_C":   0.7,   "T_LOW_C":    292,
    "P_HIGH_SS": 9.7,   "T_HIGH_SS":  975,
    "P_MID_SS":  0.65,  "T_MID_SS":   295,
    "P_LOW_SS":  0.0,   "T_LOW_SS":  1207,
}

GERMAN_TARIFF = {
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
    dt_h                 : float = 1
    n_slots              : int   = 24

    @property
    def E_min(self):
        return self.usable_capacity_kWh * self.soc_min_pct / 100.0

    @property
    def E_max(self):
        return self.usable_capacity_kWh * self.soc_max_pct / 100.0

    @property
    def E_fin(self):
        return self.usable_capacity_kWh * self.soc_departure_pct / 100.0


# =============================================================================
#  2. TRU REEFER CYCLE
# =============================================================================

def _reefer_hi_res(cycle_type, N, dt_sec=10):
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


def get_tru_1h_trace(cycle_type, W, dt_h=1.0):
    if cycle_type.strip().lower() in ("off", "noreeferstationary"):
        return np.zeros(W)
    dt_sec_hi      = 10
    slots_per_1h   = int(dt_h * 3600 / dt_sec_hi)   # 360 samples per hour
    N_hi           = W * slots_per_1h
    P_hi           = _reefer_hi_res(cycle_type, N_hi, dt_sec_hi)
    return np.array([
        float(np.mean(P_hi[i * slots_per_1h:(i + 1) * slots_per_1h]))
        for i in range(W)
    ])


def tru_avg_kw(cycle_type):
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
#  3. TARIFF & REEFER COST
# =============================================================================

def compose_all_in_price(spot_eur_kwh, tariff=None):
    if tariff is None:
        tariff = GERMAN_TARIFF
    fixed_ct = sum(v for k, v in tariff.items() if k != "VAT_pc")
    net      = np.asarray(spot_eur_kwh, dtype=float) + fixed_ct / 100.0
    return net * (1.0 + tariff["VAT_pc"] / 100.0)


def compute_reefer_costs(tru_w, buy_w, dt_h,
                          fixed_price=FIXED_PRICE_EUR_KWH,
                          diesel_price=DIESEL_PRICE_EUR_L):
    E_kWh        = float(np.sum(tru_w) * dt_h)
    cost_dynamic = float(np.sum(tru_w * buy_w) * dt_h)
    cost_fixed   = E_kWh * fixed_price
    liters        = E_kWh / max(1e-12, DIESEL_KWH_PER_L * GENSET_EFF)
    cost_diesel  = liters * diesel_price
    return {
        "E_kWh":         E_kWh,
        "cost_dynamic":  cost_dynamic,
        "cost_fixed":    cost_fixed,
        "cost_diesel":   cost_diesel,
        "diesel_liters": liters,
    }

# =============================================================================
#  5. PRICE LOADING
# =============================================================================

_CSV_CACHE = {}


def _load_csv_raw(csv_path):
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
    df["slot"]       = df.index.hour
    df["is_weekend"] = df.index.dayofweek >= 5
    df["month"]      = df.index.month
    df["date"]       = df.index.date
    _CSV_CACHE[csv_path] = df
    print(f"  CSV: {len(df):,} rows  {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def _passthrough_profile(profile: np.ndarray) -> np.ndarray:
    return profile


def load_avg_profile(csv_path, months, is_weekend):
    df      = _load_csv_raw(csv_path)
    mask    = df["month"].isin(months) & (df["is_weekend"] == is_weekend)
    sub     = df[mask]
    if len(sub) == 0:
        raise ValueError(f"No data for months={months}, weekend={is_weekend}")
    profile = sub.groupby("slot")["price"].mean().values
    if len(profile) != 24:
        raise ValueError(f"Expected 24 slots, got {len(profile)}")
    return _passthrough_profile(profile)


# =============================================================================
#  6. WINDOW + DISPLAY HELPERS
# =============================================================================

def get_wd_window(v2g, arrival_h, departure_h):
    DS = 12.0
    n  = v2g.n_slots
    dt = v2g.dt_h
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
    n   = v2g.n_slots
    pct = 100.0 / v2g.usable_capacity_kWh
    W   = dep - arr
    Pc  = np.zeros(n)
    Pd  = np.zeros(n)
    soc = np.full(n, E_init * pct)
    Pc[arr:dep]  = Pc_w[:W]
    Pd[arr:dep]  = Pd_w[:W]
    soc[arr:dep] = soc_w[:W] * pct
    if dep < n:
        soc[dep:] = soc_w[W - 1] * pct
    return Pc, Pd, soc


def soc_ramp(hours, soc_pct, init_pct):
    n  = len(hours)
    dt = (hours[1] - hours[0]) if n > 1 else 0.25
    s0 = np.concatenate([[init_pct], soc_pct[:-1]])
    x  = np.empty(2 * n)
    y  = np.empty(2 * n)
    for i in range(n):
        x[2*i]   = hours[i]
        y[2*i]   = s0[i]
        x[2*i+1] = hours[i] + dt
        y[2*i+1] = soc_pct[i]
    return x, y


# =============================================================================
#  7. MILP SOLVER
# =============================================================================

def solve_milp(v2g, buy_w, v2gp_w, E_init, E_fin,
               allow_discharge=True, tru_w=None):
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix, csc_matrix

    W  = len(buy_w)
    dt = v2g.dt_h
    ic  = np.arange(W)
    id_ = np.arange(W,   2*W)
    ie  = np.arange(2*W, 3*W)
    izc = np.arange(3*W, 4*W)
    izd = np.arange(4*W, 5*W)
    nv  = 5 * W

    if tru_w is not None and len(tru_w) == W:
        p_c_eff = np.maximum(0.0, v2g.charge_power_kW - np.asarray(tru_w))
    else:
        p_c_eff = np.full(W, v2g.charge_power_kW)

    c = np.zeros(nv)
    c[ic]  = buy_w * dt
    if allow_discharge:
        c[id_] = -v2gp_w * dt

    lb = np.zeros(nv)
    ub = np.full(nv, np.inf)
    ub[ic]  = p_c_eff
    ub[id_] = v2g.discharge_power_kW if allow_discharge else 0.0
    lb[ie]  = v2g.E_min
    ub[ie]  = v2g.E_max
    lb[izc] = 0.0;  ub[izc] = 1.0
    lb[izd] = 0.0;  ub[izd] = 1.0
    integ = np.zeros(nv)
    integ[izc] = 1
    integ[izd] = 1

    nr = 4*W + 1
    A  = lil_matrix((nr, nv))
    lo = np.full(nr, -np.inf)
    hi = np.zeros(nr)

    for t in range(W):
        A[t, ie[t]]  =  1.0
        A[t, ic[t]]  = -v2g.eta_charge * dt
        A[t, id_[t]] =  1.0 / v2g.eta_discharge * dt
        rhs = E_init if t == 0 else 0.0
        if t > 0:
            A[t, ie[t-1]] = -1.0
        lo[t] = hi[t] = rhs

    for t in range(W):
        A[W+t,   ic[t]]  =  1.0;  A[W+t,   izc[t]] = -v2g.charge_power_kW
        A[2*W+t, id_[t]] =  1.0;  A[2*W+t, izd[t]] = -v2g.discharge_power_kW
        A[3*W+t, izc[t]] =  1.0;  A[3*W+t, izd[t]] =  1.0
        hi[W+t] = hi[2*W+t] = 0.0
        hi[3*W+t] = 1.0

    A[4*W, ie[W-1]] = 1.0
    lo[4*W] = E_fin
    hi[4*W] = v2g.E_max

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
    Pc  = np.zeros(W)
    Pd  = np.zeros(W)
    soc = np.zeros(W)
    s   = E_init
    for t in range(W):
        if s < E_fin:
            p = min(float(p_c_eff[t]), (E_fin - s) / (v2g.eta_charge * dt))
            Pc[t] = p
            s = min(E_fin, s + p * v2g.eta_charge * dt)
        soc[t] = s
    return Pc, Pd, soc


def run_B_smart(v2g, buy_w, v2gp_w, E_init, tru_w=None):
    return solve_milp(v2g, buy_w, v2gp_w, E_init, v2g.E_fin,
                      allow_discharge=False, tru_w=tru_w)


def run_C_milp(v2g, buy_w, v2gp_w, E_init, tru_w=None):
    return solve_milp(v2g, buy_w, v2gp_w, E_init, v2g.E_fin,
                      allow_discharge=True, tru_w=tru_w)


def run_D_mpc(v2g, buy_w, v2gp_w, E_init, tru_w=None,
              noise_std: float = 0, seed: int = 42,
              buy_tomorrow: np.ndarray = None):
    W   = len(buy_w)
    dt  = v2g.dt_h
    s   = E_init
    rng = np.random.default_rng(seed)

    Pc_all  = np.zeros(W)
    Pd_all  = np.zeros(W)
    soc_all = np.zeros(W)

    for t in range(W):
        # Today's remaining slots
        buy_fc  = buy_w[t:].copy()
        v2gp_fc = v2gp_w[t:].copy()

        # Append tomorrow if provided
        if buy_tomorrow is not None:
            buy_fc  = np.concatenate([buy_fc,  buy_tomorrow])
            v2gp_fc = np.concatenate([v2gp_fc, buy_tomorrow])

        # Add forecast noise to today's remaining slots only
        # (tomorrow DA is already known — no noise on it)
        if noise_std > 0:
            n_today = W - t
            noise   = rng.normal(0, noise_std, size=n_today)
            buy_fc[:n_today]  = np.maximum(0.001, buy_fc[:n_today]  + noise)
            v2gp_fc[:n_today] = np.maximum(buy_fc[:n_today], v2gp_fc[:n_today] + noise)

        tw_t = tru_w[t:] if tru_w is not None else None
        # tru for tomorrow — zeros if not available
        if buy_tomorrow is not None:
            tru_tomorrow = np.zeros(24) if tru_w is None else get_tru_1h_trace("Continuous", 24, v2g.dt_h)
            tw_t = np.concatenate([tw_t, tru_tomorrow]) if tw_t is not None else None

        Pcw, Pdw, _ = solve_milp(v2g, buy_fc, v2gp_fc,
                                  s, v2g.E_fin,
                                  allow_discharge=True, tru_w=tw_t)

        pc = float(np.clip(Pcw[0], 0, v2g.charge_power_kW))
        pd = float(np.clip(Pdw[0], 0, v2g.discharge_power_kW))

        if pc > 1e-6 and pd > 1e-6:
            pc, pd = (0.0, pd) if v2gp_w[t] > buy_w[t] else (pc, 0.0)

        s = float(np.clip(
            s + pc * v2g.eta_charge * dt - pd / v2g.eta_discharge * dt,
            v2g.E_min, v2g.E_max))

        Pc_all[t]  = pc
        Pd_all[t]  = pd
        soc_all[t] = s

    return Pc_all, Pd_all, soc_all

# =============================================================================
#  9. KPI BUILDER
# =============================================================================

def make_kpi(label, v2g, Pc_w, Pd_w, soc_w, buy_w, v2gp_w, E_init_kwh,
             arr_disp=None, dep_disp=None, is_weekend_48=False, tru_w=None):
    dt       = v2g.dt_h
    chg      = float(np.sum(Pc_w * buy_w)  * dt)
    rev      = float(np.sum(Pd_w * v2gp_w) * dt)
    tru_cost = float(np.sum(tru_w * buy_w) * dt) if tru_w is not None else 0.0

    if is_weekend_48:
        pct   = 100.0 / v2g.usable_capacity_kWh
        Pc_d  = Pc_w
        Pd_d  = Pd_w
        soc_d = soc_w * pct
    else:
        Pc_d, Pd_d, soc_d = to_display_wd(v2g, Pc_w, Pd_w, soc_w,
                                           arr_disp, dep_disp, E_init_kwh)
    return {
        "label"       : label,
        "Pc_d"        : Pc_d,
        "Pd_d"        : Pd_d,
        "soc_d"       : soc_d,
        "Pc_w"        : Pc_w,
        "Pd_w"        : Pd_w,
        "soc_w_kwh"   : soc_w,
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
    ax.set_xticks(pos)
    ax.set_xticklabels(lbls, fontsize=7.5, rotation=35, ha="right")
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
#  11. KPI MULTI-TABLE CHART
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
        for k in ("Dumb", "Smart", "MILP", "MPC"):
            if k in lbl:
                return {"Dumb": "A", "Smart": "B", "MILP": "C", "MPC": "D"}[k]
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
                     bbox=dict(facecolor=hdr_col, edgecolor="none",
                               boxstyle="round,pad=0.4"))
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
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 2.1)

        hdr_bgs = ["#263238"] + [SC_COL[k] for k in res_keys]
        for ci, hc in enumerate(hdr_bgs):
            cell = tbl[0, ci]
            cell.set_facecolor(hc)
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
#  12. PRICE PROFILES CHART
# =============================================================================

def plot_price_profiles(csv_path, out):
    df   = _load_csv_raw(csv_path)
    h_ax = np.arange(24)
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

    HT = np.arange(0, 25, 2)
    HL = [f"{int(h):02d}:00" for h in HT]

    def sax(ax, title):
        ax.set_xticks(HT)
        ax.set_xticklabels(HL, fontsize=8, rotation=35, ha="right")
        ax.set_xlim(0, 24)
        ax.set_ylabel("Price (EUR/MWh)")
        ax.set_xlabel("Hour of Day")
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
        ax.grid(True, alpha=0.22)

    ax1 = axes[0, 0]
    ax1.step(h_ax, w_wd*1000, where="post", color="#1565C0", lw=2.3, label="Winter weekday")
    ax1.step(h_ax, s_wd*1000, where="post", color="#E65100", lw=2.3, label="Summer weekday")
    ax1.fill_between(h_ax, w_wd*1000, step="post", color="#1565C0", alpha=0.10)
    ax1.fill_between(h_ax, s_wd*1000, step="post", color="#E65100", alpha=0.10)
    win_allin = compose_all_in_price(w_wd) * 1000
    sum_allin = compose_all_in_price(s_wd) * 1000
    ax1.step(h_ax, win_allin, where="post", color="#1565C0", lw=1.5, ls="--", alpha=0.6,
             label="Winter all-in")
    ax1.step(h_ax, sum_allin, where="post", color="#E65100", lw=1.5, ls="--", alpha=0.6,
             label="Summer all-in")
    spread = (w_wd - s_wd) * 1000
    pk = int(np.argmax(np.abs(spread)))
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
    m_avg = df.groupby("month")["price"].mean() * 1000
    m_std = df.groupby("month")["price"].std()  * 1000
    bar_c = ["#1565C0" if m in WINTER_M else "#E65100" for m in range(1, 13)]
    ax3.bar(range(12), [m_avg.get(m, 0) for m in range(1, 13)], color=bar_c,
            alpha=0.82, width=0.65, zorder=3,
            yerr=[m_std.get(m, 0) for m in range(1, 13)],
            error_kw=dict(lw=1.2, capsize=3, capthick=1.2, ecolor="#333"))
    ax3.set_xticks(range(12))
    ax3.set_xticklabels(MONTHS, fontsize=9)
    ax3.set_ylabel("Avg Price (EUR/MWh)")
    ax3.set_title("(3) Monthly Average Spot Price (+-1 sigma)",
                  fontsize=10, fontweight="bold", loc="left")
    ax3.legend(handles=[
        mpatches.Patch(color="#1565C0", alpha=0.82, label="Winter (Oct-Mar)"),
        mpatches.Patch(color="#E65100", alpha=0.82, label="Summer (Apr-Sep)"),
    ], fontsize=9)
    ax3.grid(True, alpha=0.22, axis="y", zorder=0)
    for i, v in enumerate([m_avg.get(m, 0) for m in range(1, 13)]):
        ax3.text(i, v+1.5, f"{v:.0f}", ha="center", va="bottom",
                 fontsize=7.5, fontweight="bold")

    ax4 = axes[1, 1]
    d_max = df.groupby("date")["price"].max()
    d_min = df.groupby("date")["price"].min()
    d_spr = (d_max - d_min) * 1000
    d_mon = df.groupby("date")["month"].first()
    sprd  = [d_spr[d_mon == m].values for m in range(1, 13)]
    bp = ax4.boxplot(sprd, patch_artist=True,
                     medianprops=dict(color="black", lw=2.0),
                     flierprops=dict(marker=".", markersize=3, alpha=0.4))
    for patch, m in zip(bp["boxes"], range(1, 13)):
        patch.set_facecolor("#1565C0" if m in WINTER_M else "#E65100")
        patch.set_alpha(0.72)
    ax4.set_xticks(range(1, 13))
    ax4.set_xticklabels(MONTHS, fontsize=9)
    ax4.set_ylabel("Daily Price Spread (EUR/MWh)")
    ax4.set_title("(4) Daily Price Spread -- V2G Arbitrage Potential",
                  fontsize=10, fontweight="bold", loc="left")
    ax4.grid(True, alpha=0.22, axis="y")

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    if isinstance(out, str):
        print(f"  Saved -> {out}")


# =============================================================================
#  13. MAIN  (command-line use only)
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
        print("\n  ERROR: 2025_Electricity_Price.csv not found.\n")
        sys.exit(1)

    v2g = V2GParams()

    def ask(prompt, default, lo, hi):
        raw = input(f"  {prompt} [default {default}]: ").strip()
        try:
            val = float(raw)
            assert lo <= val <= hi
            return val
        except Exception:
            return float(default)

    departure_h           = ask("Weekday DEPARTURE hour",    6,  0, 23)
    arrival_h             = ask("Weekday ARRIVAL hour",     16,  0, 23)
    soc_winter_pct        = ask("Winter arrival SoC %",     80, 20, 100)
    soc_summer_pct        = ask("Summer arrival SoC %",     40, 20, 100)
    v2g.soc_departure_pct = ask("Departure target SoC %",  100, 20, 100)

    print("\n  Loading prices ...")
    _load_csv_raw(csv_path)

    for season, months, soc_pct in [
        ("Winter Weekday", WINTER_M, soc_winter_pct),
        ("Summer Weekday", SUMMER_M, soc_summer_pct),
    ]:
        print(f"\n  -- {season} --")
        buy   = load_avg_profile(csv_path, months, False)
        v2gp  = buy.copy()
        win, arr, dep, W = get_wd_window(v2g, arrival_h, departure_h)
        buy_w = buy[win]; v2gp_w = v2gp[win]
        E_init = v2g.usable_capacity_kWh * soc_pct / 100.0

        Pc,Pd,soc = run_A_dumb(v2g, buy_w, v2gp_w, W, E_init)
        A = make_kpi("A - Dumb", v2g, Pc, Pd, soc, buy_w, v2gp_w, E_init, arr, dep)
        Pc,Pd,soc = run_C_milp(v2g, buy_w, v2gp_w, E_init)
        C = make_kpi("C - MILP", v2g, Pc, Pd, soc, buy_w, v2gp_w, E_init, arr, dep)

        print(f"    Dumb  net cost: EUR {A['net_cost']:.4f}/day")
        print(f"    MILP  net cost: EUR {C['net_cost']:.4f}/day")
        print(f"    V2G revenue  : EUR {C['v2g_rev']:.4f}/day")
        print(f"    Savings/year : EUR {(A['net_cost']-C['net_cost'])*365:+,.0f}")


if __name__ == "__main__":
    main()