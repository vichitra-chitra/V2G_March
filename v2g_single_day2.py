#!/usr/bin/env python3
"""
S.KOe COOL — Single-Day V2G Optimisation (v3 — all fixes)
TU Dortmund IE3 x Schmitz Cargobull AG | 2026

Fixes vs v2:
  1. Arrival/departure/midnight labels removed from graph panels — legend only.
  2. SoC uses plot() for gradual linear ramp (not step jumps).
  3. Power peaks annotated with kW values.
  4. New professional insights chart (price duration, economics, timeline, utilisation).
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from dataclasses import dataclass
from pathlib import Path

DISPLAY_START_H = 12.0


# ═══════════════════════════════════════════════════════════════════════════════
#  1. PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class V2GParams:
    battery_capacity_kWh : float = 70.0
    usable_capacity_kWh  : float = 60.0
    soc_min_pct          : float = 20.0
    soc_max_pct          : float = 100.0
    charge_power_kW      : float = 22.0
    discharge_power_kW   : float = 22.0
    eta_charge           : float = 0.92
    eta_discharge        : float = 0.92
    deg_cost_eur_kwh     : float = 0.0
    dt_h                 : float = 0.25
    n_slots              : int   = 96

    @property
    def E_min(self):
        return self.usable_capacity_kWh * self.soc_min_pct / 100.0

    @property
    def E_max(self):
        return self.usable_capacity_kWh * self.soc_max_pct / 100.0


# ═══════════════════════════════════════════════════════════════════════════════
#  2. PRICE LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_winter_weekday_prices(csv_path: str) -> np.ndarray:
    df = None
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(csv_path, sep=";", encoding=enc)
            if len(df.columns) > 1: break
            df = pd.read_csv(csv_path, sep=",", encoding=enc)
            if len(df.columns) > 1: break
        except Exception:
            continue
    if df is None or df.empty:
        raise ValueError(f"Could not read CSV: {csv_path}")

    price_col = next((c for c in df.columns if "Germany" in c and "MWh" in c), None)
    if price_col is None:
        raise ValueError(f"Germany price column not found. Cols: {list(df.columns)}")

    df = df[["Start date", price_col]].copy()
    df.columns = ["dt_str", "price_eur_mwh"]
    df["dt"]   = pd.to_datetime(df["dt_str"], format="%b %d, %Y %I:%M %p", errors="coerce")
    df = df.dropna(subset=["dt", "price_eur_mwh"])
    df["price"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce") / 1000.0
    df = df.dropna(subset=["price"]).set_index("dt").sort_index()

    mask = (df.index.month.isin([1,2,3,10,11,12])) & (df.index.dayofweek < 5)
    sub  = df[mask].copy()
    sub["slot"] = sub.index.hour * 4 + sub.index.minute // 15
    profile = sub.groupby("slot")["price"].mean().values
    if len(profile) != 96:
        raise ValueError(f"Expected 96 slots, got {len(profile)}")

    n_days = int(len(sub) / 96)
    print(f"  Loaded {len(df):,} rows. Winter WD avg of {n_days} days. "
          f"Range: {profile.min()*1000:.1f}-{profile.max()*1000:.1f} EUR/MWh\n")
    return profile


# ═══════════════════════════════════════════════════════════════════════════════
#  3. WINDOW HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_window(v2g, arrival_h, departure_h):
    n  = v2g.n_slots
    dt = v2g.dt_h
    a  = round(arrival_h   / dt) % n
    d  = round(departure_h / dt) % n
    window_slots = list(range(a, n)) + list(range(0, d))
    W = len(window_slots)

    def to_disp(h):
        if h >= DISPLAY_START_H:
            return round((h - DISPLAY_START_H) / dt)
        return round((h + 24.0 - DISPLAY_START_H) / dt)

    return window_slots, to_disp(arrival_h), to_disp(departure_h), W


def to_display(v2g, P_c_w, P_d_w, soc_w_kwh, arr_disp, dep_disp, E_init_kwh):
    """
    Expand window results (W slots) into 96-slot display arrays.
    SoC returned in %.
    """
    n   = v2g.n_slots
    pct = 100.0 / v2g.usable_capacity_kWh
    P_c = np.zeros(n)
    P_d = np.zeros(n)
    # Pre-arrival: flat at arrival SoC; post-departure: flat at final SoC
    soc = np.full(n, E_init_kwh * pct)

    P_c[arr_disp:dep_disp] = P_c_w
    P_d[arr_disp:dep_disp] = P_d_w
    soc[arr_disp:dep_disp] = soc_w_kwh * pct
    soc[dep_disp:]         = soc_w_kwh[-1] * pct
    return P_c, P_d, soc


def build_display_buy_plugged(v2g, buy, arrival_h, departure_h):
    ROLL      = round(DISPLAY_START_H / v2g.dt_h)
    buy_d     = np.roll(buy, -ROLL)
    h         = np.arange(v2g.n_slots) * v2g.dt_h
    plugged_d = np.roll(((h >= arrival_h) | (h < departure_h)).astype(float), -ROLL)
    return buy_d, plugged_d


def soc_ramp(hours_d, soc_d, E_init_pct):
    """
    Build a continuous (x, y) SoC curve for plotting.
    Each 15-min slot is a linear ramp from soc[t-1] to soc[t].
    This gives gradual slopes instead of step jumps.
    """
    n  = len(hours_d)
    dt = hours_d[1] - hours_d[0]
    soc_start = np.concatenate([[E_init_pct], soc_d[:-1]])

    x = np.empty(2 * n)
    y = np.empty(2 * n)
    for i in range(n):
        x[2*i]   = hours_d[i]
        x[2*i+1] = hours_d[i] + dt
        y[2*i]   = soc_start[i]
        y[2*i+1] = soc_d[i]
    return x, y


# ═══════════════════════════════════════════════════════════════════════════════
#  4. MILP SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def solve_milp(v2g, buy_w, v2gp_w, E_init, E_fin, allow_discharge=True):
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix, csc_matrix

    W  = len(buy_w)
    dt = v2g.dt_h

    idx_c  = np.arange(W)
    idx_d  = np.arange(W,   2*W)
    idx_e  = np.arange(2*W, 3*W)
    idx_zc = np.arange(3*W, 4*W)
    idx_zd = np.arange(4*W, 5*W)
    nv     = 5 * W

    c = np.zeros(nv)
    c[idx_c] = buy_w * dt
    if allow_discharge:
        c[idx_d] = -v2gp_w * dt

    lb = np.zeros(nv); ub = np.full(nv, np.inf)
    ub[idx_c]  = v2g.charge_power_kW
    ub[idx_d]  = v2g.discharge_power_kW if allow_discharge else 0.0
    lb[idx_e]  = v2g.E_min;  ub[idx_e]  = v2g.E_max
    lb[idx_zc] = 0.0;        ub[idx_zc] = 1.0
    lb[idx_zd] = 0.0;        ub[idx_zd] = 1.0

    integrality = np.zeros(nv)
    integrality[idx_zc] = 1;  integrality[idx_zd] = 1

    n_rows = 4*W + 1
    A  = lil_matrix((n_rows, nv))
    lo = np.full(n_rows, -np.inf)
    hi = np.zeros(n_rows)

    for t in range(W):
        A[t, idx_e[t]]  =  1.0
        A[t, idx_c[t]]  = -v2g.eta_charge * dt
        A[t, idx_d[t]]  =  (1.0 / v2g.eta_discharge) * dt
        rhs = E_init if t == 0 else 0.0
        if t > 0: A[t, idx_e[t-1]] = -1.0
        lo[t] = hi[t] = rhs

    for t in range(W):
        A[W   + t, idx_c[t]]  =  1.0
        A[W   + t, idx_zc[t]] = -v2g.charge_power_kW
        A[2*W + t, idx_d[t]]  =  1.0
        A[2*W + t, idx_zd[t]] = -v2g.discharge_power_kW
        A[3*W + t, idx_zc[t]] = 1.0
        A[3*W + t, idx_zd[t]] = 1.0
        hi[W+t] = hi[2*W+t] = 0.0;  hi[3*W+t] = 1.0

    A[4*W, idx_e[W-1]] = 1.0
    lo[4*W] = E_fin;  hi[4*W] = v2g.E_max

    res = milp(c,
               constraints=LinearConstraint(csc_matrix(A), lo, hi),
               integrality=integrality,
               bounds=Bounds(lb, ub),
               options={"disp": False, "time_limit": 60})
    if not res.success:
        raise RuntimeError(f"MILP failed — {res.status!r}: {res.message!r}")
    return (np.clip(res.x[idx_c], 0, None),
            np.clip(res.x[idx_d], 0, None),
            res.x[idx_e])


# ═══════════════════════════════════════════════════════════════════════════════
#  5. SCENARIO RUNNERS
# ═══════════════════════════════════════════════════════════════════════════════

def _kpi(label, v2g, P_c_w, P_d_w, soc_w, buy_w, v2gp_w,
         arr_disp, dep_disp, E_init_kwh):
    dt  = v2g.dt_h
    chg = float(np.sum(P_c_w * buy_w)  * dt)
    rev = float(np.sum(P_d_w * v2gp_w) * dt)
    P_c_d, P_d_d, soc_d = to_display(
        v2g, P_c_w, P_d_w, soc_w, arr_disp, dep_disp, E_init_kwh)
    return {
        "label"        : label,
        "P_c_d"        : P_c_d,
        "P_d_d"        : P_d_d,
        "soc_d"        : soc_d,          # %
        "P_c_w"        : P_c_w,          # window arrays for analysis
        "P_d_w"        : P_d_w,
        "soc_w_kwh"    : soc_w,
        "net_cost"     : chg - rev,
        "charge_cost"  : chg,
        "v2g_rev"      : rev,
        "v2g_kwh"      : float(np.sum(P_d_w) * dt),
        "charge_kwh"   : float(np.sum(P_c_w) * dt),
        "E_init_pct"   : E_init_kwh * 100.0 / v2g.usable_capacity_kWh,
    }


def run_A_dumb(v2g, buy_w, v2gp_w, W, E_init, arr_disp, dep_disp):
    dt = v2g.dt_h
    P_c = np.zeros(W); P_d = np.zeros(W); soc = np.zeros(W)
    s = E_init
    for t in range(W):
        if s < v2g.E_max:
            p = min(v2g.charge_power_kW, (v2g.E_max - s) / (v2g.eta_charge * dt))
            P_c[t] = p
            s = min(v2g.E_max, s + p * v2g.eta_charge * dt)
        soc[t] = s
    return _kpi("A - Dumb", v2g, P_c, P_d, soc, buy_w, v2gp_w,
                arr_disp, dep_disp, E_init)


def run_B_smart(v2g, buy_w, v2gp_w, E_init, arr_disp, dep_disp):
    P_c, P_d, soc = solve_milp(v2g, buy_w, v2gp_w, E_init, v2g.E_max, False)
    return _kpi("B - Smart (no V2G)", v2g, P_c, P_d, soc, buy_w, v2gp_w,
                arr_disp, dep_disp, E_init)


def run_C_milp(v2g, buy_w, v2gp_w, E_init, arr_disp, dep_disp):
    P_c, P_d, soc = solve_milp(v2g, buy_w, v2gp_w, E_init, v2g.E_max, True)
    return _kpi("C - MILP Day-Ahead", v2g, P_c, P_d, soc, buy_w, v2gp_w,
                arr_disp, dep_disp, E_init)


def run_D_mpc(v2g, buy_w, v2gp_w, E_init, arr_disp, dep_disp):
    W  = len(buy_w)
    dt = v2g.dt_h
    s  = E_init
    P_c_all = np.zeros(W); P_d_all = np.zeros(W); soc_all = np.zeros(W)
    for t in range(W):
        Pc_w, Pd_w, _ = solve_milp(v2g, buy_w[t:], v2gp_w[t:],
                                    s, v2g.E_max, True)
        pc = float(np.clip(Pc_w[0], 0, v2g.charge_power_kW))
        pd = float(np.clip(Pd_w[0], 0, v2g.discharge_power_kW))
        if pc > 1e-6 and pd > 1e-6:
            pc, pd = (0.0, pd) if v2gp_w[t] > buy_w[t] else (pc, 0.0)
        s = float(np.clip(
            s + pc * v2g.eta_charge * dt - pd / v2g.eta_discharge * dt,
            v2g.E_min, v2g.E_max))
        P_c_all[t] = pc; P_d_all[t] = pd; soc_all[t] = s
    return _kpi("D - MPC (receding)", v2g, P_c_all, P_d_all, soc_all,
                buy_w, v2gp_w, arr_disp, dep_disp, E_init)


# ═══════════════════════════════════════════════════════════════════════════════
#  6. CONSOLE REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(results):
    ref = results[0]["net_cost"]
    print("\n" + "="*76)
    print("  SINGLE-DAY KPI SUMMARY  (deg=0, TRU=0, Winter WD avg)")
    print("="*76)
    print(f"  {'Scenario':<28} {'Net EUR':>9} {'Chg EUR':>9} "
          f"{'V2G EUR':>9} {'kWh out':>8} {'vs Dumb':>9}")
    print("-"*76)
    for r in results:
        print(f"  {r['label']:<28} {r['net_cost']:>9.4f} {r['charge_cost']:>9.4f} "
              f"{r['v2g_rev']:>9.4f} {r['v2g_kwh']:>8.2f} "
              f"{ref - r['net_cost']:>+9.4f}")
    print("="*76)
    cr = results[2]
    print(f"\n  Annualised x365 (Scenario C MILP):")
    print(f"    Savings vs Dumb : EUR {(ref - cr['net_cost'])*365:+,.0f}/year")
    print(f"    V2G revenue     : EUR {cr['v2g_rev']*365:,.0f}/year\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  7. PLOTTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

SC_COL = {
    "A": "#999999", "B": "#2196F3", "C": "#00ACC1", "D": "#FF7700",
    "price": "#2E7D32",
}

def _ticks():
    pos  = np.arange(DISPLAY_START_H, DISPLAY_START_H + 25, 2)
    lbls = [f"{int(h % 24):02d}:00" for h in pos]
    return pos, lbls

def _disp_x(h):
    return h if h >= DISPLAY_START_H else h + 24.0

def _midnight_line(ax):
    ax.axvline(_disp_x(0), color="#555555", lw=1.2, ls="--", alpha=0.6, zorder=5)

def _arrival_departure_lines(ax, arrival_h, departure_h):
    ax.axvline(_disp_x(arrival_h),   color="#1B5E20", lw=1.1, ls=":", alpha=0.75, zorder=5)
    ax.axvline(_disp_x(departure_h), color="#B71C1C", lw=1.1, ls=":", alpha=0.75, zorder=5)

def _format_ax(ax, ylabel, title, ylim=None):
    tp, tl = _ticks()
    ax.set_xlim(DISPLAY_START_H, DISPLAY_START_H + 24.0)
    ax.set_xticks(tp)
    ax.set_xticklabels(tl, fontsize=7.5, rotation=35, ha="right")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left", pad=4)
    ax.grid(True, alpha=0.22, zorder=0)
    if ylim: ax.set_ylim(*ylim)

def _legend_below(ax, handles, ncol=4):
    ax.legend(handles=handles, fontsize=8, ncol=ncol,
              loc="upper center", bbox_to_anchor=(0.5, -0.38),
              framealpha=0.95, edgecolor="#CCCCCC")

def _annotate_power_peaks(ax, hours_d, power_pos, power_neg, col, threshold=2.0):
    """Label max charge and max discharge values on the power panel."""
    # Charging peaks
    if power_pos.max() > threshold:
        idx  = np.argmax(power_pos)
        val  = power_pos[idx]
        ax.annotate(f"{val:.1f} kW",
                    xy=(hours_d[idx], val),
                    xytext=(hours_d[idx] + 0.3, val + 1.5),
                    fontsize=7.5, color=col, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
    # Discharge peaks (shown negative)
    if power_neg.max() > threshold:
        idx  = np.argmax(power_neg)
        val  = power_neg[idx]
        ax.annotate(f"-{val:.1f} kW",
                    xy=(hours_d[idx], -val),
                    xytext=(hours_d[idx] + 0.3, -val - 2.5),
                    fontsize=7.5, color=col, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.8))


# ═══════════════════════════════════════════════════════════════════════════════
#  8. COMPARISON CHARTS  (B vs A, C vs A, D vs A)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison(v2g, hours_d, buy_d, plugged_d,
                    sc, dumb, sc_key,
                    arrival_h, departure_h, out):

    col      = SC_COL[sc_key]
    sc_short = sc["label"].split("(")[0].strip()
    E_init_pct = sc["E_init_pct"]

    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(15, 14),
        gridspec_kw={"height_ratios": [1.0, 1.8, 1.8], "hspace": 0.72})
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        f"S.KOe COOL  —  {sc_short}  vs  A - Dumb  |  Winter Weekday Avg\n"
        f"Arrival: {int(arrival_h):02d}:00  |  Departure: {int(departure_h):02d}:00  |  "
        f"Battery: {v2g.usable_capacity_kWh:.0f} kWh usable  |  deg=0  |  TRU=0 kW",
        fontsize=11, fontweight="bold", y=0.99)

    # ── Panel 1: Price + plugged shading ──────────────────────────────────────
    for t in range(len(hours_d)):
        if plugged_d[t] > 0.5:
            ax0.axvspan(hours_d[t], hours_d[t] + v2g.dt_h,
                        color="gold", alpha=0.22, lw=0, zorder=1)
    p_line, = ax0.step(hours_d, buy_d * 1000, where="post",
                       color=SC_COL["price"], lw=2.0, zorder=3)
    ax0.fill_between(hours_d, buy_d * 1000, step="post",
                     color=SC_COL["price"], alpha=0.10, zorder=2)
    # Lines only — NO text labels inside graph
    _midnight_line(ax0)
    _arrival_departure_lines(ax0, arrival_h, departure_h)
    _legend_below(ax0, [
        p_line,
        mpatches.Patch(color="gold",    alpha=0.5,  label="Plugged-in window"),
        Line2D([0],[0], color="#1B5E20", ls=":", lw=1.3,
               label=f"Arrival  {int(arrival_h):02d}:00"),
        Line2D([0],[0], color="#B71C1C", ls=":", lw=1.3,
               label=f"Departure {int(departure_h):02d}:00"),
        Line2D([0],[0], color="#555555", ls="--", lw=1.3,
               label="Midnight"),
    ], ncol=5)
    _format_ax(ax0, "EUR / MWh", "(1) Electricity Price + Plugged-In Window")

    # ── Panel 2: Power ────────────────────────────────────────────────────────
    dm_c, = ax1.step(hours_d, dumb["P_c_d"], where="post",
                     color=SC_COL["A"], lw=1.8, alpha=0.70,
                     label="A - Dumb  charge")
    sc_c, = ax1.step(hours_d, sc["P_c_d"], where="post",
                     color=col, lw=2.4, label=f"{sc_short}  charge")
    handles_p2 = [dm_c, sc_c]

    # Annotate dumb peak
    _annotate_power_peaks(ax1, hours_d,
                          dumb["P_c_d"], np.zeros(len(hours_d)),
                          SC_COL["A"])
    # Annotate scenario peaks
    if sc["v2g_kwh"] > 0.05:
        sc_d, = ax1.step(hours_d, -sc["P_d_d"], where="post",
                         color=col, lw=2.4, ls="--", alpha=0.85,
                         label=f"{sc_short}  V2G discharge (negative)")
        handles_p2.append(sc_d)
        _annotate_power_peaks(ax1, hours_d,
                              sc["P_c_d"], sc["P_d_d"], col)
    else:
        _annotate_power_peaks(ax1, hours_d,
                              sc["P_c_d"], np.zeros(len(hours_d)), col)

    ax1.axhline(0, color="black", lw=0.6)
    _midnight_line(ax1)
    _arrival_departure_lines(ax1, arrival_h, departure_h)
    _legend_below(ax1, handles_p2, ncol=3)
    _format_ax(ax1, "Power (kW)", "(2) Charge / Discharge Power  "
               "[solid = charging above 0 | dashed = V2G discharge below 0]")

    # ── Panel 3: SoC — gradual ramp ───────────────────────────────────────────
    for t in range(len(hours_d)):
        if plugged_d[t] > 0.5:
            ax2.axvspan(hours_d[t], hours_d[t] + v2g.dt_h,
                        color="gold", alpha=0.12, lw=0, zorder=1)

    # Build gradual ramp curves
    xd, yd = soc_ramp(hours_d, dumb["soc_d"], dumb["E_init_pct"])
    xs, ys = soc_ramp(hours_d, sc["soc_d"],   E_init_pct)

    dm_s, = ax2.plot(xd, yd, color=SC_COL["A"], lw=2.0,
                     label="A - Dumb  SoC (%)")
    sc_s, = ax2.plot(xs, ys, color=col, lw=2.6,
                     label=f"{sc_short}  SoC (%)")

    ax2.axhline(v2g.soc_min_pct, color="#C62828", ls=":", lw=1.4, zorder=3,
                label=f"SoC min = {v2g.soc_min_pct:.0f}%  (cold-chain floor)")
    ax2.axhline(v2g.soc_max_pct, color="#0D47A1", ls=":", lw=1.4, zorder=3,
                label=f"SoC max = {v2g.soc_max_pct:.0f}%  (departure target)")

    _midnight_line(ax2)
    _arrival_departure_lines(ax2, arrival_h, departure_h)
    _legend_below(ax2, [
        dm_s, sc_s,
        Line2D([0],[0], color="#C62828", ls=":", lw=1.4,
               label=f"Min {v2g.soc_min_pct:.0f}% cold-chain floor"),
        Line2D([0],[0], color="#0D47A1", ls=":", lw=1.4,
               label=f"Max {v2g.soc_max_pct:.0f}% departure target"),
    ], ncol=4)
    _format_ax(ax2, "State of Charge (%)", "(3) Battery SoC Trajectory",
               ylim=(0, 112))
    ax2.set_xlabel("Time of Day", fontsize=9)

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved -> {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  9. KPI TABLE CHART
# ═══════════════════════════════════════════════════════════════════════════════

def plot_kpi_table(v2g, results, arrival_h, departure_h,
                   out="v2g_KPI_comparison.png"):
    ref   = results[0]["net_cost"]
    names = ["A — Dumb", "B — Smart", "C — MILP", "D — MPC"]
    keys  = ["A", "B", "C", "D"]

    rows = [
        ("Net cost (EUR/day)",
         [f"{r['net_cost']:.4f}" for r in results]),
        ("Charge cost (EUR/day)",
         [f"{r['charge_cost']:.4f}" for r in results]),
        ("V2G revenue (EUR/day)",
         [f"{r['v2g_rev']:.4f}" for r in results]),
        ("V2G export (kWh/day)",
         [f"{r['v2g_kwh']:.2f}" for r in results]),
        ("Grid energy purchased (kWh/day)",
         [f"{r['charge_kwh']:.2f}" for r in results]),
        ("Daily savings vs Dumb",
         ["-"] + [f"EUR {ref - r['net_cost']:+.4f}" for r in results[1:]]),
        ("Annual savings (x365)",
         ["-"] + [f"EUR {(ref - r['net_cost'])*365:+,.0f}" for r in results[1:]]),
        ("Annual V2G revenue",
         [f"EUR {r['v2g_rev']*365:,.0f}" for r in results]),
    ]

    cell_text  = [[row[0]] + row[1] for row in rows]
    col_labels = ["Metric"] + names

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        "S.KOe COOL — KPI Comparison: All 4 Scenarios  |  Winter Weekday Avg\n"
        f"Arrival {int(arrival_h):02d}:00  |  Departure {int(departure_h):02d}:00  |  "
        f"Battery {v2g.usable_capacity_kWh:.0f} kWh usable  |  deg=0  |  TRU=0 kW",
        fontsize=11, fontweight="bold", y=1.03)

    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 2.5)

    hdr_bgs = ["#263238"] + [SC_COL[k] for k in keys]
    for col_i, hc in enumerate(hdr_bgs):
        cell = tbl[0, col_i]
        cell.set_facecolor(hc)
        cell.set_text_props(color="white", fontweight="bold")

    col_bgs = ["#ECEFF1", "#E8EAF6", "#E3F2FD", "#E0F7FA", "#FFF3E0"]
    for row_i in range(1, len(rows)+1):
        for col_i in range(5):
            cell = tbl[row_i, col_i]
            cell.set_facecolor(col_bgs[col_i])
            if col_i == 0:
                cell.set_text_props(fontweight="bold")
            txt = cell.get_text().get_text()
            if "+" in txt and "EUR" in txt:
                cell.set_text_props(color="#1B5E20", fontweight="bold")

    for row_i in range(len(rows)+1):
        tbl[row_i, 0].set_width(0.36)
        for col_i in range(1, 5):
            tbl[row_i, col_i].set_width(0.16)

    ax.text(0.5, -0.04,
            "Agora Verkehrswende 2025: ~EUR 500/year arbitrage (private car)  |  "
            "Reefer trailer potential higher — larger battery, predictable depot dwell",
            ha="center", va="top", fontsize=7.5, style="italic",
            color="#555555", transform=ax.transAxes)

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved -> {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  10. INSIGHTS CHART  (new professional summary)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_insights(v2g, results, buy_w, hours_d, arrival_h, departure_h,
                  arr_disp, dep_disp, W,
                  out="v2g_insights.png"):
    """
    4-panel professional insights chart:
      Panel 1 (top-left)  : Price duration curve — when is V2G profitable?
      Panel 2 (top-right) : Economics breakdown — cost/revenue bar chart (annual)
      Panel 3 (bottom-left): Charging timeline — colour-coded activity per scenario
      Panel 4 (bottom-right): Battery SoC safety margin above cold-chain floor
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.32})
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        "S.KOe COOL — V2G Strategy Insights  |  Winter Weekday Avg\n"
        f"Arrival {int(arrival_h):02d}:00  |  Departure {int(departure_h):02d}:00  |  "
        f"Battery {v2g.usable_capacity_kWh:.0f} kWh usable  |  deg=0  |  TRU=0 kW",
        fontsize=13, fontweight="bold", y=1.00)

    A, B, C, D = results
    ref = A["net_cost"]

    # ── Panel 1: Price Duration Curve ─────────────────────────────────────────
    ax1 = axes[0, 0]
    sorted_prices = np.sort(buy_w)[::-1] * 1000   # EUR/MWh, high to low
    pct = np.linspace(0, 100, len(sorted_prices))

    ax1.fill_between(pct, sorted_prices, color="#2E7D32", alpha=0.15)
    ax1.plot(pct, sorted_prices, color="#2E7D32", lw=2.2,
             label="Day-ahead price (sorted high→low)")

    # deg=0 means any positive price is profitable for V2G
    # Draw the effective threshold (= 0 EUR/MWh since deg=0)
    ax1.axhline(0, color="#B71C1C", lw=1.5, ls="--",
                label="V2G threshold: 0 EUR/MWh\n(deg=0, sell price = buy price)")

    # Shade V2G profitable zone
    v2g_zone = sorted_prices > 0
    if v2g_zone.any():
        cutoff_pct = float(pct[v2g_zone][-1])
        ax1.axvspan(0, cutoff_pct, color="#FF7700", alpha=0.12,
                    label=f"V2G profitable: {cutoff_pct:.0f}% of plugged-in slots")
        ax1.text(cutoff_pct / 2, sorted_prices.max() * 0.55,
                 f"V2G zone\n{cutoff_pct:.0f}% of slots",
                 ha="center", va="center", fontsize=8.5,
                 color="#E65100", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor="white", edgecolor="#FF7700", alpha=0.85))

    # Mark peak and valley
    ax1.annotate(f"Peak: {sorted_prices[0]:.0f} EUR/MWh",
                 xy=(0, sorted_prices[0]),
                 xytext=(12, sorted_prices[0] - 15),
                 fontsize=8, color="#1B5E20",
                 arrowprops=dict(arrowstyle="->", color="#1B5E20", lw=0.9))
    ax1.annotate(f"Valley: {sorted_prices[-1]:.0f} EUR/MWh",
                 xy=(100, sorted_prices[-1]),
                 xytext=(72, sorted_prices[-1] + 15),
                 fontsize=8, color="#1565C0",
                 arrowprops=dict(arrowstyle="->", color="#1565C0", lw=0.9))

    ax1.set_xlabel("% of plugged-in slots (high→low price)", fontsize=9)
    ax1.set_ylabel("Price (EUR/MWh)", fontsize=9)
    ax1.set_title("(1) Price Duration Curve — V2G Opportunity Window",
                  fontsize=10, fontweight="bold", loc="left")
    ax1.legend(fontsize=7.5, loc="upper right")
    ax1.set_xlim(0, 100)
    ax1.grid(True, alpha=0.22)

    # ── Panel 2: Annual Economics Breakdown ───────────────────────────────────
    ax2 = axes[0, 1]
    sc_names   = ["A\nDumb", "B\nSmart", "C\nMILP", "D\nMPC"]
    sc_keys    = ["A",       "B",        "C",        "D"]
    ann_charge = [r["charge_cost"] * 365 for r in results]
    ann_v2g    = [r["v2g_rev"]     * 365 for r in results]
    ann_net    = [r["net_cost"]    * 365 for r in results]

    x = np.arange(4)
    w = 0.25

    bars_chg = ax2.bar(x - w, ann_charge, width=w, color="#EF5350",
                       alpha=0.85, label="Annual charge cost (EUR)")
    bars_rev = ax2.bar(x,     ann_v2g,   width=w, color="#43A047",
                       alpha=0.85, label="Annual V2G revenue (EUR)")
    bars_net = ax2.bar(x + w, ann_net,   width=w, color="#1565C0",
                       alpha=0.85, label="Annual net cost (EUR)")

    # Value labels on bars
    for bars in [bars_chg, bars_rev, bars_net]:
        for bar in bars:
            h = bar.get_height()
            va  = "bottom" if h >= 0 else "top"
            yp  = h + 10 if h >= 0 else h - 10
            ax2.text(bar.get_x() + bar.get_width() / 2, yp,
                     f"€{h:,.0f}", ha="center", va=va,
                     fontsize=7.0, fontweight="bold",
                     color=bar.get_facecolor())

    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sc_names, fontsize=9)
    ax2.set_ylabel("EUR / year  (x365)", fontsize=9)
    ax2.set_title("(2) Annual Economics Breakdown — All 4 Scenarios",
                  fontsize=10, fontweight="bold", loc="left")
    ax2.legend(fontsize=7.5, loc="upper left")
    ax2.grid(True, alpha=0.22, axis="y")

    # Annotate savings arrows B→A and C→A
    for i, r in enumerate(results[1:], 1):
        sav = (ref - r["net_cost"]) * 365
        if abs(sav) > 5:
            col_arr = "#1B5E20" if sav > 0 else "#B71C1C"
            ax2.annotate(f"€{sav:+,.0f}/yr",
                         xy=(i + w, r["net_cost"] * 365),
                         xytext=(i + w + 0.05, r["net_cost"] * 365 - 80),
                         fontsize=7, color=col_arr, fontweight="bold",
                         arrowprops=dict(arrowstyle="->",
                                         color=col_arr, lw=0.8))

    # ── Panel 3: Charging Activity Timeline ───────────────────────────────────
    ax3 = axes[1, 0]
    sc_labels = ["A - Dumb", "B - Smart", "C - MILP", "D - MPC"]
    sc_cols   = [SC_COL[k] for k in ["A","B","C","D"]]
    all_res   = [A, B, C, D]

    tp, tl = _ticks()
    hours_w = hours_d[arr_disp:dep_disp]   # hours for the plugged window

    for row_i, (r, lbl, col) in enumerate(zip(all_res, sc_labels, sc_cols)):
        Pc_w = r["P_c_w"]
        Pd_w = r["P_d_w"]
        y_base = row_i

        for t_idx, (pc, pd) in enumerate(zip(Pc_w, Pd_w)):
            if t_idx >= len(hours_w):
                break
            x0 = hours_w[t_idx]
            x1 = x0 + v2g.dt_h

            if pd > 0.5:
                c = "#E53935"    # V2G discharge = red
            elif pc > 0.5:
                c = col          # charging = scenario colour
            else:
                c = "#E0E0E0"    # idle / unplugged = light grey

            ax3.fill_between([x0, x1], [y_base + 0.1, y_base + 0.1],
                             [y_base + 0.85, y_base + 0.85],
                             color=c, alpha=0.85, lw=0)

        ax3.text(DISPLAY_START_H + 0.2, y_base + 0.47, lbl,
                 fontsize=8, va="center", fontweight="bold",
                 color="#333333")

    ax3.set_xlim(DISPLAY_START_H, DISPLAY_START_H + 24)
    ax3.set_ylim(0, 4)
    ax3.set_yticks([])
    ax3.set_xticks(tp)
    ax3.set_xticklabels(tl, fontsize=7.5, rotation=35, ha="right")
    ax3.axvline(_disp_x(0),           color="#555555", lw=1.2, ls="--", alpha=0.6)
    ax3.axvline(_disp_x(arrival_h),   color="#1B5E20", lw=1.1, ls=":", alpha=0.7)
    ax3.axvline(_disp_x(departure_h), color="#B71C1C", lw=1.1, ls=":", alpha=0.7)
    ax3.set_xlabel("Time of Day", fontsize=9)
    ax3.set_title("(3) Charging Activity Timeline — When Each Strategy Acts",
                  fontsize=10, fontweight="bold", loc="left")
    ax3.grid(True, axis="x", alpha=0.2)

    # Legend for timeline
    legend_handles = [
        mpatches.Patch(color=SC_COL["A"], label="A - Dumb (charging)"),
        mpatches.Patch(color=SC_COL["B"], label="B - Smart (charging)"),
        mpatches.Patch(color=SC_COL["C"], label="C - MILP (charging)"),
        mpatches.Patch(color=SC_COL["D"], label="D - MPC (charging)"),
        mpatches.Patch(color="#E53935",   label="V2G discharge (any)"),
        mpatches.Patch(color="#E0E0E0",   label="Idle / no action"),
    ]
    ax3.legend(handles=legend_handles, fontsize=7.5, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, -0.28),
               framealpha=0.95, edgecolor="#CCCCCC")

    # ── Panel 4: SoC Safety Margin above Cold-Chain Floor ─────────────────────
    ax4 = axes[1, 1]
    hours_w_full = hours_d[arr_disp:dep_disp]

    for r, lbl, col, ls in zip(all_res, sc_labels, sc_cols,
                                ["-", "-", "-", "--"]):
        soc_w_pct = r["soc_w_kwh"] * 100.0 / v2g.usable_capacity_kWh
        margin = soc_w_pct - v2g.soc_min_pct   # headroom above 20%

        if len(hours_w_full) > len(margin):
            h = hours_w_full[:len(margin)]
        else:
            h = hours_w_full

        ax4.plot(h, margin[:len(h)], color=col, lw=2.0, ls=ls, label=lbl)

    ax4.axhline(0, color="#B71C1C", lw=1.8, ls="--", alpha=0.8,
                label="Cold-chain floor (0% margin = risk!)")
    ax4.fill_between(hours_w_full[:1] if len(hours_w_full) > 0 else [0, 1],
                     [0, 0], [0, 0], color="#FFCDD2", alpha=0.4)

    # Shade the danger zone (margin < 5%)
    ax4.axhspan(0, 5, color="#FFCDD2", alpha=0.35, zorder=0,
                label="Danger zone: <5% margin above floor")
    ax4.axhspan(5, 15, color="#FFF9C4", alpha=0.35, zorder=0,
                label="Caution zone: 5–15% margin")

    ax4.set_xlim(_disp_x(arrival_h), _disp_x(departure_h))
    ax4.set_ylim(-5, 85)
    ax4.set_xlabel("Time of Day (plugged-in window)", fontsize=9)
    ax4.set_ylabel("SoC margin above 20% floor  (pp)", fontsize=9)
    ax4.set_title("(4) Cold-Chain Safety Margin During Plugged-In Window",
                  fontsize=10, fontweight="bold", loc="left")

    tp4  = np.arange(_disp_x(arrival_h), _disp_x(departure_h) + 0.1, 2)
    tl4  = [f"{int(h % 24):02d}:00" for h in tp4]
    ax4.set_xticks(tp4)
    ax4.set_xticklabels(tl4, fontsize=7.5, rotation=35, ha="right")
    ax4.grid(True, alpha=0.22)
    ax4.legend(fontsize=7.5, loc="lower right")

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved -> {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  11. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*62)
    print("  S.KOe COOL — Single-Day V2G Optimisation (v3)")
    print("  TU Dortmund IE3 x Schmitz Cargobull AG | 2026")
    print("="*62)

    csv_path = next(
        (str(p) for p in [
            Path(__file__).parent / "2025_Electricity_Price.csv",
            Path("2025_Electricity_Price.csv"),
        ] if p.exists()), None)
    if csv_path is None:
        print("\n  ERROR: 2025_Electricity_Price.csv not found.\n"); sys.exit(1)
    print(f"\n  Price CSV: {csv_path}")

    v2g = V2GParams()
    print(f"\n  Battery  : {v2g.battery_capacity_kWh} kWh total | "
          f"{v2g.usable_capacity_kWh} kWh usable\n"
          f"  SoC range: {v2g.soc_min_pct:.0f}%-{v2g.soc_max_pct:.0f}% | "
          f"P_max {v2g.charge_power_kW:.0f} kW | deg=0")

    def ask(prompt, default, lo, hi):
        raw = input(f"  {prompt} [default {default}]: ").strip()
        try:
            v = float(raw); assert lo <= v <= hi; return v
        except Exception:
            print(f"  -> Using default {default}"); return default

    print("\n" + "-"*50)
    departure_h  = ask("Trailer DEPARTURE hour 0-23", 6,  0, 23)
    arrival_h    = ask("Trailer ARRIVAL   hour 0-23", 16, 0, 23)
    soc_init_pct = ask("Arrival SoC % (20-100)",      45, 20, 100)
    print(f"\n  Window: {int(arrival_h):02d}:00 -> {int(departure_h):02d}:00 | "
          f"Arrival SoC: {soc_init_pct:.0f}% | Target: 100%")
    print("-"*50)

    print("\n  Loading prices ...")
    buy  = load_winter_weekday_prices(csv_path)
    v2gp = buy.copy()

    window_slots, arr_disp, dep_disp, W = get_window(v2g, arrival_h, departure_h)
    buy_w  = buy[window_slots]
    v2gp_w = v2gp[window_slots]
    E_init = v2g.usable_capacity_kWh * soc_init_pct / 100.0

    print(f"  Plugged-in window: {W} slots = {W * v2g.dt_h:.1f} h")

    buy_d, plugged_d = build_display_buy_plugged(v2g, buy, arrival_h, departure_h)
    hours_d = np.arange(v2g.n_slots) * v2g.dt_h + DISPLAY_START_H

    print("\n  Running Scenario A (Dumb)  ...")
    A = run_A_dumb(v2g, buy_w, v2gp_w, W, E_init, arr_disp, dep_disp)

    print("  Running Scenario B (Smart) ...")
    B = run_B_smart(v2g, buy_w, v2gp_w, E_init, arr_disp, dep_disp)

    print("  Running Scenario C (MILP)  ...")
    C = run_C_milp(v2g, buy_w, v2gp_w, E_init, arr_disp, dep_disp)

    print(f"  Running Scenario D (MPC)   — {W} sub-problems ...")
    D = run_D_mpc(v2g, buy_w, v2gp_w, E_init, arr_disp, dep_disp)

    results = [A, B, C, D]
    print_report(results)

    comps = [
        (B, "B", "v2g_B_smart_vs_dumb.png"),
        (C, "C", "v2g_C_milp_vs_dumb.png"),
        (D, "D", "v2g_D_mpc_vs_dumb.png"),
    ]
    for sc, sc_key, fname in comps:
        print(f"  Plotting {fname} ...")
        plot_comparison(v2g, hours_d, buy_d, plugged_d,
                        sc, A, sc_key, arrival_h, departure_h, fname)

    print("  Plotting v2g_KPI_comparison.png ...")
    plot_kpi_table(v2g, results, arrival_h, departure_h)

    print("  Plotting v2g_insights.png ...")
    plot_insights(v2g, results, buy_w, hours_d, arrival_h, departure_h,
                  arr_disp, dep_disp, W)

    print("\n  Done. Outputs:")
    for f in ["v2g_B_smart_vs_dumb.png", "v2g_C_milp_vs_dumb.png",
              "v2g_D_mpc_vs_dumb.png", "v2g_KPI_comparison.png",
              "v2g_insights.png"]:
        print(f"    {f}")
    print()


if __name__ == "__main__":
    main()