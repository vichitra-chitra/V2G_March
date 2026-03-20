#!/usr/bin/env python3
"""
S.KOe COOL — Single-Day V2G Optimisation  (v2 — corrected)
TU Dortmund IE3 x Schmitz Cargobull AG | 2026

Fixes vs v1:
  1. Optimisation runs over plugged-in window only (arrival->departure).
     SoC before arrival is flat at arrival_SoC; dumb starts at arrival.
  2. SoC y-axis shown in % (0-100), not kWh.
  3. 3 comparison charts (no KPI box) + 1 separate combined KPI table chart.
  4. Arrival/departure labels only on panel 1; panels 2-3 show lines only.
  5. Legends placed below each panel (no overlap with data).
  6. Both power and SoC use step('post') for exact 15-min alignment.
  7. X-axis labels shown on every panel with visible spacing.
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from dataclasses import dataclass
from pathlib import Path

DISPLAY_START_H = 12.0   # display axis begins at noon


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
        return self.usable_capacity_kWh * self.soc_min_pct / 100.0  # 12 kWh

    @property
    def E_max(self):
        return self.usable_capacity_kWh * self.soc_max_pct / 100.0  # 60 kWh


# ═══════════════════════════════════════════════════════════════════════════════
#  2. PRICE LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_winter_weekday_prices(csv_path: str) -> np.ndarray:
    """SMARD DE/LU 15-min day-ahead: return mean winter weekday 96-slot profile (EUR/kWh)."""
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

    price_col = next((c for c in df.columns if "Germany" in c and "MWh" in c), None)
    if price_col is None:
        raise ValueError(f"Germany price column not found. Columns: {list(df.columns)}")

    df = df[["Start date", price_col]].copy()
    df.columns = ["dt_str", "price_eur_mwh"]
    df["dt"]   = pd.to_datetime(df["dt_str"], format="%b %d, %Y %I:%M %p", errors="coerce")
    df = df.dropna(subset=["dt", "price_eur_mwh"])
    df["price"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce") / 1000.0
    df = df.dropna(subset=["price"]).set_index("dt").sort_index()

    mask = (df.index.month.isin([1, 2, 3, 10, 11, 12])) & (df.index.dayofweek < 5)
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
#  3. WINDOW & DISPLAY HELPERS
#
#  KEY DESIGN: optimise only over the plugged-in window (e.g. 16:00-06:00 = 56
#  slots). Pad the 96-slot display array with flat lines before arrival and
#  after departure. This eliminates the wrap-around initialisation bug.
# ═══════════════════════════════════════════════════════════════════════════════

def get_window(v2g, arrival_h: float, departure_h: float):
    """
    Compute the plugged-in window in both original-slot and display-slot space.

    Assumes: arrival_h > DISPLAY_START_H  (e.g. 16:00 > 12:00)
             departure_h < DISPLAY_START_H (e.g. 06:00 < 12:00)
    i.e. standard overnight depot dwell.

    Returns
    -------
    window_slots : list of original 96-slot indices  [64,65,...,95,0,1,...,23]
    arr_disp     : index in the 96-slot display array where arrival falls
    dep_disp     : index in the 96-slot display array where departure falls
    W            : number of slots in the window
    """
    n  = v2g.n_slots
    dt = v2g.dt_h

    a = round(arrival_h   / dt) % n   # e.g. 64
    d = round(departure_h / dt) % n   # e.g. 24

    # Overnight window: arrival -> end of day -> start of next day -> departure
    window_slots = list(range(a, n)) + list(range(0, d))
    W            = len(window_slots)

    # Display-frame slot: display[0] = DISPLAY_START_H = 12:00
    def to_disp(h):
        if h >= DISPLAY_START_H:
            return round((h - DISPLAY_START_H) / dt)
        return round((h + 24.0 - DISPLAY_START_H) / dt)

    return window_slots, to_disp(arrival_h), to_disp(departure_h), W


def to_display(v2g, P_c_w, P_d_w, soc_w_kwh,
               arr_disp, dep_disp, E_init_kwh):
    """
    Expand W-slot window results into 96-slot display arrays.

    Before arrival  : P=0,  SoC = arrival_SoC%  (trailer on road approaching)
    During window   : optimization results
    After departure : P=0,  SoC = final SoC%    (trailer on road, fully charged)

    SoC is returned in % (0-100).
    """
    n   = v2g.n_slots
    pct = 100.0 / v2g.usable_capacity_kWh

    P_c = np.zeros(n)
    P_d = np.zeros(n)
    soc = np.full(n, E_init_kwh * pct)      # default = arrival SoC%

    P_c[arr_disp:dep_disp] = P_c_w
    P_d[arr_disp:dep_disp] = P_d_w
    soc[arr_disp:dep_disp] = soc_w_kwh * pct
    soc[dep_disp:]         = soc_w_kwh[-1] * pct   # fully charged after departure

    return P_c, P_d, soc


def build_display_buy_plugged(v2g, buy, arrival_h, departure_h):
    """
    Roll price and plugged-mask arrays so display slot 0 = DISPLAY_START_H.
    """
    n    = v2g.n_slots
    ROLL = round(DISPLAY_START_H / v2g.dt_h)   # 48 for noon start

    buy_d = np.roll(buy, -ROLL)

    h = np.arange(n) * v2g.dt_h
    plugged_raw = ((h >= arrival_h) | (h < departure_h)).astype(float)
    plugged_d   = np.roll(plugged_raw, -ROLL)

    return buy_d, plugged_d


# ═══════════════════════════════════════════════════════════════════════════════
#  4. MILP SOLVER  (true binary mutex, all W slots plugged-in)
# ═══════════════════════════════════════════════════════════════════════════════

def solve_milp(v2g, buy_w, v2gp_w, E_init, E_fin, allow_discharge=True):
    """
    Binary-mutex MILP over W = len(buy_w) slots, all plugged-in.

    Decision variables (length 5W):
      P_c [0..W-1]    charging power  (kW, continuous)
      P_d [W..2W-1]   discharge power (kW, continuous)
      e   [2W..3W-1]  SoC trajectory  (kWh, continuous)
      z_c [3W..4W-1]  charge binary   {0,1}
      z_d [4W..5W-1]  discharge binary{0,1}

    Objective (deg=0): min sum_t [buy[t]*P_c[t] - v2g[t]*P_d[t]] * dt
    """
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

    # Cost vector
    c = np.zeros(nv)
    c[idx_c] = buy_w * dt
    if allow_discharge:
        c[idx_d] = -v2gp_w * dt

    # Variable bounds
    lb = np.zeros(nv)
    ub = np.full(nv, np.inf)
    ub[idx_c]  = v2g.charge_power_kW
    ub[idx_d]  = v2g.discharge_power_kW if allow_discharge else 0.0
    lb[idx_e]  = v2g.E_min
    ub[idx_e]  = v2g.E_max
    lb[idx_zc] = 0.0;  ub[idx_zc] = 1.0
    lb[idx_zd] = 0.0;  ub[idx_zd] = 1.0

    integrality = np.zeros(nv)
    integrality[idx_zc] = 1
    integrality[idx_zd] = 1

    # Constraint matrix
    n_rows = 4*W + 1
    A  = lil_matrix((n_rows, nv))
    lo = np.full(n_rows, -np.inf)
    hi = np.zeros(n_rows)

    # (i) SoC dynamics: e[t] = e[t-1] + eta_c*P_c*dt - P_d/eta_d*dt
    for t in range(W):
        A[t, idx_e[t]]  =  1.0
        A[t, idx_c[t]]  = -v2g.eta_charge * dt
        A[t, idx_d[t]]  =  (1.0 / v2g.eta_discharge) * dt
        rhs = E_init if t == 0 else 0.0
        if t > 0:
            A[t, idx_e[t-1]] = -1.0
        lo[t] = hi[t] = rhs

    # (ii-iv) Binary mutex: P_c <= Pmax*z_c, P_d <= Pmax*z_d, z_c+z_d<=1
    for t in range(W):
        A[W   + t, idx_c[t]]  =  1.0
        A[W   + t, idx_zc[t]] = -v2g.charge_power_kW
        hi[W  + t] = 0.0
        A[2*W + t, idx_d[t]]  =  1.0
        A[2*W + t, idx_zd[t]] = -v2g.discharge_power_kW
        hi[2*W + t] = 0.0
        A[3*W + t, idx_zc[t]] = 1.0
        A[3*W + t, idx_zd[t]] = 1.0
        hi[3*W + t] = 1.0

    # (v) Departure SoC target
    A[4*W, idx_e[W-1]] = 1.0
    lo[4*W] = E_fin
    hi[4*W] = v2g.E_max

    res = milp(
        c,
        constraints=LinearConstraint(csc_matrix(A), lo, hi),
        integrality=integrality,
        bounds=Bounds(lb, ub),
        options={"disp": False, "time_limit": 60},
    )
    if not res.success:
        raise RuntimeError(
            f"MILP failed — status: {res.status!r}, msg: {res.message!r}\n"
            "Check E_init and E_fin are feasible in the plugged window."
        )
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
        v2g, P_c_w, P_d_w, soc_w, arr_disp, dep_disp, E_init_kwh
    )
    return {
        "label"      : label,
        "P_c_d"      : P_c_d,   # 96-slot display arrays
        "P_d_d"      : P_d_d,
        "soc_d"      : soc_d,   # in %
        "net_cost"   : chg - rev,
        "charge_cost": chg,
        "v2g_rev"    : rev,
        "v2g_kwh"    : float(np.sum(P_d_w) * dt),
    }


def run_A_dumb(v2g, buy_w, v2gp_w, W, E_init, arr_disp, dep_disp):
    """
    Charges at full power from the moment of arrival (slot 0 of window).
    No price awareness. Correct: charging begins at arrival, not midnight.
    """
    dt = v2g.dt_h
    P_c = np.zeros(W); P_d = np.zeros(W); soc = np.zeros(W)
    s = E_init
    for t in range(W):
        if s < v2g.E_max:
            p      = min(v2g.charge_power_kW,
                         (v2g.E_max - s) / (v2g.eta_charge * dt))
            P_c[t] = p
            s      = min(v2g.E_max, s + p * v2g.eta_charge * dt)
        soc[t] = s
    return _kpi("A - Dumb", v2g, P_c, P_d, soc, buy_w, v2gp_w, arr_disp, dep_disp, E_init)


def run_B_smart(v2g, buy_w, v2gp_w, E_init, arr_disp, dep_disp):
    """MILP shifts charging to cheapest slots — no V2G discharge."""
    P_c, P_d, soc = solve_milp(
        v2g, buy_w, v2gp_w, E_init, v2g.E_max, allow_discharge=False
    )
    return _kpi("B - Smart (no V2G)", v2g, P_c, P_d, soc, buy_w, v2gp_w,
                arr_disp, dep_disp, E_init)


def run_C_milp(v2g, buy_w, v2gp_w, E_init, arr_disp, dep_disp):
    """Full window MILP solved once — optimal charge + V2G discharge."""
    P_c, P_d, soc = solve_milp(
        v2g, buy_w, v2gp_w, E_init, v2g.E_max, allow_discharge=True
    )
    return _kpi("C - MILP Day-Ahead", v2g, P_c, P_d, soc, buy_w, v2gp_w,
                arr_disp, dep_disp, E_init)


def run_D_mpc(v2g, buy_w, v2gp_w, E_init, arr_disp, dep_disp):
    """
    Receding-horizon MPC: re-solves MILP at every slot over the remaining
    window, applies only the first action, then advances with real SoC.
    """
    W  = len(buy_w)
    dt = v2g.dt_h
    s  = E_init
    P_c_all = np.zeros(W); P_d_all = np.zeros(W); soc_all = np.zeros(W)
    for t in range(W):
        Pc_w, Pd_w, _ = solve_milp(
            v2g, buy_w[t:], v2gp_w[t:], s, v2g.E_max, allow_discharge=True
        )
        pc = float(np.clip(Pc_w[0], 0, v2g.charge_power_kW))
        pd = float(np.clip(Pd_w[0], 0, v2g.discharge_power_kW))
        # Resolve numerical tie (binary mutex makes this rare)
        if pc > 1e-6 and pd > 1e-6:
            if v2gp_w[t] > buy_w[t]:
                pc = 0.0    # discharging more profitable
            else:
                pd = 0.0    # charging is cheaper
        s = float(np.clip(
            s + pc * v2g.eta_charge * dt - pd / v2g.eta_discharge * dt,
            v2g.E_min, v2g.E_max
        ))
        P_c_all[t] = pc; P_d_all[t] = pd; soc_all[t] = s
    return _kpi("D - MPC (receding)", v2g, P_c_all, P_d_all, soc_all, buy_w, v2gp_w,
                arr_disp, dep_disp, E_init)


# ═══════════════════════════════════════════════════════════════════════════════
#  6. CONSOLE REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(results):
    ref = results[0]["net_cost"]
    print("\n" + "="*76)
    print("  SINGLE-DAY KPI SUMMARY  (deg=0, TRU=0, Winter WD avg)")
    print("="*76)
    print(f"  {'Scenario':<28} {'Net EUR':>9} {'Chg EUR':>9} "
          f"{'V2G EUR':>9} {'kWh':>7} {'vs Dumb':>9}")
    print("-"*76)
    for r in results:
        print(f"  {r['label']:<28} {r['net_cost']:>9.4f} {r['charge_cost']:>9.4f} "
              f"{r['v2g_rev']:>9.4f} {r['v2g_kwh']:>7.2f} "
              f"{ref - r['net_cost']:>+9.4f}")
    print("="*76)
    c_r = results[2]
    print(f"\n  Annualised (x365, Scenario C MILP):")
    print(f"    Savings vs Dumb : EUR {(ref - c_r['net_cost'])*365:+,.0f} / year")
    print(f"    V2G Revenue     : EUR {c_r['v2g_rev']*365:,.0f} / year")
    print(f"  [Agora 2025: ~EUR 500/year arbitrage for private car]\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  7. PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

SC_COL = {
    "A": "#AAAAAA", "B": "#2196F3", "C": "#00BCD4", "D": "#FF7700",
    "price": "#2E7D32",
}


def _ticks():
    pos  = np.arange(DISPLAY_START_H, DISPLAY_START_H + 25, 2)
    lbls = [f"{int(h % 24):02d}:00" for h in pos]
    return pos, lbls


def _format_ax(ax, ylabel, title, ylim=None):
    tp, tl = _ticks()
    ax.set_xlim(DISPLAY_START_H, DISPLAY_START_H + 24.0)
    ax.set_xticks(tp)
    ax.set_xticklabels(tl, fontsize=7.5, rotation=35, ha="right")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold", loc="left", pad=3)
    ax.grid(True, alpha=0.22)
    if ylim:
        ax.set_ylim(*ylim)


def _disp_x(h):
    """Convert clock hour to display-axis coordinate."""
    return h if h >= DISPLAY_START_H else h + 24.0


def _midnight_line(ax):
    ax.axvline(_disp_x(0), color="#555555", lw=1.2, ls="--", alpha=0.65, zorder=5)


def _avd_lines_only(ax, arrival_h, departure_h):
    """Dotted vertical lines for arrival & departure — NO text."""
    ax.axvline(_disp_x(arrival_h),   color="#1B5E20", lw=1.1, ls=":", alpha=0.75, zorder=5)
    ax.axvline(_disp_x(departure_h), color="#B71C1C", lw=1.1, ls=":", alpha=0.75, zorder=5)


def _avd_lines_text(ax, arrival_h, departure_h):
    """Dotted lines + text labels (panel 1 only)."""
    _avd_lines_only(ax, arrival_h, departure_h)
    _midnight_line(ax)
    tr = ax.get_xaxis_transform()
    ax.text(_disp_x(arrival_h)   + 0.2, 0.96,
            f"arrival {int(arrival_h):02d}:00",
            transform=tr, fontsize=7, color="#1B5E20", va="top")
    ax.text(_disp_x(departure_h) + 0.2, 0.96,
            f"depart {int(departure_h):02d}:00",
            transform=tr, fontsize=7, color="#B71C1C", va="top")
    ax.text(_disp_x(0)           + 0.2, 0.96,
            "midnight",
            transform=tr, fontsize=7, color="#555555", va="top")


def _legend_below(ax, handles, ncol=3):
    """Place legend below the axes, never overlapping the chart."""
    ax.legend(
        handles=handles,
        fontsize=8, ncol=ncol,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        framealpha=0.95, edgecolor="#CCCCCC",
    )


def plot_comparison(v2g, hours_d, buy_d, plugged_d,
                    sc, dumb, sc_key,
                    arrival_h, departure_h, out):
    """
    3-panel comparison chart: Price | Power | SoC
    One scenario vs Dumb (A).
    """
    col      = SC_COL[sc_key]
    sc_short = sc["label"].split("(")[0].strip()

    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(15, 13),
        gridspec_kw={"height_ratios": [1.0, 1.8, 1.8], "hspace": 0.65}
    )
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        f"S.KOe COOL  —  {sc_short}  vs  A - Dumb  |  Winter Weekday Avg\n"
        f"Arrival: {int(arrival_h):02d}:00  |  Departure: {int(departure_h):02d}:00  |  "
        f"Battery: {v2g.usable_capacity_kWh:.0f} kWh usable  |  deg=0  |  TRU=0 kW",
        fontsize=11, fontweight="bold", y=0.99
    )

    # ── Panel 1: Price + plugged shading ──────────────────────────────────────
    for t in range(len(hours_d)):
        if plugged_d[t] > 0.5:
            ax0.axvspan(hours_d[t], hours_d[t] + v2g.dt_h,
                        color="gold", alpha=0.22, lw=0, zorder=1)
    p_line, = ax0.step(hours_d, buy_d * 1000, where="post",
                       color=SC_COL["price"], lw=2.0,
                       label="Day-ahead price (EUR/MWh)", zorder=3)
    ax0.fill_between(hours_d, buy_d * 1000, step="post",
                     color=SC_COL["price"], alpha=0.10, zorder=2)
    _avd_lines_text(ax0, arrival_h, departure_h)   # text labels here only
    _legend_below(ax0, [
        p_line,
        mpatches.Patch(color="gold", alpha=0.5, label="Plugged-in window"),
        Line2D([0], [0], color="#1B5E20", ls=":", lw=1.2,
               label=f"Arrival {int(arrival_h):02d}:00"),
        Line2D([0], [0], color="#B71C1C", ls=":", lw=1.2,
               label=f"Departure {int(departure_h):02d}:00"),
        Line2D([0], [0], color="#555555", ls="--", lw=1.2,
               label="Midnight"),
    ], ncol=5)
    _format_ax(ax0, "EUR/MWh", "(1) Electricity Price + Plugged-In Window")

    # ── Panel 2: Power ────────────────────────────────────────────────────────
    #  Both lines use step('post') for exact 15-min slot alignment with panel 3.
    dm_c, = ax1.step(hours_d, dumb["P_c_d"], where="post",
                     color=SC_COL["A"], lw=1.8, alpha=0.75,
                     label="A - Dumb  charge (kW)")
    sc_c, = ax1.step(hours_d, sc["P_c_d"], where="post",
                     color=col, lw=2.4,
                     label=f"{sc_short}  charge (kW)")
    handles_p2 = [dm_c, sc_c]
    if sc["v2g_kwh"] > 0.05:
        sc_d, = ax1.step(hours_d, -sc["P_d_d"], where="post",
                         color=col, lw=2.4, ls="--", alpha=0.85,
                         label=f"{sc_short}  V2G discharge (shown negative)")
        handles_p2.append(sc_d)
    ax1.axhline(0, color="black", lw=0.6)
    _midnight_line(ax1)
    _avd_lines_only(ax1, arrival_h, departure_h)   # lines only, no text
    _legend_below(ax1, handles_p2, ncol=3)
    _format_ax(ax1, "Power (kW)", "(2) Charge / Discharge Power")

    # ── Panel 3: SoC in % ─────────────────────────────────────────────────────
    #  step('post') matches exactly with panel 2 (same slot convention).
    for t in range(len(hours_d)):
        if plugged_d[t] > 0.5:
            ax2.axvspan(hours_d[t], hours_d[t] + v2g.dt_h,
                        color="gold", alpha=0.12, lw=0, zorder=1)
    dm_s, = ax2.step(hours_d, dumb["soc_d"], where="post",
                     color=SC_COL["A"], lw=2.0, label="A - Dumb  SoC (%)")
    sc_s, = ax2.step(hours_d, sc["soc_d"], where="post",
                     color=col, lw=2.6, label=f"{sc_short}  SoC (%)")
    ax2.axhline(v2g.soc_min_pct, color="#C62828", ls=":", lw=1.4,
                label=f"SoC min = {v2g.soc_min_pct:.0f}%  (cold-chain floor)")
    ax2.axhline(v2g.soc_max_pct, color="#0D47A1", ls=":", lw=1.4,
                label=f"SoC max = {v2g.soc_max_pct:.0f}%  (departure target)")
    _midnight_line(ax2)
    _avd_lines_only(ax2, arrival_h, departure_h)
    _legend_below(ax2, [
        dm_s, sc_s,
        Line2D([0], [0], color="#C62828", ls=":", lw=1.4,
               label=f"SoC min {v2g.soc_min_pct:.0f}%"),
        Line2D([0], [0], color="#0D47A1", ls=":", lw=1.4,
               label=f"SoC max {v2g.soc_max_pct:.0f}%"),
    ], ncol=4)
    _format_ax(ax2, "State of Charge (%)", "(3) Battery SoC Trajectory",
               ylim=(0, 112))
    ax2.set_xlabel("Time of Day", fontsize=9)

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved -> {out}")


def plot_kpi_table(v2g, results, arrival_h, departure_h,
                   out="v2g_KPI_comparison.png"):
    """Standalone figure: full KPI comparison table for all 4 scenarios."""
    ref   = results[0]["net_cost"]
    names = ["A — Dumb", "B — Smart", "C — MILP", "D — MPC"]
    keys  = ["A", "B", "C", "D"]

    def fmt_sav(r):
        v = ref - r["net_cost"]
        return f"EUR {v:+.4f}" if abs(v) > 1e-6 else "-"

    rows = [
        ("Net cost (EUR/day)",
         [f"{r['net_cost']:.4f}"    for r in results]),
        ("Charge cost (EUR/day)",
         [f"{r['charge_cost']:.4f}" for r in results]),
        ("V2G revenue (EUR/day)",
         [f"{r['v2g_rev']:.4f}"     for r in results]),
        ("V2G export (kWh/day)",
         [f"{r['v2g_kwh']:.2f}"     for r in results]),
        ("Daily savings vs Dumb",
         ["-"] + [fmt_sav(r)        for r in results[1:]]),
        ("Annual savings (x365)",
         ["-"] + [f"EUR {(ref - r['net_cost'])*365:+,.0f}" for r in results[1:]]),
        ("Annual V2G revenue",
         [f"EUR {r['v2g_rev']*365:,.0f}" for r in results]),
    ]

    cell_text  = [[row[0]] + row[1] for row in rows]
    col_labels = ["Metric"] + names

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        "S.KOe COOL — KPI Comparison: All 4 Scenarios  |  Winter Weekday Avg\n"
        f"Arrival {int(arrival_h):02d}:00  |  Departure {int(departure_h):02d}:00  |  "
        f"Battery {v2g.usable_capacity_kWh:.0f} kWh usable  |  deg=0  |  TRU=0 kW",
        fontsize=11, fontweight="bold", y=1.04
    )

    tbl = ax.table(
        cellText=cell_text, colLabels=col_labels,
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 2.5)

    # Header row
    hdr_colors = ["#263238"] + [SC_COL[k] for k in keys]
    for col, hc in enumerate(hdr_colors):
        cell = tbl[0, col]
        cell.set_facecolor(hc)
        cell.set_text_props(color="white", fontweight="bold")

    # Body rows
    col_bgs = ["#ECEFF1", "#E8EAF6", "#E3F2FD", "#E0F2F1", "#FFF3E0"]
    for row in range(1, len(rows) + 1):
        for col in range(5):
            cell = tbl[row, col]
            cell.set_facecolor(col_bgs[col])
            if col == 0:
                cell.set_text_props(fontweight="bold", ha="left")
            # Colour positive savings green
            txt = cell.get_text().get_text()
            if "+" in txt and "EUR" in txt:
                cell.set_text_props(color="#1B5E20", fontweight="bold")

    # Set metric column wider
    for row in range(len(rows) + 1):
        tbl[row, 0].set_width(0.34)
        for col in range(1, 5):
            tbl[row, col].set_width(0.165)

    ax.text(
        0.5, -0.06,
        "Agora Verkehrswende 2025 benchmark: ~EUR 500/year V2G arbitrage (private car)  |  "
        "Reefer trailer potential higher: larger battery, predictable depot dwell times",
        ha="center", va="top", fontsize=7.5, style="italic",
        color="#555555", transform=ax.transAxes
    )

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved -> {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  8. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*62)
    print("  S.KOe COOL — Single-Day V2G Optimisation (v2)")
    print("  TU Dortmund IE3 x Schmitz Cargobull AG | 2026")
    print("="*62)

    csv_path = next(
        (str(p) for p in [
            Path(__file__).parent / "2025_Electricity_Price.csv",
            Path("2025_Electricity_Price.csv"),
        ] if p.exists()),
        None
    )
    if csv_path is None:
        print("\n  ERROR: 2025_Electricity_Price.csv not found.\n")
        sys.exit(1)
    print(f"\n  Price CSV: {csv_path}")

    v2g = V2GParams()
    print(f"\n  Battery  : {v2g.battery_capacity_kWh} kWh total | "
          f"{v2g.usable_capacity_kWh} kWh usable\n"
          f"  SoC range: {v2g.soc_min_pct:.0f}%-{v2g.soc_max_pct:.0f}% | "
          f"P_max {v2g.charge_power_kW:.0f} kW | deg=0")

    # ── User inputs ───────────────────────────────────────────────────────────
    print("\n" + "-"*50)

    def ask(prompt, default, lo, hi):
        raw = input(f"  {prompt} [default {default}]: ").strip()
        try:
            v = float(raw)
            assert lo <= v <= hi
            return v
        except Exception:
            print(f"  -> Using default {default}")
            return default

    departure_h  = ask("Trailer DEPARTURE hour 0-23", 6,  0, 23)
    arrival_h    = ask("Trailer ARRIVAL   hour 0-23", 16, 0, 23)
    soc_init_pct = ask("Arrival SoC % (20-100)",      45, 20, 100)

    if arrival_h <= DISPLAY_START_H or departure_h >= DISPLAY_START_H:
        print(f"\n  WARNING: This script assumes arrival after {DISPLAY_START_H:.0f}:00"
              f" and departure before {DISPLAY_START_H:.0f}:00.")
        print("  Proceeding with user values — check display if unexpected.\n")

    print(f"\n  Window : {int(arrival_h):02d}:00 -> {int(departure_h):02d}:00 "
          f"(next day)  |  Arrival SoC: {soc_init_pct:.0f}%  |  Target: 100%")
    print("-"*50)

    # ── Load & build ──────────────────────────────────────────────────────────
    print("\n  Loading prices ...")
    buy  = load_winter_weekday_prices(csv_path)
    v2gp = buy.copy()   # sell price = buy price (no spread this draft)

    window_slots, arr_disp, dep_disp, W = get_window(v2g, arrival_h, departure_h)
    buy_w   = buy[window_slots]
    v2gp_w  = v2gp[window_slots]
    E_init  = v2g.usable_capacity_kWh * soc_init_pct / 100.0

    print(f"  Plugged-in window: {W} slots = {W * v2g.dt_h:.1f} h  "
          f"(display slots {arr_disp} to {dep_disp})")

    buy_d, plugged_d = build_display_buy_plugged(v2g, buy, arrival_h, departure_h)
    hours_d = np.arange(v2g.n_slots) * v2g.dt_h + DISPLAY_START_H  # 12.0 to 35.75

    # ── Run scenarios ─────────────────────────────────────────────────────────
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

    # ── Charts ────────────────────────────────────────────────────────────────
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

    print("\n  Done. Outputs:")
    print("    v2g_B_smart_vs_dumb.png")
    print("    v2g_C_milp_vs_dumb.png")
    print("    v2g_D_mpc_vs_dumb.png")
    print("    v2g_KPI_comparison.png\n")


if __name__ == "__main__":
    main()