#!/usr/bin/env python3
"""
S.KOe COOL — V2G Optimisation Suite (v4)
TU Dortmund IE3 x Schmitz Cargobull AG | 2026

Output files:
  v2g_winter_weekday.png   — all 4 scenarios, winter WD avg (overnight dwell)
  v2g_summer_weekday.png   — all 4 scenarios, summer WD avg (overnight dwell)
  v2g_winter_weekend.png   — all 4 scenarios, winter WE avg (full 24h, unlimited)
  v2g_summer_weekend.png   — all 4 scenarios, summer WE avg (full 24h, unlimited)
  v2g_KPI_multi.png        — 2x2 KPI table: A/B/C/D x 4 day types
  v2g_annual_results.png   — full-year simulation: cumulative, monthly, distribution
  v2g_price_profiles.png   — price analysis: season/daytype comparisons, spread
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

SC_COL = {"A": "#999999", "B": "#2196F3", "C": "#00ACC1", "D": "#FF7700",
          "price": "#2E7D32"}
WINTER_MONTHS = [1, 2, 3, 10, 11, 12]
SUMMER_MONTHS = [4, 5, 6, 7, 8, 9]


# =============================================================================
#  1. PARAMETERS
# =============================================================================

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
    def E_min(self): return self.usable_capacity_kWh * self.soc_min_pct  / 100.0
    @property
    def E_max(self): return self.usable_capacity_kWh * self.soc_max_pct  / 100.0


# =============================================================================
#  2. PRICE LOADING
# =============================================================================

_CSV_CACHE: dict = {}

def _load_csv_raw(csv_path: str) -> pd.DataFrame:
    if csv_path in _CSV_CACHE:
        return _CSV_CACHE[csv_path]
    df = None
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(csv_path, sep=";", encoding=enc)
            if len(df.columns) > 1: break
            df = pd.read_csv(csv_path, sep=",", encoding=enc)
            if len(df.columns) > 1: break
        except Exception: continue
    if df is None or df.empty:
        raise ValueError(f"Could not read CSV: {csv_path}")
    price_col = next((c for c in df.columns if "Germany" in c and "MWh" in c), None)
    if not price_col:
        raise ValueError(f"Germany price column not found. Columns: {list(df.columns)}")
    df = df[["Start date", price_col]].copy()
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
    print(f"  CSV: {len(df):,} rows, {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def load_avg_profile(csv_path: str, months: list, is_weekend: bool) -> np.ndarray:
    """Return mean 96-slot avg price profile (EUR/kWh) for given months + day type."""
    df   = _load_csv_raw(csv_path)
    mask = df["month"].isin(months) & (df["is_weekend"] == is_weekend)
    sub  = df[mask]
    if len(sub) == 0:
        raise ValueError(f"No data found for months={months}, weekend={is_weekend}")
    profile = sub.groupby("slot")["price"].mean().values
    if len(profile) != 96:
        raise ValueError(f"Expected 96 slots, got {len(profile)}")
    n_days = int(len(sub) / 96)
    lbl = "WE" if is_weekend else "WD"
    print(f"    [{lbl} {months[0]}-{months[-1]}] avg {n_days} days | "
          f"{profile.min()*1000:.0f}-{profile.max()*1000:.0f} EUR/MWh")
    return profile


# =============================================================================
#  3. WINDOW & DISPLAY HELPERS
# =============================================================================

def get_window(v2g, arrival_h: float, departure_h: float, is_weekend: bool):
    """
    Returns (window_slots, arr_disp, dep_disp, W, display_start_h).

    Weekend : full 24h window, display 00:00-24:00 (display_start=0).
    Weekday : overnight window, display 12:00-12:00 (display_start=12).

    All indices are in DISPLAY-FRAME coordinates so that to_display()
    and hours_d are directly aligned without any extra rolling.
    """
    if is_weekend:
        n = v2g.n_slots
        return list(range(n)), 0, n, n, 0.0

    n  = v2g.n_slots
    dt = v2g.dt_h
    DS = 12.0   # display starts at noon
    # Original-frame arrival and departure slots
    a_orig = round(arrival_h   / dt) % n
    d_orig = round(departure_h / dt) % n
    # Overnight window in original-frame
    window_slots = list(range(a_orig, n)) + list(range(0, d_orig))
    W = len(window_slots)
    # Convert clock hours to display-frame slot indices
    def to_disp(h):
        return round((h - DS) / dt) if h >= DS else round((h + 24.0 - DS) / dt)
    return window_slots, to_disp(arrival_h), to_disp(departure_h), W, DS


def build_display(v2g, buy: np.ndarray, arrival_h: float, departure_h: float,
                  is_weekend: bool, display_start: float):
    """
    Roll price array and build plugged mask so display slot 0 = display_start.
    Returns (buy_d, plugged_d, hours_d) all in display-frame.
    """
    n    = v2g.n_slots
    ROLL = round(display_start / v2g.dt_h)
    buy_d = np.roll(buy, -ROLL)

    h = np.arange(n) * v2g.dt_h
    if is_weekend:
        plug_raw = np.ones(n)
    else:
        plug_raw = ((h >= arrival_h) | (h < departure_h)).astype(float)
    plug_d  = np.roll(plug_raw, -ROLL)
    hours_d = np.arange(n) * v2g.dt_h + display_start   # e.g. 12.0..35.75 or 0..23.75
    return buy_d, plug_d, hours_d


def to_display(v2g, Pc_w, Pd_w, soc_w_kwh, arr_disp, dep_disp, E_init_kwh):
    """
    Expand W-slot window results into 96-slot display arrays (display-frame).
    SoC is returned in %.
    Pre-arrival  : flat at arrival SoC%.
    Post-departure: flat at final SoC%.
    """
    n   = v2g.n_slots
    pct = 100.0 / v2g.usable_capacity_kWh
    W   = dep_disp - arr_disp

    Pc  = np.zeros(n); Pd = np.zeros(n)
    soc = np.full(n, E_init_kwh * pct)

    Pc[arr_disp:dep_disp]  = Pc_w[:W]
    Pd[arr_disp:dep_disp]  = Pd_w[:W]
    soc[arr_disp:dep_disp] = soc_w_kwh[:W] * pct
    if dep_disp < n:
        soc[dep_disp:] = soc_w_kwh[W - 1] * pct   # hold at 100% after departure
    return Pc, Pd, soc


def soc_ramp(hours_d, soc_d, E_init_pct):
    """
    Build continuous (x, y) arrays: linear ramp from slot-start to slot-end.
    This gives gradual slopes instead of vertical step jumps.
    """
    n  = len(hours_d)
    dt = (hours_d[1] - hours_d[0]) if n > 1 else 0.25
    soc_start = np.concatenate([[E_init_pct], soc_d[:-1]])
    x = np.empty(2 * n); y = np.empty(2 * n)
    for i in range(n):
        x[2*i]   = hours_d[i];      y[2*i]   = soc_start[i]
        x[2*i+1] = hours_d[i] + dt; y[2*i+1] = soc_d[i]
    return x, y


def _disp_x(clock_h: float, display_start: float = 12.0) -> float:
    """Convert a real clock hour to display-axis x coordinate."""
    return clock_h if clock_h >= display_start else clock_h + 24.0


# =============================================================================
#  4. MILP SOLVER  (true binary mutex)
# =============================================================================

def solve_milp(v2g, buy_w, v2gp_w, E_init, E_fin, allow_discharge=True):
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix, csc_matrix

    W = len(buy_w); dt = v2g.dt_h
    idx_c  = np.arange(W);        idx_d  = np.arange(W,   2*W)
    idx_e  = np.arange(2*W, 3*W); idx_zc = np.arange(3*W, 4*W)
    idx_zd = np.arange(4*W, 5*W); nv = 5 * W

    c = np.zeros(nv)
    c[idx_c] = buy_w * dt
    if allow_discharge: c[idx_d] = -v2gp_w * dt

    lb = np.zeros(nv); ub = np.full(nv, np.inf)
    ub[idx_c]  = v2g.charge_power_kW
    ub[idx_d]  = v2g.discharge_power_kW if allow_discharge else 0.0
    lb[idx_e]  = v2g.E_min; ub[idx_e] = v2g.E_max
    lb[idx_zc] = 0.; ub[idx_zc] = 1.
    lb[idx_zd] = 0.; ub[idx_zd] = 1.

    integrality = np.zeros(nv)
    integrality[idx_zc] = 1; integrality[idx_zd] = 1

    n_rows = 4*W + 1
    A = lil_matrix((n_rows, nv))
    lo = np.full(n_rows, -np.inf); hi = np.zeros(n_rows)

    for t in range(W):
        A[t, idx_e[t]]  =  1.
        A[t, idx_c[t]]  = -v2g.eta_charge * dt
        A[t, idx_d[t]]  =  1. / v2g.eta_discharge * dt
        rhs = E_init if t == 0 else 0.
        if t > 0: A[t, idx_e[t-1]] = -1.
        lo[t] = hi[t] = rhs

    for t in range(W):
        A[W   + t, idx_c[t]]  =  1.; A[W   + t, idx_zc[t]] = -v2g.charge_power_kW
        A[2*W + t, idx_d[t]]  =  1.; A[2*W + t, idx_zd[t]] = -v2g.discharge_power_kW
        A[3*W + t, idx_zc[t]] =  1.; A[3*W + t, idx_zd[t]] =  1.
        hi[W+t] = hi[2*W+t] = 0.; hi[3*W+t] = 1.

    A[4*W, idx_e[W-1]] = 1.; lo[4*W] = E_fin; hi[4*W] = v2g.E_max

    res = milp(c, constraints=LinearConstraint(csc_matrix(A), lo, hi),
               integrality=integrality, bounds=Bounds(lb, ub),
               options={"disp": False, "time_limit": 60})
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.status!r} — {res.message!r}")
    return (np.clip(res.x[idx_c], 0, None),
            np.clip(res.x[idx_d], 0, None),
            res.x[idx_e])


# =============================================================================
#  5. SCENARIO RUNNERS
# =============================================================================

def _kpi(label, v2g, Pc, Pd, soc, buy_w, v2gp_w, arr, dep, E_init):
    dt  = v2g.dt_h
    chg = float(np.sum(Pc * buy_w)  * dt)
    rev = float(np.sum(Pd * v2gp_w) * dt)
    Pc_d, Pd_d, soc_d = to_display(v2g, Pc, Pd, soc, arr, dep, E_init)
    return {
        "label"       : label,
        "P_c_d"       : Pc_d, "P_d_d": Pd_d, "soc_d": soc_d,
        "P_c_w"       : Pc,   "P_d_w": Pd,   "soc_w_kwh": soc,
        "net_cost"    : chg - rev,
        "charge_cost" : chg,
        "v2g_rev"     : rev,
        "v2g_kwh"     : float(np.sum(Pd) * dt),
        "charge_kwh"  : float(np.sum(Pc) * dt),
        "E_init_pct"  : E_init * 100.0 / v2g.usable_capacity_kWh,
    }


def run_A_dumb(v2g, buy_w, v2gp_w, W, E_init, arr, dep):
    dt = v2g.dt_h
    Pc = np.zeros(W); Pd = np.zeros(W); soc = np.zeros(W); s = E_init
    for t in range(W):
        if s < v2g.E_max:
            p = min(v2g.charge_power_kW, (v2g.E_max - s) / (v2g.eta_charge * dt))
            Pc[t] = p; s = min(v2g.E_max, s + p * v2g.eta_charge * dt)
        soc[t] = s
    return _kpi("A - Dumb", v2g, Pc, Pd, soc, buy_w, v2gp_w, arr, dep, E_init)


def run_B_smart(v2g, buy_w, v2gp_w, E_init, arr, dep):
    Pc, Pd, soc = solve_milp(v2g, buy_w, v2gp_w, E_init, v2g.E_max, False)
    return _kpi("B - Smart (no V2G)", v2g, Pc, Pd, soc, buy_w, v2gp_w, arr, dep, E_init)


def run_C_milp(v2g, buy_w, v2gp_w, E_init, arr, dep):
    Pc, Pd, soc = solve_milp(v2g, buy_w, v2gp_w, E_init, v2g.E_max, True)
    return _kpi("C - MILP Day-Ahead", v2g, Pc, Pd, soc, buy_w, v2gp_w, arr, dep, E_init)


def run_D_mpc(v2g, buy_w, v2gp_w, E_init, arr, dep):
    W = len(buy_w); dt = v2g.dt_h; s = E_init
    Pc_all = np.zeros(W); Pd_all = np.zeros(W); soc_all = np.zeros(W)
    for t in range(W):
        Pc_w, Pd_w, _ = solve_milp(v2g, buy_w[t:], v2gp_w[t:], s, v2g.E_max, True)
        pc = float(np.clip(Pc_w[0], 0, v2g.charge_power_kW))
        pd = float(np.clip(Pd_w[0], 0, v2g.discharge_power_kW))
        if pc > 1e-6 and pd > 1e-6:
            pc, pd = (0., pd) if v2gp_w[t] > buy_w[t] else (pc, 0.)
        s = float(np.clip(
            s + pc * v2g.eta_charge * dt - pd / v2g.eta_discharge * dt,
            v2g.E_min, v2g.E_max))
        Pc_all[t] = pc; Pd_all[t] = pd; soc_all[t] = s
    return _kpi("D - MPC (receding)", v2g, Pc_all, Pd_all, soc_all,
                buy_w, v2gp_w, arr, dep, E_init)


# =============================================================================
#  6. CONSOLE REPORT
# =============================================================================

def print_report(label, results):
    ref = results[0]["net_cost"]
    print(f"\n  ── {label} ──")
    print(f"  {'Scenario':<26} {'Net €/d':>8} {'Chg €/d':>8} "
          f"{'V2G €/d':>8} {'kWh out':>7} {'vs Dumb':>8}")
    print("  " + "-"*62)
    for r in results:
        print(f"  {r['label']:<26} {r['net_cost']:>8.4f} {r['charge_cost']:>8.4f} "
              f"{r['v2g_rev']:>8.4f} {r['v2g_kwh']:>7.2f} "
              f"{ref - r['net_cost']:>+8.4f}")


# =============================================================================
#  7. PLOT HELPERS
# =============================================================================

def _format_ax(ax, ylabel, title, display_start=12.0, ylim=None):
    pos  = np.arange(display_start, display_start + 25, 2)
    lbls = [f"{int(h % 24):02d}:00" for h in pos]
    ax.set_xlim(display_start, display_start + 24.0)
    ax.set_xticks(pos); ax.set_xticklabels(lbls, fontsize=8, rotation=35, ha="right")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left", pad=4)
    ax.grid(True, alpha=0.22, zorder=0)
    if ylim: ax.set_ylim(*ylim)


def _legend_below(ax, handles, ncol=4):
    ax.legend(handles=handles, fontsize=8, ncol=ncol,
              loc="upper center", bbox_to_anchor=(0.5, -0.40),
              framealpha=0.95, edgecolor="#CCCCCC")


def _vert_lines_wd(ax, arrival_h, departure_h, DS):
    """Weekday vertical reference lines — no text."""
    ax.axvline(_disp_x(arrival_h,   DS), color="#1B5E20", lw=1.1, ls=":", alpha=0.80, zorder=5)
    ax.axvline(_disp_x(departure_h, DS), color="#B71C1C", lw=1.1, ls=":", alpha=0.80, zorder=5)
    ax.axvline(_disp_x(0,           DS), color="#555555", lw=1.2, ls="--",alpha=0.65, zorder=5)


# =============================================================================
#  8. SEASON CHART — ALL 4 SCENARIOS OVERLAID
# =============================================================================

def plot_season_chart(v2g, season_label, buy_d, plug_d, hours_d,
                      results, is_weekend, arrival_h, departure_h, out):
    """
    3-panel chart: all 4 scenarios overlaid on one figure.
    Panel 1: Price + availability (gold shading)
    Panel 2: Charge/discharge power — solid=charge, dashed=V2G discharge below 0
    Panel 3: SoC trajectory — gradual ramps, % axis
    """
    DS = hours_d[0]

    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(15, 14),
        gridspec_kw={"height_ratios": [1.0, 1.8, 1.8], "hspace": 0.75})
    fig.patch.set_facecolor("#F8F9FA")

    dwell_str = ("Full 24 h depot dwell — unlimited charge/discharge (weekend)"
                 if is_weekend else
                 f"Arrival: {int(arrival_h):02d}:00  |  Departure: {int(departure_h):02d}:00")
    fig.suptitle(
        f"S.KOe COOL  —  {season_label}  |  All 4 Scenarios\n"
        f"{dwell_str}  |  Battery: {v2g.usable_capacity_kWh:.0f} kWh usable  |  "
        f"deg=0  |  TRU=0 kW",
        fontsize=11, fontweight="bold", y=0.99)

    # ── Panel 1: Price + availability ─────────────────────────────────────────
    for t in range(len(hours_d)):
        if plug_d[t] > 0.5:
            ax0.axvspan(hours_d[t], hours_d[t] + v2g.dt_h, color="gold",
                        alpha=0.22, lw=0, zorder=1)
    p_line, = ax0.step(hours_d, buy_d * 1000, where="post",
                       color=SC_COL["price"], lw=2.0, zorder=3,
                       label="Day-ahead price (EUR/MWh)")
    ax0.fill_between(hours_d, buy_d * 1000, step="post",
                     color=SC_COL["price"], alpha=0.10, zorder=2)
    if not is_weekend:
        _vert_lines_wd(ax0, arrival_h, departure_h, DS)

    leg0 = [p_line, mpatches.Patch(color="gold", alpha=0.5, label="Plugged-in window")]
    if not is_weekend:
        leg0 += [
            Line2D([0],[0], color="#1B5E20", ls=":", lw=1.3,
                   label=f"Arrival {int(arrival_h):02d}:00"),
            Line2D([0],[0], color="#B71C1C", ls=":", lw=1.3,
                   label=f"Departure {int(departure_h):02d}:00"),
            Line2D([0],[0], color="#555555", ls="--", lw=1.3,
                   label="Midnight 00:00"),
        ]
    _legend_below(ax0, leg0, ncol=5 if not is_weekend else 2)
    _format_ax(ax0, "EUR / MWh",
               f"(1) Day-Ahead Electricity Price + Plugged-In Availability", DS)

    # ── Panel 2: Power ────────────────────────────────────────────────────────
    leg1 = []
    for key, r in zip(["A","B","C","D"], results):
        col = SC_COL[key]; lbl = r["label"].split("(")[0].strip()
        h1, = ax1.step(hours_d, r["P_c_d"], where="post",
                       color=col, lw=2.0, label=f"{lbl}  charge")
        leg1.append(h1)
        if r["v2g_kwh"] > 0.05:
            h2, = ax1.step(hours_d, -r["P_d_d"], where="post",
                           color=col, lw=2.0, ls="--", alpha=0.85,
                           label=f"{lbl}  V2G discharge (shown -)")
            leg1.append(h2)
    ax1.axhline(0, color="black", lw=0.7)
    if not is_weekend:
        _vert_lines_wd(ax1, arrival_h, departure_h, DS)
    _legend_below(ax1, leg1, ncol=4)
    _format_ax(ax1, "Power (kW)",
               "(2) Charge / Discharge Power  "
               "[solid line = charging  |  dashed line below 0 = V2G]", DS)

    # ── Panel 3: SoC — gradual linear ramps ───────────────────────────────────
    for t in range(len(hours_d)):
        if plug_d[t] > 0.5:
            ax2.axvspan(hours_d[t], hours_d[t] + v2g.dt_h, color="gold",
                        alpha=0.12, lw=0, zorder=1)
    leg2 = []
    for key, r, ls in zip(["A","B","C","D"], results, ["-","-","-","--"]):
        col = SC_COL[key]; lbl = r["label"].split("(")[0].strip()
        xr, yr = soc_ramp(hours_d, r["soc_d"], r["E_init_pct"])
        h, = ax2.plot(xr, yr, color=col, lw=2.2, ls=ls, label=f"{lbl}  SoC (%)")
        leg2.append(h)
    ax2.axhline(v2g.soc_min_pct, color="#C62828", ls=":", lw=1.5, zorder=3)
    ax2.axhline(v2g.soc_max_pct, color="#0D47A1", ls=":", lw=1.5, zorder=3)
    if not is_weekend:
        _vert_lines_wd(ax2, arrival_h, departure_h, DS)
    leg2 += [
        Line2D([0],[0], color="#C62828", ls=":", lw=1.5,
               label=f"Cold-chain floor  {v2g.soc_min_pct:.0f}%"),
        Line2D([0],[0], color="#0D47A1", ls=":", lw=1.5,
               label=f"Departure target  {v2g.soc_max_pct:.0f}%"),
    ]
    _legend_below(ax2, leg2, ncol=3)
    _format_ax(ax2, "State of Charge (%)",
               "(3) Battery SoC Trajectory  [gradual ramp = real energy flow]",
               DS, ylim=(0, 112))
    ax2.set_xlabel("Time of Day", fontsize=9)

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved -> {out}")


# =============================================================================
#  9. KPI MULTI-TABLE CHART  (2x2 grid, one subtable per day type)
# =============================================================================

def plot_kpi_multi(all_res, v2g, arrival_h, departure_h, run_mpc, out):
    day_cfg = [
        ("winter_weekday", "Winter Weekday",  "#1565C0"),
        ("summer_weekday", "Summer Weekday",  "#E65100"),
        ("winter_weekend", "Winter Weekend",  "#6A1B9A"),
        ("summer_weekend", "Summer Weekend",  "#2E7D32"),
    ]
    sc_keys  = ["A", "B", "C", "D"]
    sc_short = ["A — Dumb", "B — Smart", "C — MILP", "D — MPC"]

    metrics = [
        ("Net cost (EUR/day)",     "net_cost",    "{:.4f}"),
        ("Charge cost (EUR/day)",  "charge_cost", "{:.4f}"),
        ("V2G revenue (EUR/day)",  "v2g_rev",     "{:.4f}"),
        ("V2G export (kWh/day)",   "v2g_kwh",     "{:.2f}"),
        ("Daily savings vs Dumb",  "savings_day", "{:+.4f}"),
        ("Annual savings (x365)",  "savings_ann", "EUR {:+,.0f}"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 13),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.25})
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        "S.KOe COOL — KPI Summary: All Scenarios x All Day Types\n"
        f"Weekday arrival {int(arrival_h):02d}:00 | Departure {int(departure_h):02d}:00 | "
        f"Weekend: full 24h | Battery {v2g.usable_capacity_kWh:.0f} kWh | deg=0 | TRU=0 kW",
        fontsize=13, fontweight="bold", y=1.01)

    for ax_i, (dt_key, dt_lbl, hdr_col) in enumerate(day_cfg):
        ax = axes[ax_i // 2][ax_i % 2]
        ax.axis("off")
        ax.set_title(f"  {dt_lbl}  ", fontsize=11, fontweight="bold",
                     color="white", pad=10,
                     bbox=dict(facecolor=hdr_col, edgecolor="none",
                               boxstyle="round,pad=0.4"))

        if dt_key not in all_res:
            continue
        res = all_res[dt_key]
        ref = res[0]["net_cost"]

        # Build cell data
        cell_data = []
        for mname, mkey, mfmt in metrics:
            row = [mname]
            for i, r in enumerate(res):
                if mkey == "savings_day":
                    v = ref - r["net_cost"]
                    row.append("—" if i == 0 else mfmt.format(v))
                elif mkey == "savings_ann":
                    v = (ref - r["net_cost"]) * 365
                    row.append("—" if i == 0 else mfmt.format(v))
                else:
                    v = r[mkey]
                    row.append(mfmt.format(v))
            cell_data.append(row)

        # D placeholder note if MPC not run
        d_note = sc_short if run_mpc else sc_short[:3] + ["D — (not run)"]

        tbl = ax.table(cellText=cell_data,
                       colLabels=["Metric"] + d_note,
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 2.3)

        # Header row styling
        hdr_bgs = ["#263238"] + [SC_COL[k] for k in sc_keys]
        for ci, hc in enumerate(hdr_bgs):
            cell = tbl[0, ci]
            cell.set_facecolor(hc)
            cell.set_text_props(color="white", fontweight="bold")

        # Body styling
        col_bgs = ["#F5F5F5", "#F5F5F5", "#E3F2FD", "#E0F7FA", "#FFF3E0"]
        for ri in range(1, len(cell_data) + 1):
            for ci in range(5):
                cell = tbl[ri, ci]
                cell.set_facecolor(col_bgs[ci])
                if ci == 0:
                    cell.set_text_props(fontweight="bold")
                txt = cell.get_text().get_text()
                if "+" in txt and ri >= 5:   # savings rows
                    cell.set_text_props(color="#1B5E20", fontweight="bold")

        for ri in range(len(cell_data) + 1):
            tbl[ri, 0].set_width(0.35)
            for ci in range(1, 5):
                tbl[ri, ci].set_width(0.155)

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved -> {out}")


# =============================================================================
#  10. FULL-YEAR SIMULATION
# =============================================================================

def run_full_year(v2g, csv_path, arrival_h, departure_h,
                  soc_init_pct=45.0, run_mpc=True):
    """Run A/B/C (and optionally D) for every day in the CSV."""
    df     = _load_csv_raw(csv_path)
    E_init = v2g.usable_capacity_kWh * soc_init_pct / 100.0
    dates  = sorted(df["date"].unique())
    n_days = len(dates)
    records = []

    print(f"\n  [Annual Sim] {n_days} days | "
          f"{'A+B+C+D' if run_mpc else 'A+B+C (MPC skipped)'}...")

    for i, d in enumerate(dates):
        day_df = df[df["date"] == d]
        if len(day_df) != 96:
            continue

        buy     = day_df["price"].values
        v2gp    = buy.copy()
        is_wknd = day_df.index[0].dayofweek >= 5
        month   = int(day_df.index[0].month)

        win, arr, dep, W, DS = get_window(v2g, arrival_h, departure_h, is_wknd)
        buy_w  = buy[win]
        v2gp_w = v2gp[win]

        try:
            A = run_A_dumb( v2g, buy_w, v2gp_w, W, E_init, arr, dep)
            B = run_B_smart(v2g, buy_w, v2gp_w,    E_init, arr, dep)
            C = run_C_milp( v2g, buy_w, v2gp_w,    E_init, arr, dep)
            D = run_D_mpc(  v2g, buy_w, v2gp_w,    E_init, arr, dep) if run_mpc else None
        except RuntimeError as e:
            print(f"    SKIP {d}: {e}")
            continue

        records.append({
            "date"     : pd.Timestamp(d),
            "month"    : month,
            "is_wknd"  : is_wknd,
            "is_winter": month in WINTER_MONTHS,
            "cost_A"   : A["net_cost"],
            "cost_B"   : B["net_cost"],
            "cost_C"   : C["net_cost"],
            "cost_D"   : D["net_cost"]  if D else np.nan,
            "rev_C"    : C["v2g_rev"],
            "rev_D"    : D["v2g_rev"]   if D else np.nan,
            "chg_C"    : C["charge_cost"],
            "kwh_C"    : C["v2g_kwh"],
        })

        if (i + 1) % 60 == 0 or i == n_days - 1:
            print(f"    {i+1}/{n_days} days processed ...")

    return pd.DataFrame(records)


# =============================================================================
#  11. ANNUAL RESULTS CHART
# =============================================================================

def plot_annual_results(df, run_mpc, out):
    """
    2x2 grid:
    (1) Cumulative net cost over year — all scenarios
    (2) Monthly savings vs Dumb — bar chart
    (3) Annual totals bar chart with savings annotations
    (4) Daily cost distribution — box plots
    """
    df = df.copy()
    df["date"]  = pd.to_datetime(df["date"])
    df["sav_B"] = df["cost_A"] - df["cost_B"]
    df["sav_C"] = df["cost_A"] - df["cost_C"]
    if run_mpc:
        df["sav_D"] = df["cost_A"] - df["cost_D"]

    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.32})
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        "S.KOe COOL — Full-Year Simulation Results  |  2025 SMARD DE/LU Day-Ahead Prices\n"
        "Weekdays: overnight dwell  |  Weekends: full 24h unlimited charge/discharge",
        fontsize=13, fontweight="bold", y=1.00)

    # ── (1) Cumulative cost ────────────────────────────────────────────────────
    ax1 = axes[0, 0]
    for key, ls in [("A","-"),("B","-"),("C","-"),("D","--")]:
        col_k = f"cost_{key}"
        if key == "D" and not run_mpc: continue
        if col_k not in df: continue
        lbl = f"{key} - {'Dumb' if key=='A' else 'Smart' if key=='B' else 'MILP' if key=='C' else 'MPC'}"
        ax1.plot(df["date"], df[col_k].cumsum(),
                 color=SC_COL[key], lw=2.5, ls=ls, label=lbl)
        total = df[col_k].sum()
        ax1.annotate(f"  EUR {total:,.0f}",
                     xy=(df["date"].iloc[-1], total),
                     xytext=(5, 0), textcoords="offset points",
                     fontsize=8, color=SC_COL[key], fontweight="bold", va="center")
    ax1.set_title("(1) Cumulative Net Charging Cost Over Year",
                  fontsize=10, fontweight="bold", loc="left")
    ax1.set_ylabel("Cumulative Cost (EUR)", fontsize=9)
    ax1.set_xlabel("Date", fontsize=9)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.22)

    # ── (2) Monthly savings ────────────────────────────────────────────────────
    ax2 = axes[0, 1]
    x  = np.arange(12); w = 0.25
    for offset, key, has_it in [(-w,"B",True), (0,"C",True), (w,"D",run_mpc)]:
        if not has_it: continue
        col_s = f"sav_{key}"
        vals  = [df[df["month"] == m][col_s].sum() for m in range(1, 13)]
        lbl   = f"{key} - {'Smart' if key=='B' else 'MILP' if key=='C' else 'MPC'}"
        ax2.bar(x + offset, vals, width=w, color=SC_COL[key],
                alpha=0.85, label=lbl, zorder=3)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(MONTHS, fontsize=9)
    ax2.set_title("(2) Monthly Savings vs Dumb Charging  (includes weekdays + weekends)",
                  fontsize=10, fontweight="bold", loc="left")
    ax2.set_ylabel("Monthly Savings (EUR)", fontsize=9)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.22, axis="y", zorder=0)

    # ── (3) Annual totals ─────────────────────────────────────────────────────
    ax3 = axes[1, 0]
    sc_list = [("A","Dumb"),("B","Smart"),("C","MILP")] + ([("D","MPC")] if run_mpc else [])
    ann_costs = [df[f"cost_{k}"].sum() for k, _ in sc_list]
    sc_labels = [f"{k}\n{n}" for k, n in sc_list]
    sc_cols   = [SC_COL[k] for k, _ in sc_list]

    bars = ax3.bar(range(len(sc_list)), ann_costs, color=sc_cols,
                   alpha=0.85, width=0.55, zorder=3)
    for bar, v in zip(bars, ann_costs):
        va_s = "bottom" if v >= 0 else "top"
        ax3.text(bar.get_x() + bar.get_width()/2,
                 v + (25 if v >= 0 else -25),
                 f"EUR {v:,.0f}", ha="center", va=va_s,
                 fontsize=9, fontweight="bold", color=bar.get_facecolor())

    ref_cost = ann_costs[0]
    for i, (cost, (key, _)) in enumerate(zip(ann_costs[1:], sc_list[1:]), 1):
        sav = ref_cost - cost
        if abs(sav) > 5:
            ax3.annotate(f"saves EUR {sav:+,.0f}/yr",
                         xy=(i, cost), xytext=(i, cost - 90),
                         ha="center", fontsize=8, color="#1B5E20", fontweight="bold",
                         arrowprops=dict(arrowstyle="-", color="#1B5E20", lw=0.5))

    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_xticks(range(len(sc_list))); ax3.set_xticklabels(sc_labels, fontsize=10)
    ax3.set_title("(3) Annual Net Charging Cost — Full Year (365 Days)",
                  fontsize=10, fontweight="bold", loc="left")
    ax3.set_ylabel("Annual Net Cost (EUR / year)", fontsize=9)
    ax3.grid(True, alpha=0.22, axis="y", zorder=0)

    # ── (4) Daily cost distribution box plots ────────────────────────────────
    ax4 = axes[1, 1]
    box_keys  = ["cost_A","cost_B","cost_C"] + (["cost_D"] if run_mpc else [])
    box_data  = [df[k].dropna().values for k in box_keys]
    box_lbls  = ["A - Dumb","B - Smart","C - MILP"] + (["D - MPC"] if run_mpc else [])
    box_cols  = [SC_COL[k[5]] for k in box_keys]

    bp = ax4.boxplot(box_data, patch_artist=True, labels=box_lbls,
                     medianprops=dict(color="black", lw=2.0),
                     flierprops=dict(marker=".", markersize=3, alpha=0.5))
    for patch, col in zip(bp["boxes"], box_cols):
        patch.set_facecolor(col); patch.set_alpha(0.75)

    ax4.axhline(0, color="#B71C1C", lw=1.5, ls="--", alpha=0.7,
                label="Zero cost line (trailer earns money below this)")
    ax4.set_title("(4) Daily Net Cost Distribution — Variance Across All 365 Days",
                  fontsize=10, fontweight="bold", loc="left")
    ax4.set_ylabel("Daily Net Cost (EUR / day)", fontsize=9)
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.22, axis="y")

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved -> {out}")


# =============================================================================
#  12. PRICE PROFILES CHART  (2x2 analysis)
# =============================================================================

def plot_price_profiles(csv_path, out):
    """
    2x2 grid:
    (1) Winter vs Summer weekday avg price profile (24h overlay)
    (2) Winter weekday vs Winter weekend price profile
    (3) Monthly average price (bar chart, error bars = daily std)
    (4) Daily price spread (max-min EUR/MWh) by month = V2G arbitrage potential
    """
    df = _load_csv_raw(csv_path)
    h_axis = np.arange(96) * 0.25

    print("  Loading 4 season profiles for price analysis ...")
    w_wd = load_avg_profile(csv_path, WINTER_MONTHS, False)
    s_wd = load_avg_profile(csv_path, SUMMER_MONTHS, False)
    w_we = load_avg_profile(csv_path, WINTER_MONTHS, True)
    s_we = load_avg_profile(csv_path, SUMMER_MONTHS, True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 11),
                             gridspec_kw={"hspace": 0.48, "wspace": 0.30})
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        "S.KOe COOL — Electricity Price Analysis  |  2025 SMARD DE/LU Day-Ahead\n"
        "Price structure driving smart charging & V2G arbitrage decisions",
        fontsize=13, fontweight="bold", y=1.01)

    HTICKS = np.arange(0, 25, 2)
    HLBLS  = [f"{int(h):02d}:00" for h in HTICKS]
    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]

    def style_profile_ax(ax, title):
        ax.set_xticks(HTICKS); ax.set_xticklabels(HLBLS, fontsize=8, rotation=35, ha="right")
        ax.set_xlim(0, 24); ax.set_ylabel("Price (EUR/MWh)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
        ax.grid(True, alpha=0.22)
        ax.set_xlabel("Hour of Day", fontsize=9)

    # ── (1) Winter vs Summer weekday ──────────────────────────────────────────
    ax1 = axes[0, 0]
    ax1.step(h_axis, w_wd * 1000, where="post", color="#1565C0",
             lw=2.3, label="Winter weekday (Oct–Mar)")
    ax1.step(h_axis, s_wd * 1000, where="post", color="#E65100",
             lw=2.3, label="Summer weekday (Apr–Sep)")
    ax1.fill_between(h_axis, w_wd*1000, step="post", color="#1565C0", alpha=0.10)
    ax1.fill_between(h_axis, s_wd*1000, step="post", color="#E65100", alpha=0.10)

    # Annotate peak difference
    spread = (w_wd - s_wd) * 1000
    pk_slot = int(np.argmax(np.abs(spread)))
    ax1.annotate(
        f"Delta: {spread[pk_slot]:+.0f} EUR/MWh\nat {pk_slot*0.25:.1f}h",
        xy=(h_axis[pk_slot], max(w_wd[pk_slot], s_wd[pk_slot]) * 1000),
        xytext=(h_axis[pk_slot] + 1.5, max(w_wd[pk_slot], s_wd[pk_slot]) * 1000 + 5),
        fontsize=7.5, color="#333", fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="#333", lw=0.7))

    ax1.legend(fontsize=9)
    style_profile_ax(ax1, "(1) Winter vs Summer — Weekday Avg Price (24h)")

    # ── (2) Weekday vs Weekend (winter) ───────────────────────────────────────
    ax2 = axes[0, 1]
    ax2.step(h_axis, w_wd * 1000, where="post", color="#1565C0",
             lw=2.3, label="Winter weekday")
    ax2.step(h_axis, w_we * 1000, where="post", color="#6A1B9A",
             lw=2.3, ls="--", label="Winter weekend")
    ax2.step(h_axis, s_we * 1000, where="post", color="#C62828",
             lw=1.8, ls=":", label="Summer weekend", alpha=0.75)
    ax2.fill_between(h_axis, w_wd*1000, step="post", color="#1565C0", alpha=0.08)
    ax2.fill_between(h_axis, w_we*1000, step="post", color="#6A1B9A", alpha=0.08)

    ax2.text(0.97, 0.95,
             "Weekend prices typically\nlower & flatter — less\npeak arbitrage vs weekday",
             ha="right", va="top", fontsize=7.5, color="#333",
             transform=ax2.transAxes, style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#CCCCCC", alpha=0.9))
    ax2.legend(fontsize=9)
    style_profile_ax(ax2, "(2) Weekday vs Weekend — Winter Price Profile")

    # ── (3) Monthly average price ─────────────────────────────────────────────
    ax3 = axes[1, 0]
    monthly_avg = df.groupby("month")["price"].mean() * 1000
    monthly_std = df.groupby("month")["price"].std()  * 1000

    bar_cols = ["#1565C0" if m in WINTER_MONTHS else "#E65100"
                for m in range(1, 13)]
    bars = ax3.bar(range(12), [monthly_avg.get(m, 0) for m in range(1, 13)],
                   color=bar_cols, alpha=0.82, width=0.65, zorder=3,
                   yerr=[monthly_std.get(m, 0) for m in range(1, 13)],
                   error_kw=dict(lw=1.2, capsize=3, capthick=1.2, ecolor="#333"))
    ax3.set_xticks(range(12)); ax3.set_xticklabels(MONTHS, fontsize=9)
    ax3.set_ylabel("Avg Day-Ahead Price (EUR/MWh)", fontsize=9)
    ax3.set_title("(3) Monthly Average Electricity Price (bars ± 1σ daily std)",
                  fontsize=10, fontweight="bold", loc="left")
    ax3.legend(handles=[
        mpatches.Patch(color="#1565C0", alpha=0.82, label="Winter months (Oct–Mar)"),
        mpatches.Patch(color="#E65100", alpha=0.82, label="Summer months (Apr–Sep)"),
    ], fontsize=9, loc="upper right")
    ax3.grid(True, alpha=0.22, axis="y", zorder=0)
    for i, v in enumerate([monthly_avg.get(m, 0) for m in range(1, 13)]):
        ax3.text(i, v + 1.5, f"{v:.0f}", ha="center", va="bottom",
                 fontsize=7.5, fontweight="bold")

    # ── (4) Daily price spread by month = V2G arbitrage potential ─────────────
    ax4 = axes[1, 1]
    daily_max = df.groupby("date")["price"].max()
    daily_min = df.groupby("date")["price"].min()
    daily_spread = (daily_max - daily_min) * 1000
    daily_month  = df.groupby("date")["month"].first()
    spread_by_month = [
        daily_spread[daily_month == m].values for m in range(1, 13)
    ]

    bp = ax4.boxplot(spread_by_month, patch_artist=True,
                     medianprops=dict(color="black", lw=2.0),
                     flierprops=dict(marker=".", markersize=3, alpha=0.4))
    for i, (patch, m) in enumerate(zip(bp["boxes"], range(1, 13))):
        col = "#1565C0" if m in WINTER_MONTHS else "#E65100"
        patch.set_facecolor(col); patch.set_alpha(0.72)
    for whisker in bp["whiskers"]:
        whisker.set_linewidth(1.2)
    for cap in bp["caps"]:
        cap.set_linewidth(1.2)

    ax4.set_xticks(range(1, 13)); ax4.set_xticklabels(MONTHS, fontsize=9)
    ax4.set_ylabel("Daily Price Spread (EUR/MWh)", fontsize=9)
    ax4.set_title("(4) Daily Price Spread by Month  —  V2G Arbitrage Potential",
                  fontsize=10, fontweight="bold", loc="left")
    ax4.legend(handles=[
        mpatches.Patch(color="#1565C0", alpha=0.82, label="Winter — higher V2G potential"),
        mpatches.Patch(color="#E65100", alpha=0.82, label="Summer — lower V2G potential"),
    ], fontsize=9, loc="upper right")
    ax4.grid(True, alpha=0.22, axis="y")

    ymax = ax4.get_ylim()[1]
    ax4.text(6.5, ymax * 0.90,
             "Larger spread = more V2G profit per cycle\n"
             "Median spread shows typical arbitrage ceiling",
             ha="center", fontsize=8, style="italic", color="#555555",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#CCCCCC", alpha=0.85))

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved -> {out}")


# =============================================================================
#  13. MAIN
# =============================================================================

def main():
    print("\n" + "="*65)
    print("  S.KOe COOL — V2G Optimisation Suite (v4)")
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
    print(f"  Battery : {v2g.battery_capacity_kWh} kWh total | "
          f"{v2g.usable_capacity_kWh} kWh usable | "
          f"P_max {v2g.charge_power_kW:.0f} kW charge+discharge | deg=0")

    def ask(prompt, default, lo, hi):
        raw = input(f"  {prompt} [default {default}]: ").strip()
        try:
            val = float(raw); assert lo <= val <= hi; return val
        except Exception:
            print(f"  -> Using default {default}"); return float(default)

    print("\n" + "-"*55)
    departure_h  = ask("Weekday DEPARTURE hour 0-23",  6,  0, 23)
    arrival_h    = ask("Weekday ARRIVAL   hour 0-23", 16,  0, 23)
    soc_init_pct = ask("Arrival SoC % (20-100)",       45, 20, 100)
    raw_mpc = input("  Run MPC (D) for avg profiles — adds ~2 min [Y/n]: ").strip().lower()
    run_mpc_avg = raw_mpc not in ("n", "no")
    raw_ann = input("  Run full 365-day annual simulation [Y/n]: ").strip().lower()
    run_annual = raw_ann not in ("n", "no")
    if run_annual:
        raw_mpc_ann = input("  Include MPC in annual sim — SLOW ~15-20 min [y/N]: ").strip().lower()
        run_mpc_ann = raw_mpc_ann in ("y", "yes")
    else:
        run_mpc_ann = False

    print(f"\n  Weekday: {int(arrival_h):02d}:00 -> {int(departure_h):02d}:00  "
          f"| Arrival SoC: {soc_init_pct:.0f}%  | Departure target: 100%")
    print("  Weekend: full 24h, all 96 slots plugged, unlimited charge/discharge")
    print("-"*55)

    E_init = v2g.usable_capacity_kWh * soc_init_pct / 100.0

    # ── Season profiles + avg-day charts ─────────────────────────────────────
    print("\n  Loading price profiles ...")
    _load_csv_raw(csv_path)   # warm cache

    DAY_CONFIGS = [
        ("winter_weekday", "Winter Weekday", WINTER_MONTHS, False, 130,
         "v2g_winter_weekday.png"),
        ("summer_weekday", "Summer Weekday", SUMMER_MONTHS, False, 131,
         "v2g_summer_weekday.png"),
        ("winter_weekend", "Winter Weekend", WINTER_MONTHS, True,   52,
         "v2g_winter_weekend.png"),
        ("summer_weekend", "Summer Weekend", SUMMER_MONTHS, True,   52,
         "v2g_summer_weekend.png"),
    ]

    all_season_results = {}

    for (dt_key, dt_lbl, months, is_wknd, dpyr, out_file) in DAY_CONFIGS:
        print(f"\n  ── {dt_lbl}  ({dpyr} days/yr) ──")
        buy  = load_avg_profile(csv_path, months, is_wknd)
        v2gp = buy.copy()

        win, arr, dep, W, DS = get_window(v2g, arrival_h, departure_h, is_wknd)
        buy_w  = buy[win]; v2gp_w = v2gp[win]
        buy_d, plug_d, hours_d = build_display(v2g, buy, arrival_h, departure_h,
                                                is_wknd, DS)

        print("    A (Dumb)...", end=" ", flush=True)
        A = run_A_dumb(v2g, buy_w, v2gp_w, W, E_init, arr, dep)
        print("B (Smart)...", end=" ", flush=True)
        B = run_B_smart(v2g, buy_w, v2gp_w, E_init, arr, dep)
        print("C (MILP)...", end=" ", flush=True)
        C = run_C_milp(v2g, buy_w, v2gp_w, E_init, arr, dep)
        if run_mpc_avg:
            print(f"D (MPC, {W} sub-problems)...", flush=True)
            D = run_D_mpc(v2g, buy_w, v2gp_w, E_init, arr, dep)
        else:
            print("D (MPC) skipped.")
            D = C   # proxy — labelled below

        if not run_mpc_avg:
            D = {**C, "label": "D - MPC (not run — showing C)"}

        results = [A, B, C, D]
        all_season_results[dt_key] = results
        print_report(dt_lbl, results)

        print(f"  Plotting {out_file} ...")
        plot_season_chart(v2g, dt_lbl, buy_d, plug_d, hours_d,
                          results, is_wknd, arrival_h, departure_h, out_file)

    # ── KPI multi-table ────────────────────────────────────────────────────────
    print("\n  Plotting v2g_KPI_multi.png ...")
    plot_kpi_multi(all_season_results, v2g, arrival_h, departure_h,
                   run_mpc_avg, "v2g_KPI_multi.png")

    # ── Price analysis ─────────────────────────────────────────────────────────
    print("  Plotting v2g_price_profiles.png ...")
    plot_price_profiles(csv_path, "v2g_price_profiles.png")

    # ── Annual simulation ──────────────────────────────────────────────────────
    if run_annual:
        annual_df = run_full_year(v2g, csv_path, arrival_h, departure_h,
                                  soc_init_pct, run_mpc_ann)
        print("\n  Plotting v2g_annual_results.png ...")
        plot_annual_results(annual_df, run_mpc_ann, "v2g_annual_results.png")

        ref  = annual_df["cost_A"].sum()
        c_tot = annual_df["cost_C"].sum()
        print(f"\n{'='*65}")
        print(f"  ANNUAL SUMMARY (Scenario C — MILP Day-Ahead):")
        print(f"    Annual cost  Dumb : EUR {ref:>8,.0f}")
        print(f"    Annual cost  MILP : EUR {c_tot:>8,.0f}")
        print(f"    Annual V2G rev   : EUR {annual_df['rev_C'].sum():>8,.0f}")
        print(f"    Annual savings   : EUR {ref - c_tot:>+8,.0f}")
        print(f"{'='*65}")
    else:
        print("\n  Annual simulation skipped.")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n  Output files generated:")
    files = ["v2g_winter_weekday.png", "v2g_summer_weekday.png",
             "v2g_winter_weekend.png", "v2g_summer_weekend.png",
             "v2g_KPI_multi.png", "v2g_price_profiles.png"]
    if run_annual:
        files.append("v2g_annual_results.png")
    for f in files:
        print(f"    {f}")
    print()


if __name__ == "__main__":
    main()