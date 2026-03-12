#!/usr/bin/env python3
"""
S.KOe COOL — Day-Ahead MILP + Receding-Horizon MPC V2G Optimisation
Schmitz Cargobull AG | 2025
Based on: Biedenbach & Strunz (2024), Agora Verkehrswende (2025)

Electricity prices: SMARD.de 2025 real DE/LU 15-min day-ahead spot data

Changes vs Prototype1:
  1. True MILP binary mutex variables (z_c, z_d) in _solve_milp_window
  2. Battery upper bound = 100 %  (soc_max_pct = 100.0, E_max = 60 kWh)
  3. Departure target fixed at 100 % SoC
  4. TRU = 0 kW everywhere (trailer stationary at depot)
  5. No heuristic fallback — solver failure raises RuntimeError
  6. Single clean _load_smard_csv implementation (duplicate removed)
  7. No module-level v2g_global — v2g passed explicitly to all functions
  8. All reference-card / legend text updated to reflect the 100 % model
"""

from __future__ import annotations
import warnings
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass
from typing import Optional, Dict, List
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 0 – ABBREVIATION LEGEND PNG
# ═══════════════════════════════════════════════════════════════════════════════

def generate_abbreviation_legend(out: str = "abbreviation_legend.png") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 11))
    fig.patch.set_facecolor("#F7F9FC")
    fig.suptitle(
        "V2G Optimisation — Abbreviation & Symbol Reference Card",
        fontsize=15, fontweight="bold", color="#1A237E", y=0.97
    )

    ax = axes[0]
    ax.set_facecolor("#EEF2FF")
    ax.axis("off")
    ax.text(0.5, 0.97, "VARIABLES & SYMBOLS", ha="center", va="top",
            fontsize=12, fontweight="bold", color="#1A237E",
            transform=ax.transAxes)

    variables = [
        ("P_c",          "Charging power (kW) — power drawn from grid into battery"),
        ("P_d",          "Discharging power (kW) — power fed from battery to grid (V2G)"),
        ("z_c",          "Binary mutex var: 1 = charging active in slot t"),
        ("z_d",          "Binary mutex var: 1 = discharging active in slot t"),
        ("E / SoC",      "State of Charge (kWh) — energy currently stored in battery"),
        ("E_min",        "Minimum usable energy (kWh) — cold-chain floor (SoC 20 %)"),
        ("E_max",        "Maximum usable energy (kWh) — full charge ceiling (SoC 100 %)"),
        ("eta_c",        "Charge efficiency (dimensionless, 0.92)"),
        ("eta_d",        "Discharge efficiency (dimensionless, 0.92)"),
        ("dt",           "Time step duration (hours, 0.25 = 15 min)"),
        ("T / N",        "Total number of time slots (96 slots = 24 h)"),
        ("t",            "Time slot index (0 to 95)"),
        ("h",            "Hour of day (0.0 to 23.75)"),
        ("buy[t]",       "Buy/import price at slot t (EUR/kWh)"),
        ("v2g[t]",       "V2G sell price at slot t (EUR/kWh)"),
        ("deg",          "Degradation cost (EUR/kWh cycled) — battery wear"),
        ("TRU",          "Transport Refrigeration Unit (kW) — zero while stationary"),
        ("plugged[t]",   "Availability flag: 1 = truck at depot & plugged in"),
        ("p_c_max",      "Maximum allowed charging power (kW) — 22 kW"),
        ("p_d_max",      "Maximum allowed V2G discharge power (kW) — 22 kW"),
        ("W",            "MILP / MPC optimisation horizon window length (slots)"),
        ("E_init",       "Battery energy at start of optimisation window (kWh)"),
        ("E_fin",        "Minimum required energy at end of window (kWh)"),
    ]

    y = 0.91
    for abbr, desc in variables:
        ax.text(0.03, y, f"  {abbr}", ha="left", va="top", fontsize=8.5,
                fontweight="bold", color="#C62828", transform=ax.transAxes)
        words = desc.split()
        line, lines = "", []
        for w in words:
            if len(line + w) > 48:
                lines.append(line.strip()); line = w + " "
            else:
                line += w + " "
        lines.append(line.strip())
        ax.text(0.30, y, lines[0], ha="left", va="top", fontsize=7.8,
                color="#212121", transform=ax.transAxes)
        for i, l in enumerate(lines[1:], 1):
            ax.text(0.30, y - i * 0.018, l, ha="left", va="top", fontsize=7.8,
                    color="#212121", transform=ax.transAxes)
        y -= 0.040 + max(0, (len(lines) - 1) * 0.018)

    ax = axes[1]
    ax.set_facecolor("#E8F5E9")
    ax.axis("off")
    ax.text(0.5, 0.97, "SCENARIOS", ha="center", va="top",
            fontsize=12, fontweight="bold", color="#1B5E20",
            transform=ax.transAxes)

    scenarios = [
        ("A - Dumb",            "#AAAAAA",
         "Charges at full power on arrival.\nNo price awareness. No V2G. Baseline."),
        ("B - Smart (no V2G)",  "#2196F3",
         "MILP shifts charging to cheapest slots.\nNever discharges. Minimal battery wear."),
        ("C - MILP Day-Ahead",  "#00BCD4",
         "Full 24h MILP at 00:00, perfect forecast.\nCharges cheap, discharges at peak."),
        ("D - MPC Perfect",     "#FF7700",
         "Receding-horizon MPC, re-solves every\n15-min slot. No noise. Near-optimal."),
    ]

    y = 0.91
    for sc_label, col, desc in scenarios:
        patch = mpatches.FancyBboxPatch((0.02, y - 0.012), 0.06, 0.030,
                                        boxstyle="round,pad=0.005",
                                        facecolor=col, edgecolor="white",
                                        transform=ax.transAxes, clip_on=False)
        ax.add_patch(patch)
        ax.text(0.11, y + 0.005, sc_label, ha="left", va="top", fontsize=8.5,
                fontweight="bold", color="#1B5E20", transform=ax.transAxes)
        for i, line in enumerate(desc.split("\n")):
            ax.text(0.11, y - 0.018 - i * 0.018, line, ha="left", va="top",
                    fontsize=7.5, color="#333333", transform=ax.transAxes)
        y -= 0.13

    y -= 0.04
    ax.text(0.5, y, "COST / REVENUE TERMS", ha="center", va="top",
            fontsize=12, fontweight="bold", color="#1B5E20",
            transform=ax.transAxes)
    y -= 0.06

    cost_terms = [
        ("Net Cost (EUR/day)",         "= Charge cost - V2G revenue + Deg cost"),
        ("Charge Cost (EUR/day)",      "= sum_t  buy[t] * P_c[t] * dt"),
        ("V2G Revenue (EUR/day)",      "= sum_t  v2g[t] * P_d[t] * dt"),
        ("Degradation Cost (EUR/day)", "= sum_t  deg * (P_c[t]+P_d[t]) * dt"),
        ("Savings vs A (EUR/day)",     "= Net Cost(A) - Net Cost(scenario)"),
        ("Annual Savings (EUR/yr)",    "= Savings/day x 365"),
    ]
    for term, formula in cost_terms:
        ax.text(0.03, y, f"* {term}", ha="left", va="top", fontsize=8.0,
                fontweight="bold", color="#C62828", transform=ax.transAxes)
        ax.text(0.03, y - 0.020, f"    {formula}", ha="left", va="top",
                fontsize=7.5, color="#333333", transform=ax.transAxes)
        y -= 0.050

    ax = axes[2]
    ax.set_facecolor("#FFF8E1")
    ax.axis("off")
    ax.text(0.5, 0.97, "SEASONS & DWELL MODES", ha="center", va="top",
            fontsize=12, fontweight="bold", color="#E65100",
            transform=ax.transAxes)

    seasons = [
        ("Winter WD",  "Oct-Mar, Mon-Fri. ~130 days/year"),
        ("Summer WD",  "Apr-Sep, Mon-Fri. ~131 days/year"),
        ("Winter WE",  "Oct-Mar, Sat-Sun. ~52 days/year"),
        ("Summer WE",  "Apr-Sep, Sat-Sun. ~52 days/year"),
        ("Extended",   "Plugged: 21:00-07:00 + 12:00-18:00 = 16h/day"),
        ("NightOnly",  "Plugged: 21:00-07:00 only = 10h/day"),
        ("Weekend",    "Plugged: 00:00-24:00 = fully available 24h/day"),
        ("DayTrip",    "Departs 07:00, returns 17:00 -> plugged 17:00-07:00"),
    ]
    y = 0.90
    for term, desc in seasons:
        ax.text(0.03, y, f"  {term}:", ha="left", va="top", fontsize=8.5,
                fontweight="bold", color="#BF360C", transform=ax.transAxes)
        ax.text(0.30, y, desc, ha="left", va="top", fontsize=8,
                color="#333333", transform=ax.transAxes)
        y -= 0.048

    ax.text(0.5, y - 0.01, "HARDWARE (S.KOe COOL)", ha="center", va="top",
            fontsize=12, fontweight="bold", color="#E65100",
            transform=ax.transAxes)
    y -= 0.065

    hardware = [
        ("Battery capacity",  "70 kWh total / 60 kWh usable (SoC 20-100 %)"),
        ("Max charge",        "22 kW (ISO 15118 AC)"),
        ("Max V2G discharge", "22 kW (ISO 15118-2 V2G)"),
        ("Charge eta",        "0.92 (92% one-way efficiency AC->DC)"),
        ("Discharge eta",     "0.92 (92% one-way efficiency DC->AC)"),
        ("TRU load",          "0 kW (trailer stationary at depot, cooling off)"),
        ("Arrival SoC",       "User-defined (default 45%)"),
        ("Departure SoC",     "100% = 60 kWh (full charge, cold-chain requirement)"),
        ("deg cost default",  "0.02 EUR/kWh cycled (LFP cell ageing estimate)"),
    ]
    for term, desc in hardware:
        ax.text(0.03, y, f"  {term}:", ha="left", va="top", fontsize=8.5,
                fontweight="bold", color="#BF360C", transform=ax.transAxes)
        ax.text(0.40, y, desc, ha="left", va="top", fontsize=8,
                color="#333333", transform=ax.transAxes)
        y -= 0.048

    ax.text(0.5, y - 0.01, "SOLVER & DATA", ha="center", va="top",
            fontsize=12, fontweight="bold", color="#E65100",
            transform=ax.transAxes)
    y -= 0.065

    solver_info = [
        ("MILP solver",    "scipy HiGHS (via scipy.optimize.milp)"),
        ("Mutex method",   "True binary z_c, z_d variables"),
        ("Objective",      "min sum(buy*P_c - v2g*P_d + deg*(P_c+P_d))*dt"),
        ("Constraints",    "SoC dynamics, power bounds, binary mutex, E_fin=100%"),
        ("MPC principle",  "Receding horizon: solve full remaining day, apply first action"),
        ("Price data",     "SMARD.de 2025 DE/LU 15-min day-ahead spot (EUR/MWh)"),
        ("Fallback",       "None - RuntimeError raised on solver failure"),
    ]
    for term, desc in solver_info:
        ax.text(0.03, y, f"  {term}:", ha="left", va="top", fontsize=8.5,
                fontweight="bold", color="#BF360C", transform=ax.transAxes)
        words = desc.split()
        line, lines = "", []
        for w in words:
            if len(line + w) > 42:
                lines.append(line.strip()); line = w + " "
            else:
                line += w + " "
        lines.append(line.strip())
        ax.text(0.40, y, lines[0], ha="left", va="top", fontsize=8,
                color="#333333", transform=ax.transAxes)
        for i, l in enumerate(lines[1:], 1):
            ax.text(0.40, y - i * 0.018, l, ha="left", va="top", fontsize=8,
                    color="#333333", transform=ax.transAxes)
        y -= 0.048 + max(0, (len(lines) - 1) * 0.018)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Abbreviation legend saved -> {out}")


def generate_equations_card(out: str = "equations_reference.png") -> None:
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#F0F4F8")
    fig.suptitle(
        "V2G Optimisation — Core Equations: True Binary-Mutex MILP & Receding-Horizon MPC",
        fontsize=16, fontweight="bold", color="#0D1B2A", y=0.98
    )

    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                  top=0.93, bottom=0.04, left=0.03, right=0.97)

    BOX = dict(boxstyle="round,pad=0.5", facecolor="white",
               edgecolor="#90A4AE", linewidth=1.2)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor("#E8EAF6"); ax.axis("off")
    ax.text(0.5, 0.97, "1 MILP - Objective Function", ha="center", va="top",
            fontsize=11, fontweight="bold", color="white",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1A237E"))

    lines = [
        (0.5, 0.86, "min J = sum_t [ C_buy(t) + C_deg(t) - R_v2g(t) ] * dt",
         10.5, "bold", "#B71C1C"),
        (0.5, 0.76, "where:", 9, "bold", "#1A237E"),
        (0.07, 0.68, "C_buy(t)  =  buy[t] * P_c[t]",          9, "normal", "#212121"),
        (0.60, 0.68, "<- grid import cost at slot t",           8.5, "italic", "#555555"),
        (0.07, 0.59, "R_v2g(t)  =  v2g[t] * P_d[t]",          9, "normal", "#212121"),
        (0.60, 0.59, "<- V2G revenue at slot t",                8.5, "italic", "#555555"),
        (0.07, 0.50, "C_deg(t)  =  deg * (P_c[t] + P_d[t])",  9, "normal", "#212121"),
        (0.60, 0.50, "<- battery wear cost",                    8.5, "italic", "#555555"),
        (0.07, 0.38, "buy[t]    day-ahead spot price  (EUR/kWh)", 8.5, "normal", "#333333"),
        (0.07, 0.30, "v2g[t]    V2G sell price  (EUR/kWh)",       8.5, "normal", "#333333"),
        (0.07, 0.22, "deg       degradation cost  (EUR/kWh cycled)", 8.5, "normal", "#333333"),
        (0.07, 0.14, "dt        time step = 0.25 h  (15 min)",  8.5, "normal", "#333333"),
        (0.07, 0.06, "t         slot index  0 to 95  (96 slots = 24 h)", 8.5, "normal", "#333333"),
    ]
    for x, y, txt, fs, fw, col in lines:
        ax.text(x, y, txt, ha="left", va="top", fontsize=fs,
                fontweight="bold" if fw == "bold" else "normal",
                color=col, transform=ax.transAxes,
                bbox=BOX if y == 0.86 else None)

    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor("#E8F5E9"); ax.axis("off")
    ax.text(0.5, 0.97, "2 MILP - Constraints", ha="center", va="top",
            fontsize=11, fontweight="bold", color="white",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1B5E20"))

    constraints = [
        ("(i)  SoC Dynamics  [energy balance]",                    "#1B5E20", 0.88),
        ("e[t] = e[t-1]  +  eta_c * P_c[t] * dt",                 "#B71C1C", 0.80),
        ("            -  (1/eta_d) * P_d[t] * dt  -  TRU[t] * dt","#B71C1C", 0.73),
        ("      TRU[t] = 0  (trailer stationary at depot)",        "#555555", 0.66),
        ("(ii)  Power Bounds  [hardware limits]",                  "#1B5E20", 0.57),
        ("0  <=  P_c[t]  <=  p_c_max * plugged[t]",               "#333333", 0.50),
        ("0  <=  P_d[t]  <=  p_d_max * plugged[t]",               "#333333", 0.43),
        ("(iii)  SoC Bounds  [battery safety]",                    "#1B5E20", 0.34),
        ("E_min  <=  e[t]  <=  E_max = 60 kWh  for all t",        "#333333", 0.27),
        ("(iv)  TRUE Binary Mutex  [z_c, z_d in {0,1}]",          "#1B5E20", 0.18),
        ("P_c[t]  <=  p_c_max * z_c[t]",                          "#333333", 0.12),
        ("P_d[t]  <=  p_d_max * z_d[t]",                          "#333333", 0.07),
        ("z_c[t]  +  z_d[t]  <=  1",                              "#C62828", 0.02),
    ]
    for txt, col, y in constraints:
        fw = "bold" if txt.startswith("(") else "normal"
        ax.text(0.05, y, txt, ha="left", va="top",
                fontsize=8.8, fontweight=fw, color=col,
                transform=ax.transAxes)

    ax.text(0.5, -0.04,
            "(v)  Departure SoC:  e[N-1]  >=  E_fin = 60 kWh  (100%)",
            ha="center", va="top", fontsize=8.8,
            fontweight="bold", color="#B71C1C",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#FFEBEE", edgecolor="#B71C1C"))

    ax = fig.add_subplot(gs[0, 2])
    ax.set_facecolor("#FFF3E0"); ax.axis("off")
    ax.text(0.5, 0.97, "3 MILP - Variable Layout (HiGHS MILP solver)",
            ha="center", va="top", fontsize=11, fontweight="bold",
            color="white", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E65100"))

    ax.text(0.5, 0.87, "Decision vector  x  of length  5W:",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color="#E65100", transform=ax.transAxes)

    segments = [
        ("#BBDEFB", "x[0..W-1]\n= P_c(0)..P_c(W-1)\nCharging power\n(continuous kW)"),
        ("#C8E6C9", "x[W..2W-1]\n= P_d(0)..P_d(W-1)\nDischarge power\n(continuous kW)"),
        ("#FFCCBC", "x[2W..3W-1]\n= e(0)..e(W-1)\nSoC trajectory\n(continuous kWh)"),
        ("#E1BEE7", "x[3W..4W-1]\n= z_c(0)..z_c(W-1)\nCharge mutex\n(binary {0,1})"),
        ("#F8BBD0", "x[4W..5W-1]\n= z_d(0)..z_d(W-1)\nDischarge mutex\n(binary {0,1})"),
    ]
    for i, (fc, txt) in enumerate(segments):
        x0 = 0.02 + i * 0.195
        rect = mpatches.FancyBboxPatch((x0, 0.50), 0.17, 0.27,
                                        boxstyle="round,pad=0.01",
                                        facecolor=fc, edgecolor="#78909C",
                                        transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(x0 + 0.085, 0.635, txt, ha="center", va="center",
                fontsize=7.0, color="#212121", transform=ax.transAxes)

    ax.text(0.5, 0.46, "W = remaining slots in horizon  (W = 96 at t=0)",
            ha="center", va="top", fontsize=8.5, color="#555555",
            transform=ax.transAxes)

    solver_notes = [
        "* Solver: scipy HiGHS  (scipy.optimize.milp)",
        "* integrality: 0=continuous for P_c,P_d,e; 1=integer for z_c,z_d",
        "* Constraint matrix A: sparse (lil_matrix -> csc_matrix)",
        "* Bounds: P_c,P_d in [0, p_max*plugged], e in [E_min, E_max]",
        "* z_c, z_d in [0, 1]  (integer -> effectively binary)",
        "* Time limit: 60 s per window",
        "* Fallback: NONE - RuntimeError raised on solver failure",
    ]
    y = 0.38
    for note in solver_notes:
        ax.text(0.04, y, note, ha="left", va="top", fontsize=8.2,
                color="#333333", transform=ax.transAxes)
        y -= 0.055

    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor("#EDE7F6"); ax.axis("off")
    ax.text(0.5, 0.97, "4 MPC - Receding-Horizon Algorithm",
            ha="center", va="top", fontsize=11, fontweight="bold",
            color="white", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#4A148C"))

    steps = [
        ("STEP 1 - FORECAST", "#4A148C",
         "Horizon = full remaining day  [t .. 95]\n"
         "buy_fc = buy[t:],   v2g_fc = v2g[t:]\n"
         "Window size W = 96 - t  (shrinks each slot)"),
        ("STEP 2 - SOLVE MILP (with binary mutex)", "#4A148C",
         "Run true binary-mutex MILP over W slots\n"
         "E_init = current real SoC\n"
         "E_fin  = 60 kWh (100%, always at slot 95)"),
        ("STEP 3 - EXECUTE FIRST ACTION", "#4A148C",
         "Apply only P_c[0] and P_d[0]\n"
         "Binary mutex guarantees at most one > 0\n"
         "Discard the rest of the optimal schedule"),
        ("STEP 4 - ADVANCE SoC", "#4A148C",
         "soc += P_c*eta_c*dt - P_d/eta_d*dt - TRU*dt\n"
         "TRU = 0  (trailer parked)\n"
         "soc = clip(soc, E_min, E_max);  t -> t+1"),
    ]

    y = 0.87
    for title, tc, body in steps:
        ax.text(0.05, y, title, ha="left", va="top", fontsize=8.8,
                fontweight="bold", color=tc, transform=ax.transAxes)
        for i, line in enumerate(body.split("\n")):
            ax.text(0.07, y - 0.055 - i * 0.045, line, ha="left", va="top",
                    fontsize=8.0, color="#212121", transform=ax.transAxes)
        y -= 0.22

    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor("#E0F2F1"); ax.axis("off")
    ax.text(0.5, 0.97, "5 MILP Day-Ahead vs MPC - Key Differences",
            ha="center", va="top", fontsize=11, fontweight="bold",
            color="white", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#004D40"))

    headers = ["Property", "C - MILP", "D - MPC"]
    rows = [
        ["Solved at",         "Once at t=0",          "Every slot t"],
        ["Horizon W",         "96 slots (fixed)",     "96-t (shrinking)"],
        ["Price info",        "Full day known",        "Full day known"],
        ["SoC update",        "Open-loop",             "Closed-loop"],
        ["Re-optimises",      "Never",                 "Every 15 min"],
        ["Disturbance corr.", "None",                  "Yes (real SoC)"],
        ["Compute cost",      "1x per day",            "96x per day"],
        ["Mutex model",       "Binary z_c,z_d",        "Binary z_c,z_d"],
        ["Result quality",    "Global optimum",        "Near-optimal"],
    ]

    col_x  = [0.03, 0.38, 0.70]
    col_bg = ["#B2DFDB", "#B2EBF2", "#FFE0B2"]
    y = 0.84
    for cx, hdr, bg in zip(col_x, headers, col_bg):
        ax.text(cx, y, hdr, ha="left", va="top", fontsize=8.8,
                fontweight="bold", color="#004D40",
                transform=ax.transAxes,
                bbox=dict(boxstyle="square,pad=0.2",
                          facecolor=bg, edgecolor="#90A4AE"))
    y -= 0.09
    for row in rows:
        for cx, val in zip(col_x, row):
            ax.text(cx, y, val, ha="left", va="top", fontsize=8.0,
                    color="#212121", transform=ax.transAxes)
        y -= 0.072

    ax.text(0.5, 0.02,
            "In practice: MPC ~ MILP because full-day horizon\n"
            "eliminates the myopic planning problem.",
            ha="center", va="bottom", fontsize=8.0,
            fontstyle="italic", color="#004D40",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#E0F7FA", edgecolor="#004D40"))

    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor("#FCE4EC"); ax.axis("off")
    ax.text(0.5, 0.97, "6 SoC Dynamics & Efficiency Chain",
            ha="center", va="top", fontsize=11, fontweight="bold",
            color="white", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#880E4F"))

    ax.text(0.5, 0.88,
            "e[t] = e[t-1] + eta_c*P_c[t]*dt - P_d[t]/eta_d*dt - TRU*dt",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color="#B71C1C", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#FFEBEE", edgecolor="#B71C1C"))

    flow_items = [
        ("GRID",            "#1565C0", 0.10, 0.73),
        ("P_c*eta_c ->",    "#1565C0", 0.30, 0.73),
        ("BATTERY\ne[t] kWh","#2E7D32", 0.50, 0.73),
        ("-> P_d/eta_d",    "#C62828", 0.70, 0.73),
        ("GRID\n(V2G)",     "#C62828", 0.87, 0.73),
    ]
    for txt, col, x, y_pos in flow_items:
        is_box = txt in ["GRID", "BATTERY\ne[t] kWh", "GRID\n(V2G)"]
        ax.text(x, y_pos, txt, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=col,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#E3F2FD" if "GRID" in txt else "#E8F5E9",
                          edgecolor=col) if is_box else None)

    ax.text(0.50, 0.61, "TRU = 0 kW\n(trailer stationary)", ha="center", va="center",
            fontsize=7.8, color="#888888", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#F3E5F5", edgecolor="#888888"))

    params = [
        ("eta_c = 0.92",  "Charge efficiency (8% lost to heat on AC->DC)"),
        ("eta_d = 0.92",  "Discharge efficiency (8% lost on DC->AC invert)"),
        ("E_min = 12 kWh","SoC floor = 20% x 60 kWh usable"),
        ("E_max = 60 kWh","SoC ceiling = 100% x 60 kWh usable"),
        ("TRU = 0 kW",    "Trailer parked; refrigeration off"),
        ("dt = 0.25 h",   "15-min slot = 1/4 hour"),
    ]
    y = 0.50
    for param, desc in params:
        ax.text(0.03, y, param, ha="left", va="top", fontsize=8.2,
                fontweight="bold", color="#880E4F", transform=ax.transAxes)
        ax.text(0.28, y, desc, ha="left", va="top", fontsize=8.0,
                color="#333333", transform=ax.transAxes)
        y -= 0.063

    ax.text(0.5, 0.03,
            "Round-trip efficiency = eta_c x eta_d = 0.92^2 ~ 84.6%\n"
            "Every kWh discharged costs 1/0.92 ~ 1.087 kWh from battery",
            ha="center", va="bottom", fontsize=7.8,
            fontstyle="italic", color="#555555",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.2",
                      facecolor="#FCE4EC", edgecolor="#880E4F"))

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Equations reference card saved -> {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 – PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class V2GParams:
    battery_capacity_kWh:  float = 70.0
    usable_capacity_kWh:   float = 60.0     # usable window (SoC 20-100%)
    soc_min_pct:           float = 20.0     # cold-chain floor
    soc_max_pct:           float = 100.0    # full-charge ceiling (CHANGED: was 95)
    charge_power_kW:       float = 22.0
    discharge_power_kW:    float = 22.0
    eta_charge:            float = 0.92
    eta_discharge:         float = 0.92
    deg_cost_eur_kwh:      float = 0.02
    dt_h:                  float = 0.25
    n_slots:               int   = 96
    depot_connection_kVA:  float = 0.0
    transformer_limit_kVA: float = 0.0

    @property
    def E_min(self) -> float:
        return self.usable_capacity_kWh * self.soc_min_pct / 100.0

    @property
    def E_max(self) -> float:
        return self.usable_capacity_kWh * self.soc_max_pct / 100.0

    @property
    def _grid_kw_cap(self) -> float:
        PF = 0.95
        limits = [v for v in [self.depot_connection_kVA,
                               self.transformer_limit_kVA] if v > 0]
        return min(limits) * PF if limits else float("inf")

    @property
    def p_c_max(self) -> float:
        return min(self.charge_power_kW, self._grid_kw_cap)

    @property
    def p_d_max(self) -> float:
        return min(self.discharge_power_kW, self._grid_kw_cap)


@dataclass
class V2GResult:
    scenario:            str
    p_charge:            np.ndarray
    p_discharge:         np.ndarray
    soc:                 np.ndarray
    cost_eur_day:        float
    v2g_revenue_eur_day: float
    v2g_export_kwh_day:  float
    charge_cost_eur_day: float
    deg_cost_eur_day:    float
    price_buy:           np.ndarray
    price_v2g:           np.ndarray
    plugged:             np.ndarray
    tru_load:            np.ndarray


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 – PRICE DATA
# ═══════════════════════════════════════════════════════════════════════════════

_PRICE_CACHE: dict = {}


def _load_smard_csv(csv_path: str) -> pd.DataFrame:
    global _PRICE_CACHE
    if "df" in _PRICE_CACHE:
        return _PRICE_CACHE["df"]

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

    if df is None or df.empty or len(df.columns) < 2:
        raise ValueError(
            f"Could not read CSV at {csv_path} — "
            "check the file exists and is a valid SMARD export."
        )

    col = "Germany/Luxembourg [EUR/MWh] Original resolutions"
    for possible_col in df.columns:
        if "Germany" in possible_col and "MWh" in possible_col:
            col = possible_col
            break

    if col not in df.columns:
        raise ValueError(
            f"Expected price column not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[["Start date", col]].copy()
    df.columns = ["datetime_str", "price_eur_mwh"]
    df["datetime"] = pd.to_datetime(
        df["datetime_str"], format="%b %d, %Y %I:%M %p", errors="coerce"
    )
    df = df.dropna(subset=["datetime", "price_eur_mwh"])
    df["price_eur_kwh"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce") / 1000.0
    df = df.dropna(subset=["price_eur_kwh"])
    df = df.set_index("datetime").sort_index()
    df["slot"]       = df.index.hour * 4 + df.index.minute // 15
    df["is_weekend"] = df.index.dayofweek >= 5
    df["month"]      = df.index.month
    df["is_winter"]  = df["month"].isin([1, 2, 3, 10, 11, 12])

    _PRICE_CACHE["df"] = df
    print(
        f"  Loaded {len(df):,} price slots from SMARD CSV "
        f"({df.index[0].date()} -> {df.index[-1].date()})"
    )
    return df


def load_prices_from_csv(csv_path: str, v2g: V2GParams,
                         season: str = "winter") -> tuple:
    df = _load_smard_csv(csv_path)

    is_wknd   = season.endswith("_weekend")
    is_winter = season.startswith("winter")

    mask = (df["is_winter"] == is_winter) & (df["is_weekend"] == is_wknd)
    sub  = df[mask]

    if len(sub) == 0:
        raise ValueError(f"No price data found for season='{season}'")

    profile = sub.groupby("slot")["price_eur_kwh"].mean().values
    if len(profile) != 96:
        raise ValueError(f"Expected 96 slots, got {len(profile)}")

    buy   = profile.copy()
    v2g_p = profile.copy()

    n_days    = int(len(sub) / 96)
    season_lbl = {
        "winter":         "Oct-Mar WD",
        "summer":         "Apr-Sep WD",
        "winter_weekend": "Oct-Mar WE",
        "summer_weekend": "Apr-Sep WE",
    }
    source = (
        f"REAL - SMARD.de 2025 DE/LU spot  |  {season_lbl.get(season, season)}"
        f"  |  avg of {n_days} days"
        f"  |  range {profile.min()*1000:.1f}-{profile.max()*1000:.1f} EUR/MWh"
    )
    return buy, v2g_p, source


def load_deg_sensitivity(v2g: V2GParams) -> np.ndarray:
    return np.linspace(0.02, 0.15, 10)


def build_load_and_availability(v2g: V2GParams, dwell: str = "Extended") -> tuple:
    """
    TRU is always zero — trailer is stationary and refrigeration is off.
    """
    N = v2g.n_slots
    h = np.arange(N) * v2g.dt_h
    tru = np.zeros(N)   # TRU always zero at depot

    if dwell == "Weekend":
        plugged = np.ones(N)
    elif dwell == "NightOnly":
        plugged = ((h >= 21) | (h < 7)).astype(float)
    elif dwell == "DayTrip":
        plugged = ((h >= 17) | (h < 7)).astype(float)
    else:  # Extended (default)
        plugged = ((h >= 21) | (h < 7) | ((h >= 12) & (h < 18))).astype(float)

    return tru, plugged


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 – MILP INNER SOLVER (TRUE binary mutex)
# ═══════════════════════════════════════════════════════════════════════════════

def _solve_milp_window(
    v2g:     V2GParams,
    buy:     np.ndarray,
    v2g_p:   np.ndarray,
    tru:     np.ndarray,
    plugged: np.ndarray,
    E_init:  float,
    E_fin:   float,
    deg:     float,
) -> tuple:
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix, csc_matrix

    W  = len(buy)
    dt = v2g.dt_h

    idx_c  = np.arange(W)
    idx_d  = np.arange(W,   2 * W)
    idx_e  = np.arange(2*W, 3 * W)
    idx_zc = np.arange(3*W, 4 * W)
    idx_zd = np.arange(4*W, 5 * W)
    nv     = 5 * W

    c_vec         = np.zeros(nv)
    c_vec[idx_c]  =  buy   * dt + deg * dt
    c_vec[idx_d]  = -v2g_p * dt + deg * dt

    lb = np.zeros(nv)
    ub = np.full(nv, np.inf)
    ub[idx_c]  = v2g.p_c_max * plugged
    ub[idx_d]  = v2g.p_d_max * plugged
    lb[idx_e]  = v2g.E_min
    ub[idx_e]  = v2g.E_max
    lb[idx_zc] = 0.0;  ub[idx_zc] = 1.0
    lb[idx_zd] = 0.0;  ub[idx_zd] = 1.0

    integrality = np.zeros(nv)
    integrality[idx_zc] = 1
    integrality[idx_zd] = 1

    n_rows = 4 * W + 1
    A  = lil_matrix((n_rows, nv))
    lo = np.full(n_rows, -np.inf)
    hi = np.zeros(n_rows)

    for t in range(W):
        A[t, idx_e[t]]  =  1.0
        A[t, idx_c[t]]  = -v2g.eta_charge * dt
        A[t, idx_d[t]]  =  (1.0 / v2g.eta_discharge) * dt
        rhs = -tru[t] * dt
        if t == 0:
            rhs += E_init
        else:
            A[t, idx_e[t - 1]] = -1.0
        lo[t] = hi[t] = rhs

    for t in range(W):
        row = W + t
        A[row, idx_c[t]]  =  1.0
        A[row, idx_zc[t]] = -v2g.p_c_max
        lo[row] = -np.inf
        hi[row] =  0.0

    for t in range(W):
        row = 2 * W + t
        A[row, idx_d[t]]  =  1.0
        A[row, idx_zd[t]] = -v2g.p_d_max
        lo[row] = -np.inf
        hi[row] =  0.0

    for t in range(W):
        row = 3 * W + t
        A[row, idx_zc[t]] = 1.0
        A[row, idx_zd[t]] = 1.0
        lo[row] = -np.inf
        hi[row] =  1.0

    A[4 * W, idx_e[W - 1]] = 1.0
    lo[4 * W] = E_fin
    hi[4 * W] = v2g.E_max

    res = milp(
        c_vec,
        constraints=LinearConstraint(csc_matrix(A), lo, hi),
        integrality=integrality,
        bounds=Bounds(lb, ub),
        options={"disp": False, "time_limit": 60},
    )

    if not res.success:
        raise RuntimeError(
            f"[MILP] HiGHS solver failed - status: {res.status!r}, "
            f"message: {res.message!r}. "
            "Check that the problem is feasible (E_init, E_fin, plugged window)."
        )

    P_c = np.clip(res.x[idx_c], 0.0, None)
    P_d = np.clip(res.x[idx_d], 0.0, None)
    e   = res.x[idx_e]
    return P_c, P_d, e


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 – SCENARIO A: DUMB BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_dumb(
    v2g: V2GParams,
    buy: np.ndarray, v2g_p: np.ndarray,
    tru: np.ndarray, plugged: np.ndarray,
    soc_init_pct: float = 45.0,
    soc_final_pct: float = 100.0,
) -> V2GResult:
    N, dt  = v2g.n_slots, v2g.dt_h
    E_init = v2g.usable_capacity_kWh * soc_init_pct / 100.0
    P_c = np.zeros(N); P_d = np.zeros(N); e = np.zeros(N)
    soc = E_init
    for t in range(N):
        soc -= tru[t] * dt
        soc  = max(v2g.E_min, soc)
        if plugged[t] and soc < v2g.E_max:
            p = min(v2g.p_c_max,
                    (v2g.E_max - soc) / (v2g.eta_charge * dt))
            P_c[t] = p
            soc    = min(v2g.E_max, soc + p * v2g.eta_charge * dt)
        e[t] = soc
    return _make_result(
        "A - Dumb (uncontrolled)", v2g, P_c, P_d, e,
        buy, v2g_p, plugged, tru, deg=0.0
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 – SCENARIO B: SMART CHARGING ONLY
# ═══════════════════════════════════════════════════════════════════════════════

def run_smart_no_v2g(
    v2g: V2GParams,
    buy: np.ndarray, v2g_p: np.ndarray,
    tru: np.ndarray, plugged: np.ndarray,
    soc_init_pct: float = 45.0,
    soc_final_pct: float = 100.0,
) -> V2GResult:
    E_init = v2g.usable_capacity_kWh * soc_init_pct  / 100.0
    E_fin  = min(v2g.usable_capacity_kWh * soc_final_pct / 100.0, v2g.E_max)
    P_c, P_d, e = _solve_milp_window(
        v2g, buy, np.zeros_like(v2g_p), tru, plugged, E_init, E_fin, deg=0.0
    )
    return _make_result(
        "B - Smart (no V2G)", v2g, P_c, P_d, e,
        buy, v2g_p, plugged, tru, deg=0.0
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 – SCENARIO C: FULL DAY-AHEAD MILP
# ═══════════════════════════════════════════════════════════════════════════════

def run_milp_day_ahead(
    v2g: V2GParams,
    buy: np.ndarray, v2g_p: np.ndarray,
    tru: np.ndarray, plugged: np.ndarray,
    soc_init_pct: float = 45.0,
    soc_final_pct: float = 100.0,
) -> V2GResult:
    deg    = v2g.deg_cost_eur_kwh
    E_init = v2g.usable_capacity_kWh * soc_init_pct  / 100.0
    E_fin  = min(v2g.usable_capacity_kWh * soc_final_pct / 100.0, v2g.E_max)
    P_c, P_d, e = _solve_milp_window(
        v2g, buy, v2g_p, tru, plugged, E_init, E_fin, deg
    )
    return _make_result(
        "C - MILP Day-Ahead (perfect)", v2g, P_c, P_d, e,
        buy, v2g_p, plugged, tru, deg
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 – SCENARIO D: RECEDING-HORIZON MPC
# ═══════════════════════════════════════════════════════════════════════════════

def run_mpc_day_ahead(
    v2g: V2GParams,
    buy_day_ahead: np.ndarray,
    v2g_day_ahead: np.ndarray,
    tru: np.ndarray,
    plugged: np.ndarray,
    soc_init_pct: float = 45.0,
    soc_final_pct: float = 100.0,
    label: str = "D - MPC perfect",
) -> V2GResult:
    deg    = v2g.deg_cost_eur_kwh
    E_init = v2g.usable_capacity_kWh * soc_init_pct  / 100.0
    E_fin  = min(v2g.usable_capacity_kWh * soc_final_pct / 100.0, v2g.E_max)
    N, dt  = v2g.n_slots, v2g.dt_h

    P_c_all = np.zeros(N)
    P_d_all = np.zeros(N)
    e_all   = np.zeros(N)
    soc     = E_init

    for t in range(N):
        buy_fc = buy_day_ahead[t:].copy()
        v2g_fc = v2g_day_ahead[t:].copy()
        tru_w  = tru[t:]
        plug_w = plugged[t:]

        P_c_w, P_d_w, _ = _solve_milp_window(
            v2g, buy_fc, v2g_fc, tru_w, plug_w,
            E_init=soc, E_fin=E_fin, deg=deg
        )

        pc_t = float(np.clip(P_c_w[0], 0.0, v2g.p_c_max * plugged[t]))
        pd_t = float(np.clip(P_d_w[0], 0.0, v2g.p_d_max * plugged[t]))

        if pc_t > 1e-6 and pd_t > 1e-6:
            if (v2g_day_ahead[t] - deg) > (buy_day_ahead[t] + deg):
                pc_t = 0.0
            else:
                pd_t = 0.0

        soc -= tru[t] * dt
        soc += pc_t * v2g.eta_charge      * dt
        soc -= pd_t / v2g.eta_discharge   * dt
        soc  = float(np.clip(soc, v2g.E_min, v2g.E_max))

        P_c_all[t] = pc_t
        P_d_all[t] = pd_t
        e_all[t]   = soc

    return _make_result(
        label, v2g, P_c_all, P_d_all, e_all,
        buy_day_ahead, v2g_day_ahead, plugged, tru, deg
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 – KPI BUILDER & SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════

def _make_result(
    label: str, v2g: V2GParams,
    P_c: np.ndarray, P_d: np.ndarray, e: np.ndarray,
    buy: np.ndarray, v2g_p: np.ndarray,
    plugged: np.ndarray, tru: np.ndarray,
    deg: float,
) -> V2GResult:
    dt  = v2g.dt_h
    chg = float(np.sum(P_c * buy)   * dt)
    rev = float(np.sum(P_d * v2g_p) * dt)
    dgc = float(np.sum((P_c + P_d) * deg) * dt)
    return V2GResult(
        scenario=label, p_charge=P_c, p_discharge=P_d, soc=e,
        cost_eur_day=chg - rev + dgc,
        v2g_revenue_eur_day=rev,
        v2g_export_kwh_day=float(np.sum(P_d) * dt),
        charge_cost_eur_day=chg,
        deg_cost_eur_day=dgc,
        price_buy=buy, price_v2g=v2g_p, plugged=plugged, tru_load=tru,
    )


def deg_sensitivity(
    v2g: V2GParams,
    buy: np.ndarray, v2g_p: np.ndarray,
    tru: np.ndarray, plugged: np.ndarray,
    deg_values: Optional[np.ndarray] = None,
    soc_init: float = 45.0,
    soc_final: float = 100.0,
) -> pd.DataFrame:
    if deg_values is None:
        deg_values = np.linspace(0.02, 0.15, 10)
    rows = []
    for dv in deg_values:
        E_i = v2g.usable_capacity_kWh * soc_init  / 100.0
        E_f = min(v2g.usable_capacity_kWh * soc_final / 100.0, v2g.E_max)
        P_c, P_d, e = _solve_milp_window(
            v2g, buy, v2g_p, tru, plugged, E_i, E_f, dv
        )
        r = _make_result(
            f"deg={dv:.3f}", v2g, P_c, P_d, e, buy, v2g_p, plugged, tru, dv
        )
        rows.append({
            "DegCost_EUR_kWh": dv,
            "NetCost_EUR_day": r.cost_eur_day,
            "V2G_Rev_EUR_day": r.v2g_revenue_eur_day,
            "V2G_kWh_day":     r.v2g_export_kwh_day,
            "V2G_active":      r.v2g_export_kwh_day > 0.1,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 – MAIN RESULTS PLOT
# ═══════════════════════════════════════════════════════════════════════════════

COL = {
    "dumb":  "#AAAAAA",
    "smart": "#2196F3",
    "milp":  "#00BCD4",
    "mpc":   "#FF7700",
    "price": "#007700",
    "tru":   "#AA0000",
}


def plot_all(
    v2g: V2GParams,
    hours: np.ndarray,
    A: V2GResult, B: V2GResult, C: V2GResult, D: V2GResult,
    deg_df: pd.DataFrame,
    season: str = "winter",
    out: str = "results.png",
) -> None:
    ROLL = 68
    N    = len(hours)

    def roll_arr(a):
        return np.concatenate([a[ROLL:], a[:ROLL]])

    def roll_r(r):
        return V2GResult(
            scenario=r.scenario,
            p_charge=roll_arr(r.p_charge),
            p_discharge=roll_arr(r.p_discharge),
            soc=roll_arr(r.soc),
            cost_eur_day=r.cost_eur_day,
            v2g_revenue_eur_day=r.v2g_revenue_eur_day,
            v2g_export_kwh_day=r.v2g_export_kwh_day,
            charge_cost_eur_day=r.charge_cost_eur_day,
            deg_cost_eur_day=r.deg_cost_eur_day,
            price_buy=roll_arr(r.price_buy),
            price_v2g=roll_arr(r.price_v2g),
            plugged=roll_arr(r.plugged),
            tru_load=roll_arr(r.tru_load),
        )

    rA, rB, rC, rD  = roll_r(A), roll_r(B), roll_r(C), roll_r(D)
    results_rolled   = [rA, rB, rC, rD]

    hours_disp = np.concatenate([hours[ROLL:], hours[:ROLL] + 24.0])
    tick_pos   = np.arange(17, 42, 2, dtype=float)
    tick_lbls  = [f"{int(h % 24):02d}:00" for h in tick_pos]

    labels_full = [
        "A - Dumb (uncontrolled)",
        "B - Smart (no V2G)",
        "C - MILP Day-Ahead",
        "D - MPC Perfect",
    ]
    colors_list = [COL["dumb"], COL["smart"], COL["milp"], COL["mpc"]]

    fig = plt.figure(figsize=(22, 16))
    gs  = GridSpec(4, 3, figure=fig,
                   width_ratios=[1.8, 1.5, 1.5],
                   hspace=0.08, wspace=0.38,
                   top=0.93, bottom=0.07, left=0.06, right=0.97)
    fig.suptitle(
        f"S.KOe COOL  -  True Binary-Mutex MILP + MPC V2G Optimisation  ({season})\n"
        f"Time axis: 17:00 arrival -> next day 17:00  |  "
        f"Departure SoC target = 100%  |  TRU = 0 kW",
        fontsize=12, fontweight="bold",
    )

    y_max = max(r.p_charge.max() for r in results_rolled) * 1.18 or 25

    for i, (r, lbl, col) in enumerate(zip(results_rolled, labels_full, colors_list)):
        ax  = fig.add_subplot(gs[i, 0])
        ax2 = ax.twinx()

        ax2.step(hours_disp, r.price_v2g * 1000, where="post",
                 color=COL["price"], lw=1.0, alpha=0.55, ls="--")
        ax2.set_ylabel("EUR/MWh", fontsize=6, color=COL["price"])
        ax2.tick_params(axis="y", labelsize=6, colors=COL["price"])

        ax.fill_between(hours_disp, r.p_charge, step="pre",
                        color=col, alpha=0.80)
        if r.p_discharge.max() > 0.01:
            ax.fill_between(hours_disp, -r.p_discharge, step="pre",
                            color="#E53935", alpha=0.70)
            ax.text(0.98, 0.05, "v P_d V2G", transform=ax.transAxes,
                    fontsize=6.5, color="#E53935", ha="right", va="bottom")

        plug = r.plugged
        for t in range(1, N):
            if plug[t] > 0.5 > plug[t - 1]:
                ax.axvline(hours_disp[t], color="green", lw=1.2, ls="--", alpha=0.6)
                ax.text(hours_disp[t] + 0.1, y_max * 0.88, "IN",
                        fontsize=6, color="green", va="top")
            elif plug[t] < 0.5 < plug[t - 1]:
                ax.axvline(hours_disp[t], color="red", lw=1.2, ls="--", alpha=0.6)
                ax.text(hours_disp[t] + 0.1, y_max * 0.88, "OUT",
                        fontsize=6, color="red", va="top")

        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlim(17, 41)
        ax.set_ylim(-v2g.p_d_max * 1.15, y_max)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, alpha=0.25)
        ax.text(0.01, 0.95, f"({i+1}) {lbl}",
                transform=ax.transAxes, fontsize=8.5,
                fontweight="bold", color=col, va="top")
        ax.set_ylabel("kW", fontsize=7)

        if i < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_lbls, fontsize=7, rotation=35, ha="right")
            ax.set_xlabel("Hour  (17:00 = trailer arrival at depot)", fontsize=8)

    ax = fig.add_subplot(gs[0:2, 1])
    for r, lbl, col, ls in zip(results_rolled, labels_full, colors_list,
                                ["-", "-", "-", "--"]):
        ax.plot(hours_disp, r.soc, color=col, lw=2, ls=ls,
                label=lbl.split("-")[0].strip())
    ax.axhline(v2g.E_min, color="red",  ls=":", lw=1.2,
               label=f"E_min = {v2g.E_min:.0f} kWh (20%)")
    ax.axhline(v2g.E_max, color="navy", ls=":", lw=1.2,
               label=f"E_max = {v2g.E_max:.0f} kWh (100%)")
    ax.axvline(31, color="grey", ls=":", lw=0.8, alpha=0.5)
    ax.text(31.1, v2g.E_min + 1, "midnight", fontsize=7,
            color="grey", va="bottom")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbls, fontsize=7, rotation=35, ha="right")
    ax.set_xlim(17, 41)
    ax.set_title(
        f"(5) SoC - Battery State of Charge\n"
        f"(17:00 -> next day 17:00 | target = {v2g.E_max:.0f} kWh = 100%)",
        fontsize=9, fontweight="bold",
    )
    ax.set_ylabel("E  (kWh)"); ax.set_xlabel("Hour")
    ax.legend(fontsize=7.5, loc="lower right"); ax.grid(True, alpha=0.3)

    ax  = fig.add_subplot(gs[0:2, 2])
    ax2 = ax.twinx()
    w   = 0.18
    ax.bar(hours_disp - w / 2, rC.p_discharge, width=w,
           color=COL["milp"], alpha=0.85, label="C - MILP P_d")
    ax.bar(hours_disp + w / 2, rD.p_discharge, width=w,
           color=COL["mpc"],  alpha=0.75, label="D - MPC P_d")
    ax2.step(hours_disp, rC.price_v2g * 1000, where="post",
             color=COL["price"], lw=1.8, label="V2G price (EUR/MWh)")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbls, fontsize=7, rotation=35, ha="right")
    ax.set_xlim(17, 41)
    ax.set_title("(6) P_d - V2G Discharge vs Price\n(C-MILP vs D-MPC)",
                 fontsize=9, fontweight="bold")
    ax.set_ylabel("P_d  (kW)"); ax.set_xlabel("Hour")
    ax2.set_ylabel("Price  (EUR/MWh)", color=COL["price"], fontsize=8)
    ax.legend(loc="upper left", fontsize=7.5)
    ax2.legend(loc="upper right", fontsize=7.5)
    ax.grid(True, alpha=0.3)

    ax  = fig.add_subplot(gs[2:4, 1])
    ax2 = ax.twinx()
    ax.plot(deg_df["DegCost_EUR_kWh"], deg_df["NetCost_EUR_day"],
            "o-", color=COL["milp"], lw=2, label="Net Cost (EUR/day)")
    ax.plot(deg_df["DegCost_EUR_kWh"], deg_df["V2G_Rev_EUR_day"],
            "s--", color=COL["mpc"], lw=2, label="V2G Revenue (EUR/day)")
    ax2.bar(deg_df["DegCost_EUR_kWh"], deg_df["V2G_kWh_day"],
            width=0.008, color=COL["mpc"], alpha=0.22, label="V2G kWh/day")
    tipping = deg_df[deg_df["V2G_active"]]["DegCost_EUR_kWh"].max()
    if not np.isnan(tipping):
        ax.axvline(tipping, color="red", ls=":", lw=1.5,
                   label=f"V2G cutoff ~ {tipping:.3f}")
    ax.axvline(v2g.deg_cost_eur_kwh, color="black", ls="--", lw=1.2,
               label=f"Active deg = {v2g.deg_cost_eur_kwh:.3f}")
    ax.set_title("(7) Degradation (deg) Sensitivity",
                 fontsize=9, fontweight="bold")
    ax.set_xlabel("deg  (EUR/kWh cycled)"); ax.set_ylabel("EUR/day")
    ax2.set_ylabel("V2G export  (kWh/day)", color=COL["mpc"], fontsize=8)
    ax.legend(loc="upper left", fontsize=7.5)
    ax2.legend(loc="upper right", fontsize=7.5)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2:4, 2])
    ax.axis("off")
    ref = A.cost_eur_day
    table_data = [
        ["Metric",               "A\nDumb",   "B\nSmart",  "C\nMILP",   "D\nMPC"],
        ["Net cost\n(EUR/day)",
         f"{A.cost_eur_day:.3f}",   f"{B.cost_eur_day:.3f}",
         f"{C.cost_eur_day:.3f}",   f"{D.cost_eur_day:.3f}"],
        ["Charge cost\n(EUR/day)",
         f"{A.charge_cost_eur_day:.3f}", f"{B.charge_cost_eur_day:.3f}",
         f"{C.charge_cost_eur_day:.3f}", f"{D.charge_cost_eur_day:.3f}"],
        ["V2G revenue\n(EUR/day)",
         f"{A.v2g_revenue_eur_day:.3f}", f"{B.v2g_revenue_eur_day:.3f}",
         f"{C.v2g_revenue_eur_day:.3f}", f"{D.v2g_revenue_eur_day:.3f}"],
        ["deg cost\n(EUR/day)",
         f"{A.deg_cost_eur_day:.3f}", f"{B.deg_cost_eur_day:.3f}",
         f"{C.deg_cost_eur_day:.3f}", f"{D.deg_cost_eur_day:.3f}"],
        ["V2G export\n(kWh/day)",
         f"{A.v2g_export_kwh_day:.2f}", f"{B.v2g_export_kwh_day:.2f}",
         f"{C.v2g_export_kwh_day:.2f}", f"{D.v2g_export_kwh_day:.2f}"],
        ["Savings vs A\n(EUR/day)",
         "-", f"{ref - B.cost_eur_day:+.3f}",
         f"{ref - C.cost_eur_day:+.3f}", f"{ref - D.cost_eur_day:+.3f}"],
        ["Annual savings\n(EUR/yr)",
         "-", f"{(ref - B.cost_eur_day)*365:+,.0f}",
         f"{(ref - C.cost_eur_day)*365:+,.0f}",
         f"{(ref - D.cost_eur_day)*365:+,.0f}"],
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.1, 2.15)

    col_colors = ["#EEEEEE", "#BBDEFB", "#B2EBF2", "#FFE0B2"]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#90A4AE")
        if r == 0:
            cell.set_facecolor("#263238")
            cell.set_text_props(color="white", fontweight="bold", fontsize=8)
        elif c == 0:
            cell.set_facecolor("#ECEFF1")
            cell.set_text_props(fontweight="bold", fontsize=8)
            cell.set_width(0.38)
        else:
            cell.set_facecolor(col_colors[c - 1])

    ax.set_title("(8) KPI Summary Table", fontsize=9, fontweight="bold", pad=14)

    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved -> {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 – FULL YEAR SIMULATION & PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_year(
    v2g: V2GParams,
    csv_path: str,
    soc_init_pct: float = 45.0,
    soc_final_pct: float = 100.0,
) -> pd.DataFrame:
    df = _load_smard_csv(csv_path)
    df['date'] = df.index.date
    dates = df['date'].unique()

    records = []
    print(f"\n  [Full Year] Starting accurate 365-day day-by-day simulation...")
    print(f"  (Running A, B, C, and D for every day. MPC solves ~35,000 times, this may take 2-5 minutes...)")

    for i, d in enumerate(dates):
        day_df = df[df['date'] == d]
        if len(day_df) != 96:
            continue

        buy = day_df["price_eur_kwh"].values
        v2g_p = buy.copy()
        month = day_df.index[0].month

        is_weekend = (pd.Timestamp(d).dayofweek >= 5)
        dwell = "Weekend" if is_weekend else "DayTrip"

        tru, plugged = build_load_and_availability(v2g, dwell=dwell)

        if dwell == "DayTrip":
            ROLL = 68
            buy     = np.roll(buy, -ROLL)
            v2g_p   = np.roll(v2g_p, -ROLL)
            tru     = np.roll(tru, -ROLL)
            plugged = np.roll(plugged, -ROLL)

        A = run_dumb(v2g, buy, v2g_p, tru, plugged, soc_init_pct, soc_final_pct)
        B = run_smart_no_v2g(v2g, buy, v2g_p, tru, plugged, soc_init_pct, soc_final_pct)
        C = run_milp_day_ahead(v2g, buy, v2g_p, tru, plugged, soc_init_pct, soc_final_pct)
        D = run_mpc_day_ahead(v2g, buy, v2g_p, tru, plugged, soc_init_pct, soc_final_pct, label="D")

        records.append({
            "Date": d,
            "Month": month,
            "Dwell": dwell,
            "Cost_A": A.cost_eur_day,
            "Cost_B": B.cost_eur_day,
            "Cost_C": C.cost_eur_day,
            "Cost_D": D.cost_eur_day,
            "Rev_C": C.v2g_revenue_eur_day,
            "Rev_D": D.v2g_revenue_eur_day,
            "Export_C": C.v2g_export_kwh_day,
            "Export_D": D.v2g_export_kwh_day,
        })

        if (i + 1) % 30 == 0 or i == len(dates) - 1:
            print(f"    ... processed {i + 1}/{len(dates)} days")

    return pd.DataFrame(records)


def plot_full_year_analysis(
    df: pd.DataFrame,
    v2g: V2GParams,
    out: str = "full_year_analysis.png"
) -> None:
    fig = plt.figure(figsize=(24, 18))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    fig.patch.set_facecolor("#F8F9FA")

    fig.suptitle(
        "V2G Full-Year Profitability Analysis (True 365-Day Simulation)\n"
        f"SoC 20-100% | E_max = {v2g.E_max:.0f} kWh | deg = {v2g.deg_cost_eur_kwh:.3f} EUR/kWh",
        fontsize=18, fontweight="bold", color="#1A237E", y=0.96
    )

    df["Date"] = pd.to_datetime(df["Date"])
    dates = df["Date"]

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(dates, df["Cost_A"].cumsum(), label="A - Dumb", color=COL["dumb"], lw=2.5)
    ax1.plot(dates, df["Cost_B"].cumsum(), label="B - Smart", color=COL["smart"], lw=2.5)
    ax1.plot(dates, df["Cost_C"].cumsum(), label="C - MILP", color=COL["milp"], lw=2.5)
    ax1.plot(dates, df["Cost_D"].cumsum(), label="D - MPC", color=COL["mpc"], lw=2.5, ls="--")
    ax1.set_title("1. Cumulative Net Charging Cost Over Year", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Cumulative Cost (EUR)", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.text(dates.iloc[-1], df["Cost_A"].sum(), f"  €{df['Cost_A'].sum():,.0f}", color="#555", va="center", fontweight="bold")
    ax1.text(dates.iloc[-1], df["Cost_C"].sum(), f"  €{df['Cost_C'].sum():,.0f}", color=COL["milp"], va="center", fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 1])
    df["Sav_B"] = df["Cost_A"] - df["Cost_B"]
    df["Sav_C"] = df["Cost_A"] - df["Cost_C"]
    df["Sav_D"] = df["Cost_A"] - df["Cost_D"]

    monthly = df.groupby("Month")[["Sav_B", "Sav_C", "Sav_D"]].sum()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x = np.arange(len(monthly))
    w = 0.25

    ax2.bar(x - w, monthly["Sav_B"], width=w, label="B Savings (vs A)", color=COL["smart"], alpha=0.9)
    ax2.bar(x,     monthly["Sav_C"], width=w, label="C Savings (vs A)", color=COL["milp"], alpha=0.9)
    ax2.bar(x + w, monthly["Sav_D"], width=w, label="D Savings (vs A)", color=COL["mpc"], alpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([months[i-1] for i in monthly.index], fontsize=10)
    ax2.set_title("2. Monthly Savings compared to Dumb Charging", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Monthly Savings (EUR)", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(fontsize=11)

    ax3 = fig.add_subplot(gs[1, 0])
    box_data = [df["Cost_A"], df["Cost_B"], df["Cost_C"], df["Cost_D"]]
    bp = ax3.boxplot(box_data, patch_artist=True, labels=["A - Dumb", "B - Smart", "C - MILP", "D - MPC"])
    colors = [COL["dumb"], COL["smart"], COL["milp"], COL["mpc"]]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in bp['medians']:
        median.set(color="black", linewidth=2)
    ax3.set_title("3. Distribution of Daily Net Costs (Variance over 365 Days)", fontsize=14, fontweight="bold")
    ax3.set_ylabel("Daily Net Cost (EUR/day)", fontsize=11)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.axhline(0, color="#D32F2F", lw=1.5, ls="--", label="Zero Cost Line")
    ax3.legend(fontsize=10)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4_2 = ax4.twinx()
    ax4.scatter(dates, df["Rev_C"], alpha=0.3, color=COL["milp"], s=20, label="Daily V2G Rev (MILP)")
    ax4.plot(dates, df["Rev_C"].rolling(14, center=True).mean(), color="#006064", lw=3, label="14-day Avg Rev")
    ax4_2.plot(dates, df["Export_C"].rolling(14, center=True).mean(), color="#2E7D32", lw=2, ls=":", label="14-day Avg Export (kWh)")
    ax4.set_title("4. Daily V2G Revenue & Grid Export (Scenario C - MILP)", fontsize=14, fontweight="bold")
    ax4.set_ylabel("Daily Revenue (EUR)", fontsize=11)
    ax4_2.set_ylabel("Daily Grid Export (kWh)", color="#2E7D32", fontsize=11)
    ax4_2.tick_params(axis="y", colors="#2E7D32")
    ax4.grid(True, alpha=0.3)
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_2.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=11)

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Full-year chart saved -> {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 10b – ADDITIONAL CROSS-SEASON ANALYSIS  (called by GUI)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_additional_analysis(
    v2g: V2GParams,
    hours: np.ndarray,
    A: V2GResult, B: V2GResult, C: V2GResult, D: V2GResult,
    buy: np.ndarray,
    v2g_p: np.ndarray,
    all_season_results: dict,
    csv_path: str,
    out: str = "additional_analysis.png",
) -> None:
    """
    4-panel cross-season summary chart generated by the GUI after all seasons run.

    Panel 1 — Grouped bar: net cost EUR/day per scenario × season
    Panel 2 — Grouped bar: annual savings (B, C, D vs A) by season
    Panel 3 — Price duration curve for winter + V2G profitable zone
    Panel 4 — V2G export kWh/day and revenue/day by season (Scenario C MILP)
    """
    season_keys   = ["winter", "summer", "winter_weekend", "summer_weekend"]
    season_labels = ["Winter WD", "Summer WD", "Winter WE", "Summer WE"]
    days_per_yr   = [130, 131, 52, 52]
    sc_keys       = ["A", "B", "C", "D"]
    sc_labels     = ["A Dumb", "B Smart", "C MILP", "D MPC"]
    sc_colors     = [COL["dumb"], COL["smart"], COL["milp"], COL["mpc"]]

    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#F7F9FC")
    gs  = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32,
                   top=0.92, bottom=0.07, left=0.07, right=0.97)
    fig.suptitle(
        "V2G Additional Analysis — Cross-Season Summary\n"
        f"Battery: {v2g.usable_capacity_kWh:.0f} kWh usable | "
        f"P_c/d max: {v2g.p_c_max:.0f}/{v2g.p_d_max:.0f} kW | "
        f"eta: {v2g.eta_charge:.2f}/{v2g.eta_discharge:.2f} | "
        f"deg: {v2g.deg_cost_eur_kwh:.3f} EUR/kWh",
        fontsize=13, fontweight="bold", color="#1A237E", y=0.97,
    )

    x      = np.arange(len(season_keys))
    n_sc   = len(sc_keys)
    bar_w  = 0.18

    # ── Panel 1: Net cost per scenario per season ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#EEF2FF")

    for i, (sc, lbl, col) in enumerate(zip(sc_keys, sc_labels, sc_colors)):
        costs = []
        for sk in season_keys:
            if sk in all_season_results and sc in all_season_results[sk]:
                costs.append(all_season_results[sk][sc].cost_eur_day)
            else:
                costs.append(0.0)
        offset = (i - n_sc / 2 + 0.5) * bar_w
        bars = ax1.bar(x + offset, costs, width=bar_w, color=col,
                       alpha=0.85, label=lbl, zorder=3)
        for bar, v in zip(bars, costs):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=6.5, color=col,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(season_labels, fontsize=9)
    ax1.set_ylabel("Net Cost (EUR/day)", fontsize=10)
    ax1.set_title("(1) Net Cost per Scenario × Season", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8.5, loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y", zorder=0)
    ax1.axhline(0, color="black", lw=0.8)

    # ── Panel 2: Annual savings contribution by season ─────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#E8F5E9")

    sc_save   = ["B", "C", "D"]
    sc_col2   = [COL["smart"], COL["milp"], COL["mpc"]]
    sc_lbl2   = ["B Smart", "C MILP", "D MPC"]
    bar_w2    = 0.25

    for i, (sc, lbl, col) in enumerate(zip(sc_save, sc_lbl2, sc_col2)):
        ann_sav = []
        for sk, dpyr in zip(season_keys, days_per_yr):
            if sk in all_season_results:
                r_a  = all_season_results[sk]["A"].cost_eur_day
                r_sc = all_season_results[sk][sc].cost_eur_day
                ann_sav.append((r_a - r_sc) * dpyr)
            else:
                ann_sav.append(0.0)
        offset = (i - len(sc_save) / 2 + 0.5) * bar_w2
        bars = ax2.bar(x + offset, ann_sav, width=bar_w2, color=col,
                       alpha=0.85, label=lbl, zorder=3)
        for bar, v in zip(bars, ann_sav):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (1.5 if v >= 0 else -6),
                f"{v:+.0f}",
                ha="center", va="bottom", fontsize=7, color=col,
            )

    ax2.set_xticks(x)
    ax2.set_xticklabels(season_labels, fontsize=9)
    ax2.set_ylabel("Annual Savings vs A (EUR/year)", fontsize=10)
    ax2.set_title("(2) Annual Savings Contribution by Season", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8.5, loc="upper right")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.grid(True, alpha=0.3, axis="y", zorder=0)

    # ── Panel 3: Price duration curve (winter) + V2G zone ─────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#FFF8E1")

    sorted_buy = np.sort(buy)[::-1] * 1000  # convert to EUR/MWh
    n_slots    = len(sorted_buy)
    pct_time   = np.linspace(0, 100, n_slots)

    ax3.fill_between(pct_time, sorted_buy, alpha=0.30, color=COL["price"], zorder=2)
    ax3.plot(pct_time, sorted_buy, color=COL["price"], lw=2,
             label="Winter buy price (sorted)", zorder=3)

    deg_threshold_mwh = v2g.deg_cost_eur_kwh * 1000  # EUR/MWh
    ax3.axhline(deg_threshold_mwh, color=COL["mpc"], lw=1.8, ls="--",
                label=f"Deg cost threshold ({deg_threshold_mwh:.0f} EUR/MWh)")
    ax3.axhline(0, color="grey", lw=0.8, ls=":")

    v2g_zone = sorted_buy > deg_threshold_mwh
    if v2g_zone.any():
        pct_cutoff = float(pct_time[v2g_zone][-1])
        ax3.axvspan(0, pct_cutoff, alpha=0.12, color=COL["mpc"],
                    label=f"V2G profitable zone ({pct_cutoff:.1f}% of slots)")
        ax3.text(pct_cutoff / 2, sorted_buy.max() * 0.55,
                 f"V2G\nzone\n{pct_cutoff:.0f}%",
                 ha="center", va="center", fontsize=8.5,
                 color=COL["mpc"], fontweight="bold")

    ax3.set_xlabel("% of 15-min slots (high → low price)", fontsize=9)
    ax3.set_ylabel("Electricity price (EUR/MWh)", fontsize=10)
    ax3.set_title("(3) Price Duration Curve (Winter) + V2G Profitable Zone",
                  fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8.5)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 100)

    # ── Panel 4: V2G export kWh/day + revenue/day by season (MILP C) ──────────
    ax4  = fig.add_subplot(gs[1, 1])
    ax4b = ax4.twinx()
    ax4.set_facecolor("#FCE4EC")

    export_kwh = []
    v2g_rev    = []
    for sk in season_keys:
        if sk in all_season_results:
            export_kwh.append(all_season_results[sk]["C"].v2g_export_kwh_day)
            v2g_rev.append(all_season_results[sk]["C"].v2g_revenue_eur_day)
        else:
            export_kwh.append(0.0)
            v2g_rev.append(0.0)

    bar_w4 = 0.30
    b1 = ax4.bar(x - bar_w4 / 2, export_kwh, width=bar_w4,
                 color=COL["milp"], alpha=0.85,
                 label="V2G Export (kWh/day)", zorder=3)
    b2 = ax4b.bar(x + bar_w4 / 2, v2g_rev, width=bar_w4,
                  color=COL["mpc"], alpha=0.75,
                  label="V2G Revenue (EUR/day)", zorder=3)

    for bar, v in zip(b1, export_kwh):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.08,
                 f"{v:.1f} kWh",
                 ha="center", va="bottom", fontsize=7.5, color=COL["milp"])
    for bar, v in zip(b2, v2g_rev):
        ax4b.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.001,
                  f"€{v:.4f}",
                  ha="center", va="bottom", fontsize=7.5, color=COL["mpc"])

    ax4.set_xticks(x)
    ax4.set_xticklabels(season_labels, fontsize=9)
    ax4.set_ylabel("V2G Export (kWh/day)", fontsize=10, color=COL["milp"])
    ax4b.set_ylabel("V2G Revenue (EUR/day)", fontsize=10, color=COL["mpc"])
    ax4.tick_params(axis="y", colors=COL["milp"])
    ax4b.tick_params(axis="y", colors=COL["mpc"])
    ax4.set_title("(4) V2G Export & Revenue by Season (Scenario C — MILP)",
                  fontsize=11, fontweight="bold")

    lines1, lbl1 = ax4.get_legend_handles_labels()
    lines2, lbl2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8.5, loc="upper right")
    ax4.grid(True, alpha=0.3, axis="y", zorder=0)

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Additional analysis chart saved -> {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 – CONSOLE REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(
    v2g: V2GParams,
    results: dict,
    deg_df: pd.DataFrame,
    season: str = "winter",
    price_source: str = "",
) -> None:
    ref = results["A"].cost_eur_day
    print("\n" + "=" * 84)
    print(f"  RESULTS - {season.upper()} (Averaged Season Data) | {price_source[:65]}")
    print("=" * 84)
    print(
        f"  {'Scenario':<32} {'Net EUR/day':>11} {'Charge EUR':>10} "
        f"{'V2G Rev EUR':>11} {'deg EUR':>8} {'V2G kWh':>8} {'vs Dumb':>10}"
    )
    print("-" * 84)
    for r in results.values():
        print(
            f"  {r.scenario:<32} {r.cost_eur_day:>11.4f} "
            f"{r.charge_cost_eur_day:>10.4f} {r.v2g_revenue_eur_day:>11.4f} "
            f"{r.deg_cost_eur_day:>8.4f} {r.v2g_export_kwh_day:>8.2f} "
            f"  {ref - r.cost_eur_day:>+.4f}"
        )
    print("=" * 84)

    tipping = deg_df[deg_df["V2G_active"]]["DegCost_EUR_kWh"].max()
    tipping_ok = "OK" if v2g.deg_cost_eur_kwh <= tipping else "WARN"
    print(
        f"  Degradation tipping point: V2G profitable up to ~EUR{tipping:.3f}/kWh  "
        f"({tipping_ok})\n"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "=" * 65)
    print("  S.KOe COOL  -  Day-Ahead MILP + Receding-Horizon MPC V2G")
    print("  Schmitz Cargobull AG  |  2025")
    print("  Binary mutex | SoC 20-100% | TRU=0 kW | Full Year Analysis")
    print("=" * 65)

    csv_candidates = [
        Path(__file__).parent / "2025_Electricity_Price.csv",
        Path("2025_Electricity_Price.csv"),
    ]
    csv_path = None
    for p in csv_candidates:
        if p.exists():
            csv_path = str(p)
            break
    if csv_path is None:
        print("\n  ERROR: 2025_Electricity_Price.csv not found.")
        print("  Place the SMARD CSV in the same folder as this script.\n")
        sys.exit(1)
    print(f"\n  Price data: {csv_path}")

    v2g = V2GParams()
    print(
        f"\n  Battery: {v2g.battery_capacity_kWh} kWh total, "
        f"{v2g.usable_capacity_kWh} kWh usable  "
        f"(SoC 20-100% -> E_min={v2g.E_min:.0f} kWh, E_max={v2g.E_max:.0f} kWh)\n"
        f"  P_c_max = {v2g.p_c_max:.0f} kW  |  P_d_max = {v2g.p_d_max:.0f} kW  |  "
        f"TRU = 0 kW (stationary)"
    )

    print("\n" + "-" * 55)
    print("  BATTERY DEGRADATION COST")
    print(f"  Current value: deg = EUR{v2g.deg_cost_eur_kwh:.3f} / kWh cycled")
    answer = input("  Include battery degradation cost in optimisation? [Y/n]: ").strip().lower()
    if answer in ("n", "no"):
        v2g.deg_cost_eur_kwh = 0.0
        print("  -> Degradation cost set to 0.")
    else:
        answer2 = input(f"  Keep default EUR{v2g.deg_cost_eur_kwh:.3f}/kWh? [Y / enter value]: ").strip()
        if answer2 and answer2.lower() not in ("y", "yes"):
            try:
                v2g.deg_cost_eur_kwh = float(answer2)
                print(f"  -> Degradation cost set to EUR{v2g.deg_cost_eur_kwh:.3f}/kWh")
            except ValueError:
                print(f"  -> Invalid input. Keeping EUR{v2g.deg_cost_eur_kwh:.3f}/kWh")
        else:
            print(f"  -> Degradation cost: EUR{v2g.deg_cost_eur_kwh:.3f}/kWh")
    print("-" * 55)

    print("\n" + "-" * 55)
    print("  ARRIVAL STATE OF CHARGE")
    soc_input = input("  Enter trailer arrival SoC % (e.g. 45): ").strip()
    try:
        soc_init_pct = float(soc_input)
        assert 20.0 <= soc_init_pct <= 100.0
    except Exception:
        soc_init_pct = 45.0
        print("  -> Invalid input. Using default 45%")
    print(f"  -> Arrival SoC: {soc_init_pct:.0f}%   Departure SoC: 100% = {v2g.E_max:.0f} kWh (fixed)")
    print("-" * 55)

    soc_final_pct = 100.0

    print("\n  Generating abbreviation legend ...")
    generate_abbreviation_legend("abbreviation_legend.png")

    print("  Generating equations reference card ...")
    generate_equations_card("equations_reference.png")

    DAY_TYPES = [
        ("winter",         "DayTrip", 130, "Winter weekday  (Mon-Fri, Oct-Mar)"),
        ("summer",         "DayTrip", 131, "Summer weekday  (Mon-Fri, Apr-Sep)"),
        ("winter_weekend", "Weekend",  52, "Winter weekend  (Sat-Sun, Oct-Mar)"),
        ("summer_weekend", "Weekend",  52, "Summer weekend  (Sat-Sun, Apr-Sep)"),
    ]

    deg_values = load_deg_sensitivity(v2g)
    hours = np.arange(v2g.n_slots) * v2g.dt_h

    for season, dwell_type, days_per_year, label in DAY_TYPES:
        tru, plugged = build_load_and_availability(v2g, dwell=dwell_type)
        buy, v2g_p, price_source = load_prices_from_csv(csv_path, v2g, season=season)

        if dwell_type == "DayTrip":
            ROLL = 68
            buy, v2g_p = np.roll(buy, -ROLL), np.roll(v2g_p, -ROLL)
            tru, plugged = np.roll(tru, -ROLL), np.roll(plugged, -ROLL)

        A = run_dumb(v2g, buy, v2g_p, tru, plugged, soc_init_pct, soc_final_pct)
        B = run_smart_no_v2g(v2g, buy, v2g_p, tru, plugged, soc_init_pct, soc_final_pct)
        C = run_milp_day_ahead(v2g, buy, v2g_p, tru, plugged, soc_init_pct, soc_final_pct)
        D = run_mpc_day_ahead(v2g, buy, v2g_p, tru, plugged, soc_init_pct, soc_final_pct, label="D - MPC perfect")

        results = {"A": A, "B": B, "C": C, "D": D}
        deg_df = deg_sensitivity(v2g, buy, v2g_p, tru, plugged, deg_values, soc_init_pct, soc_final_pct)

        print_report(v2g, results, deg_df, season=label, price_source=price_source)
        plot_all(v2g, hours, A, B, C, D, deg_df, season=label, out=f"results_{season}.png")

    print("\n" + "=" * 65)
    print("  RUNNING FULL 365-DAY YEAR ANALYSIS")
    print("=" * 65)

    full_year_df = run_full_year(v2g, csv_path, soc_init_pct, soc_final_pct)
    plot_full_year_analysis(full_year_df, v2g, out="full_year_analysis.png")

    annual_cost_a       = full_year_df["Cost_A"].sum()
    annual_cost_milp    = full_year_df["Cost_C"].sum()
    annual_v2g_milp     = full_year_df["Rev_C"].sum()
    annual_savings_dumb = annual_cost_a - annual_cost_milp

    print(f"\n{'='*65}")
    print(f"  TRUE ANNUAL SUMMARY (365 Days) - Single Trailer (Scenario C MILP)")
    print(f"{'='*65}")
    print(f"  Annual energy cost (Dumb):        EUR{annual_cost_a:>8,.0f}/year")
    print(f"  Annual energy cost (MILP):        EUR{annual_cost_milp:>8,.0f}/year")
    print(f"  Annual V2G revenue (MILP):        EUR{annual_v2g_milp:>8,.0f}/year")
    print(f"  Annual savings vs Dumb charging:  EUR{annual_savings_dumb:>8,.0f}/year")
    print(f"  [Agora 2025 benchmark for car:    ~EUR500/year for arbitrage only]")
    print()


def run_headless(
    soc_init_pct: float = 45.0,
    soc_final_pct: float = 100.0,
    deg_cost: float = 0.02,
    csv_path: str = "",
) -> dict:
    """Non-interactive entry point — callable from GUI or scripts."""
    if not csv_path:
        for p in [Path(__file__).parent / "2025_Electricity_Price.csv", Path("2025_Electricity_Price.csv")]:
            if p.exists():
                csv_path = str(p)
                break
        if not csv_path:
            raise FileNotFoundError("2025_Electricity_Price.csv not found")

    v2g = V2GParams()
    v2g.deg_cost_eur_kwh = deg_cost
    return {"status": "ok"}


if __name__ == "__main__":
    main()