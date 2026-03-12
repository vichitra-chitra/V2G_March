#!/usr/bin/env python3
"""
S.KOe COOL — V2G Web GUI
Flask front-end that calls Prototype2optimization functions directly
(bypasses interactive input() prompts).
All V2GParams fields are exposed as editable form inputs.
"""

import os
import sys
import json
import time
import traceback
import threading
import queue
from pathlib import Path

from flask import (
    Flask, render_template_string, request,
    jsonify, Response, send_from_directory,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import Prototype2optimization as v2g_mod

app = Flask(__name__)

progress_queue: queue.Queue = queue.Queue()
run_lock = threading.Lock()

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  CORE: Run optimisation WITHOUT interactive input()
# ═══════════════════════════════════════════════════════════════════

def run_optimisation(
    # ── SoC / mission ───────────────────────────────────────────
    soc_init_pct:          float = 45.0,
    soc_final_pct:         float = 100.0,
    # ── Battery hardware ────────────────────────────────────────
    usable_capacity_kWh:   float = 60.0,
    soc_min_pct:           float = 20.0,
    soc_max_pct:           float = 100.0,
    charge_power_kW:       float = 22.0,
    discharge_power_kW:    float = 22.0,
    # ── Efficiency ──────────────────────────────────────────────
    eta_charge:            float = 0.92,
    eta_discharge:         float = 0.92,
    # ── Economics ───────────────────────────────────────────────
    deg_cost:              float = 0.02,
    # ── Grid connection (0 = unlimited) ─────────────────────────
    depot_connection_kVA:  float = 0.0,
    transformer_limit_kVA: float = 0.0,
    # ── Data ────────────────────────────────────────────────────
    csv_path:              str   = "",
) -> dict:
    """
    Runs the full 4-season V2G optimisation.
    All V2GParams fields are set from the GUI form before any solver call.
    """

    def log(msg: str):
        print(f"  [GUI] {msg}")
        progress_queue.put(msg)

    # ── Locate CSV ──────────────────────────────────────────────
    if not csv_path:
        candidates = [
            Path(__file__).parent / "2025_Electricity_Price.csv",
            Path("2025_Electricity_Price.csv"),
        ]
        for p in candidates:
            if p.exists():
                csv_path = str(p)
                break
        if not csv_path:
            raise FileNotFoundError(
                "2025_Electricity_Price.csv not found. "
                "Place it in the same folder as gui.py."
            )

    log(f"CSV: {csv_path}")

    # ── Build V2GParams from ALL form values ────────────────────
    v2g = v2g_mod.V2GParams()
    v2g.usable_capacity_kWh   = usable_capacity_kWh
    v2g.soc_min_pct            = soc_min_pct
    v2g.soc_max_pct            = soc_max_pct
    v2g.charge_power_kW        = charge_power_kW
    v2g.discharge_power_kW     = discharge_power_kW
    v2g.eta_charge             = eta_charge
    v2g.eta_discharge          = eta_discharge
    v2g.deg_cost_eur_kwh       = deg_cost
    v2g.depot_connection_kVA   = depot_connection_kVA
    v2g.transformer_limit_kVA  = transformer_limit_kVA

    log(
        f"Battery: {v2g.usable_capacity_kWh} kWh usable | "
        f"SoC {v2g.soc_min_pct}-{v2g.soc_max_pct}% | "
        f"P_c/d {v2g.p_c_max:.0f}/{v2g.p_d_max:.0f} kW | "
        f"eta {v2g.eta_charge}/{v2g.eta_discharge} | "
        f"deg={v2g.deg_cost_eur_kwh} EUR/kWh"
    )
    log(f"Arrival SoC: {soc_init_pct}% | Departure SoC: {soc_final_pct}%")
    if depot_connection_kVA > 0 or transformer_limit_kVA > 0:
        log(f"Grid cap: depot={depot_connection_kVA} kVA | transformer={transformer_limit_kVA} kVA")

    # ── Generate reference cards ────────────────────────────────
    log("Generating abbreviation legend...")
    v2g_mod.generate_abbreviation_legend(str(OUT_DIR / "abbreviation_legend.png"))

    log("Generating equations reference card...")
    v2g_mod.generate_equations_card(str(OUT_DIR / "equations_reference.png"))

    # ── Run all seasons ─────────────────────────────────────────
    hours      = np.arange(v2g.n_slots) * v2g.dt_h
    deg_values = v2g_mod.load_deg_sensitivity(v2g)

    all_season_results: dict = {}
    season_summary: list     = []

    DAY_TYPES = [
        ("winter",         "DayTrip", 130, "Winter weekday (Mon-Fri, Oct-Mar)"),
        ("summer",         "DayTrip", 131, "Summer weekday (Mon-Fri, Apr-Sep)"),
        ("winter_weekend", "Weekend",  52, "Winter weekend (Sat-Sun, Oct-Mar)"),
        ("summer_weekend", "Weekend",  52, "Summer weekend (Sat-Sun, Apr-Sep)"),
    ]

    annual_cost_milp    = 0.0
    annual_v2g_milp     = 0.0
    annual_savings_dumb = 0.0

    for season, dwell_type, days_per_year, label in DAY_TYPES:
        log(f"━━━ {label} ({days_per_year} days/yr) ━━━")

        tru, plugged = v2g_mod.build_load_and_availability(v2g, dwell=dwell_type)
        buy, v2g_p, price_source = v2g_mod.load_prices_from_csv(
            csv_path, v2g, season=season
        )

        if dwell_type == "DayTrip":
            ROLL = 68
            buy     = np.roll(buy,     -ROLL)
            v2g_p   = np.roll(v2g_p,   -ROLL)
            tru     = np.roll(tru,     -ROLL)
            plugged = np.roll(plugged, -ROLL)

        log(f"  Buy: {buy.min()*1000:.1f}-{buy.max()*1000:.1f} EUR/MWh | "
            f"Plugged: {int(plugged.sum()*v2g.dt_h)}h/day")

        log(f"  Running A - Dumb...")
        A = v2g_mod.run_dumb(v2g, buy, v2g_p, tru, plugged,
                             soc_init_pct, soc_final_pct)

        log(f"  Running B - Smart (no V2G)...")
        B = v2g_mod.run_smart_no_v2g(v2g, buy, v2g_p, tru, plugged,
                                      soc_init_pct, soc_final_pct)

        log(f"  Running C - MILP Day-Ahead...")
        C = v2g_mod.run_milp_day_ahead(v2g, buy, v2g_p, tru, plugged,
                                        soc_init_pct, soc_final_pct)

        log(f"  Running D - MPC (96 solves)...")
        D = v2g_mod.run_mpc_day_ahead(v2g, buy, v2g_p, tru, plugged,
                                       soc_init_pct, soc_final_pct,
                                       label="D - MPC perfect")

        results = {"A": A, "B": B, "C": C, "D": D}
        all_season_results[season] = results

        log(f"  Running degradation sensitivity...")
        deg_df = v2g_mod.deg_sensitivity(
            v2g, buy, v2g_p, tru, plugged,
            deg_values, soc_init_pct, soc_final_pct
        )

        out_png = str(OUT_DIR / f"results_{season}.png")
        log(f"  Plotting -> results_{season}.png")
        v2g_mod.plot_all(v2g, hours, A, B, C, D, deg_df,
                         season=label, out=out_png)

        ref = A.cost_eur_day
        season_summary.append({
            "season":         label,
            "days_per_year":  days_per_year,
            "A_cost":         round(A.cost_eur_day, 4),
            "B_cost":         round(B.cost_eur_day, 4),
            "C_cost":         round(C.cost_eur_day, 4),
            "D_cost":         round(D.cost_eur_day, 4),
            "C_v2g_rev":      round(C.v2g_revenue_eur_day, 4),
            "D_v2g_rev":      round(D.v2g_revenue_eur_day, 4),
            "C_v2g_kwh":      round(C.v2g_export_kwh_day, 2),
            "C_savings_vs_A": round(ref - C.cost_eur_day, 4),
        })

        annual_cost_milp    += C.cost_eur_day * days_per_year
        annual_v2g_milp     += C.v2g_revenue_eur_day * days_per_year
        annual_savings_dumb += (A.cost_eur_day - C.cost_eur_day) * days_per_year

    # ── Additional cross-season analysis plot ───────────────────
    log("Generating additional analysis graphs...")
    buy_w, v2g_p_w, _ = v2g_mod.load_prices_from_csv(csv_path, v2g, season="winter")
    v2g_mod.plot_additional_analysis(
        v2g, hours,
        all_season_results["winter"]["A"],
        all_season_results["winter"]["B"],
        all_season_results["winter"]["C"],
        all_season_results["winter"]["D"],
        buy_w, v2g_p_w,
        all_season_results,
        csv_path,
        out=str(OUT_DIR / "additional_analysis.png"),
    )

    summary = {
        "annual_cost_milp":       round(annual_cost_milp, 0),
        "annual_v2g_revenue":     round(annual_v2g_milp, 0),
        "annual_savings_vs_dumb": round(annual_savings_dumb, 0),
        "seasons":                season_summary,
        "params_used": {
            "usable_capacity_kWh":   v2g.usable_capacity_kWh,
            "soc_min_pct":           v2g.soc_min_pct,
            "soc_max_pct":           v2g.soc_max_pct,
            "charge_power_kW":       v2g.charge_power_kW,
            "discharge_power_kW":    v2g.discharge_power_kW,
            "eta_charge":            v2g.eta_charge,
            "eta_discharge":         v2g.eta_discharge,
            "deg_cost_eur_kwh":      v2g.deg_cost_eur_kwh,
            "soc_init_pct":          soc_init_pct,
            "soc_final_pct":         soc_final_pct,
            "depot_connection_kVA":  v2g.depot_connection_kVA,
            "transformer_limit_kVA": v2g.transformer_limit_kVA,
        },
        "images": [
            "abbreviation_legend.png",
            "equations_reference.png",
            "results_winter.png",
            "results_summer.png",
            "results_winter_weekend.png",
            "results_summer_weekend.png",
            "additional_analysis.png",
        ],
    }

    log(f"✓ DONE — Annual savings vs Dumb: EUR{annual_savings_dumb:,.0f}/yr")
    log("__DONE__")
    return summary


# ═══════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/run", methods=["POST"])
def run():
    """Start optimisation in a background thread."""
    if run_lock.locked():
        return jsonify({"error": "Optimisation already running"}), 409

    data = request.get_json(silent=True) or {}

    # ── Extract & validate every parameter ──────────────────────
    def fget(key, default):
        try:
            return float(data.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    soc_init              = fget("soc_init",              45.0)
    soc_final             = fget("soc_final",             100.0)
    usable_capacity_kWh   = fget("usable_capacity_kWh",   60.0)
    soc_min_pct           = fget("soc_min_pct",           20.0)
    soc_max_pct           = fget("soc_max_pct",           100.0)
    charge_power_kW       = fget("charge_power_kW",       22.0)
    discharge_power_kW    = fget("discharge_power_kW",    22.0)
    eta_charge            = fget("eta_charge",            0.92)
    eta_discharge         = fget("eta_discharge",         0.92)
    deg_cost              = fget("deg_cost",              0.02)
    depot_connection_kVA  = fget("depot_connection_kVA",  0.0)
    transformer_limit_kVA = fget("transformer_limit_kVA", 0.0)

    # Basic validation
    errors = []
    if not (20 <= soc_init <= 100):
        errors.append("Arrival SoC must be 20–100%")
    if not (20 <= soc_final <= 100):
        errors.append("Departure SoC must be 20–100%")
    if soc_final < soc_init:
        errors.append("Departure SoC must be >= Arrival SoC")
    if not (5 <= usable_capacity_kWh <= 500):
        errors.append("Usable capacity must be 5–500 kWh")
    if not (0 <= soc_min_pct < soc_max_pct <= 100):
        errors.append("SoC min must be < SoC max, both 0–100%")
    if not (1 <= charge_power_kW <= 350):
        errors.append("Charge power must be 1–350 kW")
    if not (1 <= discharge_power_kW <= 350):
        errors.append("Discharge power must be 1–350 kW")
    if not (0.5 <= eta_charge <= 1.0):
        errors.append("Charge efficiency must be 0.50–1.00")
    if not (0.5 <= eta_discharge <= 1.0):
        errors.append("Discharge efficiency must be 0.50–1.00")
    if not (0 <= deg_cost <= 1):
        errors.append("Degradation cost must be 0–1 EUR/kWh")
    if errors:
        return jsonify({"error": " | ".join(errors)}), 400

    # Clear old messages
    while not progress_queue.empty():
        try:
            progress_queue.get_nowait()
        except queue.Empty:
            break

    def worker():
        with run_lock:
            try:
                result = run_optimisation(
                    soc_init_pct          = soc_init,
                    soc_final_pct         = soc_final,
                    usable_capacity_kWh   = usable_capacity_kWh,
                    soc_min_pct           = soc_min_pct,
                    soc_max_pct           = soc_max_pct,
                    charge_power_kW       = charge_power_kW,
                    discharge_power_kW    = discharge_power_kW,
                    eta_charge            = eta_charge,
                    eta_discharge         = eta_discharge,
                    deg_cost              = deg_cost,
                    depot_connection_kVA  = depot_connection_kVA,
                    transformer_limit_kVA = transformer_limit_kVA,
                )
                progress_queue.put(json.dumps({
                    "type": "result",
                    "data": result,
                }))
            except Exception as e:
                tb = traceback.format_exc()
                print(f"  [GUI ERROR]\n{tb}")
                progress_queue.put(json.dumps({
                    "type":      "error",
                    "message":   str(e),
                    "traceback": tb,
                }))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/stream")
def stream():
    """Server-Sent Events — streams progress messages to the browser."""
    def generate():
        while True:
            try:
                msg = progress_queue.get(timeout=120)
            except queue.Empty:
                yield "data: {\"type\": \"error\", \"message\": \"Timeout\"}\n\n"
                return

            if msg.startswith("{"):
                yield f"data: {msg}\n\n"
                return
            elif msg == "__DONE__":
                return
            else:
                payload = json.dumps({"type": "progress", "message": msg})
                yield f"data: {payload}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/output/<path:filename>")
def output_file(filename):
    return send_from_directory(str(OUT_DIR), filename)


# ═══════════════════════════════════════════════════════════════════
#  HTML / JS FRONTEND
# ═══════════════════════════════════════════════════════════════════

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>S.KOe COOL — V2G Optimisation</title>
<style>
  :root {
    --bg: #f4f6fb;
    --card: #ffffff;
    --primary: #1a237e;
    --accent: #00bcd4;
    --danger: #e53935;
    --success: #43a047;
    --warn: #f57c00;
    --text: #212121;
    --muted: #78909c;
    --border: #e0e0e0;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
  }
  .header {
    background: linear-gradient(135deg, var(--primary), #0d47a1);
    color: white;
    padding: 22px 32px;
    text-align: center;
  }
  .header h1 { font-size: 1.7rem; margin-bottom: 4px; }
  .header p  { opacity: 0.85; font-size: 0.9rem; }

  .container { max-width: 1440px; margin: 0 auto; padding: 20px; }
  .grid { display: grid; grid-template-columns: 370px 1fr; gap: 20px; }

  /* ── Panel ── */
  .panel {
    background: var(--card);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border: 1px solid var(--border);
  }
  .panel > h2 {
    font-size: 1rem;
    color: var(--primary);
    margin-bottom: 14px;
    padding-bottom: 6px;
    border-bottom: 2px solid var(--accent);
  }

  /* ── Collapsible section ── */
  .section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 12px;
    margin: 10px 0 4px;
    cursor: pointer;
    user-select: none;
    font-weight: 700;
    font-size: 0.85rem;
    color: var(--primary);
    transition: background 0.15s;
  }
  .section-header:hover { background: #e8eaf6; }
  .section-header .arrow { font-size: 0.75rem; transition: transform 0.2s; }
  .section-header.open .arrow { transform: rotate(90deg); }
  .section-body { display: none; padding: 4px 0 6px; }
  .section-body.open { display: block; }

  /* ── Fields ── */
  .field { margin-bottom: 10px; }
  .field label {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-weight: 600;
    margin-bottom: 2px;
    font-size: 0.82rem;
    color: #37474f;
  }
  .field label .unit {
    font-weight: 400;
    font-size: 0.75rem;
    color: var(--muted);
  }
  .field .hint {
    font-size: 0.73rem;
    color: var(--muted);
    margin-bottom: 3px;
  }
  .field input[type="number"],
  .field input[type="text"] {
    width: 100%;
    padding: 8px 10px;
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 0.95rem;
    transition: border-color 0.2s;
    background: white;
  }
  .field input:focus { border-color: var(--accent); outline: none; }
  .field input:disabled { background: #f5f5f5; color: #9e9e9e; }

  /* ── Run button ── */
  .btn {
    display: block;
    width: 100%;
    padding: 13px;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s;
    margin-top: 10px;
  }
  .btn-run {
    background: linear-gradient(135deg, var(--accent), #0097a7);
    color: white;
  }
  .btn-run:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,188,212,.4); }
  .btn-run:disabled { opacity: 0.5; cursor: not-allowed; transform: none; box-shadow: none; }

  /* ── Log ── */
  .log-box {
    background: #263238;
    color: #e0e0e0;
    border-radius: 8px;
    padding: 14px;
    font-family: 'Cascadia Code', 'Fira Code', monospace;
    font-size: 0.78rem;
    max-height: 200px;
    overflow-y: auto;
    margin-top: 12px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .log-ok   { color: #69f0ae; }
  .log-err  { color: #ff5252; }
  .log-info { color: #80deea; }

  /* ── Results panel ── */
  .results { display: none; }
  .results.show { display: block; }
  .results h2 {
    font-size: 1.1rem;
    color: var(--primary);
    margin: 20px 0 10px;
  }

  /* ── Params echo ── */
  .params-echo {
    background: #e8f5e9;
    border: 1px solid #a5d6a7;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #1b5e20;
    margin-bottom: 16px;
    font-family: monospace;
    white-space: pre-wrap;
  }

  /* ── KPI cards ── */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 10px;
    margin-bottom: 18px;
  }
  .kpi-card {
    background: var(--card);
    border-radius: 10px;
    padding: 14px;
    text-align: center;
    border: 1px solid var(--border);
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
  }
  .kpi-card .kpi-value { font-size: 1.4rem; font-weight: 700; color: var(--primary); }
  .kpi-card .kpi-label { font-size: 0.78rem; color: var(--muted); margin-top: 3px; }
  .kpi-card.green .kpi-value { color: var(--success); }

  /* ── Season table ── */
  .season-table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0 20px;
    font-size: 0.85rem;
  }
  .season-table th {
    background: var(--primary);
    color: white;
    padding: 9px 7px;
    text-align: center;
    font-size: 0.78rem;
  }
  .season-table td {
    padding: 7px;
    text-align: center;
    border-bottom: 1px solid var(--border);
  }
  .season-table tr:hover { background: #e8eaf6; }

  /* ── Gallery ── */
  .gallery img {
    width: 100%;
    border-radius: 8px;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,.12);
    cursor: pointer;
    transition: transform 0.2s;
  }
  .gallery img:hover { transform: scale(1.01); }

  @media (max-width: 920px) {
    .grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<div class="header">
  <h1>S.KOe COOL — V2G Optimisation Dashboard</h1>
  <p>Day-Ahead MILP + Receding-Horizon MPC &nbsp;|&nbsp;
     Schmitz Cargobull AG 2025 &nbsp;|&nbsp;
     Binary Mutex &nbsp;|&nbsp; All parameters editable</p>
</div>

<div class="container">
<div class="grid">

  <!-- ─── LEFT: Controls ─── -->
  <div>
    <div class="panel">
      <h2>⚙️ Optimisation Parameters</h2>

      <!-- ══ SECTION 1: Mission ══ -->
      <div class="section-header open" onclick="toggleSection(this)">
        🚛 Mission / SoC Targets <span class="arrow">▶</span>
      </div>
      <div class="section-body open">
        <div class="field">
          <label>Arrival SoC <span class="unit">%  (20–100)</span></label>
          <div class="hint">State of charge when trailer arrives at depot</div>
          <input type="number" id="soc_init" value="45" min="20" max="100" step="1">
        </div>
        <div class="field">
          <label>Departure SoC target <span class="unit">%  (20–100)</span></label>
          <div class="hint">Required SoC before truck departs (cold-chain)</div>
          <input type="number" id="soc_final" value="100" min="20" max="100" step="1">
        </div>
        <div class="field">
          <label>SoC floor (E_min) <span class="unit">%  (0–50)</span></label>
          <div class="hint">Minimum allowed SoC during optimisation</div>
          <input type="number" id="soc_min_pct" value="20" min="0" max="50" step="1">
        </div>
        <div class="field">
          <label>SoC ceiling (E_max) <span class="unit">%  (50–100)</span></label>
          <div class="hint">Maximum allowed SoC (physical battery limit)</div>
          <input type="number" id="soc_max_pct" value="100" min="50" max="100" step="1">
        </div>
      </div>

      <!-- ══ SECTION 2: Battery ══ -->
      <div class="section-header open" onclick="toggleSection(this)">
        🔋 Battery Hardware <span class="arrow">▶</span>
      </div>
      <div class="section-body open">
        <div class="field">
          <label>Usable capacity <span class="unit">kWh  (5–500)</span></label>
          <div class="hint">Energy window between SoC floor and ceiling</div>
          <input type="number" id="usable_capacity_kWh" value="60" min="5" max="500" step="1">
        </div>
        <div class="field">
          <label>Max charge power <span class="unit">kW  (1–350)</span></label>
          <div class="hint">P_c_max — limited by onboard charger (default 22 kW)</div>
          <input type="number" id="charge_power_kW" value="22" min="1" max="350" step="0.5">
        </div>
        <div class="field">
          <label>Max V2G discharge power <span class="unit">kW  (1–350)</span></label>
          <div class="hint">P_d_max — limited by inverter (default 22 kW)</div>
          <input type="number" id="discharge_power_kW" value="22" min="1" max="350" step="0.5">
        </div>
      </div>

      <!-- ══ SECTION 3: Efficiency & Economics ══ -->
      <div class="section-header open" onclick="toggleSection(this)">
        ⚡ Efficiency &amp; Economics <span class="arrow">▶</span>
      </div>
      <div class="section-body open">
        <div class="field">
          <label>Charge efficiency η_c <span class="unit">0.50–1.00</span></label>
          <div class="hint">AC→DC one-way efficiency (default 0.92 = 92%)</div>
          <input type="number" id="eta_charge" value="0.92" min="0.5" max="1.0" step="0.01">
        </div>
        <div class="field">
          <label>Discharge efficiency η_d <span class="unit">0.50–1.00</span></label>
          <div class="hint">DC→AC one-way efficiency (default 0.92 = 92%)</div>
          <input type="number" id="eta_discharge" value="0.92" min="0.5" max="1.0" step="0.01">
        </div>
        <div class="field">
          <label>Degradation cost <span class="unit">EUR/kWh cycled</span></label>
          <div class="hint">Battery wear cost per kWh (0 = ignore, 0.02 = LFP default)</div>
          <input type="number" id="deg_cost" value="0.02" min="0" max="1" step="0.005">
        </div>
      </div>

      <!-- ══ SECTION 4: Grid Connection ══ -->
      <div class="section-header" onclick="toggleSection(this)">
        🔌 Grid Connection Limits <span class="arrow">▶</span>
      </div>
      <div class="section-body">
        <div class="field">
          <label>Depot connection <span class="unit">kVA  (0 = unlimited)</span></label>
          <div class="hint">Maximum kVA of the depot grid connection</div>
          <input type="number" id="depot_connection_kVA" value="0" min="0" max="2000" step="10">
        </div>
        <div class="field">
          <label>Transformer limit <span class="unit">kVA  (0 = unlimited)</span></label>
          <div class="hint">Transformer capacity cap (overrides depot if smaller)</div>
          <input type="number" id="transformer_limit_kVA" value="0" min="0" max="2000" step="10">
        </div>
        <div class="hint" style="margin-top:2px; padding: 6px 8px; background:#fff3e0; border-radius:5px;">
          ℹ️ Effective P_c/d_max = min(hardware kW, connection kVA × 0.95 PF).
          Set both to 0 to use hardware limits only.
        </div>
      </div>

      <button class="btn btn-run" id="btnRun" onclick="startRun()">
        ▶ Run Optimisation
      </button>

      <div class="log-box" id="logBox">Ready. Configure parameters and click Run.\n</div>
    </div>
  </div>

  <!-- ─── RIGHT: Results ─── -->
  <div>
    <div class="panel results" id="resultsPanel">

      <h2>✅ Parameters Used</h2>
      <div class="params-echo" id="paramsEcho"></div>

      <h2>📊 Annual Summary (Scenario C — MILP)</h2>
      <div class="kpi-grid" id="kpiGrid"></div>

      <h2>📋 Season Breakdown</h2>
      <table class="season-table" id="seasonTable">
        <thead>
          <tr>
            <th>Season</th><th>Days/yr</th>
            <th>A Dumb<br>EUR/day</th>
            <th>B Smart<br>EUR/day</th>
            <th>C MILP<br>EUR/day</th>
            <th>D MPC<br>EUR/day</th>
            <th>C V2G Rev<br>EUR/day</th>
            <th>C Savings<br>vs A</th>
          </tr>
        </thead>
        <tbody id="seasonBody"></tbody>
      </table>

      <h2>📈 Generated Charts</h2>
      <div class="gallery" id="gallery"></div>
    </div>
  </div>

</div>
</div>

<script>
// ── Collapsible sections ─────────────────────────────────────────
function toggleSection(header) {
  header.classList.toggle("open");
  const body = header.nextElementSibling;
  body.classList.toggle("open");
}

// ── DOM refs ─────────────────────────────────────────────────────
const logBox     = document.getElementById("logBox");
const btnRun     = document.getElementById("btnRun");
const results    = document.getElementById("resultsPanel");
const kpiGrid    = document.getElementById("kpiGrid");
const seasonBody = document.getElementById("seasonBody");
const gallery    = document.getElementById("gallery");
const paramsEcho = document.getElementById("paramsEcho");

function appendLog(msg, cls = "") {
  const span = document.createElement("span");
  span.className = cls;
  span.textContent = msg + "\n";
  logBox.appendChild(span);
  logBox.scrollTop = logBox.scrollHeight;
}

function getNum(id, fallback) {
  const v = parseFloat(document.getElementById(id).value);
  return isNaN(v) ? fallback : v;
}

function startRun() {
  // Collect all parameters
  const params = {
    soc_init:              getNum("soc_init",              45),
    soc_final:             getNum("soc_final",             100),
    soc_min_pct:           getNum("soc_min_pct",           20),
    soc_max_pct:           getNum("soc_max_pct",           100),
    usable_capacity_kWh:   getNum("usable_capacity_kWh",   60),
    charge_power_kW:       getNum("charge_power_kW",       22),
    discharge_power_kW:    getNum("discharge_power_kW",    22),
    eta_charge:            getNum("eta_charge",            0.92),
    eta_discharge:         getNum("eta_discharge",         0.92),
    deg_cost:              getNum("deg_cost",              0.02),
    depot_connection_kVA:  getNum("depot_connection_kVA",  0),
    transformer_limit_kVA: getNum("transformer_limit_kVA", 0),
  };

  // Client-side sanity checks
  if (params.soc_init < 20 || params.soc_init > 100) {
    alert("Arrival SoC must be 20–100%."); return;
  }
  if (params.soc_final < 20 || params.soc_final > 100) {
    alert("Departure SoC must be 20–100%."); return;
  }
  if (params.soc_final < params.soc_init) {
    alert("Departure SoC must be ≥ Arrival SoC."); return;
  }
  if (params.soc_min_pct >= params.soc_max_pct) {
    alert("SoC floor must be less than SoC ceiling."); return;
  }
  if (params.eta_charge < 0.5 || params.eta_charge > 1) {
    alert("Charge efficiency must be 0.50–1.00."); return;
  }
  if (params.eta_discharge < 0.5 || params.eta_discharge > 1) {
    alert("Discharge efficiency must be 0.50–1.00."); return;
  }

  // Reset UI
  logBox.innerHTML = "";
  results.classList.remove("show");
  kpiGrid.innerHTML = "";
  seasonBody.innerHTML = "";
  gallery.innerHTML = "";
  paramsEcho.textContent = "";
  btnRun.disabled = true;
  btnRun.textContent = "⏳ Running...";

  appendLog("Starting optimisation...", "log-info");
  appendLog(`  Arrival SoC: ${params.soc_init}%  |  Departure SoC: ${params.soc_final}%`, "log-info");
  appendLog(`  Battery: ${params.usable_capacity_kWh} kWh usable  |  P_c/d: ${params.charge_power_kW}/${params.discharge_power_kW} kW`, "log-info");
  appendLog(`  η_c/d: ${params.eta_charge}/${params.eta_discharge}  |  deg: ${params.deg_cost} EUR/kWh`, "log-info");
  appendLog("");

  fetch("/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  })
  .then(resp => {
    if (!resp.ok) {
      return resp.json().then(d => { throw new Error(d.error || "Server error"); });
    }
    return resp.json();
  })
  .then(() => {
    const evtSource = new EventSource("/stream");

    evtSource.onmessage = function(event) {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "progress") {
          let cls = "log-info";
          if (data.message.includes("✓") || data.message.includes("DONE")) cls = "log-ok";
          if (data.message.includes("ERROR") || data.message.includes("WARN")) cls = "log-err";
          appendLog(data.message, cls);
        }
        else if (data.type === "result") {
          appendLog("\n✓ Optimisation complete!", "log-ok");
          evtSource.close();
          showResults(data.data);
          btnRun.disabled = false;
          btnRun.textContent = "▶ Run Optimisation";
        }
        else if (data.type === "error") {
          appendLog("\n✗ ERROR: " + data.message, "log-err");
          if (data.traceback) appendLog(data.traceback, "log-err");
          evtSource.close();
          btnRun.disabled = false;
          btnRun.textContent = "▶ Run Optimisation";
        }
      } catch (e) {
        appendLog(event.data);
      }
    };

    evtSource.onerror = function() {
      evtSource.close();
      appendLog("\n[Connection closed]", "log-info");
      btnRun.disabled = false;
      btnRun.textContent = "▶ Run Optimisation";
    };
  })
  .catch(err => {
    appendLog("✗ " + err.message, "log-err");
    btnRun.disabled = false;
    btnRun.textContent = "▶ Run Optimisation";
  });
}

function showResults(data) {
  results.classList.add("show");

  // Params echo
  if (data.params_used) {
    const p = data.params_used;
    paramsEcho.textContent =
      `Usable cap: ${p.usable_capacity_kWh} kWh  |  ` +
      `SoC: ${p.soc_min_pct}–${p.soc_max_pct}%  |  ` +
      `Arrival: ${p.soc_init_pct}%  Departure: ${p.soc_final_pct}%\n` +
      `P_c/d: ${p.charge_power_kW}/${p.discharge_power_kW} kW  |  ` +
      `η_c/d: ${p.eta_charge}/${p.eta_discharge}  |  ` +
      `deg: ${p.deg_cost_eur_kwh} EUR/kWh\n` +
      `Grid limits: depot=${p.depot_connection_kVA} kVA  transformer=${p.transformer_limit_kVA} kVA`;
  }

  // KPI cards
  const kpis = [
    { label: "Annual Cost (MILP)",    value: `€${Number(data.annual_cost_milp).toLocaleString()}`,        green: false },
    { label: "Annual V2G Revenue",    value: `€${Number(data.annual_v2g_revenue).toLocaleString()}`,      green: true  },
    { label: "Annual Savings vs Dumb",value: `€${Number(data.annual_savings_vs_dumb).toLocaleString()}`,  green: true  },
  ];
  kpiGrid.innerHTML = kpis.map(k => `
    <div class="kpi-card ${k.green ? 'green' : ''}">
      <div class="kpi-value">${k.value}</div>
      <div class="kpi-label">${k.label}</div>
    </div>
  `).join("");

  // Season table
  seasonBody.innerHTML = data.seasons.map(s => `
    <tr>
      <td><strong>${s.season}</strong></td>
      <td>${s.days_per_year}</td>
      <td>${s.A_cost.toFixed(4)}</td>
      <td>${s.B_cost.toFixed(4)}</td>
      <td>${s.C_cost.toFixed(4)}</td>
      <td>${s.D_cost.toFixed(4)}</td>
      <td>${s.C_v2g_rev.toFixed(4)}</td>
      <td style="color:${s.C_savings_vs_A > 0 ? '#43a047' : '#e53935'}; font-weight:700">
        ${s.C_savings_vs_A > 0 ? '+' : ''}${s.C_savings_vs_A.toFixed(4)}
      </td>
    </tr>
  `).join("");

  // Image gallery
  const ts = Date.now();
  gallery.innerHTML = data.images.map(img => `
    <a href="/output/${img}?t=${ts}" target="_blank">
      <img src="/output/${img}?t=${ts}" alt="${img}"
           loading="lazy" onerror="this.style.display='none'">
    </a>
  `).join("");

  results.scrollIntoView({ behavior: "smooth", block: "start" });
}
</script>

</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  S.KOe COOL — V2G Web GUI")
    print("  Open in browser: http://localhost:5000")
    print("  In Codespaces: use the 'Ports' tab to open the forwarded port.\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)