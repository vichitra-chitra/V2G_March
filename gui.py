#!/usr/bin/env python3
"""
S.KOe COOL — V2G Web GUI
Flask front-end that calls v2g_optimisation functions directly
(bypasses interactive input() prompts).
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

# ── Make sure our script is importable ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import Prototype2optimization as v2g_mod

app = Flask(__name__)

# ── Global progress queue for SSE streaming ─────────────────────
progress_queue: queue.Queue = queue.Queue()
run_lock = threading.Lock()

# ── Output folder ────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  CORE: Run optimisation WITHOUT interactive input()
# ═══════════════════════════════════════════════════════════════════

def run_optimisation(
    soc_init_pct: float = 45.0,
    deg_cost: float = 0.02,
    csv_path: str = "",
) -> dict:
    """
    Runs the full 4-season V2G optimisation.
    Sends progress messages via progress_queue.
    Returns a summary dict.
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

    # ── Build params (NO input() calls) ─────────────────────────
    v2g = v2g_mod.V2GParams()
    v2g.deg_cost_eur_kwh = deg_cost
    soc_final_pct = 100.0

    log(
        f"Battery: {v2g.usable_capacity_kWh} kWh usable | "
        f"SoC {v2g.soc_min_pct}-{v2g.soc_max_pct}% | "
        f"deg={v2g.deg_cost_eur_kwh} EUR/kWh"
    )
    log(f"Arrival SoC: {soc_init_pct}% | Departure SoC: {soc_final_pct}%")

    # ── Generate reference cards ────────────────────────────────
    log("Generating abbreviation legend...")
    v2g_mod.generate_abbreviation_legend(str(OUT_DIR / "abbreviation_legend.png"))

    log("Generating equations reference card...")
    v2g_mod.generate_equations_card(str(OUT_DIR / "equations_reference.png"))

    # ── Run all seasons ─────────────────────────────────────────
    hours = np.arange(v2g.n_slots) * v2g.dt_h
    deg_values = v2g_mod.load_deg_sensitivity(v2g)

    all_season_results: dict = {}
    season_summary: list = []

    DAY_TYPES = [
        ("winter",         "DayTrip", 130, "Winter weekday (Mon-Fri, Oct-Mar)"),
        ("summer",         "DayTrip", 131, "Summer weekday (Mon-Fri, Apr-Sep)"),
        ("winter_weekend", "Weekend",  52, "Winter weekend (Sat-Sun, Oct-Mar)"),
        ("summer_weekend", "Weekend",  52, "Summer weekend (Sat-Sun, Apr-Sep)"),
    ]

    annual_cost_milp = 0.0
    annual_v2g_milp = 0.0
    annual_savings_dumb = 0.0

    for season, dwell_type, days_per_year, label in DAY_TYPES:
        log(f"━━━ {label} ({days_per_year} days/yr) ━━━")

        tru, plugged = v2g_mod.build_load_and_availability(v2g, dwell=dwell_type)
        buy, v2g_p, price_source = v2g_mod.load_prices_from_csv(
            csv_path, v2g, season=season
        )

        # Roll for DayTrip (arrival at 17:00)
        if dwell_type == "DayTrip":
            ROLL = 68
            buy     = np.roll(buy,     -ROLL)
            v2g_p   = np.roll(v2g_p,   -ROLL)
            tru     = np.roll(tru,     -ROLL)
            plugged = np.roll(plugged, -ROLL)

        log(f"  Buy: {buy.min()*1000:.1f}-{buy.max()*1000:.1f} EUR/MWh | "
            f"Plugged: {int(plugged.sum()*v2g.dt_h)}h/day")

        # ── Scenario A: Dumb ────────────────────────────────────
        log(f"  Running A - Dumb...")
        A = v2g_mod.run_dumb(v2g, buy, v2g_p, tru, plugged,
                             soc_init_pct, soc_final_pct)

        # ── Scenario B: Smart (no V2G) ──────────────────────────
        log(f"  Running B - Smart (no V2G)...")
        B = v2g_mod.run_smart_no_v2g(v2g, buy, v2g_p, tru, plugged,
                                      soc_init_pct, soc_final_pct)

        # ── Scenario C: MILP Day-Ahead ──────────────────────────
        log(f"  Running C - MILP Day-Ahead...")
        C = v2g_mod.run_milp_day_ahead(v2g, buy, v2g_p, tru, plugged,
                                        soc_init_pct, soc_final_pct)

        # ── Scenario D: MPC ─────────────────────────────────────
        log(f"  Running D - MPC (96 solves)...")
        D = v2g_mod.run_mpc_day_ahead(v2g, buy, v2g_p, tru, plugged,
                                       soc_init_pct, soc_final_pct,
                                       label="D - MPC perfect")

        results = {"A": A, "B": B, "C": C, "D": D}
        all_season_results[season] = results

        # ── Degradation sensitivity ─────────────────────────────
        log(f"  Running degradation sensitivity...")
        deg_df = v2g_mod.deg_sensitivity(
            v2g, buy, v2g_p, tru, plugged,
            deg_values, soc_init_pct, soc_final_pct
        )

        # ── Plot ────────────────────────────────────────────────
        out_png = str(OUT_DIR / f"results_{season}.png")
        log(f"  Plotting -> results_{season}.png")
        v2g_mod.plot_all(v2g, hours, A, B, C, D, deg_df,
                         season=label, out=out_png)

        # ── Collect KPIs ────────────────────────────────────────
        ref = A.cost_eur_day
        season_summary.append({
            "season": label,
            "days_per_year": days_per_year,
            "A_cost": round(A.cost_eur_day, 4),
            "B_cost": round(B.cost_eur_day, 4),
            "C_cost": round(C.cost_eur_day, 4),
            "D_cost": round(D.cost_eur_day, 4),
            "C_v2g_rev": round(C.v2g_revenue_eur_day, 4),
            "D_v2g_rev": round(D.v2g_revenue_eur_day, 4),
            "C_v2g_kwh": round(C.v2g_export_kwh_day, 2),
            "C_savings_vs_A": round(ref - C.cost_eur_day, 4),
        })

        annual_cost_milp += C.cost_eur_day * days_per_year
        annual_v2g_milp += C.v2g_revenue_eur_day * days_per_year
        annual_savings_dumb += (A.cost_eur_day - C.cost_eur_day) * days_per_year

    # ── Additional analysis plot ────────────────────────────────
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
        "annual_cost_milp": round(annual_cost_milp, 0),
        "annual_v2g_revenue": round(annual_v2g_milp, 0),
        "annual_savings_vs_dumb": round(annual_savings_dumb, 0),
        "seasons": season_summary,
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
    soc_init = float(data.get("soc_init", 45))
    deg_cost = float(data.get("deg_cost", 0.02))

    # Validate
    if not (20 <= soc_init <= 100):
        return jsonify({"error": "SoC must be 20-100%"}), 400
    if not (0 <= deg_cost <= 1):
        return jsonify({"error": "Degradation cost must be 0-1 EUR/kWh"}), 400

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
                    soc_init_pct=soc_init,
                    deg_cost=deg_cost,
                )
                progress_queue.put(json.dumps({
                    "type": "result",
                    "data": result,
                }))
            except Exception as e:
                tb = traceback.format_exc()
                print(f"  [GUI ERROR]\n{tb}")
                progress_queue.put(json.dumps({
                    "type": "error",
                    "message": str(e),
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

            # Check if it's a JSON result/error
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
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/output/<path:filename>")
def output_file(filename):
    """Serve generated PNG files."""
    return send_from_directory(str(OUT_DIR), filename)


# ═══════════════════════════════════════════════════════════════════
#  HTML / JS FRONTEND (single-file, no templates needed)
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
    padding: 24px 32px;
    text-align: center;
  }
  .header h1 { font-size: 1.8rem; margin-bottom: 4px; }
  .header p { opacity: 0.85; font-size: 0.95rem; }
  .container { max-width: 1400px; margin: 0 auto; padding: 24px; }
  .grid { display: grid; grid-template-columns: 340px 1fr; gap: 24px; }

  /* ── Controls Panel ── */
  .panel {
    background: var(--card);
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border: 1px solid var(--border);
  }
  .panel h2 {
    font-size: 1.1rem;
    color: var(--primary);
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--accent);
  }
  .field { margin-bottom: 16px; }
  .field label {
    display: block;
    font-weight: 600;
    margin-bottom: 4px;
    font-size: 0.9rem;
    color: var(--primary);
  }
  .field .hint {
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 4px;
  }
  .field input, .field select {
    width: 100%;
    padding: 10px 12px;
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 1rem;
    transition: border-color 0.2s;
  }
  .field input:focus { border-color: var(--accent); outline: none; }
  .btn {
    display: block;
    width: 100%;
    padding: 14px;
    border: none;
    border-radius: 8px;
    font-size: 1.05rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s;
    margin-top: 8px;
  }
  .btn-run {
    background: linear-gradient(135deg, var(--accent), #0097a7);
    color: white;
  }
  .btn-run:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,188,212,0.4); }
  .btn-run:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

  /* ── Progress Log ── */
  .log-box {
    background: #263238;
    color: #e0e0e0;
    border-radius: 8px;
    padding: 16px;
    font-family: 'Cascadia Code', 'Fira Code', monospace;
    font-size: 0.82rem;
    max-height: 220px;
    overflow-y: auto;
    margin-top: 16px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .log-box .log-ok { color: #69f0ae; }
  .log-box .log-err { color: #ff5252; }
  .log-box .log-info { color: #80deea; }

  /* ── Results ── */
  .results { display: none; }
  .results.show { display: block; }
  .results h2 {
    font-size: 1.2rem;
    color: var(--primary);
    margin: 24px 0 12px;
  }
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
  }
  .kpi-card {
    background: var(--card);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    border: 1px solid var(--border);
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  .kpi-card .kpi-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--primary);
  }
  .kpi-card .kpi-label {
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: 4px;
  }
  .kpi-card.green .kpi-value { color: var(--success); }

  /* ── Season table ── */
  .season-table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 24px;
    font-size: 0.88rem;
  }
  .season-table th {
    background: var(--primary);
    color: white;
    padding: 10px 8px;
    text-align: center;
  }
  .season-table td {
    padding: 8px;
    text-align: center;
    border-bottom: 1px solid var(--border);
  }
  .season-table tr:hover { background: #e8eaf6; }

  /* ── Image gallery ── */
  .gallery { margin-top: 16px; }
  .gallery img {
    width: 100%;
    max-width: 100%;
    border-radius: 8px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    cursor: pointer;
    transition: transform 0.2s;
  }
  .gallery img:hover { transform: scale(1.01); }

  /* ── Responsive ── */
  @media (max-width: 900px) {
    .grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<div class="header">
  <h1>S.KOe COOL — V2G Optimisation Dashboard</h1>
  <p>Day-Ahead MILP + Receding-Horizon MPC &nbsp;|&nbsp;
     Schmitz Cargobull AG 2025 &nbsp;|&nbsp;
     Binary Mutex &nbsp;|&nbsp; SoC 20–100%</p>
</div>

<div class="container">
<div class="grid">

  <!-- ─── LEFT: Controls ─── -->
  <div>
    <div class="panel">
      <h2>⚙️ Parameters</h2>

      <div class="field">
        <label>Arrival SoC (%)</label>
        <div class="hint">State of Charge when trailer arrives at depot (20–100)</div>
        <input type="number" id="soc_init" value="45" min="20" max="100" step="1">
      </div>

      <div class="field">
        <label>Degradation Cost (EUR/kWh)</label>
        <div class="hint">Battery wear cost per kWh cycled (0 = ignore, 0.02 = default LFP)</div>
        <input type="number" id="deg_cost" value="0.02" min="0" max="1" step="0.005">
      </div>

      <div class="field">
        <label>Departure SoC</label>
        <div class="hint">Fixed at 100% (cold-chain requirement)</div>
        <input type="text" value="100% (fixed)" disabled
               style="background:#f5f5f5; color:#999;">
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
const logBox   = document.getElementById("logBox");
const btnRun   = document.getElementById("btnRun");
const results  = document.getElementById("resultsPanel");
const kpiGrid  = document.getElementById("kpiGrid");
const seasonBody = document.getElementById("seasonBody");
const gallery  = document.getElementById("gallery");

function appendLog(msg, cls = "") {
    const span = document.createElement("span");
    span.className = cls;
    span.textContent = msg + "\n";
    logBox.appendChild(span);
    logBox.scrollTop = logBox.scrollHeight;
}

function startRun() {
    const socInit = parseFloat(document.getElementById("soc_init").value);
    const degCost = parseFloat(document.getElementById("deg_cost").value);

    if (isNaN(socInit) || socInit < 20 || socInit > 100) {
        alert("Arrival SoC must be between 20 and 100.");
        return;
    }
    if (isNaN(degCost) || degCost < 0 || degCost > 1) {
        alert("Degradation cost must be between 0 and 1.");
        return;
    }

    // Reset UI
    logBox.innerHTML = "";
    results.classList.remove("show");
    kpiGrid.innerHTML = "";
    seasonBody.innerHTML = "";
    gallery.innerHTML = "";
    btnRun.disabled = true;
    btnRun.textContent = "⏳ Running...";

    appendLog("Starting optimisation...", "log-info");
    appendLog(`  SoC init: ${socInit}%  |  deg: ${degCost} EUR/kWh`, "log-info");
    appendLog("");

    // POST to start
    fetch("/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ soc_init: socInit, deg_cost: degCost }),
    })
    .then(resp => {
        if (!resp.ok) {
            return resp.json().then(d => { throw new Error(d.error || "Server error"); });
        }
        return resp.json();
    })
    .then(() => {
        // Open SSE stream
        const evtSource = new EventSource("/stream");

        evtSource.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);

                if (data.type === "progress") {
                    let cls = "log-info";
                    if (data.message.includes("✓") || data.message.includes("DONE"))
                        cls = "log-ok";
                    if (data.message.includes("ERROR") || data.message.includes("WARN"))
                        cls = "log-err";
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
                    if (data.traceback) {
                        appendLog(data.traceback, "log-err");
                    }
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

    // KPI cards
    const kpis = [
        { label: "Annual Cost (MILP)",        value: `€${Number(data.annual_cost_milp).toLocaleString()}`,  green: false },
        { label: "Annual V2G Revenue",         value: `€${Number(data.annual_v2g_revenue).toLocaleString()}`, green: true },
        { label: "Annual Savings vs Dumb",     value: `€${Number(data.annual_savings_vs_dumb).toLocaleString()}`, green: true },
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
            <td style="color: ${s.C_savings_vs_A > 0 ? '#43a047' : '#e53935'}">
                ${s.C_savings_vs_A > 0 ? '+' : ''}${s.C_savings_vs_A.toFixed(4)}
            </td>
        </tr>
    `).join("");

    // Image gallery
    const ts = Date.now();
    gallery.innerHTML = data.images.map(img => `
        <a href="/output/${img}?t=${ts}" target="_blank">
            <img src="/output/${img}?t=${ts}" alt="${img}"
                 loading="lazy"
                 onerror="this.style.display='none'">
        </a>
    `).join("");

    // Scroll to results
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