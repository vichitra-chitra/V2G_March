#!/usr/bin/env python3
"""
S.KOe COOL — V2G Optimisation Web GUI
Schmitz Cargobull AG | 2025

Flask-based parameter panel — works in GitHub Codespaces, devcontainers,
and any environment without a display server.

Run:  python gui.py
Then open the forwarded port (default 5000) in your browser.
"""

from __future__ import annotations
import sys
import io
import json
import queue
import threading
import importlib
import traceback
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, Response, stream_with_context

app = Flask(__name__)

# ── Shared state ──────────────────────────────────────────────────────────────
_log_queue: queue.Queue = queue.Queue()
_running   = threading.Event()


def _enqueue(text: str):
    _log_queue.put(text)


class _QueueStream(io.TextIOBase):
    def write(self, s: str) -> int:
        if s:
            _enqueue(s)
        return len(s)
    def flush(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  HTML TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════════

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>S.KOe COOL — V2G Optimisation</title>
<style>
  /* ── Reset & base ── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:        #1E2127;
    --panel:     #282C34;
    --card:      #2E3440;
    --input-bg:  #3B4252;
    --border:    #434C5E;
    --text:      #ECEFF4;
    --muted:     #7B8290;
    --accent:    #88C0D0;
    --accent2:   #A3BE8C;
    --warn:      #EBCB8B;
    --error:     #BF616A;
    --btn:       #5E81AC;
    --btn-hover: #6E91BC;
    --btn-reset: #4C566A;
  }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* ── Banner ── */
  header {
    background: var(--btn);
    padding: 12px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
  }
  header h1 { font-size: 1.15rem; font-weight: 700; color: #fff; }
  header span { font-size: 0.82rem; color: #BCD0E8; }

  /* ── Main layout ── */
  .main {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  /* ── Left param panel ── */
  .params {
    width: 480px;
    min-width: 380px;
    background: var(--bg);
    overflow-y: auto;
    padding: 16px 16px 80px;
    flex-shrink: 0;
    border-right: 1px solid var(--border);
  }

  /* ── Right log panel ── */
  .log-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: var(--panel);
    min-width: 0;
  }
  .log-header {
    padding: 10px 16px;
    color: var(--accent);
    font-weight: 700;
    font-size: 0.95rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
  }
  #log {
    flex: 1;
    overflow-y: auto;
    background: #13151A;
    color: #D8DEE9;
    font-family: 'Consolas', 'Fira Code', monospace;
    font-size: 0.82rem;
    padding: 12px 16px;
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* ── Section headers ── */
  .section-header {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--accent);
    font-size: 0.9rem;
    font-weight: 700;
    padding: 14px 0 6px;
  }
  .section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  /* ── Cards ── */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
  }

  /* ── Param rows ── */
  .param-row {
    display: flex;
    align-items: center;
    padding: 5px 0;
    gap: 8px;
  }
  .param-row label {
    flex: 1;
    font-size: 0.875rem;
    color: #B0BAC8;
    cursor: default;
  }
  .param-row input {
    width: 110px;
    background: var(--input-bg);
    border: 1px solid var(--border);
    border-radius: 5px;
    color: var(--text);
    font-family: 'Consolas', monospace;
    font-size: 0.875rem;
    padding: 4px 8px;
    outline: none;
    transition: border-color 0.15s;
  }
  .param-row input:focus {
    border-color: var(--accent);
  }
  .param-row input.error {
    border-color: var(--error);
  }
  .param-row .unit {
    width: 52px;
    color: var(--muted);
    font-size: 0.8rem;
  }

  /* CSV row */
  .csv-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 0;
  }
  .csv-row label { font-size: 0.875rem; color: #B0BAC8; white-space: nowrap; }
  .csv-row input[type="text"] {
    flex: 1;
    background: var(--input-bg);
    border: 1px solid var(--border);
    border-radius: 5px;
    color: var(--text);
    font-family: 'Consolas', monospace;
    font-size: 0.8rem;
    padding: 4px 8px;
    outline: none;
  }
  .csv-row input[type="text"]:focus { border-color: var(--accent); }

  /* ── Derived display ── */
  .derived-row {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    font-size: 0.85rem;
  }
  .derived-row .d-key { color: var(--muted); }
  .derived-row .d-val { color: var(--accent2); font-family: 'Consolas', monospace; font-weight: 700; }

  /* ── Tooltip ── */
  [data-tip] { position: relative; cursor: help; }
  [data-tip]:hover::after {
    content: attr(data-tip);
    position: absolute;
    left: 50%; bottom: calc(100% + 6px);
    transform: translateX(-50%);
    background: #2E3440;
    color: #ECEFF4;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 0.78rem;
    white-space: normal;
    width: 240px;
    z-index: 999;
    pointer-events: none;
    line-height: 1.4;
  }

  /* ── Bottom button bar ── */
  .btn-bar {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 480px;
    background: var(--bg);
    border-top: 1px solid var(--border);
    padding: 10px 16px;
    display: flex;
    gap: 10px;
    align-items: center;
    z-index: 100;
  }
  button {
    padding: 8px 20px;
    border: none;
    border-radius: 6px;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s, opacity 0.15s;
  }
  #btn-run {
    background: var(--btn);
    color: #fff;
    flex: 1;
  }
  #btn-run:hover:not(:disabled) { background: var(--btn-hover); }
  #btn-run:disabled { opacity: 0.55; cursor: not-allowed; }
  #btn-reset {
    background: var(--btn-reset);
    color: var(--text);
  }
  #btn-reset:hover { background: #5C6A80; }
  #btn-clear {
    background: var(--btn-reset);
    color: var(--text);
  }
  #btn-clear:hover { background: #5C6A80; }
  .status {
    margin-left: auto;
    font-size: 0.82rem;
    color: var(--accent2);
    white-space: nowrap;
  }

  /* ── Validation banner ── */
  #err-banner {
    display: none;
    background: #3B1B1B;
    border: 1px solid var(--error);
    color: var(--error);
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 0.85rem;
    margin-bottom: 8px;
  }

  /* ── Log colours ── */
  .log-info    { color: #88C0D0; }
  .log-success { color: #A3BE8C; }
  .log-warn    { color: #EBCB8B; }
  .log-error   { color: #BF616A; }

  /* scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>

<!-- Banner -->
<header>
  <h1>⚡ S.KOe COOL &nbsp;—&nbsp; V2G Optimisation Control Panel</h1>
  <span>Schmitz Cargobull AG &nbsp;·&nbsp; 2025</span>
</header>

<div class="main">

  <!-- ═══ LEFT: Parameters ═══ -->
  <div class="params" id="params-panel">

    <div id="err-banner"></div>

    <!-- CSV -->
    <div class="section-header">📂 Price Data</div>
    <div class="card">
      <div class="csv-row">
        <label for="csv_path">CSV file path</label>
        <input type="text" id="csv_path" value="2025_Electricity_Price.csv">
      </div>
    </div>

    <!-- Battery -->
    <div class="section-header">🔋 Battery Parameters</div>
    <div class="card" id="card-battery"></div>

    <!-- Power -->
    <div class="section-header">⚡ Power Limits</div>
    <div class="card" id="card-power"></div>

    <!-- Efficiency -->
    <div class="section-header">⚙️ Efficiency</div>
    <div class="card" id="card-eff"></div>

    <!-- Economics -->
    <div class="section-header">💶 Economics</div>
    <div class="card" id="card-econ"></div>

    <!-- Grid -->
    <div class="section-header">🔌 Grid Connection <span style="font-weight:400;color:var(--muted);font-size:0.8rem">(0 = unlimited)</span></div>
    <div class="card" id="card-grid"></div>

    <!-- Simulation -->
    <div class="section-header">🕐 Simulation Settings</div>
    <div class="card" id="card-sim"></div>

    <!-- Derived -->
    <div class="section-header">📐 Derived Values <span style="font-weight:400;color:var(--muted);font-size:0.8rem">(auto)</span></div>
    <div class="card" id="card-derived">
      <div class="derived-row"><span class="d-key">E_min (kWh)</span><span class="d-val" id="d-emin">—</span></div>
      <div class="derived-row"><span class="d-key">E_max (kWh)</span><span class="d-val" id="d-emax">—</span></div>
      <div class="derived-row"><span class="d-key">Round-trip efficiency</span><span class="d-val" id="d-rt">—</span></div>
      <div class="derived-row"><span class="d-key">p_c_max (kW)</span><span class="d-val" id="d-pc">—</span></div>
      <div class="derived-row"><span class="d-key">p_d_max (kW)</span><span class="d-val" id="d-pd">—</span></div>
    </div>

  </div><!-- /params -->

  <!-- ═══ RIGHT: Log ═══ -->
  <div class="log-panel">
    <div class="log-header">
      <span>📋 Console Output</span>
      <span id="log-status" style="color:var(--accent2);font-size:0.82rem;font-weight:400">Ready</span>
    </div>
    <div id="log">  S.KOe COOL V2G Optimisation GUI — web edition.
  Edit parameters on the left, then click RUN OPTIMISATION.

</div>
  </div>

</div><!-- /main -->

<!-- Button bar -->
<div class="btn-bar">
  <button id="btn-run"   onclick="runOpt()">▶&nbsp; RUN OPTIMISATION</button>
  <button id="btn-reset" onclick="resetDefaults()">↺&nbsp; Reset</button>
  <button id="btn-clear" onclick="clearLog()">✕&nbsp; Clear Log</button>
  <span class="status" id="status-txt">Ready</span>
</div>

<script>
// ── Parameter definitions ─────────────────────────────────────────────────
const PARAMS = {
  battery: [
    { key:"battery_capacity_kWh", label:"Total capacity",    unit:"kWh",  default:70.0,  tip:"Total physical battery nameplate capacity in kWh." },
    { key:"usable_capacity_kWh",  label:"Usable capacity",   unit:"kWh",  default:60.0,  tip:"Energy window for charge/discharge (SoC 20–100%)." },
    { key:"soc_min_pct",          label:"SoC minimum",       unit:"%",    default:20.0,  tip:"Minimum SoC % — cold-chain safety floor." },
    { key:"soc_max_pct",          label:"SoC maximum",       unit:"%",    default:100.0, tip:"Maximum SoC % — 100% = full charge at departure." },
  ],
  power: [
    { key:"charge_power_kW",    label:"Max charge power",  unit:"kW", default:22.0, tip:"Maximum AC charging power (ISO 15118 limit)." },
    { key:"discharge_power_kW", label:"Max V2G discharge", unit:"kW", default:22.0, tip:"Maximum V2G discharge power (ISO 15118-2)." },
  ],
  eff: [
    { key:"eta_charge",    label:"Charge efficiency η_c",    unit:"—", default:0.92, tip:"One-way AC→DC efficiency. Typical: 0.92." },
    { key:"eta_discharge", label:"Discharge efficiency η_d", unit:"—", default:0.92, tip:"One-way DC→AC efficiency. Typical: 0.92." },
  ],
  econ: [
    { key:"deg_cost_eur_kwh", label:"Degradation cost", unit:"€/kWh", default:0.02, tip:"Battery wear cost in €/kWh cycled. LFP default: 0.02." },
  ],
  grid: [
    { key:"depot_connection_kVA",  label:"Depot connection",  unit:"kVA", default:0.0, tip:"Depot grid connection limit in kVA. 0 = no limit." },
    { key:"transformer_limit_kVA", label:"Transformer limit", unit:"kVA", default:0.0, tip:"Depot transformer limit in kVA. 0 = no limit." },
  ],
  sim: [
    { key:"dt_h",         label:"Time step",    unit:"h", default:0.25, tip:"Slot length in hours. 0.25 = 15-minute resolution." },
    { key:"n_slots",      label:"Slots per 24h",unit:"—", default:96,   tip:"96 = 4 slots/hour × 24 hours." },
    { key:"soc_init_pct", label:"Arrival SoC",  unit:"%", default:45.0, tip:"Trailer arrival State-of-Charge. Typically 45%." },
  ],
};

// ── Build input rows ───────────────────────────────────────────────────────
function buildCard(cardId, params) {
  const card = document.getElementById(cardId);
  params.forEach(p => {
    const row = document.createElement('div');
    row.className = 'param-row';
    row.innerHTML = `
      <label for="${p.key}" data-tip="${p.tip}">${p.label}</label>
      <input type="number" id="${p.key}" value="${p.default}"
             step="any" oninput="updateDerived()">
      <span class="unit">${p.unit}</span>`;
    card.appendChild(row);
  });
}

buildCard('card-battery', PARAMS.battery);
buildCard('card-power',   PARAMS.power);
buildCard('card-eff',     PARAMS.eff);
buildCard('card-econ',    PARAMS.econ);
buildCard('card-grid',    PARAMS.grid);
buildCard('card-sim',     PARAMS.sim);

// ── Derived values ─────────────────────────────────────────────────────────
function updateDerived() {
  const g = id => { const el = document.getElementById(id); return el ? parseFloat(el.value) : NaN; };
  const usable  = g('usable_capacity_kWh');
  const socMin  = g('soc_min_pct');
  const socMax  = g('soc_max_pct');
  const etaC    = g('eta_charge');
  const etaD    = g('eta_discharge');
  const pc      = g('charge_power_kW');
  const pd      = g('discharge_power_kW');
  const depKva  = g('depot_connection_kVA');
  const traKva  = g('transformer_limit_kVA');

  const set = (id, v) => { const el = document.getElementById(id); if(el) el.textContent = v; };

  if (isNaN(usable)||isNaN(socMin)||isNaN(socMax)) return;
  set('d-emin', (usable * socMin / 100).toFixed(2) + ' kWh');
  set('d-emax', (usable * socMax / 100).toFixed(2) + ' kWh');
  set('d-rt',   isNaN(etaC)||isNaN(etaD) ? '—' : (etaC*etaD*100).toFixed(1) + ' %');

  const limits = [depKva, traKva].filter(v => !isNaN(v) && v > 0).map(v => v * 0.95);
  const cap    = limits.length ? Math.min(...limits) : Infinity;
  set('d-pc', isNaN(pc) ? '—' : (cap===Infinity ? pc.toFixed(1) : Math.min(pc,cap).toFixed(1)) + ' kW');
  set('d-pd', isNaN(pd) ? '—' : (cap===Infinity ? pd.toFixed(1) : Math.min(pd,cap).toFixed(1)) + ' kW');
}
updateDerived();

// ── Reset ──────────────────────────────────────────────────────────────────
function resetDefaults() {
  Object.values(PARAMS).flat().forEach(p => {
    const el = document.getElementById(p.key);
    if (el) el.value = p.default;
  });
  document.getElementById('csv_path').value = '2025_Electricity_Price.csv';
  updateDerived();
  hideBanner();
}

// ── Log helpers ────────────────────────────────────────────────────────────
function clearLog() {
  document.getElementById('log').textContent = '';
}

function appendLog(text) {
  const el = document.getElementById('log');
  el.textContent += text;
  el.scrollTop = el.scrollHeight;
}

// ── Validation banner ──────────────────────────────────────────────────────
function showBanner(msg) {
  const b = document.getElementById('err-banner');
  b.textContent = '⚠  ' + msg;
  b.style.display = 'block';
}
function hideBanner() {
  document.getElementById('err-banner').style.display = 'none';
}

// ── Collect values ─────────────────────────────────────────────────────────
function collectValues() {
  const vals = {};
  Object.values(PARAMS).flat().forEach(p => {
    const el = document.getElementById(p.key);
    vals[p.key] = el ? el.value : String(p.default);
  });
  vals['csv_path'] = document.getElementById('csv_path').value.trim();
  return vals;
}

// ── Client-side validation ─────────────────────────────────────────────────
function validate(v) {
  const num = k => { const n = parseFloat(v[k]); return isNaN(n) ? null : n; };
  const checks = [
    [num('soc_min_pct') >= num('soc_max_pct'), 'SoC minimum must be less than SoC maximum.'],
    [!(num('eta_charge')>0 && num('eta_charge')<=1), 'Charge efficiency must be between 0 and 1.'],
    [!(num('eta_discharge')>0 && num('eta_discharge')<=1), 'Discharge efficiency must be between 0 and 1.'],
    [!v['csv_path'], 'CSV file path cannot be empty.'],
  ];
  for (const [fail, msg] of checks) if (fail) return msg;
  return null;
}

// ── RUN ────────────────────────────────────────────────────────────────────
function setStatus(txt, color) {
  const s = document.getElementById('status-txt');
  const l = document.getElementById('log-status');
  s.textContent = l.textContent = txt;
  s.style.color = l.style.color = color;
}

async function runOpt() {
  hideBanner();
  const vals = collectValues();
  const err  = validate(vals);
  if (err) { showBanner(err); return; }

  const btn = document.getElementById('btn-run');
  btn.disabled = true;
  btn.textContent = '⏳  Running…';
  setStatus('Running…', '#EBCB8B');
  clearLog();

  // POST params, then stream log via EventSource
  const res = await fetch('/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(vals),
  });

  if (!res.ok) {
    showBanner('Server error starting run.');
    btn.disabled = false;
    btn.textContent = '▶  RUN OPTIMISATION';
    setStatus('Error', '#BF616A');
    return;
  }

  // Stream log
  const es = new EventSource('/stream');
  es.onmessage = e => {
    if (e.data === '__DONE__') {
      es.close();
      btn.disabled = false;
      btn.textContent = '▶  RUN OPTIMISATION';
      setStatus('Completed ✓', '#A3BE8C');
    } else if (e.data === '__ERROR__') {
      es.close();
      btn.disabled = false;
      btn.textContent = '▶  RUN OPTIMISATION';
      setStatus('Error ✗', '#BF616A');
    } else {
      appendLog(e.data + '\n');
    }
  };
  es.onerror = () => {
    es.close();
    btn.disabled = false;
    btn.textContent = '▶  RUN OPTIMISATION';
    setStatus('Stream error', '#BF616A');
  };
}
</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/run", methods=["POST"])
def run_endpoint():
    if _running.is_set():
        return jsonify({"error": "Already running"}), 409

    params = request.get_json(force=True)
    thread = threading.Thread(target=_run_worker, args=(params,), daemon=True)
    thread.start()
    return jsonify({"status": "started"})


@app.route("/stream")
def stream():
    def generate():
        while True:
            try:
                msg = _log_queue.get(timeout=120)
                if msg in ("__DONE__", "__ERROR__"):
                    yield f"data: {msg}\n\n"
                    break
                # Escape newlines for SSE
                for line in msg.splitlines(keepends=False):
                    yield f"data: {line}\n\n"
            except queue.Empty:
                yield "data: \n\n"   # keep-alive
    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTIMISATION WORKER
# ═══════════════════════════════════════════════════════════════════════════════

def _run_worker(params: dict):
    _running.set()
    old_out, old_err = sys.stdout, sys.stderr
    stream_obj       = _QueueStream()
    sys.stdout       = stream_obj
    sys.stderr       = stream_obj

    try:
        _execute(params)
        _enqueue("__DONE__")
    except Exception:
        traceback.print_exc()
        _enqueue("__ERROR__")
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        _running.clear()


def _execute(params: dict):
    import numpy as np

    script_dir = str(Path(__file__).parent.resolve())
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    import Prototype1optimization_v2 as v2g_mod
    importlib.reload(v2g_mod)

    csv_path   = params["csv_path"]
    soc_init   = float(params["soc_init_pct"])
    soc_final  = 100.0

    v2g = v2g_mod.V2GParams(
        battery_capacity_kWh  = float(params["battery_capacity_kWh"]),
        usable_capacity_kWh   = float(params["usable_capacity_kWh"]),
        soc_min_pct           = float(params["soc_min_pct"]),
        soc_max_pct           = float(params["soc_max_pct"]),
        charge_power_kW       = float(params["charge_power_kW"]),
        discharge_power_kW    = float(params["discharge_power_kW"]),
        eta_charge            = float(params["eta_charge"]),
        eta_discharge         = float(params["eta_discharge"]),
        deg_cost_eur_kwh      = float(params["deg_cost_eur_kwh"]),
        dt_h                  = float(params["dt_h"]),
        n_slots               = int(params["n_slots"]),
        depot_connection_kVA  = float(params["depot_connection_kVA"]),
        transformer_limit_kVA = float(params["transformer_limit_kVA"]),
    )

    print("=" * 65)
    print("  S.KOe COOL — V2G Optimisation  (launched from Web GUI)")
    print("=" * 65)
    print(f"\n  Battery : {v2g.battery_capacity_kWh} kWh total | "
          f"{v2g.usable_capacity_kWh} kWh usable")
    print(f"  SoC     : {v2g.soc_min_pct:.0f}%–{v2g.soc_max_pct:.0f}%  "
          f"(E_min={v2g.E_min:.1f} kWh, E_max={v2g.E_max:.1f} kWh)")
    print(f"  Power   : charge {v2g.p_c_max:.0f} kW | discharge {v2g.p_d_max:.0f} kW")
    print(f"  deg     : {v2g.deg_cost_eur_kwh:.4f} EUR/kWh cycled")
    print(f"  Arrival SoC : {soc_init:.0f}%  |  Departure SoC : 100%")
    print(f"  CSV     : {csv_path}\n")

    print("  Generating reference cards...")
    v2g_mod.generate_abbreviation_legend("abbreviation_legend.png")
    v2g_mod.generate_equations_card("equations_reference.png")

    deg_values = v2g_mod.load_deg_sensitivity(v2g)
    hours      = np.arange(v2g.n_slots) * v2g.dt_h

    all_season_results: dict = {}
    annual_cost_milp    = 0.0
    annual_v2g_milp     = 0.0
    annual_savings_dumb = 0.0

    DAY_TYPES = [
        ("winter",         "DayTrip", 130, "Winter weekday  (Mon-Fri, Oct-Mar)"),
        ("summer",         "DayTrip", 131, "Summer weekday  (Mon-Fri, Apr-Sep)"),
        ("winter_weekend", "Weekend",  52, "Winter weekend  (Sat-Sun, Oct-Mar)"),
        ("summer_weekend", "Weekend",  52, "Summer weekend  (Sat-Sun, Apr-Sep)"),
    ]

    for season, dwell_type, days_per_year, label in DAY_TYPES:
        print(f"\n{'='*65}")
        print(f"  {label}  ({days_per_year} days/year)")
        print(f"{'='*65}")

        tru, plugged = v2g_mod.build_load_and_availability(v2g, dwell=dwell_type)
        buy, v2g_p, price_source = v2g_mod.load_prices_from_csv(
            csv_path, v2g, season=season)

        if dwell_type == "DayTrip":
            ROLL    = 68
            buy     = np.roll(buy,     -ROLL)
            v2g_p   = np.roll(v2g_p,   -ROLL)
            tru     = np.roll(tru,     -ROLL)
            plugged = np.roll(plugged, -ROLL)

        print(f"  Prices: {price_source}")
        print(f"  Buy range : {buy.min()*1000:.1f}–{buy.max()*1000:.1f} EUR/MWh  |  "
              f"Plugged : {int(plugged.sum()*v2g.dt_h)}h/day")

        A = v2g_mod.run_dumb(
            v2g, buy, v2g_p, tru, plugged, soc_init, soc_final)
        B = v2g_mod.run_smart_no_v2g(
            v2g, buy, v2g_p, tru, plugged, soc_init, soc_final)
        C = v2g_mod.run_milp_day_ahead(
            v2g, buy, v2g_p, tru, plugged, soc_init, soc_final)
        D = v2g_mod.run_mpc_day_ahead(
            v2g, buy, v2g_p, tru, plugged, soc_init, soc_final,
            label="D - MPC perfect")

        results = {"A": A, "B": B, "C": C, "D": D}
        deg_df  = v2g_mod.deg_sensitivity(
            v2g, buy, v2g_p, tru, plugged, deg_values, soc_init, soc_final)

        all_season_results[season] = results
        v2g_mod.print_report(v2g, results, deg_df,
                              season=label, price_source=price_source)

        v2g_mod.plot_all(v2g, hours, A, B, C, D, deg_df,
                          season=label, out=f"results_{season}.png")

        annual_cost_milp    += C.cost_eur_day        * days_per_year
        annual_v2g_milp     += C.v2g_revenue_eur_day * days_per_year
        annual_savings_dumb += (A.cost_eur_day - C.cost_eur_day) * days_per_year

    print("\n  Generating additional analysis graphs...")
    tru_w, _ = v2g_mod.build_load_and_availability(v2g, dwell="Extended")
    buy_w, v2g_p_w, _ = v2g_mod.load_prices_from_csv(csv_path, v2g, season="winter")
    v2g_mod.plot_additional_analysis(
        v2g, hours,
        all_season_results["winter"]["A"],
        all_season_results["winter"]["B"],
        all_season_results["winter"]["C"],
        all_season_results["winter"]["D"],
        buy_w, v2g_p_w, all_season_results, csv_path,
        out="additional_analysis.png",
    )

    print(f"\n{'='*65}")
    print(f"  ANNUAL SUMMARY — Single Trailer (Scenario C MILP)")
    print(f"{'='*65}")
    print(f"  Annual energy cost  : EUR {annual_cost_milp:>8,.0f} / year")
    print(f"  Annual V2G revenue  : EUR {annual_v2g_milp:>8,.0f} / year")
    print(f"  Annual savings vs A : EUR {annual_savings_dumb:>8,.0f} / year")
    print(f"\n  All output PNG files saved to working directory.")
    print(f"    abbreviation_legend.png")
    print(f"    equations_reference.png")
    print(f"    results_winter.png")
    print(f"    results_summer.png")
    print(f"    results_winter_weekend.png")
    print(f"    results_summer_weekend.png")
    print(f"    additional_analysis.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = 5000
    print(f"\n  S.KOe COOL — V2G Web GUI")
    print(f"  Open in browser: http://localhost:{port}")
    print(f"  In Codespaces: use the 'Ports' tab to open the forwarded port.\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)