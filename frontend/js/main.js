/* ============================================================
   main.js  –  App Initialisation, Tab Routing, API Helpers
   ============================================================ */

const API = "http://localhost:5000/api";

// ── Global state ─────────────────────────────────────────
const State = {
  options:    { crops: [], states: [], models: [], msp_crops: [] },
  production: {},
  area:       {},
  yieldData:  {},
  topStates:  {},
  yoy:        {},
  evalData:   [],
  cvData:     [],
  residuals:  null,
  mspTrend:   {},
  fcMethod:   "arima",
  fcChart:    null,
  mspFcChart: null,
  mspDistChart: null,
};

// ── API fetch helper ──────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.message || `HTTP ${res.status}`);
  }
  const json = await res.json();
  if (json.status !== "ok") throw new Error(json.message || "API error");
  return json.data;
}

// ── Loader ────────────────────────────────────────────────
function showLoader(msg = "Loading…") {
  document.getElementById("loader").classList.remove("hidden");
  document.getElementById("loaderText").textContent = msg;
}
function hideLoader() {
  document.getElementById("loader").classList.add("hidden");
}

// ── Toast ─────────────────────────────────────────────────
function showToast(msg, type = "") {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className   = "toast " + type;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 3200);
}

// ── Status dot ───────────────────────────────────────────
function setStatus(online) {
  const dot  = document.getElementById("statusDot");
  const text = document.getElementById("statusText");
  dot.className  = "status-dot " + (online ? "online" : "");
  text.textContent = online ? "API Online" : "API Offline";
}

// ── Tab routing ───────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    const id = "panel-" + btn.dataset.tab;
    document.getElementById(id).classList.add("active");
  });
});

// ── EDA metric switcher (guarded — panel removed) ─────────
const _edaMetric = document.getElementById("edaMetric");
if (_edaMetric) _edaMetric.addEventListener("change", function () {
  renderEdaTrend(this.value);
});

// ── Top states crop switcher (guarded — panel removed) ────
const _topStatesCrop = document.getElementById("topStatesCrop");
if (_topStatesCrop) _topStatesCrop.addEventListener("change", function () {
  renderTopStates(this.value);
});

// ── Feature importance model switcher ─────────────────────
function loadFeatureImportance() {
  const sel = document.getElementById("fiModelSelect");
  if (!sel) return;
  const model = sel.value;
  const lbl = document.getElementById("fiModelLabel");
  if (lbl) lbl.textContent = model;
  apiFetch(`/xai/global?model=${encodeURIComponent(model)}`)
    .then(data => renderFeatureImportance(data))
    .catch(e => showToast("Global SHAP unavailable for this model", "error"));
}

// ── Heatmap ───────────────────────────────────────────────
async function loadHeatmap() {
  const crop  = document.getElementById("heatmapCrop").value;
  const topN  = document.getElementById("heatmapTopN").value;
  document.getElementById("heatmapCropLabel").textContent =
    crop.charAt(0).toUpperCase() + crop.slice(1);
  try {
    const data = await apiFetch(`/eda/heatmap?crop=${crop}&top_n=${topN}`);
    renderHeatmap("heatmapTable", data);
  } catch (e) { showToast(e.message, "error"); }
}

// ── Heatmap renderer ─────────────────────────────────────
function renderHeatmap(tableId, data) {
  const table  = document.getElementById(tableId);
  const { states, years, matrix } = data;

  // Find max for colour scaling
  const maxVal = Math.max(...matrix.flat());

  let html = "<thead><tr><th>State</th>";
  years.forEach(y => { html += `<th>${y}</th>`; });
  html += "</tr></thead><tbody>";

  states.forEach((state, si) => {
    html += `<tr><td style="font-weight:600;white-space:nowrap">${state}</td>`;
    matrix[si].forEach(val => {
      const pct   = maxVal > 0 ? val / maxVal : 0;
      const alpha = (0.1 + pct * 0.85).toFixed(2);
      const color = `rgba(37,99,235,${alpha})`;
      const text  = pct > 0.5 ? "#fff" : "#1e293b";
      html += `<td style="background:${color};color:${text}">${val.toFixed(2)}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody>";
  table.innerHTML = html;
}

// ── Populate select options ───────────────────────────────
function populateSelect(id, items, caps = true) {
  const sel = document.getElementById(id);
  if (!sel) return;
  sel.innerHTML = items.map(v => {
    const label = caps ? (v.charAt(0).toUpperCase() + v.slice(1)) : v;
    return `<option value="${v}">${label}</option>`;
  }).join("");
}

// ── Main init ─────────────────────────────────────────────
async function init() {
  showLoader("Connecting to ML server…");

  // 1. Health check
  try {
    await apiFetch("/health");
    setStatus(true);
    document.getElementById("apiBanner").classList.remove("show");
  } catch {
    setStatus(false);
    document.getElementById("apiBanner").classList.add("show");
    hideLoader();
    return;
  }

  // 2. Load options
  try {
    State.options = await apiFetch("/options");
    populateSelect("pCrop",     State.options.crops);
    populateSelect("pState",    State.options.states, false);
    populateSelect("pModel",    State.options.models, false);
    populateSelect("fcCrop",    State.options.crops);
    populateSelect("fcState",   State.options.states, false);

    // Default fcState to Rajasthan
    const fcSel = document.getElementById("fcState");
    [...fcSel.options].forEach(o => {
      if (o.value === "Rajasthan") o.selected = true;
    });
  } catch (e) { showToast("Could not load options: " + e.message, "error"); }

  // 3. Load stats
  try {
    const stats = await apiFetch("/stats");
    document.getElementById("s-rows").textContent   = stats.total_rows.toLocaleString();
    if (document.getElementById("s-crops")) document.getElementById("s-crops").textContent = stats.num_crops;
    if (document.getElementById("s-years")) document.getElementById("s-years").textContent = `${stats.year_max - stats.year_min + 1}`;
    if (document.getElementById("s-ml-models")) document.getElementById("s-ml-models").textContent = stats.num_ml_models || 22;
    if (document.getElementById("s-dl-models")) document.getElementById("s-dl-models").textContent = stats.num_dl_models || 2;
  } catch (e) { console.error("Stats error", e); }

  // 4. Load EDA data in parallel
  showLoader("Loading charts…");
  try {
    const [prod, area, yld, top, yoy] = await Promise.all([
      apiFetch("/eda/production"),
      apiFetch("/eda/area"),
      apiFetch("/eda/yield"),
      apiFetch("/eda/top_states"),
      apiFetch("/eda/yoy"),
    ]);
    State.production = prod;
    State.area       = area;
    State.yieldData  = yld;
    State.topStates  = top;
    State.yoy        = yoy;
  } catch (e) { showToast("EDA load error: " + e.message, "error"); }

  // 5. Load model data
  try {
    const [ev, cv] = await Promise.all([
      apiFetch("/models/evaluation"),
      apiFetch("/models/cv"),
    ]);
    State.evalData = ev;
    State.cvData   = cv;

    // Show best R² in overview card
    if (ev.length) {
      document.getElementById("s-best-r2").textContent = ev[0].r2.toFixed(4);
    }
  } catch {}

  // 6. Residuals
  try {
    State.residuals = await apiFetch("/models/residuals?model=Random+Forest");
  } catch {}

  // 7. MSP
  try {
    State.mspTrend = await apiFetch("/msp/trend");
  } catch {}

  // 8. Render charts
  renderOverview();
  renderEvalTable();
  renderModelCharts();
  renderFeatureImportance(null);
  renderResiduals();
  renderMspTrend();
  renderNationalForecast();
  loadMspDistrict();

  hideLoader();
  showToast("Dashboard loaded ✓", "success");
}

window.addEventListener("DOMContentLoaded", init);
