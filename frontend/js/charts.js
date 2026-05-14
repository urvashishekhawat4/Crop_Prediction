/* ============================================================
   charts.js  –  All Chart.js rendering functions
   ============================================================ */

const CROP_COLORS = {
  bajra:  "#ef4444",
  jowar:  "#f97316",
  maize:  "#eab308",
  rice:   "#22c55e",
  wheat:  "#3b82f6",
  gram:   "#a855f7",
  barley: "#06b6d4",
};

const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: true,
  plugins: {
    legend: { labels: { font: { size: 11 }, boxWidth: 14 } },
    tooltip: { mode: "index", intersect: false },
  },
  scales: {
    x: { grid: { color: "#f1f5f9" }, ticks: { font: { size: 10 } } },
    y: { grid: { color: "#f1f5f9" }, ticks: { font: { size: 10 } } },
  },
};

// Registry to destroy before recreating
const _charts = {};
function makeChart(id, config) {
  if (_charts[id]) { _charts[id].destroy(); }
  const ctx = document.getElementById(id);
  if (!ctx) return null;
  _charts[id] = new Chart(ctx, config);
  return _charts[id];
}

// ── Overview ─────────────────────────────────────────────
function renderOverview() {
  const prod = State.production;
  const crops = Object.keys(prod);
  if (!crops.length) return;

  // All years from first crop
  const allYears = prod[crops[0]]?.years ?? [];

  // Line chart: all crops production
  makeChart("overviewProdChart", {
    type: "line",
    data: {
      labels: allYears,
      datasets: crops.map(crop => ({
        label: crop.charAt(0).toUpperCase() + crop.slice(1),
        data:  prod[crop].values,
        borderColor: CROP_COLORS[crop] || "#888",
        backgroundColor: (CROP_COLORS[crop] || "#888") + "18",
        borderWidth: 2, pointRadius: 2, tension: 0.35, fill: false,
      })),
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: { ...CHART_DEFAULTS.plugins, legend: { position: "top", labels: { font: { size: 10 }, boxWidth: 12 } } },
      scales: {
        x: { ...CHART_DEFAULTS.scales.x, ticks: { maxTicksLimit: 8, font: { size: 9 } } },
        y: { ...CHART_DEFAULTS.scales.y, title: { display: true, text: "M Tonnes", font: { size: 10 } } },
      },
    },
  });

  // Bar chart: top 5 states wheat latest year
  const top = State.topStates["wheat"];
  if (top) {
    const top5States = top.states.slice(0, 5);
    const top5Vals   = top.values.slice(0, 5);
    makeChart("overviewStateChart", {
      type: "bar",
      data: {
        labels: top5States,
        datasets: [{
          label: "Cumulative Production (B Tonnes)",
          data: top5Vals,
          backgroundColor: ["#3b82f6","#22c55e","#f97316","#a855f7","#06b6d4"],
          borderRadius: 8,
        }],
      },
      options: {
        ...CHART_DEFAULTS,
        plugins: { legend: { display: false }, tooltip: CHART_DEFAULTS.plugins.tooltip },
        scales: {
          x: { ...CHART_DEFAULTS.scales.x },
          y: { ...CHART_DEFAULTS.scales.y, title: { display: true, text: "B Tonnes", font: { size: 10 } } },
        },
      },
    });
  }
}

// ── EDA Trend ────────────────────────────────────────────
function renderEdaTrend(metric) {
  const dataMap = {
    production: State.production,
    area:       State.area,
    yield:      State.yieldData,
  };
  const labels = { production: "M Tonnes", area: "M Hectares", yield: "Tonnes/Ha" };
  const titles = {
    production: '<i class="fa-solid fa-chart-column ico"></i> National Production Trends (1997–2023)',
    area:       '<i class="fa-solid fa-chart-column ico"></i> National Cultivated Area Trends (1997–2023)',
    yield:      '<i class="fa-solid fa-chart-column ico"></i> Average Crop Yield Trends (1997–2023)',
  };

  document.getElementById("edaChartTitle").innerHTML = titles[metric];

  const src   = dataMap[metric];
  const crops = Object.keys(src);
  if (!crops.length) return;

  const allYears = src[crops[0]]?.years ?? [];

  makeChart("edaTrendChart", {
    type: "line",
    data: {
      labels: allYears,
      datasets: crops.map(crop => ({
        label: crop.charAt(0).toUpperCase() + crop.slice(1),
        data:  src[crop].values,
        borderColor: CROP_COLORS[crop] || "#888",
        backgroundColor: "transparent",
        borderWidth: 2.2, pointRadius: 2.5, tension: 0.3,
      })),
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: { ...CHART_DEFAULTS.plugins, legend: { position: "top" } },
      scales: {
        x: { ...CHART_DEFAULTS.scales.x, ticks: { maxTicksLimit: 10, font: { size: 9 } } },
        y: { ...CHART_DEFAULTS.scales.y, title: { display: true, text: labels[metric] } },
      },
    },
  });
}

// ── Box chart (simulated from min/max/median) ─────────────
function renderBoxChart() {
  const prod  = State.production;
  const crops = Object.keys(prod);
  if (!crops.length) return;

  // Compute per-crop stats from values array
  const labels = crops.map(c => c.charAt(0).toUpperCase() + c.slice(1));
  const medians = [], mins = [], maxs = [];

  crops.forEach(c => {
    const vals = [...prod[c].values].sort((a, b) => a - b);
    const n    = vals.length;
    medians.push(vals[Math.floor(n / 2)]);
    mins.push(vals[0]);
    maxs.push(vals[n - 1]);
  });

  makeChart("boxChart", {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Median",  data: medians, backgroundColor: crops.map(c => CROP_COLORS[c] + "cc"), borderRadius: 6 },
        { label: "Min",     data: mins,    backgroundColor: "#94a3b844", borderRadius: 6 },
        { label: "Max",     data: maxs,    backgroundColor: "#1e3a5f22", borderRadius: 6 },
      ],
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: { ...CHART_DEFAULTS.plugins, legend: { position: "top" } },
      scales: {
        y: { ...CHART_DEFAULTS.scales.y, title: { display: true, text: "M Tonnes", font: { size: 10 } } },
      },
    },
  });
}

// ── YoY Growth ───────────────────────────────────────────
function renderYoYChart() {
  const yoy   = State.yoy;
  const crops = Object.keys(yoy);
  if (!crops.length) return;

  const allYears = yoy[crops[0]]?.years ?? [];

  makeChart("yoyChart", {
    type: "bar",
    data: {
      labels: allYears,
      datasets: crops.map(crop => ({
        label: crop.charAt(0).toUpperCase() + crop.slice(1),
        data:  yoy[crop].values,
        backgroundColor: CROP_COLORS[crop] + "bb",
        borderRadius: 3,
      })),
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: { ...CHART_DEFAULTS.plugins, legend: { position: "top" } },
      scales: {
        x: { stacked: false, ...CHART_DEFAULTS.scales.x, ticks: { maxTicksLimit: 8, font: { size: 8 } } },
        y: { ...CHART_DEFAULTS.scales.y, title: { display: true, text: "YoY Change (%)" },
             ticks: { callback: v => v + "%" } },
      },
    },
  });
}

// ── Top states ────────────────────────────────────────────
function renderTopStates(crop) {
  document.getElementById("topStatesLabel").textContent =
    crop.charAt(0).toUpperCase() + crop.slice(1);
  const d = State.topStates[crop];
  if (!d) return;

  makeChart("topStatesChart", {
    type: "bar",
    data: {
      labels: d.states,
      datasets: [{
        label: "Cumulative Production (B Tonnes)",
        data:  d.values,
        backgroundColor: CROP_COLORS[crop] || "#3b82f6",
        borderRadius: 7,
      }],
    },
    options: {
      indexAxis: "y",
      ...CHART_DEFAULTS,
      plugins: { legend: { display: false }, tooltip: CHART_DEFAULTS.plugins.tooltip },
      scales: {
        x: { ...CHART_DEFAULTS.scales.x, title: { display: true, text: "B Tonnes" } },
        y: { ...CHART_DEFAULTS.scales.y, ticks: { font: { size: 10 } } },
      },
    },
  });
}

// ── Eval table ───────────────────────────────────────────
function renderEvalTable() {
  const data = State.evalData;
  const tbody = document.getElementById("evalTbody");
  
  if (!data || !data.length) {
    if (tbody) tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:30px;color:#94a3b8;"><i class="fa-solid fa-spinner fa-spin"></i> Models are still training on the backend. Please wait a minute and refresh the page!</td></tr>`;
    return;
  }
  const medals = ["🥇", "🥈", "🥉"];
  tbody.innerHTML = data.map((r, i) => `
    <tr class="${i === 0 ? "best" : ""}">
      <td>${medals[i] || (i+1)}</td>
      <td>${r.model}</td>
      <td><strong>${r.r2}</strong></td>
      <td>
        <div class="r2-bar">
          <div class="r2-bar-inner" style="width:${r.r2 * 100}%"></div>
          <span>${(r.r2 * 100).toFixed(1)}%</span>
        </div>
      </td>
      <td>${r.mae.toLocaleString()}</td>
      <td>${r.rmse.toLocaleString()}</td>
      <td>${r.mape !== undefined ? r.mape.toFixed(2) : '—'}</td>
      <td>${r.evs !== undefined ? r.evs : '—'}</td>
    </tr>`).join("");

  const cvTbody = document.getElementById("cvTbody");
  cvTbody.innerHTML = State.cvData.map((r, i) => `
    <tr class="${i === 0 ? "best" : ""}">
      <td>${i + 1}</td>
      <td>${r.model}</td>
      <td><strong>${r.mean_r2}</strong></td>
      <td>± ${r.std_r2}</td>
      <td>${r.min_r2}</td>
      <td>${r.max_r2}</td>
    </tr>`).join("");
}

// ── Model comparison charts ───────────────────────────────
function renderModelCharts() {
  const data = State.evalData;
  if (!data.length) return;

  const labels = data.map(d => {
    const shorts = { "Linear Regression": "Lin. Reg.", "Ridge Regression": "Ridge",
      "Lasso Regression": "Lasso", "Bayesian Ridge": "Bayes. R.",
      "Decision Tree": "Dec. Tree", "Random Forest": "Rand. Forest",
      "Extra Trees": "Extra Trees", "Gradient Boosting": "Grad. Boost",
      "KNN Regressor": "KNN", "MLP Neural Net": "MLP NN" };
    return shorts[d.model] || d.model;
  });
  
  const colors = [
    "#22c55e","#3b82f6","#f97316","#a855f7","#06b6d4",
    "#ec4899","#eab308","#14b8a6","#8b5cf6","#f43f5e",
    "#0ea5e9","#84cc16","#d946ef","#fb923c","#ef4444",
    "#64748b","#334155","#10b981","#6366f1","#8b5cf6",
    "#d946ef","#f43f5e","#fca5a5","#93c5fd","#86efac"
  ];

  const minR2 = Math.min(...data.map(d => d.r2));
  const xMin  = Math.max(0, Math.floor(minR2 * 10) / 10 - 0.05);

  makeChart("r2BarChart", {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "R² Score",
        data:  data.map(d => d.r2),
        backgroundColor: colors.slice(0, data.length),
        borderRadius: 7,
      }],
    },
    options: {
      indexAxis: "y",
      ...CHART_DEFAULTS,
      plugins: { legend: { display: false } },
      scales: {
        x: { min: xMin, max: 1.0, ticks: { font: { size: 9 } } },
        y: { ticks: { font: { size: 9 } } },
      },
    },
  });

  makeChart("errorChart", {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "MAE",  data: data.map(d => d.mae / 1e6),  backgroundColor: "#3b82f6bb", borderRadius: 5 },
        { label: "RMSE", data: data.map(d => d.rmse / 1e6), backgroundColor: "#ef4444bb", borderRadius: 5 },
      ],
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: { ...CHART_DEFAULTS.plugins, legend: { position: "top" } },
      scales: {
        y: { title: { display: true, text: "M Tonnes", font: { size: 10 } } },
      },
    },
  });
}

// ── Feature importance ────────────────────────────────────
function renderFeatureImportance(data) {
  if (!data) {
    // Load via API
    const model = document.getElementById("fiModelSelect")?.value || "Random Forest";
    apiFetch(`/xai/global?model=${encodeURIComponent(model)}`)
      .then(d => renderFeatureImportance(d))
      .catch(() => {});
    return;
  }
  const labels = data.map(d => d.feature);
  const vals   = data.map(d => d.importance);
  const colors = ["#3b82f6","#22c55e","#f97316","#a855f7","#06b6d4"];

  makeChart("featureChart", {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Mean |SHAP| Value",
        data:  vals,
        backgroundColor: colors,
        borderRadius: 7,
      }],
    },
    options: {
      indexAxis: "y",
      ...CHART_DEFAULTS,
      plugins: { legend: { display: false } },
      scales: {
        y: { title: { display: true, text: "Importance Score", font: { size: 10 } } },
      },
    },
  });
}

// ── Residuals ────────────────────────────────────────────
function renderResiduals() {
  const r = State.residuals;
  if (!r) return;

  // Scatter: actual vs predicted
  const pts = r.actual.map((a, i) => ({ x: a, y: r.fitted[i] }));
  const mn  = Math.min(...r.actual, ...r.fitted);
  const mx  = Math.max(...r.actual, ...r.fitted);

  makeChart("residScatter", {
    type: "scatter",
    data: {
      datasets: [
        { label: "Actual vs Predicted", data: pts,
          backgroundColor: "#3b82f620", borderColor: "#3b82f660", pointRadius: 3 },
        { label: "Perfect fit", type: "line",
          data: [{ x: mn, y: mn }, { x: mx, y: mx }],
          borderColor: "#ef4444", borderDash: [5, 5], borderWidth: 1.5,
          pointRadius: 0 },
      ],
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: { legend: { position: "top" } },
      scales: {
        x: { ...CHART_DEFAULTS.scales.x, title: { display: true, text: "Actual (M T)" } },
        y: { ...CHART_DEFAULTS.scales.y, title: { display: true, text: "Predicted (M T)" } },
      },
    },
  });

  // Histogram: residual distribution
  makeChart("residHist", {
    type: "bar",
    data: {
      labels: r.hist_edges.slice(0, -1).map(v => v.toFixed(2)),
      datasets: [{
        label: "Count",
        data:  r.hist_counts,
        backgroundColor: "#22c55e88",
        borderColor:     "#16a34a",
        borderWidth: 1,
        borderRadius: 3,
      }],
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { maxTicksLimit: 8, font: { size: 8 } },
             title: { display: true, text: "Residual (M T)", font: { size: 10 } } },
        y: { title: { display: true, text: "Count", font: { size: 10 } } },
      },
    },
  });
}

// ── MSP Trend ────────────────────────────────────────────
function renderMspTrend() {
  const d = State.mspTrend;
  const crops = Object.keys(d);
  if (!crops.length) return;

  const allYears = d[crops[0]]?.years ?? [];

  makeChart("mspTrendChart", {
    type: "line",
    data: {
      labels: allYears,
      datasets: crops.map(crop => ({
        label: crop.charAt(0).toUpperCase() + crop.slice(1),
        data:  d[crop].values,
        borderColor: CROP_COLORS[crop] || "#888",
        backgroundColor: (CROP_COLORS[crop] || "#888") + "18",
        borderWidth: 2.5, pointRadius: 5, tension: 0.3,
      })),
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: { ...CHART_DEFAULTS.plugins, legend: { position: "top" } },
      scales: {
        x: { ticks: { font: { size: 9 } } },
        y: { title: { display: true, text: "₹ per Quintal" } },
      },
    },
  });
}

// ── National forecast chart ───────────────────────────────
async function renderNationalForecast() {
  // Build from ARIMA endpoint for all 5 crops with state=all aggregated
  // We use Rajasthan for wheat, West Bengal for rice etc.
  const pairs = [
    { crop: "wheat", state: "Rajasthan"   },
    { crop: "rice",  state: "West Bengal" },
    { crop: "maize", state: "Karnataka"   },
    { crop: "bajra", state: "Rajasthan"   },
    { crop: "jowar", state: "Maharashtra" },
  ];

  const datasets = [];
  let allHistYears = [], allFcYears = [];

  for (const { crop, state } of pairs) {
    try {
      const fc = await apiFetch(`/forecast?crop=${crop}&state=${encodeURIComponent(state)}&steps=8&method=arima`);
      if (!allHistYears.length) {
        allHistYears = fc.historical.years;
        allFcYears   = fc.forecast.years;
      }
      datasets.push({
        label: crop.charAt(0).toUpperCase() + crop.slice(1) + " (hist)",
        data:  fc.historical.values,
        borderColor: CROP_COLORS[crop],
        backgroundColor: "transparent",
        borderWidth: 1.5, pointRadius: 1, tension: 0.3,
      });
      datasets.push({
        label: crop.charAt(0).toUpperCase() + crop.slice(1) + " (forecast)",
        data:  [...Array(fc.historical.values.length - 1).fill(null),
                fc.historical.values.at(-1),
                ...fc.forecast.values],
        borderColor: CROP_COLORS[crop],
        backgroundColor: "transparent",
        borderWidth: 2.5, borderDash: [5, 3], pointRadius: 4,
        tension: 0.3,
      });
    } catch {}
  }

  const allYears = [...allHistYears, ...allFcYears];
  makeChart("nationalFcChart", {
    type: "line",
    data: {
      labels: allYears,
      datasets,
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: {
        legend: { display: false },
        tooltip: { mode: "index", intersect: false },
      },
      scales: {
        x: { ticks: { maxTicksLimit: 12, font: { size: 8 } },
             title: { display: true, text: "Year" } },
        y: { title: { display: true, text: "M Tonnes" } },
      },
    },
  });
}
