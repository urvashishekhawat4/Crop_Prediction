/* ============================================================
   msp.js  –  MSP Analysis Tab
   ============================================================ */

async function loadMSPForecast() {
  const crop  = document.getElementById("mspFcCrop").value;
  const steps = document.getElementById("mspFcSteps").value;
  document.getElementById("mspFcTitle").innerHTML =
    `<i class="fa-solid fa-chart-column ico"></i> MSP Forecast – ${crop.charAt(0).toUpperCase() + crop.slice(1)}`;

  try {
    const data = await apiFetch(
      `/msp/forecast?crop=${encodeURIComponent(crop)}&steps=${steps}`
    );
    renderMSPForecastChart(data);
    showToast("MSP forecast ready ✓", "success");
  } catch (e) {
    showToast("MSP forecast error: " + e.message, "error");
  }
}

function renderMSPForecastChart(data) {
  const color     = CROP_COLORS[data.crop] || "#3b82f6";
  const histYrs   = data.historical.years;
  const histVls   = data.historical.values;
  const fcYrs     = data.forecast.years;
  const fcVls     = data.forecast.values;
  const allYears  = [...histYrs, ...fcYrs];

  // Confidence band ±5%
  const fcLow  = fcVls.map(v => +(v * 0.95).toFixed(0));
  const fcHigh = fcVls.map(v => +(v * 1.05).toFixed(0));

  makeChart("mspFcChart", {
    type: "line",
    data: {
      labels: allYears,
      datasets: [
        {
          label: "Historical MSP",
          data:  histVls,
          borderColor: color, backgroundColor: color + "20",
          borderWidth: 2.5, pointRadius: 5, tension: 0.3, fill: true,
        },
        {
          label: "ARIMA Forecast",
          data:  [...Array(histYrs.length - 1).fill(null), histVls.at(-1), ...fcVls],
          borderColor: "#ef4444", backgroundColor: "transparent",
          borderWidth: 2.5, borderDash: [6, 3],
          pointRadius: 6, pointStyle: "rectRot",
        },
        {
          label: "+5% Band",
          data:  [...Array(histYrs.length - 1).fill(null), histVls.at(-1), ...fcHigh],
          borderColor: "#ef444420", backgroundColor: "#ef444410",
          borderWidth: 0, pointRadius: 0, fill: "+1",
        },
        {
          label: "−5% Band",
          data:  [...Array(histYrs.length - 1).fill(null), histVls.at(-1), ...fcLow],
          borderColor: "#ef444420", backgroundColor: "#ef444410",
          borderWidth: 0, pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          position: "top",
          labels: { filter: item => !item.text.includes("Band"), font: { size: 10 } },
        },
        tooltip: {
          mode: "index", intersect: false,
          callbacks: { label: ctx => `${ctx.dataset.label}: ₹${ctx.parsed.y?.toLocaleString()}` },
        },
      },
      scales: {
        x: { ticks: { font: { size: 9 } }, grid: { color: "#f1f5f9" } },
        y: {
          title: { display: true, text: "MSP (₹/Quintal)", font: { size: 10 } },
          grid:  { color: "#f1f5f9" },
          ticks: { callback: v => "₹" + v.toLocaleString() },
        },
      },
    },
  });

  // Update MSP cards with latest historical values
  const crops    = ["wheat", "gram", "barley"];
  const cardIds  = ["mspWheat", "mspGram", "mspBarley"];
  if (State.mspTrend && Object.keys(State.mspTrend).length) {
    crops.forEach((c, i) => {
      const d = State.mspTrend[c];
      if (d && d.values.length) {
        const latest = d.values.at(-1);
        document.getElementById(cardIds[i]).textContent =
          "₹" + Number(latest).toLocaleString();
      }
    });
  }
}

// ── District Heatmap ─────────────────────────────────────
async function loadMspDistrict() {
  const crop = document.getElementById("mspDistCrop")?.value || "wheat";
  try {
    const data = await apiFetch(`/msp/district?crop=${encodeURIComponent(crop)}`);
    renderMspDistrictHeatmap(data);
  } catch (e) {
    showToast("MSP district data error: " + e.message, "error");
  }
}

function renderMspDistrictHeatmap(data) {
  const { districts, years, matrix } = data;
  if (!districts.length) return;

  const maxVal = Math.max(...matrix.flat());
  const table  = document.getElementById("mspDistTable");

  let html = "<thead><tr><th>District</th>";
  years.forEach(y => { html += `<th style="white-space:nowrap;font-size:0.72rem;">${y}</th>`; });
  html += "</tr></thead><tbody>";

  districts.forEach((dist, di) => {
    html += `<tr><td style="font-weight:600;white-space:nowrap;font-size:0.78rem;">${dist}</td>`;
    matrix[di].forEach(val => {
      const pct   = maxVal > 0 ? val / maxVal : 0;
      const alpha = (0.1 + pct * 0.85).toFixed(2);
      const color = `rgba(22,163,74,${alpha})`;   // green palette for MSP
      const text  = pct > 0.55 ? "#fff" : "#1e293b";
      const disp  = val > 0 ? val.toFixed(1) : "—";
      html += `<td style="background:${color};color:${text};text-align:right;font-size:0.72rem;">${disp}</td>`;
    });
    html += "</tr>";
  });

  html += "</tbody>";
  table.innerHTML = html;
}

// Initialise MSP cards on page load from msp trend data
function initMspCards() {
  const map   = { wheat: "mspWheat", gram: "mspGram", barley: "mspBarley" };
  const trend = State.mspTrend || {};
  Object.entries(map).forEach(([crop, id]) => {
    const d = trend[crop];
    if (d && d.values.length) {
      document.getElementById(id).textContent =
        "₹" + Number(d.values.at(-1)).toLocaleString();
    }
  });
}

// Override original renderMspTrend to also init cards after
const _origRenderMspTrend = typeof renderMspTrend !== "undefined" ? renderMspTrend : null;
document.addEventListener("mspTrendReady", () => initMspCards());
