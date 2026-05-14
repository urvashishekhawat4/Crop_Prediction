/* ============================================================
   forecast.js  –  ARIMA / LSTM Forecast Tab
   ============================================================ */

function setMethod(method) {
  State.fcMethod = method;
  document.getElementById("btnARIMA").classList.toggle("active", method === "arima");
  document.getElementById("btnLSTM").classList.toggle("active",  method === "lstm");
}

async function runForecast() {
  const crop   = document.getElementById("fcCrop").value;
  const state  = document.getElementById("fcState").value;
  const steps  = document.getElementById("fcSteps").value;
  const method = State.fcMethod;

  const btn = document.querySelector(".forecast-sidebar .btn-run");
  btn.disabled   = true;
  btn.innerHTML   = '<i class="fa-solid fa-spinner fa-spin ico"></i> Running…';

  try {
    const data = await apiFetch(
      `/forecast?crop=${encodeURIComponent(crop)}&state=${encodeURIComponent(state)}&steps=${steps}&method=${method}`
    );
    renderForecastChart(data, method);
    renderForecastTable(data);
    showToast(`${method.toUpperCase()} forecast complete ✓`, "success");
  } catch (e) {
    showToast("Forecast error: " + e.message, "error");
  } finally {
    btn.disabled   = false;
    btn.innerHTML   = '<i class="fa-solid fa-play ico"></i> Run Forecast';
  }
}

function renderForecastChart(data, method) {
  const crop  = data.crop.charAt(0).toUpperCase() + data.crop.slice(1);
  const state = data.state;
  document.getElementById("fcChartTitle").innerHTML =
    `<i class="fa-solid fa-chart-line ico"></i> ${crop} – ${state} | ${method.toUpperCase()} Forecast`;

  const color   = CROP_COLORS[data.crop] || "#3b82f6";
  const histYrs = data.historical.years;
  const histVls = data.historical.values;
  const fcYrs   = data.forecast.years;
  const fcVls   = data.forecast.values;

  // Confidence band (±8%)
  const fcLow  = fcVls.map(v => +(v * 0.92).toFixed(4));
  const fcHigh = fcVls.map(v => +(v * 1.08).toFixed(4));

  const allYears = [...histYrs, ...fcYrs];

  makeChart("fcChart", {
    type: "line",
    data: {
      labels: allYears,
      datasets: [
        // Historical
        {
          label: "Historical",
          data:  histVls,
          borderColor: color, backgroundColor: color + "18",
          borderWidth: 2.5, pointRadius: 3, tension: 0.35, fill: true,
        },
        // Forecast line
        {
          label: `${method.toUpperCase()} Forecast`,
          data:  [...Array(histYrs.length - 1).fill(null), histVls.at(-1), ...fcVls],
          borderColor: "#ef4444", backgroundColor: "transparent",
          borderWidth: 2.5, borderDash: [6, 3],
          pointRadius: 5, pointStyle: "diamond", tension: 0.3,
        },
        // Upper band
        {
          label: "+8% Band",
          data:  [...Array(histYrs.length - 1).fill(null), histVls.at(-1), ...fcHigh],
          borderColor: "#ef444422", backgroundColor: "#ef444412",
          borderWidth: 1, borderDash: [3, 3], pointRadius: 0, fill: "+1",
        },
        // Lower band
        {
          label: "−8% Band",
          data:  [...Array(histYrs.length - 1).fill(null), histVls.at(-1), ...fcLow],
          borderColor: "#ef444422", backgroundColor: "#ef444412",
          borderWidth: 1, borderDash: [3, 3], pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { position: "top", labels: { filter: item => !item.text.includes("Band"), font: { size: 10 } } },
        tooltip: { mode: "index", intersect: false,
          callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y?.toFixed(3)} M T` }
        },
      },
      scales: {
        x: { ticks: { maxTicksLimit: 12, font: { size: 9 } }, grid: { color: "#f1f5f9" } },
        y: { title: { display: true, text: "Production (M Tonnes)", font: { size: 10 } },
             grid: { color: "#f1f5f9" } },
      },
    },
  });
}

function renderForecastTable(data) {
  const wrap = document.getElementById("fcTableWrap");
  const body = document.getElementById("fcTableBody");
  wrap.style.display = "block";
  body.innerHTML = data.forecast.years.map((yr, i) => `
    <tr>
      <td>${yr}</td>
      <td><strong>${data.forecast.values[i].toFixed(4)}</strong> M T</td>
    </tr>`).join("");
}
