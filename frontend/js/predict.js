/* ============================================================
   predict.js  –  Prediction Form & Result Rendering
   ============================================================ */

async function runPrediction() {
  const btn = document.getElementById("predictBtn");
  btn.disabled = true;
  btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin ico"></i> Predicting…`;

  const payload = {
    crop:  document.getElementById("pCrop").value,
    state: document.getElementById("pState").value,
    area:  parseFloat(document.getElementById("pArea").value),
    year:  parseInt(document.getElementById("pYear").value),
    yield: parseFloat(document.getElementById("pYield").value),
    model: document.getElementById("pModel").value,
  };

  try {
    const [result, xaiResult] = await Promise.all([
      apiFetch("/predict", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(payload),
      }),
      apiFetch("/xai/local", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(payload),
      }).catch(e => { console.warn("XAI failed", e); return null; })
    ]);
    renderPredictionResult(result, xaiResult);
    showToast("Prediction complete ✓", "success");
  } catch (e) {
    showToast("Prediction failed: " + e.message, "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles ico"></i> Predict Production`;
  }
}

function renderPredictionResult(r, xai) {
  const box = document.getElementById("predictResult");
  const xaiBox = document.getElementById("localXaiBox");
  const cropLabel  = r.crop.charAt(0).toUpperCase() + r.crop.slice(1);
  const predMT     = (r.prediction / 1e6).toFixed(4);
  const lowerMT    = (r.lower_bound / 1e6).toFixed(4);
  const upperMT    = (r.upper_bound / 1e6).toFixed(4);
  const predFormatted = r.prediction.toLocaleString("en-IN", { maximumFractionDigits: 0 });

  box.innerHTML = `
    <h3 style="font-size:0.95rem;font-weight:700;margin-bottom:16px;color:#1e293b;">
      <i class="fa-solid fa-chart-column ico"></i> Prediction Result
    </h3>

    <div class="result-hero">
      <div class="result-crop"><i class="fa-solid fa-wheat-awn ico"></i> ${cropLabel} in ${r.state} (${r.year})</div>
      <div class="result-num">${predMT}</div>
      <div class="result-unit">Million Tonnes predicted production</div>
      <div style="margin-top:10px;font-size:0.82rem;color:#475569;">
        = ${predFormatted} Tonnes &nbsp;|&nbsp; Model: <strong>${r.model}</strong>
      </div>
    </div>

    <div class="result-bands">
      <div class="band-card low">
        <div class="band-label"><i class="fa-solid fa-arrow-down ico"></i> Lower Bound (−10%)</div>
        <div class="band-val">${lowerMT} M T</div>
      </div>
      <div class="band-card high">
        <div class="band-label"><i class="fa-solid fa-arrow-up ico"></i> Upper Bound (+10%)</div>
        <div class="band-val">${upperMT} M T</div>
      </div>
    </div>

    <div class="result-meta">
      <div class="meta-item">
        <div class="m-lbl"><i class="fa-solid fa-seedling ico"></i> Crop</div>
        <div class="m-val">${cropLabel}</div>
      </div>
      <div class="meta-item">
        <div class="m-lbl"><i class="fa-solid fa-location-dot ico"></i> State</div>
        <div class="m-val">${r.state}</div>
      </div>
      <div class="meta-item">
        <div class="m-lbl"><i class="fa-solid fa-ruler-combined ico"></i> Area Input</div>
        <div class="m-val">${Number(r.area).toLocaleString()} Ha</div>
      </div>
      <div class="meta-item">
        <div class="m-lbl"><i class="fa-solid fa-leaf ico"></i> Yield Input</div>
        <div class="m-val">${r.yield} T/Ha</div>
      </div>
      <div class="meta-item">
        <div class="m-lbl"><i class="fa-solid fa-calendar-days ico"></i> Year</div>
        <div class="m-val">${r.year}</div>
      </div>
      <div class="meta-item">
        <div class="m-lbl"><i class="fa-solid fa-robot ico"></i> Model Used</div>
        <div class="m-val" style="font-size:0.82rem;">${r.model}</div>
      </div>
    </div>

    <div style="margin-top:18px;">
      <canvas id="predGaugeChart" height="120"></canvas>
    </div>
  `;

  // Mini bar chart comparing the 3 values
  setTimeout(() => {
    makeChart("predGaugeChart", {
      type: "bar",
      data: {
        labels: ["Lower Bound", "Predicted", "Upper Bound"],
        datasets: [{
          data:            [parseFloat(lowerMT), parseFloat(predMT), parseFloat(upperMT)],
          backgroundColor: ["#fecaca", "#3b82f6", "#bbf7d0"],
          borderRadius:    8,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: { legend: { display: false },
          tooltip: { callbacks: { label: ctx => `${ctx.parsed.y} M Tonnes` } }
        },
        scales: {
          y: { title: { display: true, text: "M Tonnes", font: { size: 10 } },
               grid: { color: "#f1f5f9" } },
          x: { grid: { display: false } },
        },
      },
    });

    // Render XAI if available
    if (xai && xai.contributions) {
      xaiBox.style.display = "block";
      const labels = xai.contributions.map(c => `${c.feature} (${c.value})`);
      const data   = xai.contributions.map(c => c.contribution);
      const colors = data.map(v => v > 0 ? "#22c55e" : "#ef4444");
    
      makeChart("localXaiChart", {
        type: "bar",
        data: {
          labels: labels,
          datasets: [{
            label: "SHAP Impact",
            data: data,
            backgroundColor: colors,
            borderRadius: 4
          }]
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  const val = ctx.raw;
                  return `Impact: ${val > 0 ? '+' : ''}${val.toLocaleString()} Tonnes`;
                }
              }
            }
          },
          scales: {
            x: { title: { display: true, text: "Impact on Prediction (Tonnes)", font: { size: 10 } } }
          }
        }
      });
    } else {
      xaiBox.style.display = "none";
    }
  }, 50);
}
