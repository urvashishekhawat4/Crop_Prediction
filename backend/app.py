"""
app.py  –  Flask REST API
Crop Production Prediction & MSP Trend Analysis
IIS (Deemed to be University), Jaipur
Student: Ms. Urvashi Shekhawat | Roll: 242207

Run:
    python app.py

API Base URL: http://localhost:5000/api
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import traceback

from data_loader import (
    load_crop_df, load_msp_df, build_feature_matrix,
    get_summary_stats, get_production_trend, get_area_trend,
    get_yield_trend, get_state_heatmap, get_msp_trend,
    get_msp_district_data, get_top_states, get_yoy_growth,
)
from ml_engine  import MLEngine
from forecaster import arima_forecast, lstm_forecast, msp_arima_forecast, msp_lstm_forecast, msp_gru_forecast

# ─────────────────────────────────────────────────────────
# App Init
# ─────────────────────────────────────────────────────────
app    = Flask(__name__)
CORS(app)   # Allow all origins (needed for VS Code Live Server)

# ─────────────────────────────────────────────────────────
# Load & train once at startup
# ─────────────────────────────────────────────────────────
print("[startup] Loading datasets …")
crop_df = load_crop_df()
msp_df  = load_msp_df()
X_sc, y, le_crop, le_state, scaler, features = build_feature_matrix(crop_df)

print("[startup] Training ML models …")
engine = MLEngine()
engine.fit(X_sc, y, le_crop, le_state, scaler)
print("[startup] Ready ✓")


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def ok(data):
    return jsonify({"status": "ok", "data": data})

def err(msg, code=400):
    return jsonify({"status": "error", "message": str(msg)}), code


# ─────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────

# ── Health check ─────────────────────────────────────────
@app.route("/api/health")
def health():
    return ok({"message": "Crop ML API is running", "models_ready": engine._fitted})


# ── Overview / Stats ─────────────────────────────────────
@app.route("/api/stats")
def stats():
    st = get_summary_stats(crop_df)
    models = engine.model_names
    # Count DL vs ML
    dl_models = [m for m in models if "Neural Net" in m or "Boltzmann" in m or "Belief" in m]
    st["num_ml_models"] = len(models) - len(dl_models)
    st["num_dl_models"] = len(dl_models)
    return ok(st)


# ── EDA: Production trend ────────────────────────────────
@app.route("/api/eda/production")
def eda_production():
    return ok(get_production_trend(crop_df))


# ── EDA: Area trend ──────────────────────────────────────
@app.route("/api/eda/area")
def eda_area():
    return ok(get_area_trend(crop_df))


# ── EDA: Yield trend ─────────────────────────────────────
@app.route("/api/eda/yield")
def eda_yield():
    return ok(get_yield_trend(crop_df))


# ── EDA: State heatmap ───────────────────────────────────
@app.route("/api/eda/heatmap")
def eda_heatmap():
    crop  = request.args.get("crop", "wheat")
    top_n = int(request.args.get("top_n", 12))
    return ok(get_state_heatmap(crop_df, crop, top_n))


# ── EDA: Top states ──────────────────────────────────────
@app.route("/api/eda/top_states")
def eda_top_states():
    return ok(get_top_states(crop_df))


# ── EDA: YoY growth ──────────────────────────────────────
@app.route("/api/eda/yoy")
def eda_yoy():
    return ok(get_yoy_growth(crop_df))


# ── MSP: trends ──────────────────────────────────────────
@app.route("/api/msp/trend")
def msp_trend():
    return ok(get_msp_trend(msp_df))


# ── MSP: district data ───────────────────────────────────
@app.route("/api/msp/district")
def msp_district():
    crop = request.args.get("crop", "wheat")
    return ok(get_msp_district_data(msp_df, crop))


# ── MSP: forecast ────────────────────────────────────────
@app.route("/api/msp/forecast")
def msp_forecast_route():
    crop  = request.args.get("crop", "wheat")
    steps = int(request.args.get("steps", 4))
    method = request.args.get("method", "arima").lower()
    
    sub   = (msp_df[~msp_df["anticipated"]]
             .drop_duplicates(["year", "crop"])
             .query("crop == @crop")[["year", "msp"]]
             .dropna()
             .sort_values("year"))
    if len(sub) < 4:
        return err(f"Not enough MSP data for crop='{crop}'")
    msp_vals   = sub["msp"].tolist()
    msp_years  = sub["year"].tolist()
    try:
        if method == "lstm":
            fc = msp_lstm_forecast(msp_vals, steps=steps)
        elif method == "gru":
            fc = msp_gru_forecast(msp_vals, steps=steps)
        else:
            fc = msp_arima_forecast(msp_vals, steps=steps)
    except Exception as e:
        return err(str(e))

    last_yr = msp_years[-1]
    start   = int(last_yr[:4])
    future  = [f"{start+i}-{str(start+i+1)[-2:]}" for i in range(1, steps+1)]
    return ok({
        "crop":           crop,
        "method":         method,
        "historical":     {"years": msp_years, "values": msp_vals},
        "forecast":       {"years": future,    "values": [round(v, 0) for v in fc]},
    })


# ── Models: evaluation results ───────────────────────────
@app.route("/api/models/evaluation")
def models_eval():
    return ok(engine.eval_results)


# ── MSP Models: evaluation results ───────────────────────
@app.route("/api/msp/models/evaluation")
def msp_models_eval():
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    
    crops = ["wheat", "gram", "barley"]
    arima_errors = {"mae": [], "rmse": [], "mape": []}
    lstm_errors = {"mae": [], "rmse": [], "mape": []}
    gru_errors = {"mae": [], "rmse": [], "mape": []}
    
    for crop in crops:
        sub = (msp_df[~msp_df["anticipated"]]
                 .drop_duplicates(["year", "crop"])
                 .query("crop == @crop")[["year", "msp"]]
                 .dropna()
                 .sort_values("year"))
        if len(sub) < 5:
            continue
        msp_vals = np.array(sub["msp"].tolist(), dtype=float)
        
        train = msp_vals[:-2]
        test = msp_vals[-2:]
        
        try:
            fc_arima = msp_arima_forecast(train.tolist(), steps=2)
            arima_errors["mae"].append(mean_absolute_error(test, fc_arima))
            arima_errors["rmse"].append(np.sqrt(mean_squared_error(test, fc_arima)))
            arima_errors["mape"].append(mean_absolute_percentage_error(test, fc_arima) * 100)
        except:
            pass

        try:
            fc_lstm = msp_lstm_forecast(train.tolist(), steps=2)
            lstm_errors["mae"].append(mean_absolute_error(test, fc_lstm))
            lstm_errors["rmse"].append(np.sqrt(mean_squared_error(test, fc_lstm)))
            lstm_errors["mape"].append(mean_absolute_percentage_error(test, fc_lstm) * 100)
        except:
            pass
            
        try:
            fc_gru = msp_gru_forecast(train.tolist(), steps=2)
            gru_errors["mae"].append(mean_absolute_error(test, fc_gru))
            gru_errors["rmse"].append(np.sqrt(mean_squared_error(test, fc_gru)))
            gru_errors["mape"].append(mean_absolute_percentage_error(test, fc_gru) * 100)
        except:
            pass

    def avg(lst):
        return round(sum(lst)/len(lst), 2) if lst else 0.0

    res = [
        {
            "model": "ARIMA",
            "mae": avg(arima_errors["mae"]),
            "rmse": avg(arima_errors["rmse"]),
            "mape": avg(arima_errors["mape"])
        },
        {
            "model": "LSTM",
            "mae": avg(lstm_errors["mae"]),
            "rmse": avg(lstm_errors["rmse"]),
            "mape": avg(lstm_errors["mape"])
        },
        {
            "model": "GRU",
            "mae": avg(gru_errors["mae"]),
            "rmse": avg(gru_errors["rmse"]),
            "mape": avg(gru_errors["mape"])
        }
    ]
    return ok(res)


# ── Models: cross-validation ─────────────────────────────
@app.route("/api/models/cv")
def models_cv():
    return ok(engine.cv_results)


# ── XAI: Global SHAP Importance ────────────────────────────
@app.route("/api/xai/global")
def xai_global():
    model_name = request.args.get("model", "Random Forest")
    return ok(engine.global_shap(model_name))


# ── XAI: Local SHAP Explanation ──────────────────────────
@app.route("/api/xai/local", methods=["POST"])
def xai_local():
    body = request.get_json(force=True)
    required = ["crop", "state", "area", "year", "yield"]
    missing  = [k for k in required if k not in body]
    if missing:
        return err(f"Missing fields: {missing}")
    try:
        result = engine.local_shap(
            crop       = str(body["crop"]),
            state      = str(body["state"]),
            area       = float(body["area"]),
            year       = int(body["year"]),
            yield_val  = float(body["yield"]),
            model_name = str(body.get("model", "Random Forest")),
        )
        return ok(result)
    except Exception:
        return err(traceback.format_exc(), 500)


# ── Models: residuals ────────────────────────────────────
@app.route("/api/models/residuals")
def models_residuals():
    model_name = request.args.get("model", "Random Forest")
    return ok(engine.residuals(X_sc, y, model_name))


# ── Predict: single prediction ───────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    required = ["crop", "state", "area", "year", "yield"]
    missing  = [k for k in required if k not in body]
    if missing:
        return err(f"Missing fields: {missing}")
    try:
        result = engine.predict(
            crop       = str(body["crop"]),
            state      = str(body["state"]),
            area       = float(body["area"]),
            year       = int(body["year"]),
            yield_val  = float(body["yield"]),
            model_name = str(body.get("model", "Random Forest")),
        )
        return ok(result)
    except (ValueError, RuntimeError) as e:
        return err(str(e))
    except Exception:
        return err(traceback.format_exc(), 500)


# ── Forecast: ARIMA or LSTM for crop+state ───────────────
@app.route("/api/forecast")
def forecast_route():
    crop    = request.args.get("crop",   "wheat")
    state   = request.args.get("state",  "Rajasthan")
    steps   = int(request.args.get("steps",  8))
    method  = request.args.get("method", "arima").lower()

    sub = (crop_df[(crop_df["crop"] == crop.lower()) &
                   (crop_df["state"] == state)]
           .sort_values("year"))
    if len(sub) < 8:
        return err(f"Not enough data for crop='{crop}', state='{state}'")

    series   = sub["production"].values.astype(float)
    years    = sub["year"].values.tolist()
    last_yr  = int(years[-1])
    future   = list(range(last_yr + 1, last_yr + 1 + steps))

    try:
        if method == "lstm":
            fc = lstm_forecast(series, steps=steps)
        else:
            fc = arima_forecast(series, steps=steps)
    except Exception as e:
        return err(str(e))

    return ok({
        "crop":       crop,
        "state":      state,
        "method":     method,
        "steps":      steps,
        "historical": {
            "years":  years,
            "values": [round(v / 1e6, 4) for v in series.tolist()]
        },
        "forecast":   {
            "years":  future,
            "values": [round(v / 1e6, 4) for v in fc]
        },
    })


# ── Available options (crops, states) ────────────────────
@app.route("/api/options")
def options():
    return ok({
        "crops":   sorted(crop_df["crop"].unique().tolist()),
        "states":  sorted(crop_df["state"].unique().tolist()),
        "models":  engine.model_names,
        "importance_models": engine.importance_models,
        "msp_crops": sorted(msp_df["crop"].unique().tolist()),
    })


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
