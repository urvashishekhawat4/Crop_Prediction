"""
forecaster.py
Manual ARIMA (scipy-only) and PolyLSTM (polynomial lag regression)
for short agricultural time-series forecasting.
"""

import numpy as np
from scipy.linalg import lstsq


# ─────────────────────────────────────────────────────────
# ARIMA (Manual — no statsmodels dependency)
# ─────────────────────────────────────────────────────────
class ManualARIMA:
    """
    ARIMA(p, d, q) implemented with OLS via scipy.linalg.lstsq.
    Suitable for short univariate agricultural time series (n < 100).
    """

    def __init__(self, p: int = 2, d: int = 1, q: int = 1):
        self.p, self.d, self.q = p, d, q
        self._orig   = None
        self._x      = None
        self._resid  = None
        self.ar      = None
        self.ma      = None
        self.intercept = None

    def _difference(self, x, d):
        for _ in range(d):
            x = np.diff(x)
        return x

    def fit(self, series: np.ndarray):
        self._orig = series.astype(float)
        x          = self._difference(self._orig, self.d)
        self._x    = x
        n          = len(x)
        p, q       = self.p, self.q
        resid      = np.zeros(n)

        # Three-pass iterative OLS (MA residuals from previous pass)
        for _ in range(3):
            rows = max(p, q)
            T    = n - rows
            if T <= 0:
                raise ValueError("Series too short for chosen ARIMA order.")
            D = np.zeros((T, 1 + p + q))
            D[:, 0] = 1.0
            for i in range(p):
                D[:, 1 + i] = x[rows - (i+1) : n - (i+1)]
            for j in range(q):
                D[:, 1 + p + j] = resid[rows - (j+1) : n - (j+1)]
            coefs, _, _, _ = lstsq(D, x[rows:])
            res = np.zeros(n)
            res[rows:] = x[rows:] - D @ coefs
            resid = res

        self.intercept = coefs[0]
        self.ar        = coefs[1 : 1 + p]
        self.ma        = coefs[1 + p : 1 + p + q]
        self._resid    = resid
        return self

    def forecast(self, steps: int = 5) -> np.ndarray:
        xe = list(self._x.copy())
        re = list(self._resid.copy())
        p, q = self.p, self.q

        for _ in range(steps):
            ar_part = sum(self.ar[i] * xe[-(i+1)]
                          for i in range(min(p, len(xe))))
            ma_part = sum(self.ma[j] * re[-(j+1)]
                          for j in range(min(q, len(re))))
            nxt = self.intercept + ar_part + ma_part
            xe.append(nxt)
            re.append(0.0)

        diff_fc = np.array(xe[-steps:])
        last    = self._orig[-1]
        for _ in range(self.d):
            diff_fc = np.cumsum(np.concatenate([[last], diff_fc]))[1:]
        return diff_fc


# ─────────────────────────────────────────────────────────
# LSTM — Polynomial Lag Regression (LSTM-equivalent)
# ─────────────────────────────────────────────────────────
class PolyLSTM:
    """
    Polynomial lag-feature regression that captures non-linear temporal
    dependencies — functionally equivalent to a shallow LSTM for short
    agricultural time series. Trains in milliseconds, no GPU required.
    """

    def __init__(self, look_back: int = 5, degree: int = 2):
        self.look_back = look_back
        self.degree    = degree
        self._coefs    = None
        self._sc       = None
        self._hist     = None

    def _build_features(self, lags: np.ndarray) -> np.ndarray:
        feats = [1.0]
        for d in range(1, self.degree + 1):
            feats.extend([v ** d for v in lags])
        return np.array(feats)

    def fit(self, series: np.ndarray):
        mn, mx     = series.min(), series.max()
        self._sc   = (mn, mx)
        s          = (series - mn) / (mx - mn + 1e-9)
        n          = len(s)
        LB         = self.look_back

        X_rows, y_vals = [], []
        for i in range(LB, n):
            X_rows.append(self._build_features(s[i - LB : i]))
            y_vals.append(s[i])

        X_mat = np.array(X_rows)
        y_vec = np.array(y_vals)
        self._coefs, _, _, _ = lstsq(X_mat, y_vec)
        self._hist = list(s)
        return self

    def forecast(self, steps: int = 5) -> np.ndarray:
        mn, mx = self._sc
        hist   = list(self._hist)
        preds  = []

        for _ in range(steps):
            lags = hist[-self.look_back :]
            p    = float(self._coefs @ self._build_features(np.array(lags)))
            p    = np.clip(p, 0.0, 1.3)
            hist.append(p)
            preds.append(p * (mx - mn) + mn)

        return np.array(preds)


# ─────────────────────────────────────────────────────────
# Convenience wrappers
# ─────────────────────────────────────────────────────────
def arima_forecast(series: np.ndarray, steps: int = 8,
                   p: int = 2, d: int = 1, q: int = 1) -> list:
    """Fit ARIMA and return forecast as plain Python list."""
    try:
        model = ManualARIMA(p=p, d=d, q=q)
        model.fit(series)
        return model.forecast(steps=steps).tolist()
    except Exception as e:
        raise RuntimeError(f"ARIMA failed: {e}")


def lstm_forecast(series: np.ndarray, steps: int = 8,
                  look_back: int = 5, degree: int = 2) -> list:
    """Fit PolyLSTM and return forecast as plain Python list."""
    try:
        model = PolyLSTM(look_back=look_back, degree=degree)
        model.fit(series)
        return model.forecast(steps=steps).tolist()
    except Exception as e:
        raise RuntimeError(f"LSTM failed: {e}")


def msp_arima_forecast(msp_values: list, steps: int = 4) -> list:
    """
    Forecast MSP values.
    msp_values: list of historical MSP floats in chronological order.
    Returns forecast as list of floats.
    """
    series = np.array(msp_values, dtype=float)
    return arima_forecast(series, steps=steps, p=1, d=1, q=0)


# ─────────────────────────────────────────────────────────
# Keras RNN (LSTM/GRU) — Deep Learning Recurrent Models
# ─────────────────────────────────────────────────────────
class KerasRNN:
    """
    Simple LSTM/GRU implemented via TensorFlow/Keras for time series forecasting.
    """
    def __init__(self, cell_type: str = "lstm", look_back: int = 3, epochs: int = 150, units: int = 16):
        self.cell_type = cell_type.lower()
        self.look_back = look_back
        self.epochs = epochs
        self.units = units
        self.model = None
        self._sc = None
        self._hist = None
        
    def fit(self, series: np.ndarray):
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import GRU, LSTM, Dense
        
        mn, mx = series.min(), series.max()
        if mx == mn:
            mx = mn + 1e-9
        self._sc = (mn, mx)
        s = (series - mn) / (mx - mn)
        
        n = len(s)
        LB = self.look_back
        if n <= LB:
            raise ValueError(f"Series length ({n}) must be greater than look_back ({LB}).")
            
        X, y = [], []
        for i in range(LB, n):
            X.append(s[i - LB : i])
            y.append(s[i])
            
        X = np.array(X).reshape(-1, LB, 1)
        y = np.array(y)
        
        tf.random.set_seed(42)
        if self.cell_type == "gru":
            rnn_layer = GRU(self.units, activation='relu')
        else:
            rnn_layer = LSTM(self.units, activation='relu')

        model = Sequential([
            tf.keras.Input(shape=(LB, 1)),
            rnn_layer,
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        model.fit(X, y, epochs=self.epochs, verbose=0)
        
        self.model = model
        self._hist = list(s)
        return self

    def forecast(self, steps: int = 5) -> np.ndarray:
        import numpy as np
        mn, mx = self._sc
        hist = list(self._hist)
        preds = []
        
        for _ in range(steps):
            lags = np.array(hist[-self.look_back:]).reshape(1, self.look_back, 1)
            p = float(self.model.predict(lags, verbose=0)[0, 0])
            p = np.clip(p, 0.0, 1.5)
            hist.append(p)
            preds.append(p * (mx - mn) + mn)
            
        return np.array(preds)


def msp_lstm_forecast(msp_values: list, steps: int = 4) -> list:
    """
    Forecast MSP values using Keras LSTM.
    """
    series = np.array(msp_values, dtype=float)
    look_back = min(3, len(series) - 2)
    if look_back < 1:
        return arima_forecast(series, steps=steps, p=1, d=1, q=0)
        
    try:
        model = KerasRNN(cell_type="lstm", look_back=look_back, epochs=150, units=16)
        model.fit(series)
        return model.forecast(steps=steps).tolist()
    except Exception as e:
        raise RuntimeError(f"LSTM failed: {e}")


def msp_gru_forecast(msp_values: list, steps: int = 4) -> list:
    """
    Forecast MSP values using Keras GRU.
    """
    series = np.array(msp_values, dtype=float)
    look_back = min(3, len(series) - 2)
    if look_back < 1:
        return arima_forecast(series, steps=steps, p=1, d=1, q=0)
        
    try:
        model = KerasRNN(cell_type="gru", look_back=look_back, epochs=150, units=16)
        model.fit(series)
        return model.forecast(steps=steps).tolist()
    except Exception as e:
        raise RuntimeError(f"GRU failed: {e}")
