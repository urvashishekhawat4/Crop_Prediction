"""
ml_engine.py
Comprehensive ML & DL model comparison for crop production prediction.
Trains 14 models: Linear, Ridge, Lasso, ElasticNet, Bayesian Ridge,
Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost,
XGBoost, SVR, KNN, and MLP Neural Network.

Evaluation metrics: R², MAE, RMSE, MAPE, Explained Variance.
"""

import warnings
import numpy as np
import pandas as pd
import shap

# ── Linear Models ─────────────────────────────────────────
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
)

# ── Tree-based Models ─────────────────────────────────────
from sklearn.tree     import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, AdaBoostRegressor,
    BaggingRegressor, StackingRegressor
)

# ── XGBoost ───────────────────────────────────────────────
from xgboost import XGBRegressor

# ── Other ML Models ───────────────────────────────────────
from sklearn.svm       import SVR
from sklearn.neighbors import KNeighborsRegressor

# ── Deep Learning (MLP, RBM) ──────────────────────────────
from sklearn.neural_network import MLPRegressor, BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin

# ── Evaluation ────────────────────────────────────────────
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    explained_variance_score, mean_absolute_percentage_error
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    # ── Linear Models ──────────────────────────────────
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1, max_iter=5000),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
    "Bayesian Ridge": BayesianRidge(),

    # ── Tree-based Models ──────────────────────────────
    "Decision Tree": DecisionTreeRegressor(max_depth=15, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=200, learning_rate=0.1, random_state=42),

    # ── Ensembles ──────────────────────────────────────
    "Bagging Regressor": BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=10), n_estimators=50, random_state=42, n_jobs=-1),
    "Stacking Regressor": StackingRegressor(
        estimators=[
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42))
        ],
        final_estimator=Ridge()
    ),

    # ── XGBoost ────────────────────────────────────────
    "XGBoost": XGBRegressor(
        n_estimators=300, learning_rate=0.1, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0
    ),

    # ── Support Vector Regression ──────────────────────
    "SVR": SVR(kernel="rbf", C=100, epsilon=0.1),

    # ── K-Nearest Neighbours ───────────────────────────
    "KNN Regressor": KNeighborsRegressor(n_neighbors=7, weights="distance", n_jobs=-1),

    # ── Deep Learning (MLP & RBM/DBN) ──────────────────
    "MLP Neural Net": MLPRegressor(
        hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
        max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1
    ),
    "Restricted Boltzmann Machine": Pipeline(steps=[
        ('rbm', BernoulliRBM(n_components=64, learning_rate=0.01, n_iter=20, random_state=42)),
        ('regressor', Ridge())
    ]),
    "Deep Belief Network (Simulated)": MLPRegressor(
        hidden_layer_sizes=(256, 128, 64, 32), activation="logistic", solver="adam",
        max_iter=500, random_state=42, early_stopping=True
    ),
}


class MLEngine:
    """
    Trains all 14 models at startup, evaluates with 5 metrics,
    runs 5-fold CV, and serves predictions via the Flask API.
    """

    def __init__(self):
        self.trained_models  = {}
        self.le_crop         = None
        self.le_state        = None
        self.scaler          = None
        self.feature_names   = ["area", "yield", "year", "crop_enc", "state_enc"]
        self.eval_results    = []
        self.cv_results      = []
        self._fitted         = False
        self.X_bg            = None  # Background dataset for SHAP

    # ── Training ────────────────────────────────────────
    def fit(self, X_sc: np.ndarray, y: np.ndarray,
            le_crop: LabelEncoder, le_state: LabelEncoder,
            scaler: StandardScaler):
        """
        Train all models on the scaled feature matrix.
        Call once at server startup.
        """
        self.le_crop  = le_crop
        self.le_state = le_state
        self.scaler   = scaler

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sc, y, test_size=0.2, random_state=42
        )
        
        # Save a small background sample for SHAP explainers (keep it fast)
        np.random.seed(42)
        idx = np.random.choice(X_tr.shape[0], min(200, X_tr.shape[0]), replace=False)
        self.X_bg = X_tr[idx]

        # ── Train & evaluate each model ─────────────────
        self.eval_results = []
        for name, model in MODEL_CONFIGS.items():
            print(f"  Training {name}…")
            model.fit(X_tr, y_tr)
            self.trained_models[name] = model

            preds = model.predict(X_te)

            # Guard against division by zero in MAPE
            mask = y_te != 0
            if mask.sum() > 0:
                mape = float(mean_absolute_percentage_error(
                    y_te[mask], preds[mask]
                )) * 100
            else:
                mape = 0.0

            self.eval_results.append({
                "model":    name,
                "r2":       round(float(r2_score(y_te, preds)), 4),
                "mae":      round(float(mean_absolute_error(y_te, preds)), 2),
                "rmse":     round(float(np.sqrt(mean_squared_error(y_te, preds))), 2),
                "mape":     round(mape, 2),
                "evs":      round(float(explained_variance_score(y_te, preds)), 4),
            })

        # Sort best first by R²
        self.eval_results.sort(key=lambda x: x["r2"], reverse=True)

        # ── 5-fold Cross-Validation ─────────────────────
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        self.cv_results = []
        for name, model in MODEL_CONFIGS.items():
            print(f"  CV {name}…")
            try:
                scores = cross_val_score(model, X_sc, y, cv=kf,
                                         scoring="r2", n_jobs=-1)
                self.cv_results.append({
                    "model":    name,
                    "mean_r2":  round(float(scores.mean()), 4),
                    "std_r2":   round(float(scores.std()),  4),
                    "min_r2":   round(float(scores.min()),  4),
                    "max_r2":   round(float(scores.max()),  4),
                })
            except Exception as e:
                print(f"    CV failed for {name}: {e}")
                self.cv_results.append({
                    "model": name, "mean_r2": 0, "std_r2": 0,
                    "min_r2": 0, "max_r2": 0,
                })

        self.cv_results.sort(key=lambda x: x["mean_r2"], reverse=True)

        self._fitted = True
        print(f"[MLEngine] Trained {len(self.trained_models)} models. ✓")

    # ── Prediction ───────────────────────────────────────
    def predict(self, crop: str, state: str, area: float,
                year: int, yield_val: float,
                model_name: str = "Random Forest") -> dict:
        """
        Predict production (tonnes) for the given inputs.
        Returns dict with prediction and confidence band (±10 %).
        """
        if not self._fitted:
            raise RuntimeError("MLEngine not fitted yet.")

        # Encode
        try:
            crop_enc  = int(self.le_crop.transform([crop.lower()])[0])
        except ValueError:
            raise ValueError(f"Unknown crop '{crop}'. "
                             f"Valid: {list(self.le_crop.classes_)}")
        try:
            state_enc = int(self.le_state.transform([state])[0])
        except ValueError:
            raise ValueError(f"Unknown state '{state}'. "
                             f"Valid: {list(self.le_state.classes_)}")

        X = np.array([[area, yield_val, year, crop_enc, state_enc]])
        X_sc = self.scaler.transform(X)

        model = self.trained_models.get(model_name)
        if model is None:
            model = self.trained_models["Random Forest"]
            model_name = "Random Forest"

        pred = float(model.predict(X_sc)[0])
        return {
            "crop":        crop,
            "state":       state,
            "area":        area,
            "year":        year,
            "yield":       yield_val,
            "model":       model_name,
            "prediction":  round(pred, 0),
            "pred_mT":     round(pred / 1e6, 4),
            "lower_bound": round(pred * 0.90, 0),
            "upper_bound": round(pred * 1.10, 0),
        }

    # ── SHAP Global Importance ───────────────────────────
    def global_shap(self, model_name: str = "Random Forest") -> list:
        model = self.trained_models.get(model_name)
        if model is None or self.X_bg is None:
            return []
            
        try:
            # Try TreeExplainer first for speed
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(self.X_bg)
        except Exception:
            # Fallback to KernelExplainer for non-tree models (can be slow)
            explainer = shap.KernelExplainer(model.predict, self.X_bg[:50])
            shap_values = explainer.shap_values(self.X_bg[:50])
            
        # Calculate mean absolute SHAP values per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0).tolist()
        
        # Return sorted by importance
        res = [{"feature": f, "importance": round(v, 5)}
               for f, v in zip(self.feature_names, mean_abs_shap)]
        res.sort(key=lambda x: x["importance"], reverse=True)
        return res

    # ── SHAP Local Explanation ───────────────────────────
    def local_shap(self, crop: str, state: str, area: float,
                   year: int, yield_val: float,
                   model_name: str = "Random Forest") -> dict:
        if not self._fitted:
            raise RuntimeError("MLEngine not fitted yet.")
            
        model = self.trained_models.get(model_name)
        if model is None:
            return {}

        crop_enc  = int(self.le_crop.transform([crop.lower()])[0])
        state_enc = int(self.le_state.transform([state])[0])

        # Create instance matching the feature order
        X = np.array([[area, yield_val, year, crop_enc, state_enc]])
        X_sc = self.scaler.transform(X)
        
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_sc)[0]
            base_val  = explainer.expected_value
            if isinstance(base_val, np.ndarray):
                base_val = base_val[0]
        except Exception:
            explainer = shap.KernelExplainer(model.predict, self.X_bg[:20])
            shap_vals = explainer.shap_values(X_sc)[0]
            base_val  = explainer.expected_value

        # Return exactly how each feature pushed the prediction
        contributions = []
        for i, f in enumerate(self.feature_names):
            # Include original raw value for the UI context
            val = [area, yield_val, year, crop_enc, state_enc][i]
            contributions.append({
                "feature": f,
                "value": round(val, 2),
                "contribution": round(float(shap_vals[i]), 2)
            })
            
        # Sort by absolute contribution impact
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        return {
            "base_value": round(float(base_val), 2),
            "prediction": round(float(model.predict(X_sc)[0]), 2),
            "contributions": contributions
        }

    # ── Models that support SHAP efficiently ─────────────
    @property
    def importance_models(self) -> list:
        # For responsiveness, we only list tree-based models here
        tree_models = [
            "Decision Tree", "Random Forest", "Extra Trees", 
            "Gradient Boosting", "AdaBoost", "XGBoost"
        ]
        return [name for name in self.trained_models.keys() if name in tree_models]

    # ── Residual data ────────────────────────────────────
    def residuals(self, X_sc: np.ndarray, y: np.ndarray,
                  model_name: str = "Random Forest") -> dict:
        model  = self.trained_models.get(model_name,
                    self.trained_models["Random Forest"])
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sc, y, test_size=0.2, random_state=42
        )
        preds  = model.predict(X_te)
        resid  = (y_te - preds).tolist()
        fitted = preds.tolist()
        actual = y_te.tolist()
        # Histogram bins
        hist, edges = np.histogram(resid, bins=30)
        return {
            "fitted":    [round(v/1e6, 4) for v in fitted],
            "actual":    [round(v/1e6, 4) for v in actual],
            "residuals": [round(v/1e6, 4) for v in resid],
            "hist_counts": hist.tolist(),
            "hist_edges":  [round(e/1e6, 4) for e in edges.tolist()],
        }

    # ── Model names list ─────────────────────────────────
    @property
    def model_names(self) -> list:
        return list(MODEL_CONFIGS.keys())

    # ── Best model name ──────────────────────────────────
    @property
    def best_model_name(self) -> str:
        if self.eval_results:
            return self.eval_results[0]["model"]
        return "Random Forest"
