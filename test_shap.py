import sys
import json
sys.path.append("/Users/urvashishekhawat/Downloads/crop-ml-project/backend")
from data_loader import load_crop_df, build_feature_matrix
from ml_engine import MLEngine

df = load_crop_df()
X_sc, y, le_crop, le_state, scaler, features = build_feature_matrix(df)

engine = MLEngine()
engine.fit(X_sc, y, le_crop, le_state, scaler)
res = engine.local_shap("wheat", "Rajasthan", 2000000, 2025, 2.5, "Linear Regression")
print(json.dumps(res, indent=2))
