# ============================================================
# ✅ FINAL RANDOM FOREST PIPELINE (Regression)
# - log1p on skewed numeric features (optional list)
# - StandardScaler (not required for RF, but ok to keep consistent)
# - Train/Test evaluation + Cross-Validation
# - Save trained pipeline
# ============================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os


# ---------------------------
# 1) Load data
# ---------------------------
df = pd.read_csv("data/boston.csv")  # <-- update if your path differs

target_col = "MEDV"
X = df.drop(columns=[target_col])
y = df[target_col]


# ---------------------------
# 2) Train/Test split (NO leakage)
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)


# ---------------------------
# 3) Feature lists
# ---------------------------
# These are the same ones you used in your script. :contentReference[oaicite:1]{index=1}
log_features = ["CRIM", "ZN", "DIS", "RAD"]  # apply log1p + scale
all_numeric = X.columns.tolist()
numeric_no_log = [c for c in all_numeric if c not in log_features]


# ---------------------------
# 4) Preprocessing
# ---------------------------
log_pipeline = Pipeline(steps=[
    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scaler", StandardScaler())
])

num_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("log", log_pipeline, log_features),
        ("num", num_pipeline, numeric_no_log),
    ],
    remainder="drop"
)


# ---------------------------
# 5) Final Random Forest model
# ---------------------------
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=1,
    # You can add: max_depth=..., min_samples_split=..., max_features=...
)

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", rf)
])


# ---------------------------
# 6) Train
# ---------------------------
rf_pipeline.fit(X_train, y_train)


# ---------------------------
# 7) Evaluate (Train + Test)
# ---------------------------
def regression_report(name, model, Xtr, ytr, Xte, yte):
    ytr_pred = model.predict(Xtr)
    yte_pred = model.predict(Xte)

    train_r2 = r2_score(ytr, ytr_pred)
    test_r2 = r2_score(yte, yte_pred)

    train_rmse = np.sqrt(mean_squared_error(ytr, ytr_pred))
    test_rmse = np.sqrt(mean_squared_error(yte, yte_pred))

    print(f"\n{name} Performance")
    print("-" * 40)
    print("Train R2  :", train_r2)
    print("Train RMSE:", train_rmse)
    print("Test  R2  :", test_r2)
    print("Test  RMSE:", test_rmse)

regression_report("Random Forest", rf_pipeline, X_train, y_train, X_test, y_test)


# ---------------------------
# 8) Cross-Validation (on TRAIN only)
# ---------------------------
# Use train set only so test set stays "final unseen"
cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_r2 = cross_val_score(rf_pipeline, X_train, y_train, cv=cv, scoring="r2")
cv_rmse = -cross_val_score(rf_pipeline, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")

print("\nCV Results (Train Only)")
print("-" * 40)
print("R2   : mean =", cv_r2.mean(), " | std =", cv_r2.std())
print("RMSE : mean =", cv_rmse.mean(), " | std =", cv_rmse.std())


# ---------------------------
# 9) Save the trained pipeline
# ---------------------------
os.makedirs("artifacts", exist_ok=True)
joblib.dump(rf_pipeline, "artifacts/rf_housing_pipeline.joblib")
print("\nSaved: artifacts/rf_housing_pipeline.joblib")