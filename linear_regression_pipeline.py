# linear_logtarget_pipeline.py

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib


# ----------------------------
# 1) Load data
# ----------------------------
df = pd.read_csv("data/boston.csv")  # change if needed

TARGET = "MEDV"
X = df.drop(columns=[TARGET])
y = df[TARGET]


# ----------------------------
# 2) Train/Test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ----------------------------
# 3) Preprocessing
#    (kept similar to your approach: log some skewed features + scale everything)
# ----------------------------
# You used these in your file. Keep/edit as you like. :contentReference[oaicite:1]{index=1}
log_features = ["CRIM", "ZN", "DIS", "RAD"]

# numeric features present in X
numeric_features = X.columns.tolist()
numeric_no_log = [c for c in numeric_features if c not in log_features]

log_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scaler", StandardScaler()),
])

num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("log", log_pipe, log_features),
        ("num", num_pipe, numeric_no_log),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)


# ----------------------------
# 4) Linear model + LogTarget wrapper
# ----------------------------
base_regressor = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model = TransformedTargetRegressor(
    regressor=base_regressor,
    func=np.log1p,        # y -> log(1+y)
    inverse_func=np.expm1 # back to original scale
)


# ----------------------------
# 5) Train
# ----------------------------
model.fit(X_train, y_train)


# ----------------------------
# 6) Evaluate (on ORIGINAL target scale)
# ----------------------------
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2  = r2_score(y_test, y_pred_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse  = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\nLinear + LogTarget Performance:")
print("Train R2  :", train_r2)
print("Train RMSE:", train_rmse)
print("Test  R2  :", test_r2)
print("Test  RMSE:", test_rmse)


# ----------------------------
# 7) Save pipeline (fixes your earlier artifacts/ path error)
# ----------------------------
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/linear_logtarget_pipeline.joblib")
print("\nSaved: artifacts/linear_logtarget_pipeline.joblib")