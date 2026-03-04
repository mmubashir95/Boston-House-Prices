# complete_onnx_tflite_conversion.py

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# ─────────────────────────────────────────
# STEP 1: Train or Load the Model
# ─────────────────────────────────────────
def train_model():
    df = pd.read_csv("data/boston.csv")
    TARGET = "MEDV"
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    log_features = ["CRIM", "ZN", "DIS", "RAD"]
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

    base_regressor = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    model = TransformedTargetRegressor(
        regressor=base_regressor,
        func=np.log1p,
        inverse_func=np.expm1
    )

    model.fit(X_train, y_train)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/linear_logtarget_pipeline.joblib")
    print("✅ Model trained and saved.")
    return model, X_test


def load_model():
    model = joblib.load("artifacts/linear_logtarget_pipeline.joblib")
    print("✅ Model loaded from artifacts/")
    return model


# ─────────────────────────────────────────
# STEP 2: Verify Model Predictions
# ─────────────────────────────────────────
def verify_model(model, X_sample):
    preds = model.predict(X_sample[:5])
    print("\n📊 Sample sklearn predictions (original scale):")
    print(preds)
    return preds


# ─────────────────────────────────────────
# STEP 3: Convert to ONNX
# ─────────────────────────────────────────
def convert_to_onnx(model, X_sample):
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnx
    import onnxruntime as rt

    # Extract inner pipeline from TransformedTargetRegressor
    inner_pipeline = model.regressor_
    n_features = X_sample.shape[1]

    print(f"\n🔄 Converting to ONNX (input features: {n_features})...")

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        inner_pipeline,
        initial_types=initial_type,
        target_opset=12
    )

    os.makedirs("artifacts", exist_ok=True)
    onnx_path = "artifacts/model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"✅ ONNX model saved: {onnx_path}")

    # Verify ONNX output matches sklearn (before log-target inverse)
    print("\n🔍 Verifying ONNX predictions...")
    sess = rt.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    X_input = X_sample[:5].astype(np.float32).values

    onnx_raw_preds = sess.run(None, {input_name: X_input})[0].flatten()

    # Apply inverse transform manually (expm1) to match sklearn output
    onnx_final_preds = np.expm1(onnx_raw_preds)

    print("ONNX predictions (after expm1):", onnx_final_preds)
    print("(Compare these with sklearn predictions above ↑)")

    return onnx_path


# ─────────────────────────────────────────
# STEP 4: Convert ONNX → TFLite
# ─────────────────────────────────────────
def convert_to_tflite(onnx_path):
    import tensorflow as tf
    import subprocess

    tf_saved_model_path = "artifacts/tf_saved_model"
    tflite_path = "artifacts/model.tflite"

    print("\n🔄 Converting ONNX → TensorFlow SavedModel...")
    result = subprocess.run([
        "python", "-m", "tf2onnx.convert",
        "--onnx", onnx_path,
        "--output", tflite_path,
        "--target", "tflite",
        "--saved-model", tf_saved_model_path
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ tf2onnx error:", result.stderr)
        raise RuntimeError("ONNX to TFLite conversion failed.")

    print(f"✅ TFLite model saved: {tflite_path}")
    return tflite_path


# ─────────────────────────────────────────
# STEP 5: Verify TFLite Model
# ─────────────────────────────────────────
def verify_tflite(tflite_path, X_sample):
    import tensorflow as tf

    print("\n🔍 Verifying TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input shape  :", input_details[0]['shape'])
    print("Output shape :", output_details[0]['shape'])

    # Run single sample prediction
    sample = X_sample[:1].astype(np.float32).values
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()

    raw_output = interpreter.get_tensor(output_details[0]['index'])[0][0]
    final_pred  = np.expm1(raw_output)   # ← apply inverse transform on Android too

    print(f"\n✅ TFLite prediction (single sample): {final_pred:.4f}")
    print("⚠️  Remember: apply np.expm1() on Android after getting raw output!")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":

    # Load or train
    if os.path.exists("artifacts/linear_logtarget_pipeline.joblib"):
        model = load_model()
        df = pd.read_csv("data/boston.csv")
        X = df.drop(columns=["MEDV"])
        _, X_test, _, _ = train_test_split(X, df["MEDV"], test_size=0.2, random_state=42)
    else:
        model, X_test = train_model()

    # Pipeline
    sklearn_preds = verify_model(model, X_test)
    onnx_path     = convert_to_onnx(model, X_test)
    tflite_path   = convert_to_tflite(onnx_path)
    verify_tflite(tflite_path, X_test)

    print("\n🎉 Done! Files in artifacts/:")
    print("   • linear_logtarget_pipeline.joblib  ← original sklearn model")
    print("   • model.onnx                        ← intermediate ONNX")
    print("   • model.tflite                      ← deploy this on Android")
    print("\n⚠️  On Android: after model output → apply Math.exp(output) - 1")