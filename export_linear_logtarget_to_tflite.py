import numpy as np
import joblib
import tensorflow as tf

JOBLIB_PATH = "artifacts/linear_logtarget_pipeline.joblib"
TFLITE_OUT  = "artifacts/linear_logtarget.tflite"


def main():
    model = joblib.load(JOBLIB_PATH)

    # TransformedTargetRegressor -> regressor_ is fitted Pipeline
    base = model.regressor_
    pre  = base.named_steps["preprocessor"]
    lin  = base.named_steps["regressor"]

    # Expect two blocks: ("log", log_pipe, log_features) and ("num", num_pipe, other_numeric)
    log_cols = list(pre.transformers_[0][2])
    num_cols = list(pre.transformers_[1][2])

    log_pipe = pre.named_transformers_["log"]
    num_pipe = pre.named_transformers_["num"]

    log_imputer = log_pipe.named_steps["imputer"]
    log_scaler  = log_pipe.named_steps["scaler"]

    num_imputer = num_pipe.named_steps["imputer"]
    num_scaler  = num_pipe.named_steps["scaler"]

    W = lin.coef_.astype(np.float32)      # (n_features,)
    b = np.float32(lin.intercept_)

    n_log = len(log_cols)
    n_num = len(num_cols)
    n_in  = n_log + n_num

    @tf.function(input_signature=[tf.TensorSpec([None, n_in], tf.float32)])
    def forward(x):
        x_log = x[:, :n_log]
        x_num = x[:, n_log:]

        # NaN -> median
        log_median = tf.constant(log_imputer.statistics_.astype(np.float32))
        num_median = tf.constant(num_imputer.statistics_.astype(np.float32))
        x_log = tf.where(tf.math.is_nan(x_log), log_median[None, :], x_log)
        x_num = tf.where(tf.math.is_nan(x_num), num_median[None, :], x_num)

        # log1p only for log block
        x_log = tf.math.log1p(x_log)

        # standardize
        log_mean  = tf.constant(log_scaler.mean_.astype(np.float32))
        log_scale = tf.constant(log_scaler.scale_.astype(np.float32))
        num_mean  = tf.constant(num_scaler.mean_.astype(np.float32))
        num_scale = tf.constant(num_scaler.scale_.astype(np.float32))

        x_log = (x_log - log_mean[None, :]) / log_scale[None, :]
        x_num = (x_num - num_mean[None, :]) / num_scale[None, :]

        feat = tf.concat([x_log, x_num], axis=1)

        # Linear: predicts log1p(y)
        W_tf = tf.constant(W.reshape(-1, 1))
        y_log = tf.matmul(feat, W_tf) + b

        # inverse target transform -> original y
        y = tf.math.expm1(y_log)
        return y

    concrete_fn = forward.get_concrete_function()

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    # simplest: keep float32
    converter.optimizations = []
    tflite_model = converter.convert()

    with open(TFLITE_OUT, "wb") as f:
        f.write(tflite_model)

    print("✅ Saved:", TFLITE_OUT)
    print("✅ Input order for Android:", log_cols + num_cols)
    print("✅ n_inputs =", n_in)


if __name__ == "__main__":
    main()