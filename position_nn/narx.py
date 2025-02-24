import sys
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from scipy.ndimage import gaussian_filter1d


def load_training_data(folder):
    # loads all csv files from a folder and combines them into one dataframe
    # useful when training data is spread across multiple files
    all_dfs = []
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):  # find all csv files
        df = pd.read_csv(f)
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)  # merge all data


def make_narx_features(X, Y, x_lag=4, y_lag=4):
    # creates lagged features for a narx model (nonlinear autoregressive with exogenous inputs)
    # adds past values of inputs (X) and outputs (Y) as new columns to help the model learn patterns over time
    N = len(X)
    m = max(x_lag, y_lag)  # the largest lag determines how much we trim
    cols = []

    # add past input values
    for lag in range(1, x_lag + 1):
        if lag < len(X):
            cols.append(X[lag:])

    # add past output values
    for lag in range(1, y_lag + 1):
        if lag < len(Y):
            cols.append(Y[lag:])

    # trim all arrays to match in length
    trimmed = [c[: (N - m)] for c in cols]
    features = np.column_stack(trimmed)
    targets = Y[m:]

    return features, targets


def safe_log(x):
    # apply log transform while handling zeros and negative values
    # log normally can't handle negatives, so we take the absolute value first
    # we also use log1p(x) instead of log(x) to prevent issues when x is close to zero
    # sign is preserved so negative values don’t just disappear
    return np.sign(x) * np.log1p(np.abs(x))


def inverse_safe_log(x):
    # reverses the safe_log transformation
    # since we used log1p (which is log(1 + x)), we do exp(x) - 1 to undo it
    # sign is restored so negative numbers go back to their original values
    return np.sign(x) * (np.exp(np.abs(x)) - 1)


def narx_closed_loop_predict(
    model, x_scaler, y_scaler, X_full, init_y, x_lag=10, y_lag=10
):
    # closed-loop prediction: the model predicts one step, then uses its own output as input for the next step
    # instead of predicting absolute values, we predict delta changes over time
    # using deltas helps keep the system stable and prevents drift
    N, d = len(X_full), init_y.shape[1]
    preds = np.zeros((N, d))  # store predictions
    preds[:y_lag] = init_y[:y_lag]  # initialize with known values

    for k in range(y_lag, N):
        x_features = []
        for lag in range(1, x_lag + 1):
            if k - lag >= 0:
                x_features.append(X_full[k - lag])  # past input states
            else:
                x_features.append(
                    np.zeros_like(X_full[0])
                )  # zero padding for early steps

        y_features = []
        for lag in range(1, y_lag + 1):
            if k - lag >= 0:
                y_features.append(
                    preds[k - lag] - preds[k - lag - 1]
                )  # past output deltas
            else:
                y_features.append(np.zeros(d))  # zero padding

        f = np.hstack(x_features + y_features).reshape(1, -1)
        fs = x_scaler.transform(f)
        delta_pred = y_scaler.inverse_transform(model.predict(fs))[0]

        # compute next step using the last known state and predicted change
        preds[k] = preds[k - 1] + delta_pred

    return preds


# check command-line arguments
if len(sys.argv) != 3:
    print(f"usage: {sys.argv[0]} <training_folder> <simulation_input.csv>")
    sys.exit(1)

training_folder = sys.argv[1]
sim_input_file = sys.argv[2]

print("loading training data...")
df_train = load_training_data(training_folder)

# extract input and output variables
Xf = df_train[["potX", "potY", "potZ"]].values.astype(np.float64)
Yf = df_train[["RX", "RY", "RZ", "TX", "TY", "TZ"]].values.astype(np.float64)

print(f"xf shape: {Xf.shape}, yf shape: {Yf.shape}")

# compute delta changes (instead of absolute values)
delta_Yf = np.diff(Yf, axis=0)
delta_Yf = np.vstack([np.zeros(Yf.shape[1]), delta_Yf])

print(f"delta yf shape: {delta_Yf.shape}")

# generate narx-style input-output pairs
X_narx, Y_narx = make_narx_features(Xf, delta_Yf, x_lag=10, y_lag=10)

# train-test split & scaling
X_train, X_test, y_train, y_test = train_test_split(
    X_narx, Y_narx, test_size=0.2, random_state=42
)
xs, ys = StandardScaler(), StandardScaler()
X_train_s, X_test_s = xs.fit_transform(X_train), xs.transform(X_test)
y_train_s, y_test_s = ys.fit_transform(y_train), ys.transform(y_test)

print("training model...")
mdl = MLPRegressor(
    hidden_layer_sizes=(32, 16),  # moderate complexity
    activation="relu",
    solver="adam",
    max_iter=500,
    learning_rate_init=0.001,
    tol=1e-4,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,
    alpha=0.008,  # regularization to reduce overfitting
    random_state=42,
    batch_size=32,
)
mdl.fit(X_train_s, y_train_s)

torch.save((mdl, xs, ys), "narx_model.pth")
print("model saved!")

# load simulation data & expand based on duration column
print("loading simulation input...")
df_sim_input = pd.read_csv(sim_input_file)
expanded_rows = [
    [row["potX"], row["potY"], row["potZ"]]
    for _, row in df_sim_input.iterrows()
    for _ in range(int(row["duration"]))
]
X_sim = np.array(expanded_rows, dtype=np.float64)

init_y = np.zeros((10, 6))
raw_preds = narx_closed_loop_predict(mdl, xs, ys, X_sim, init_y, x_lag=10, y_lag=10)


# apply simple smoothing to reduce noise
def smooth_predictions(predictions, window):
    return np.convolve(predictions, np.ones(window) / window, mode="same")


preds = np.apply_along_axis(lambda x: smooth_predictions(x, 5), axis=0, arr=raw_preds)

df_pred = pd.DataFrame(preds, columns=["RX", "RY", "RZ", "TX", "TY", "TZ"])
df_pred.insert(0, "Frame", range(1, len(df_pred) + 1))
df_pred.to_csv("narx_predictions.csv", index=False)

print("predictions saved!")

# compute velocity & acceleration
df_pred[["V_RX", "V_RY", "V_RZ", "V_TX", "V_TY", "V_TZ"]] = (
    df_pred.iloc[:, 1:7].diff().fillna(0)
)
df_pred[["A_RX", "A_RY", "A_RZ", "A_TX", "A_TY", "A_TZ"]] = (
    df_pred.iloc[:, 7:].diff().fillna(0)
)

# save plots
plt.savefig("narx_predictions.png")
print("plots saved!")
