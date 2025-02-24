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
    # load all csv files in a folder and merge them into one dataframe
    # useful when training data is split across multiple files
    all_dfs = []
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        df = pd.read_csv(f)
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def make_narx_features(X, Y, x_lag=4, y_lag=4):
    # create lagged features for narx model
    # this adds past values of inputs (X) and outputs (Y) as features to help the model
    # learn relationships over time
    N = len(X)
    m = max(x_lag, y_lag)  # we need to trim to the largest lag
    cols = []

    # add past input values
    for lag in range(1, x_lag + 1):
        if lag < len(X):
            cols.append(X[lag:])

    # add past output values
    for lag in range(1, y_lag + 1):
        if lag < len(Y):
            cols.append(Y[lag:])

    # trim all arrays to match in length before stacking
    trimmed = [c[: (N - m)] for c in cols]
    features = np.column_stack(trimmed)
    targets = Y[m:]

    return features, targets


def narx_closed_loop_predict(
    model, x_scaler, y_scaler, X_full, init_y, x_lag=10, y_lag=10
):
    # closed-loop prediction method
    # after predicting a new value, we feed it back into the model for future predictions
    # uses past X values and the differences in predicted Y values to generate new inputs

    N, d = len(X_full), init_y.shape[1]
    preds = np.zeros((N, d))

    # initialize first few predictions with given values
    preds[:y_lag] = init_y[:y_lag]

    for k in range(y_lag, N):
        x_features = []
        for lag in range(1, x_lag + 1):
            if k - lag >= 0:
                x_features.append(X_full[k - lag])  # use past input values
            else:
                x_features.append(np.zeros_like(X_full[0]))  # pad with zeros if needed

        y_features = []
        for lag in range(1, y_lag + 1):
            if k - lag >= 0:
                # difference between previous predicted values (velocity-like feature)
                y_features.append(preds[k - lag] - preds[k - lag - 1])
            else:
                y_features.append(np.zeros(d))

        f = np.hstack(x_features + y_features).reshape(1, -1)
        fs = x_scaler.transform(f)

        # predicting change in output (delta)
        delta_pred = y_scaler.inverse_transform(model.predict(fs))[0]

        # update prediction by adding the delta to the previous prediction
        preds[k] = preds[k - 1] + delta_pred

    return preds


if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <training_folder> <simulation_input.csv>")
    sys.exit(1)

training_folder = sys.argv[1]
sim_input_file = sys.argv[2]
print("Loading training data...")
df_train = load_training_data(training_folder)

# select input features (potentiometer values) and output features (rx, ry, rz, tx, ty, tz)
Xf = df_train[["potX", "potY", "potZ"]].values.astype(np.float64)
Yf = df_train[["RX", "RY", "RZ", "TX", "TY", "TZ"]].values.astype(np.float64)

print(f"Training data shapes:")
print(f"Xf shape: {Xf.shape}")
print(f"Yf shape: {Yf.shape}")

# calculate differences in output (delta Y)
delta_Yf = np.diff(Yf, axis=0)
delta_Yf = np.vstack([np.zeros(Yf.shape[1]), delta_Yf])  # pad first row with zeros
print(f"Delta Yf shape: {delta_Yf.shape}")

# generate narx features using a lag of 10 for both X and Y
print("Generating NARX features...")
X_narx, Y_narx = make_narx_features(Xf, delta_Yf, x_lag=10, y_lag=10)

assert Y_narx.shape[1] == 6, f"Expected 6 columns in Y_narx but got {Y_narx.shape[1]}"

# split data and scale features
X_train, X_test, y_train, y_test = train_test_split(
    X_narx, Y_narx, test_size=0.2, random_state=42
)
xs, ys = StandardScaler(), StandardScaler()
X_train_s, X_test_s = xs.fit_transform(X_train), xs.transform(X_test)
y_train_s, y_test_s = ys.fit_transform(y_train), ys.transform(y_test)

print("Training model...")
mdl = MLPRegressor(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    max_iter=500,
    learning_rate="constant",
    learning_rate_init=0.001,
    tol=1e-4,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,
    alpha=0.008,
    random_state=42,
    batch_size=32,
)
mdl.fit(X_train_s, y_train_s)

# save trained model
torch.save((mdl, xs, ys), "narx_model.pth")
print("Model trained and saved as narx_model.pth!")

print("Loading simulation input...")
df_sim_input = pd.read_csv(sim_input_file)

expanded_rows = []
for _, row in df_sim_input.iterrows():
    for _ in range(int(row["duration"])):
        expanded_rows.append([row["potX"], row["potY"], row["potZ"]])
X_sim = np.array(expanded_rows, dtype=np.float64)

init_y = np.zeros((10, 6))
raw_preds = narx_closed_loop_predict(mdl, xs, ys, X_sim, init_y, x_lag=10, y_lag=10)

print(f"Raw predictions shape: {raw_preds.shape}")


def smooth_predictions(predictions, window):
    return np.convolve(predictions, np.ones(window) / window, mode="same")


preds = np.apply_along_axis(lambda x: smooth_predictions(x, 5), axis=0, arr=raw_preds)

assert preds.shape[1] == 6, f"Expected 6 columns but got {preds.shape[1]}"
df_pred = pd.DataFrame(preds, columns=["RX", "RY", "RZ", "TX", "TY", "TZ"])

df_pred.insert(0, "Frame", range(1, len(df_pred) + 1))
df_pred.to_csv("narx_predictions.csv", index=False)
print("Predictions saved to narx_predictions.csv!")

# compute velocity and acceleration
df_pred[["V_RX", "V_RY", "V_RZ", "V_TX", "V_TY", "V_TZ"]] = (
    df_pred[["RX", "RY", "RZ", "TX", "TY", "TZ"]].diff().fillna(0)
)
df_pred[["A_RX", "A_RY", "A_RZ", "A_TX", "A_TY", "A_TZ"]] = (
    df_pred[["V_RX", "V_RY", "V_RZ", "V_TX", "V_TY", "V_TZ"]].diff().fillna(0)
)

# generate plots
print("Generating plots...")
fig, axs = plt.subplots(3, 1, figsize=(12, 15))

for label in ["RX", "RY", "RZ", "TX", "TY", "TZ"]:
    axs[0].plot(df_pred["Frame"], df_pred[label], label=label)
axs[0].set_title("Position Over Time")
axs[0].legend()

for label in ["V_RX", "V_RY", "V_RZ", "V_TX", "V_TY", "V_TZ"]:
    axs[1].plot(df_pred["Frame"], df_pred[label], label=label)
axs[1].set_title("Velocity Over Time")
axs[1].legend()

for label in ["A_RX", "A_RY", "A_RZ", "A_TX", "A_TY", "A_TZ"]:
    axs[2].plot(df_pred["Frame"], df_pred[label], label=label)
axs[2].set_title("Acceleration Over Time")
axs[2].legend()

plt.tight_layout()
plt.savefig("narx_predictions.png")
print("Plots saved as narx_predictions.png!")
