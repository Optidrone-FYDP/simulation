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


def load_training_data(folder):
    """
    Load all CSV files from a folder and merge them into a single DataFrame.
    This is useful when your training data is spread across multiple files.
    """
    all_dfs = []
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):  # Find all CSV files
        df = pd.read_csv(f)
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)  # Merge all data


def make_narx_features(X, Y, x_lag=4, y_lag=4):
    """
    Create lagged features for a NARX model.
    Basically, we are adding past values of inputs (X) and outputs (Y)
    as new columns to help the model learn relationships over time.
    """
    N = len(X)
    m = max(x_lag, y_lag)  # The largest lag determines how much we trim
    cols = []

    # Add past input values
    for lag in range(1, x_lag + 1):
        cols.append(X[lag:])

    # Add past output values
    for lag in range(1, y_lag + 1):
        cols.append(Y[lag:])

    # Trim all arrays to the same length before stacking
    trimmed = [c[len(c) - (N - m) :] for c in cols]

    return np.hstack(trimmed), Y[m:]


def narx_closed_loop_predict(
    model, x_scaler, y_scaler, X_full, init_y, x_lag=4, y_lag=4
):
    """
    Predict future values using a closed-loop approach.
    This means that after predicting a value, we feed it back into the model
    as input for the next prediction (like a chain reaction).
    """
    N, d = len(X_full), init_y.shape[1]  # Number of data points and output dimensions
    preds = np.zeros((N, d))  # Store predictions
    preds[:y_lag] = init_y[:y_lag]  # Fill in the first few rows with initial values

    for k in range(y_lag, N):
        # Get previous `x_lag` inputs
        lx = [X_full[k - lag] for lag in range(1, x_lag + 1)]
        # Get previous `y_lag` outputs (which we are predicting)
        ly = [preds[k - lag] for lag in range(1, y_lag + 1)]
        f = np.hstack(lx + ly).reshape(1, -1)  # Stack and reshape for the model

        # Scale the features and predict
        fs = x_scaler.transform(f)
        ps = model.predict(fs)

        # Transform the prediction back to the original scale
        preds[k] = y_scaler.inverse_transform(ps)[0]

    return preds


# Make sure the script is run with the correct arguments
if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <training_folder> <simulation_input.csv>")
    sys.exit(1)

# Load training data
training_folder = sys.argv[1]
sim_input_file = sys.argv[2]
print("Loading training data...")
df_train = load_training_data(training_folder)

# Select relevant columns (inputs and outputs)
Xf = df_train[["potX", "potY", "potZ"]].values.astype(np.float64)  # Inputs
Yf = df_train[["Frame", "RX", "RY", "RZ", "TX", "TY", "TZ"]].values.astype(
    np.float64
)  # Outputs

# Generate features for training using past values (NARX-style)
print("Generating NARX features...")
X_narx, Y_narx = make_narx_features(Xf, Yf)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_narx, Y_narx, test_size=0.2, random_state=42
)

# Standardize the data (important for neural networks)
print("Scaling features...")
xs, ys = StandardScaler(), StandardScaler()
X_train_s, X_test_s = xs.fit_transform(X_train), xs.transform(X_test)
y_train_s, y_test_s = ys.fit_transform(y_train), ys.transform(y_test)

# Define and train a simple neural network
print("Training model...")
mdl = MLPRegressor(
    hidden_layer_sizes=(256, 256, 256),  # Three hidden layers with 256 neurons each
    activation="tanh",  # Activation function
    solver="adam",  # Optimization algorithm
    max_iter=500,  # Number of training iterations
    random_state=42,
)
mdl.fit(X_train_s, y_train_s)

# Save the trained model
torch.save((mdl, xs, ys), "narx_model.pth")
print("Model trained and saved as narx_model.pth!")

# Load simulation input data
print("Loading simulation input...")
df_sim_input = pd.read_csv(sim_input_file)

# Expand input rows based on duration column
expanded_rows = []
for _, row in df_sim_input.iterrows():
    for _ in range(int(row["duration"])):
        expanded_rows.append([row["potX"], row["potY"], row["potZ"]])
X_sim = np.array(expanded_rows, dtype=np.float64)

# Generate predictions
print("Generating predictions...")
init_y = np.zeros((4, 7))  # Initial conditions for the feedback loop
raw_preds = narx_closed_loop_predict(mdl, xs, ys, X_sim, init_y)


# Apply smoothing to reduce noise in predictions
def smooth_predictions(predictions, window):
    return np.convolve(predictions, np.ones(window) / window, mode="same")


preds = np.apply_along_axis(lambda x: smooth_predictions(x, 9), axis=0, arr=raw_preds)

# Save predictions
df_pred = pd.DataFrame(preds, columns=["Frame", "RX", "RY", "RZ", "TX", "TY", "TZ"])
if "Frame" in df_pred.columns:
    df_pred.drop(columns=["Frame"], inplace=True)

df_pred.insert(0, "Frame", range(1, len(df_pred) + 1))
df_pred.to_csv("narx_predictions.csv", index=False)
print("Predictions saved to narx_predictions.csv!")

# Compute velocity and acceleration
df_pred[["V_RX", "V_RY", "V_RZ", "V_TX", "V_TY", "V_TZ"]] = (
    df_pred[["RX", "RY", "RZ", "TX", "TY", "TZ"]].diff().fillna(0)
)
df_pred[["A_RX", "A_RY", "A_RZ", "A_TX", "A_TY", "A_TZ"]] = (
    df_pred[["V_RX", "V_RY", "V_RZ", "V_TX", "V_TY", "V_TZ"]].diff().fillna(0)
)

# Plot results
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
