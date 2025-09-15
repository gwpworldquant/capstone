import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Initial parameters to prepare the data
INPUT_CSV = "dip_tip.csv"
OUTPUT_DIR = "prepared_data"
WINDOW_SIZE = 5                 # number of past timesteps to include in each observation by the RL agent
TRAIN_FRACTION = 0.8            # fraction of data used for training
INCLUDE_SIGNAL_IN_STATE = True  # we can opt to omit the dip tip signal to test behavior of RL agent
RANDOM_SEED = 42

# Here we load the data and prepare the df
def load_data(path):
    # Loads data and drops useless columns
    df = pd.read_csv(path, parse_dates=["time"])
    df.drop("open", axis=1, inplace=True)
    df.drop("high", axis=1, inplace=True)
    df.drop("low", axis=1, inplace=True)
    df.drop("tick_volume", axis=1, inplace=True)
    df.drop("spread", axis=1, inplace=True)
    df.drop("real_volume", axis=1, inplace=True)
    df.drop("converged", axis=1, inplace=True)
    df.drop("ema_34", axis=1, inplace=True)
    df.drop("ema_55", axis=1, inplace=True)
    df.drop("ema_84", axis=1, inplace=True)
    df.drop("ema_mid", axis=1, inplace=True)
    df.drop("max_ema_gap", axis=1, inplace=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df

# We do feature engineering
def add_basic_features(df):
    # Log returns
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    # Next period return
    df["next_log_return"] = df["log_return"].shift(-1)

    # Volume normalization: log(volume) / rolling mean of log(volume)
    df["log_volume"] = np.log1p(df["volume"].fillna(0))
    df["vol_rolling_mean_20"] = df["log_volume"].rolling(window=20, min_periods=1).mean()
    df["norm_volume"] = df["log_volume"] / (df["vol_rolling_mean_20"] + 1e-9)

    # EMA differences and ratios
    df["ema34_minus_ema84"] = df["EMA34"] - df["EMA84"]
    df["ema34_minus_ema55"] = df["EMA34"] - df["EMA55"]
    df["ema34_div_close"] = df["EMA34"] / (df["close"] + 1e-9)
    df["ema55_div_close"] = df["EMA55"] / (df["close"] + 1e-9)
    df["ema84_div_close"] = df["EMA84"] / (df["close"] + 1e-9)

    # Rolling volatility
    df["volatility_20"] = df["log_return"].rolling(window=20, min_periods=1).std().fillna(0)

    # Ensure ema_converged is numeric (0/1)
    if df["ema_converged"].dtype == "bool":
        df["ema_converged"] = df["ema_converged"].astype(int)
    else:
        df["ema_converged"] = pd.to_numeric(df["ema_converged"], errors="coerce").fillna(0).astype(int)

    # Make sure signal is binary numeric
    if df["signal"].dtype == "bool":
        df["signal_bool"] = df["signal"].astype(int)
    else:
        df["signal_bool"] = pd.to_numeric(df["signal"], errors="coerce").fillna(0).astype(int)

    return df

def drop_initial_nans(df, min_valid=1):
    # Drop the first rows with NaN due to shifts
    df2 = df.dropna(subset=["log_return"]).reset_index(drop=True)
    return df2

# We build the sequence of past time steps
def build_sequences(feature_array, window_size):
    T = feature_array.shape[0]
    if T < window_size:
        return np.zeros((0, window_size, feature_array.shape[1]))
    N = T - window_size + 1
    X = np.zeros((N, window_size, feature_array.shape[1]), dtype=np.float32)
    for i in range(N):
        X[i] = feature_array[i:i+window_size]
    return X

# We prepare the data pipeline for the df
def prepare_dataset(df,
                    feature_cols,
                    window_size=WINDOW_SIZE,
                    train_frac=TRAIN_FRACTION,
                    include_signal_in_state=INCLUDE_SIGNAL_IN_STATE,
                    output_dir=OUTPUT_DIR,
                    basename="dataset",
                    random_seed=RANDOM_SEED):
    os.makedirs(output_dir, exist_ok=True)

    # Keep time and price columns for alignment & later reward calculation
    times = df["time"].copy()
    prices = df["close"].copy()
    y_signal = df["signal_bool"].copy()

    # Split sequentially
    T = len(df)
    train_T = int(np.floor(T * train_frac))
    if train_T <= window_size:
        raise ValueError("Training set too small for the chosen window_size. Reduce window_size or increase train_frac.")

    # Select features for scaler
    X_df = df[feature_cols].copy()

    # Fit scaler on training portion to avoid leakage
    scaler = StandardScaler()
    scaler.fit(X_df.iloc[:train_T].values)
    X_scaled = scaler.transform(X_df.values)

    # Save processed CSV (unscaled)
    proc_csv_path = Path(output_dir) / f"processed_{basename}.csv"
    df_to_save = df.copy()
    # Drop intermediate helper columns
    df_to_save.to_csv(proc_csv_path, index=False)

    # Build sequences from scaled features
    X_seq = build_sequences(X_scaled, window_size)
    # For alignment, we drop first (window_size-1) timestamps for sequence-based arrays
    seq_times = times.iloc[window_size-1:].reset_index(drop=True)
    seq_prices = prices.iloc[window_size-1:].reset_index(drop=True)
    seq_y_signal = y_signal.iloc[window_size-1:].reset_index(drop=True)
    seq_next_return = df["next_log_return"].iloc[window_size-1:].reset_index(drop=True)

    # Now split into train/test over the sequences
    N = X_seq.shape[0]
    train_N = int(np.floor(N * train_frac))
    # Training arrays
    X_train = X_seq[:train_N]
    y_train_signal = seq_y_signal.iloc[:train_N].values.astype(np.int8)
    times_train = seq_times.iloc[:train_N].astype(str).values
    prices_train = seq_prices.iloc[:train_N].values
    next_return_train = seq_next_return.iloc[:train_N].values

    # Test arrays
    X_test = X_seq[train_N:]
    y_test_signal = seq_y_signal.iloc[train_N:].values.astype(np.int8)
    times_test = seq_times.iloc[train_N:].astype(str).values
    prices_test = seq_prices.iloc[train_N:].values
    next_return_test = seq_next_return.iloc[train_N:].values

    # Here we include signal inside the observation/state
    if include_signal_in_state:
        # We'll append signal as last feature in each timestep of the window
        # Create expanded versions and re-save
        def append_signal_to_X(X_arr, seq_y_signal_full):
            # seq_y_signal_full is signal aligned to last timestep of each sequence
            X_new = X_arr.copy()
            # need to append a constant feature for each row in window equal to the signal at that sequence's last time
            N_local, w, nfeat = X_arr.shape
            X_e = np.zeros((N_local, w, nfeat + 1), dtype=X_arr.dtype)
            X_e[:, :, :nfeat] = X_arr
            # Broadcast the signal value along the window axis
            for i in range(N_local):
                X_e[i, :, nfeat] = seq_y_signal_full.iloc[i]
            return X_e

        X_train = append_signal_to_X(X_train, seq_y_signal.iloc[:train_N])
        X_test  = append_signal_to_X(X_test,  seq_y_signal.iloc[train_N:])

    # Save npz and scalers
    train_path = Path(output_dir) / f"train_{basename}.npz"
    test_path  = Path(output_dir) / f"test_{basename}.npz"
    np.savez_compressed(train_path,
                        X=X_train,
                        y_signal=y_train_signal,
                        times=times_train,
                        prices=prices_train,
                        next_return=next_return_train)
    np.savez_compressed(test_path,
                        X=X_test,
                        y_signal=y_test_signal,
                        times=times_test,
                        prices=prices_test,
                        next_return=next_return_test)

    scaler_path = Path(output_dir) / f"scaler_{basename}.joblib"
    joblib.dump(scaler, scaler_path)

    print("Saved processed CSV:", proc_csv_path)
    print("Saved train npz:", train_path)
    print("Saved test npz:", test_path)
    print("Saved scaler:", scaler_path)
    return {
        "csv": str(proc_csv_path),
        "train_npz": str(train_path),
        "test_npz": str(test_path),
        "scaler": str(scaler_path),
        "feature_columns": feature_cols
    }

# Code execution
def main(args):
    np.random.seed(args.random_seed)
    df = load_data(args.input_csv)
    df = add_basic_features(df)
    df = drop_initial_nans(df)

    # list of features to use as input to the model
    feature_columns = [
        "log_return",
        "norm_volume",
        "ema34_minus_ema84",
        "ema34_minus_ema55",
        "ema34_div_close",
        "ema55_div_close",
        "ema84_div_close",
        "volatility_20",
        "ema_converged"
    ]

    # We make sure all these exist
    for c in feature_columns:
        if c not in df.columns:
            raise KeyError(f"Expected feature column '{c}' not found in dataframe. Available columns: {df.columns.tolist()}")

    # If we opted to use the dip tip signal
    if "signal_bool" not in df.columns:
        raise KeyError("signal_bool column not found. Make sure the input CSV has 'signal'.")

    basename = Path(args.input_csv).stem
    out = prepare_dataset(df,
                          feature_cols=feature_columns,
                          window_size=args.window_size,
                          train_frac=args.train_frac,
                          include_signal_in_state=args.include_signal_in_state,
                          output_dir=args.output_dir,
                          basename=basename,
                          random_seed=args.random_seed)
    
    print("Prepared files:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default=INPUT_CSV, help="Path to input CSV with required columns.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Where to save processed data.")
    parser.add_argument("--window_size", type=int, default=WINDOW_SIZE, help="Window size for sequence building.")
    parser.add_argument("--train_frac", type=float, default=TRAIN_FRACTION, help="Fraction for train split (sequential).")
    parser.add_argument("--include_signal_in_state", action="store_true", help="If set, add 'signal' as a feature in the state (NOT recommended).")
    parser.add_argument("--random_seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    main(args)
