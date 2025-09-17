import numpy as np
import pandas as pd

def prepare_rl_data(df, feature_cols=None, signal_col="signal", window=32, train_ratio=0.8):
    """
    Prepares data for RL training.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV + indicators + optional signal column.
    feature_cols : list[str], optional
        Columns to use as features. Defaults to standard OHLCV + EMA columns.
    signal_col : str
        Column name of trading signal (BUY/SELL/False).
    window : int
        Length of sliding window for RL state.
    train_ratio : float
        Fraction of data for training.

    Returns
    -------
    dict
        Dictionary with train/test arrays:
        X_train, X_test, nr_train, nr_test,
        sig_train, sig_test, t_train, t_test,
        p_train, p_test
    """
    df = df.copy()
    
    # Convert signal to boolean 0/1
    df[signal_col] = df[signal_col].apply(lambda x: 1.0 if x == 'BUY' else 0.0)
    
    # Compute log returns
    df["log_return"] = np.log(df["close"]).diff()
    df["next_return"] = df["log_return"].shift(-1)
    
    # Features
    if feature_cols is None:
        feature_cols = ["open", "high", "low", "close", 
                        "tick_volume", "spread", 
                        "EMA34", "EMA55", "EMA84"]
    
    X_raw = df[feature_cols].values.astype(np.float32)
    y_signal = df[signal_col].fillna(0).values.astype(np.float32)
    times = df["time"].values
    prices = df["close"].values

    # Sliding window builder
    X_seq, y_out, sig_out, t_out, p_out = [], [], [], [], []
    for i in range(len(X_raw) - window - 1):
        X_seq.append(X_raw[i:i+window])
        y_out.append(df["next_return"].values[i+window])
        sig_out.append(y_signal[i+window])
        t_out.append(times[i+window])
        p_out.append(prices[i+window])

    X_seq = np.array(X_seq, dtype=np.float32)
    y_out = np.array(y_out, dtype=np.float32)
    sig_out = np.array(sig_out, dtype=np.float32)
    t_out = np.array(t_out)
    p_out = np.array(p_out, dtype=np.float32)

    # Train/test split
    split_idx = int(train_ratio * len(X_seq))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    nr_train, nr_test = y_out[:split_idx], y_out[split_idx:]
    sig_train, sig_test = sig_out[:split_idx], sig_out[split_idx:]
    t_train, t_test = t_out[:split_idx], t_out[split_idx:]
    p_train, p_test = p_out[:split_idx], p_out[split_idx:]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "nr_train": nr_train,
        "nr_test": nr_test,
        "sig_train": sig_train,
        "sig_test": sig_test,
        "t_train": t_train,
        "t_test": t_test,
        "p_train": p_train,
        "p_test": p_test
    }
