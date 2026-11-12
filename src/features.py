"""Exactly the setup in the original post.

Six features, built from the past only (info available at the close of day t):
  returns over 1, 5, 10, 20 days | 20-day volatility | distance to the 50-day average
Target: is the index HIGHER in 5 trading days?  (close[t+5] > close[t])
"""
import pandas as pd

HORIZON = 5


def make_features(df: pd.DataFrame, horizon: int = HORIZON):
    c = df["Close"]
    X = pd.DataFrame(index=df.index)
    for k in (1, 5, 10, 20):
        X[f"ret_{k}d"] = c.pct_change(k)
    X["vol_20d"] = c.pct_change().rolling(20).std()
    X["dist_ma_50"] = c / c.rolling(50).mean() - 1

    future = c.shift(-horizon)
    y = (future > c).astype(int)
    ok = future.notna() & X.notna().all(axis=1)        # drop warm-up rows and last `horizon` rows
    return X[ok], y[ok]
