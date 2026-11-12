"""Load ~20 years of S&P 500 daily data (real via yfinance, or synthetic for testing)."""
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[1] / "data" / "sp500.csv"


def load_real(start="1999-01-01", end="2019-01-01", refresh=False) -> pd.DataFrame:
    if CACHE.exists() and not refresh:
        return pd.read_csv(CACHE, index_col=0, parse_dates=True)
    import yfinance as yf
    df = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):          # newer yfinance returns 2-level columns
        df.columns = df.columns.get_level_values(0)
    df = df[["Close", "Volume"]].dropna()
    df.to_csv(CACHE)
    return df


def make_synthetic(n=5200, seed=0) -> pd.DataFrame:
    """Random-walk with volatility clustering and NO predictable direction.
    Used only to sanity-check the pipeline: true skill is zero, so any honest score is just the base rate."""
    rng = np.random.default_rng(seed)
    vol = np.empty(n); vol[0] = 0.01
    ret = np.empty(n)
    for t in range(n):
        if t > 0:
            vol[t] = np.sqrt(1e-6 + 0.08 * ret[t-1]**2 + 0.90 * vol[t-1]**2)
        ret[t] = 0.0003 + vol[t] * rng.standard_normal()
    close = 1000 * np.exp(np.cumsum(ret))
    volume = rng.lognormal(20, 0.3, n) * (1 + 20 * vol)
    idx = pd.bdate_range("1999-01-04", periods=n)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=idx)
