"""Same model, same data, different splits."""
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, TimeSeriesSplit


def _run(model, X, y, splits):
    accs, base = [], []
    for tr, te in splits:
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        accs.append((m.predict(X.iloc[te]) == y.iloc[te].values).mean())
        base.append(y.iloc[te].mean())                  # accuracy of "always up" on this test fold
    return np.array(accs), np.array(base)


def shuffled_cv(model, X, y, n_splits=5, seed=42):
    """LEAKY: test days sit between training days, so overlapping 5-day outcomes are shared."""
    return _run(model, X, y, KFold(n_splits, shuffle=True, random_state=seed).split(X))


def walk_forward(model, X, y, n_splits=5, gap=0):
    """HONEST: every test day comes after every training day. gap = days skipped between them."""
    return _run(model, X, y, TimeSeriesSplit(n_splits=n_splits, gap=gap).split(X))
