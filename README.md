# An Honest Backtest of a Market Model

20 years of S&P 500 data. A random forest scores about **57%** with shuffled cross-validation and about **50%** with walk-forward
validation, below the 57.4% you get by always guessing "up". Same model, same data, different split.

This repo reproduces that experiment exactly, then lets you test other algorithms under the same protocol.

## The setup (as in the original post)
- **Data:** S&P 500 (`^GSPC`) daily closes, 1999-2018
- **Features (6, past only):** returns over 1, 5, 10, 20 days; 20-day volatility; distance to the 50-day average
- **Question:** will the index be higher in **5 trading days**?
- **Model:** random forest
- **Scoring:** shuffled 5-fold CV vs walk-forward (every test day after every training day) vs an "always up" baseline

## Run it
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_backtest.py              # the main reproduction -> results/leak_sp500.png
python compare_models.py            # extra algorithms through the same protocol
python run_backtest.py --synthetic  # offline check on pure noise (see below)
```

## Files
| File | Role |
|---|---|
| `src/data.py` | Downloads and caches S&P 500 data; synthetic generator |
| `src/features.py` | The six features and the 5-day-ahead target |
| `src/evaluate.py` | Shuffled K-fold, walk-forward, walk-forward with purge gap |
| `src/neuron.py` | A single neuron from scratch in NumPy (start of the neurons-to-networks path) |
| `run_backtest.py` | Main experiment, prints the table and draws the chart |
| `compare_models.py` | Add models to `MODELS` and compare them honestly |

## Why the leak happens
The target looks 5 days ahead, so the target for Monday and the target for Tuesday share 4 of their 5 future days. The features overlap too
(a 20-day return changes by only 2 of its 20 days from one day to the next). With shuffled folds, almost every test day has near-duplicate
neighbours in training, so the forest looks up answers it has effectively already seen:

    leak = acc(shuffled) - acc(walk-forward)

Walk-forward removes this. The **purge gap** variant also skips 5 days between train and test, so no training label can overlap a test label.

## Sanity check: the leak appears even on pure noise
`--synthetic` generates a random walk with **no predictable signal**. Result on my test run:

| Method | Accuracy |
|---|---|
| Random 5-fold | 58.8% |
| Walk-forward | 51.7% |
| Always up | 54.0% |

A model with zero skill "scores" 58.8% under shuffled CV. That is the leak, isolated.

## Caveats
- Exact numbers depend on the data source (yfinance prices are adjusted), random seeds and fold counts. The *gap* is the finding.
- Accuracy is not profitability: no costs, slippage or position sizing.
- Not financial advice.

## Roadmap
1. [x] Reproduce the leak
2. [ ] From-scratch layer and multi-layer network with backpropagation
3. [ ] SVM, PyTorch MLP, LSTM under the same walk-forward protocol
4. [ ] Equity curve, Sharpe ratio, transaction costs
5. [ ] Regime analysis (2000-02, 2008, 2018 drawdowns)
