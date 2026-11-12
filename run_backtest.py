"""Reproduces the post: random forest, shuffled 5-fold CV vs walk-forward vs 'always up'.

  python run_backtest.py              real S&P 500, 1999-2018 (needs internet once, then cached)
  python run_backtest.py --synthetic  offline pipeline check on a no-skill random walk
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from src.data import load_real, make_synthetic
from src.features import make_features, HORIZON
from src.evaluate import shuffled_cv, walk_forward

ap = argparse.ArgumentParser()
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--splits", type=int, default=5)
ap.add_argument("--trees", type=int, default=300)
args = ap.parse_args()

df = make_synthetic() if args.synthetic else load_real()
X, y = make_features(df)
base_all = y.mean() * 100
print(f"{len(X)} rows | {X.index[0].date()} -> {X.index[-1].date()} | up in {HORIZON} days: {base_all:.1f}% of the time")

rf = RandomForestClassifier(n_estimators=args.trees, n_jobs=-1, random_state=0)

sh, _ = shuffled_cv(rf, X, y, args.splits)
wf, wf_base = walk_forward(rf, X, y, args.splits)
pg, _ = walk_forward(rf, X, y, args.splits, gap=HORIZON)

rows = [
    ("Random 5-fold (leaks the future)", sh.mean() * 100, sh.std() * 100),
    ("Walk-forward (honest)", wf.mean() * 100, wf.std() * 100),
    (f"Walk-forward + {HORIZON}d purge gap", pg.mean() * 100, pg.std() * 100),
    ("Always say 'up' (no model), whole sample", base_all, 0.0),
    ("Always say 'up', walk-forward test windows", wf_base.mean() * 100, wf_base.std() * 100),
]
res = pd.DataFrame(rows, columns=["method", "accuracy_pct", "std_pct"])
print(res.round(1).to_string(index=False))
print(f"\nLEAK = shuffled - walk-forward = {sh.mean()*100 - wf.mean()*100:+.1f} points")

tag = "synthetic" if args.synthetic else "sp500"
res.to_csv(f"results/results_{tag}.csv", index=False)

# --- chart in the style of the original post ---
labels = ["random 5-fold\n(leaks the future)", "walk-forward\n(honest)", "always say \"up\"\n(no model)"]
vals = [sh.mean() * 100, wf.mean() * 100, base_all]
fig, ax = plt.subplots(figsize=(6.5, 5))
bars = ax.bar(labels, vals, color=["#d6455d", "#1f9e89", "#5b6270"], width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}%", ha="center", fontweight="bold")
ax.axhline(50, color="gray", ls="--", lw=0.8)
ax.set_ylim(40, 62); ax.set_ylabel("accuracy (%)")
span = f"{X.index[0].year}-{X.index[-1].year}"
ax.set_title(f"S&P 500, {span}: is the index higher in {HORIZON} days?" + ("  [SYNTHETIC]" if args.synthetic else ""), fontsize=10)
plt.tight_layout(); plt.savefig(f"results/leak_{tag}.png", dpi=150)
print(f"Saved results/results_{tag}.csv and results/leak_{tag}.png")
