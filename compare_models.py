"""Extension: run other algorithms through the SAME honest protocol.
Add a model to MODELS and re-run. Scores shuffled vs walk-forward vs purged walk-forward."""
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_real, make_synthetic
from src.features import make_features, HORIZON
from src.evaluate import shuffled_cv, walk_forward
from src.neuron import Neuron

ap = argparse.ArgumentParser()
ap.add_argument("--synthetic", action="store_true")
args = ap.parse_args()

df = make_synthetic() if args.synthetic else load_real()
X, y = make_features(df)
print(f"always-up baseline: {y.mean()*100:.1f}%")

MODELS = {
    "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0),
    "GradientBoosting": GradientBoostingClassifier(random_state=0),
    "LogisticReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Neuron (scratch)": Neuron(),
}
rows = []
for name, m in MODELS.items():
    s, _ = shuffled_cv(m, X, y)
    w, _ = walk_forward(m, X, y)
    p, _ = walk_forward(m, X, y, gap=HORIZON)
    rows.append({"model": name, "shuffled": s.mean()*100, "walk_fwd": w.mean()*100,
                 "walk_fwd_purged": p.mean()*100, "leak": (s.mean()-w.mean())*100})
out = pd.DataFrame(rows).round(1)
print(out.to_string(index=False))
out.to_csv("results/model_comparison.csv", index=False)
