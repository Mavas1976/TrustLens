import io, contextlib, logging
logging.basicConfig(level=logging.ERROR)
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
from trustlens import analyze
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
X, y = make_classification(n_samples=1000, n_features=5, random_state=0)
m = LogisticRegression().fit(X[:500], y[:500])
Xt = pd.DataFrame(X[500:]); yt = pd.Series(y[500:])
perm = np.random.RandomState(0).permutation(500)
Xs, ys = Xt.iloc[perm], yt.iloc[perm]   # shuffled, non-default index (typical after train_test_split)
g = pd.Series(np.random.RandomState(1).choice(["A","B"], 500), index=ys.index)
def summ(r):
    b = r.results["bias"]; sp = b.get("subgroup_performance", {})
    return r.trust_score.score, r.results["calibration"]["brier_score"], {k: {kk: vv.get("accuracy") if isinstance(vv, dict) else vv for kk, vv in v.items()} for k, v in sp.items()}
with contextlib.redirect_stdout(io.StringIO()):
    a = analyze(m, Xs.values, ys.values, sensitive_features={"g": g.values}, verbose=False)
    b = analyze(m, Xs, ys, sensitive_features={"g": g}, verbose=False)
    g2 = g.sort_index()  # same values, different order, aligned by index
    c = analyze(m, Xs, ys, sensitive_features={"g": g2}, verbose=False)
print("numpy          :", summ(a))
print("pandas shuffled:", summ(b))
print("pandas sens sorted-index (label-aligned same data):", summ(c))
print("y_true type stored in report:", type(b.y_true))
