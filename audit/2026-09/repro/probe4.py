import io, contextlib, logging, warnings
logging.basicConfig(level=logging.WARNING, format="LOG %(levelname)s %(name)s: %(message)s")
import numpy as np, matplotlib; matplotlib.use("Agg")
from trustlens import analyze
from trustlens.comparison import compare
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=2000, n_features=8, flip_y=0.1, random_state=0)
Xtr, Xt, ytr, yt = X[:1000], X[1000:], y[:1000], y[1000:]
good = GaussianNB().fit(Xtr, ytr)
rs = {}
for mods in (["bias"], ["calibration"], ["failure"], ["representation"]):
    with warnings.catch_warnings(record=True) as w, contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("always")
        r = analyze(good, Xt, yt, modules=mods, verbose=False)
    ts = r.trust_score
    rs[mods[0]] = r
    print(mods, "score", ts.score, ts.grade, "blocked", ts.is_blocked, "subs", ts.sub_scores, "weights", ts.weights_used, "warnings", [str(x.message)[:80] for x in w])
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rf = analyze(good, Xt, yt, verbose=False)
    compare([rf, rs["bias"]])
print(buf.getvalue()[-700:])
