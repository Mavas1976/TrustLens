import io, contextlib, logging
logging.basicConfig(level=logging.WARNING, format="LOG %(levelname)s %(name)s: %(message)s")
import numpy as np, matplotlib; matplotlib.use("Agg")
from trustlens import analyze
from trustlens.comparison import compare
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=2000, n_features=8, flip_y=0.1, random_state=0)
Xtr, Xt, ytr, yt = X[:1000], X[1000:], y[:1000], y[1000:]
good = GaussianNB().fit(Xtr, ytr)
q = io.StringIO()
with contextlib.redirect_stdout(q):
    r_full = analyze(good, Xt, yt, verbose=False)
    # a terrible model given as y_pred only (degraded): 45% wrong
    rng = np.random.RandomState(0); bad_pred = np.where(rng.rand(len(yt))<0.45, 1-yt, yt)
    r_deg = analyze(None, Xt, yt, y_pred=bad_pred, verbose=False)
for n, r in (("full GaussianNB", r_full), ("degraded 45%-error y_pred-only", r_deg)):
    ts = r.trust_score
    print(n, "score", ts.score, ts.grade, "blocked", ts.is_blocked, "subs", ts.sub_scores, "verdict", ts.verdict, "degraded", r.backend_metadata.get("degraded_mode"))
buf = io.StringIO()
with contextlib.redirect_stdout(buf): compare([r_full, r_deg])
print(buf.getvalue())
# XSS via feature name with forced subgroup gap
g = np.where(yt != good.predict(Xt), "A", np.random.RandomState(2).choice(["A","B"], len(yt)))
with contextlib.redirect_stdout(io.StringIO()):
    rx = analyze(good, Xt, yt, sensitive_features={"<script>alert(1)</script>": g}, verbose=False)
html = rx._repr_html_()
print("XSS payload raw in _repr_html_:", "<script>alert(1)</script>" in html)
print([l.strip()[:160] for l in html.splitlines() if "alert(1)" in l])
