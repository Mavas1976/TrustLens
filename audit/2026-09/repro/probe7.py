import io, contextlib, logging, os, pathlib
logging.basicConfig(level=logging.ERROR)
import numpy as np, matplotlib; matplotlib.use("Agg")
from trustlens import analyze, quick_analyze
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import tempfile
SP = tempfile.mkdtemp(prefix="trustlens_audit_")  # never write into the repo
X, y = make_classification(n_samples=400, n_features=5, random_state=0)
m = LogisticRegression().fit(X[:200], y[:200])
with contextlib.redirect_stdout(io.StringIO()):
    r = analyze(m, X[200:], y[200:], verbose=False)
try: print("save(Path) ->", r.save(pathlib.Path(SP)/"p.json"))
except Exception as e: print("save(Path) EXC", type(e).__name__, e)
# X length mismatch with manual predictions
with contextlib.redirect_stdout(io.StringIO()):
    try:
        r2 = analyze(None, X[:10], y[200:], y_pred=m.predict(X[200:]), y_prob=m.predict_proba(X[200:]), verbose=False); print("X len 10 vs y 200 manual: OK (no validation), score", r2.trust_score.score)
    except Exception as e: print("X mismatch EXC", e)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    q = quick_analyze(m, X[200:], y[200:])
print("quick_analyze(user model) prints 'demo' banner:", [l for l in buf.getvalue().splitlines()[:3]])
