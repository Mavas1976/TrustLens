import warnings, logging, io, contextlib, traceback, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import trustlens
from trustlens import analyze
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import make_classification
logging.basicConfig(level=logging.WARNING, format="LOG %(levelname)s %(name)s: %(message)s")

def run(name, fn):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            r = fn()
        cal = r.results.get("calibration", {})
        print(f"[{name}] OK task={r.task_type} score={r.trust_score.score} grade={r.trust_score.grade} "
              f"brier={cal.get('brier_score')} ece={cal.get('ece')} degraded={r.backend_metadata.get('degraded_mode')} "
              f"n_classes={r.metadata.get('n_classes')} err={r.results.get('failure',{}).get('misclassification_summary',{}).get('__overall__')}")
        return r
    except Exception as e:
        print(f"[{name}] EXC {type(e).__name__}: {str(e)[:200]}")

X, y = make_classification(n_samples=400, n_features=5, random_state=0)
m = LogisticRegression().fit(X[:200], y[:200]); Xt, yt = X[200:], y[200:]
P = m.predict_proba(Xt)
run("baseline", lambda: analyze(m, Xt, yt, verbose=False))
run("pandas X/y", lambda: analyze(m, pd.DataFrame(Xt), pd.Series(yt), verbose=False))
run("pandas y_prob DF", lambda: analyze(m, Xt, yt, y_prob=pd.DataFrame(P), verbose=False))
run("list y_prob", lambda: analyze(m, Xt, yt, y_prob=P.tolist(), verbose=False))
run("list y_true", lambda: analyze(m, Xt, list(yt), verbose=False))
# string labels
ys = np.where(y=="x", "a", np.where(y==1, "yes", "no"))
ms = LogisticRegression().fit(X[:200], ys[:200])
run("string labels sklearn", lambda: analyze(ms, Xt, ys[200:], verbose=False))
run("string labels manual y_prob only", lambda: analyze(None, Xt, ys[200:], y_prob=ms.predict_proba(Xt), verbose=False))
run("string labels manual +class_labels", lambda: analyze(None, Xt, ys[200:], y_prob=ms.predict_proba(Xt), class_labels=ms.classes_, verbose=False))
# labels 1/2 binary
y12 = y+1
m12 = LogisticRegression().fit(X[:200], y12[:200])
run("labels {1,2} sklearn", lambda: analyze(m12, Xt, y12[200:], verbose=False))
run("labels {1,2} manual y_prob", lambda: analyze(None, Xt, y12[200:], y_prob=m12.predict_proba(Xt), verbose=False))
run("labels {1,2} manual y_pred+y_prob", lambda: analyze(None, Xt, y12[200:], y_pred=m12.predict(Xt), y_prob=m12.predict_proba(Xt), verbose=False))
# multiclass 1..3
X3, y3 = make_classification(n_samples=600, n_features=6, n_informative=4, n_classes=3, random_state=1)
y3b = y3 + 1
m3 = LogisticRegression(max_iter=500).fit(X3[:300], y3b[:300])
run("mc labels 1..3 sklearn", lambda: analyze(m3, X3[300:], y3b[300:], verbose=False))
run("mc labels 1..3 manual y_prob", lambda: analyze(None, X3[300:], y3b[300:], y_prob=m3.predict_proba(X3[300:]), verbose=False))
run("mc labels 1..3 manual y_pred+y_prob", lambda: analyze(None, X3[300:], y3b[300:], y_pred=m3.predict(X3[300:]), y_prob=m3.predict_proba(X3[300:]), verbose=False))
# misaligned columns: reversed
run("binary reversed y_prob cols (sklearn model)", lambda: analyze(m, Xt, yt, y_prob=P[:, ::-1], verbose=False))
# 1-D binary
run("1-D y_prob sklearn", lambda: analyze(m, Xt, yt, y_prob=P[:,1], verbose=False))
run("1-D y_prob manual", lambda: analyze(None, Xt, yt, y_prob=P[:,1], verbose=False))
run("1-D y_prob manual + y_pred", lambda: analyze(None, Xt, yt, y_pred=m.predict(Xt), y_prob=P[:,1], verbose=False))
# NaN
Pn = P.copy(); Pn[0,0]=np.nan
run("NaN y_prob", lambda: analyze(m, Xt, yt, y_prob=Pn, verbose=False))
Xn = Xt.copy(); Xn[0,0]=np.nan
run("NaN X", lambda: analyze(m, Xn, yt, verbose=False))
ytn = yt.astype(float); ytn[0]=np.nan
run("NaN y_true", lambda: analyze(m, Xt, ytn, verbose=False))
# row sums not 1
run("y_prob rows not summing to 1", lambda: analyze(m, Xt, yt, y_prob=P*0.5, verbose=False))
# mismatched lengths
run("y_true shorter", lambda: analyze(m, Xt, yt[:-10], verbose=False))
run("y_true longer (manual)", lambda: analyze(None, Xt, np.r_[yt, yt[:10]], y_pred=m.predict(Xt), y_prob=P, verbose=False))
# single class
run("single-class y_true", lambda: analyze(m, Xt, np.ones_like(yt), verbose=False))
# regression misrouting
Xr = np.random.RandomState(0).randn(300, 3)
yr_int = (100 + 50*Xr[:,0]).round().astype(int)
mr = LinearRegression().fit(Xr, yr_int)
run("int regression target auto", lambda: analyze(mr, Xr, yr_int, verbose=False))
run("int-valued float regression target auto", lambda: analyze(mr, Xr, yr_int.astype(float), verbose=False))
run("int regression target task=regression", lambda: analyze(mr, Xr, yr_int, task="regression", verbose=False))
run("regression y_pred manual int target", lambda: analyze(None, Xr, yr_int, y_pred=mr.predict(Xr), verbose=False))
run("bad task", lambda: analyze(m, Xt, yt, task="Regression", verbose=False))
# sensitive features variants
g = np.random.RandomState(1).choice(["A","B"], len(yt))
run("sens dict np", lambda: analyze(m, Xt, yt, sensitive_features={"g": g}, verbose=False))
run("sens dict pandas", lambda: analyze(m, Xt, yt, sensitive_features={"g": pd.Series(g)}, verbose=False))
run("sens dict list", lambda: analyze(m, Xt, yt, sensitive_features={"g": list(g)}, verbose=False))
run("sens wrong length", lambda: analyze(m, Xt, yt, sensitive_features={"g": g[:50]}, verbose=False))
run("sens DataFrame", lambda: analyze(m, Xt, yt, sensitive_features=pd.DataFrame({"g": g}), verbose=False))
run("sens single group", lambda: analyze(m, Xt, yt, sensitive_features={"g": np.array(["A"]*len(yt))}, verbose=False))
run("sens with NaN", lambda: analyze(m, Xt, yt, sensitive_features={"g": np.where(g=="A", np.nan, 1.0)}, verbose=False))
run("modules=[] ", lambda: analyze(m, Xt, yt, modules=[], verbose=False))
run("modules typo", lambda: analyze(m, Xt, yt, modules=["calibartion"], verbose=False))
run("unknown plugin", lambda: analyze(m, Xt, yt, plugins=["nope"], verbose=False))
