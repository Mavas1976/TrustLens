import io, contextlib, json, time, os, sys, logging
logging.basicConfig(level=logging.WARNING, format="LOG %(levelname)s %(name)s: %(message)s")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl
from trustlens import analyze
from trustlens.comparison import compare
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import tempfile
SP = tempfile.mkdtemp(prefix="trustlens_audit_")  # never write into the repo
X, y = make_classification(n_samples=400, n_features=5, random_state=0)
m = LogisticRegression().fit(X[:200], y[:200]); Xt, yt = X[200:], y[200:]
g = np.random.RandomState(1).choice(["A","B"], len(yt))
q = io.StringIO()
with contextlib.redirect_stdout(q):
    r = analyze(m, Xt, yt, sensitive_features={"<img src=x onerror=alert(1)>": g}, verbose=False)
print("stdout chars printed by analyze(verbose=False):", len(q.getvalue()), repr(q.getvalue()[:120]))
# XSS
rc_before = dict(mpl.rcParams)
html = r._repr_html_()
print("XSS raw tag in _repr_html_:", "<img src=x onerror=alert(1)>" in html)
for line in html.splitlines():
    if "onerror" in line: print("   ", line.strip()[:200])
rc_after = dict(mpl.rcParams)
diff = {k:(rc_before[k], rc_after[k]) for k in rc_before if str(rc_before[k])!=str(rc_after[k])}
print("rcParams diff after _repr_html_:", diff)
print("open figs after _repr_html_:", len(plt.get_fignums()))
# to_dict json
d = r.to_dict()
try: json.dumps(d); print("to_dict json OK, keys", len(d))
except Exception as e: print("to_dict json FAIL", e)
# save variants
os.chdir(SP)
p = r.save(os.path.join(SP, "out_a.json")); print("save json ->", p)
p2 = r.save(os.path.join(SP, "out_a.json")); print("save again same path (overwrite silently?) ->", p2)
p3 = r.save(os.path.join(SP, "bundle")); print("bundle ->", sorted(os.listdir(p3)))
p4 = r.save(os.path.join(SP, "report.PNG")); print("save .PNG path ->", p4, "isdir", os.path.isdir(p4))
p5 = r.save(os.path.join(SP, "x/../trav.json")); print("traversal path ->", p5)
print("open figs after save:", len(plt.get_fignums()))
# repeated calls & fig leak
for i in range(5):
    with contextlib.redirect_stdout(io.StringIO()):
        rr = analyze(m, Xt, yt, verbose=False); rr.summary_plot(show=False); rr.plot_bias if False else None
print("open figs after 5x analyze+summary_plot:", len(plt.get_fignums()))
with contextlib.redirect_stdout(io.StringIO()):
    try:
        r.plot(); print_ok=True
    except Exception as e: print("plot() EXC", e)
print("open figs after r.plot():", len(plt.get_fignums()))
# modules typo detail
with contextlib.redirect_stdout(io.StringIO()):
    rt = analyze(m, Xt, yt, modules=["calibartion"], verbose=False)
print("modules typo: results keys", list(rt.results), "score", rt.trust_score.score, rt.trust_score.grade, "verdict", rt.trust_score.verdict, "sub", rt.trust_score.sub_scores)
# compare
buf = io.StringIO()
with contextlib.redirect_stdout(buf): compare([r, rt])
print("compare output:\n" + buf.getvalue())
# regression report save/html
from sklearn.linear_model import LinearRegression
Xr = np.random.RandomState(0).randn(300, 3); yr = Xr[:,0]*3.3 + np.random.RandomState(1).randn(300)
with contextlib.redirect_stdout(io.StringIO()):
    rr = analyze(LinearRegression().fit(Xr, yr), Xr, yr, verbose=False)
print("regression task", rr.task_type, rr.trust_score.score, "html len", len(rr._repr_html_()))
json.dumps(rr.to_dict()); print("regression to_dict json OK")
print(rr.save(os.path.join(SP, "reg.json")))
# perf
for n in (20000, 200000):
    Xb, yb = make_classification(n_samples=n, n_features=10, random_state=0)
    mb = LogisticRegression().fit(Xb[:5000], yb[:5000])
    gb = np.random.RandomState(0).choice(["a","b","c"], n)
    t=time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        rb = analyze(mb, Xb, yb, sensitive_features={"g": gb}, verbose=False)
    t1=time.time()-t; t=time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        rb.save(os.path.join(SP, f"perf_{n}.json"))
    t2=time.time()-t
    print(f"perf n={n}: analyze {t1:.2f}s save.json {t2:.2f}s size {os.path.getsize(os.path.join(SP, f'perf_{n}.json'))}B")
import trustlens.trust_score
print("matplotlib imported by trustlens import:", 'matplotlib' in sys.modules)
