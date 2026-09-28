import warnings, logging, io, contextlib
import numpy as np
warnings.simplefilter("ignore"); logging.disable(logging.CRITICAL)
from trustlens import analyze
from trustlens.trust_score import regression_trust_score
def run(*a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return analyze(*a, verbose=False, **kw)
yc = np.full(100, 3.3)
rep = run(None, None, yc, y_pred=yc + 0.1, task="regression")
print("G14 constant target, pred off by .1:", rep.trust_score.score, rep.trust_score.grade, rep.trust_score.sub_scores, rep.trust_score.verdict)
yc2 = np.full(100, 3.3) + np.linspace(0, 1e-7, 100)
rep = run(None, None, yc2, y_pred=yc2 + 1e-6, task="regression")
print("G14b near-constant target (var 8e-16), pred off by 1e-6:", rep.results['regression']['error_distribution']['rmse'], rep.results['regression']['target_variance'], rep.trust_score.score, rep.trust_score.grade)
# task auto-detect: continuous target with <=20 unique values -> classification
y = np.round(np.random.default_rng(0).normal(size=500) * 2) / 2 + 0.25
print("T auto-detect: n_unique", len(np.unique(y)), "->", end=" ")
try:
    rep = run(None, None, y, y_pred=y)
    print(rep.task_type if hasattr(rep, 'task_type') else rep.results.keys())
except Exception as e:
    print("RAISES", type(e).__name__, str(e)[:100])
