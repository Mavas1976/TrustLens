"""WP1 T5: failure sub-score/blocker on perfectly calibrated binary models of increasing sharpness + real sklearn models."""
import logging, io, contextlib, warnings
import numpy as np
warnings.simplefilter("ignore"); logging.disable(logging.CRITICAL)
from trustlens import analyze
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
rng = np.random.default_rng(5)
def run(*a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return analyze(*a, verbose=False, **kw)
print("sd  acc    ECE    gap    fail_sub  score grade blocked")
for sd in [1, 2, 4, 8, 16, 32, 64]:
    n = 20000; z = rng.normal(0, sd, n); p = 1/(1+np.exp(-z)); y = (rng.random(n) < p).astype(int)
    r = run(None, None, y, y_pred=(p>.5).astype(int), y_prob=np.c_[1-p, p])
    t = r.trust_score
    print(f"{sd:<3} {np.mean((p>.5)==y):.4f} {r.results['calibration']['ece']:.4f} {r.results['failure']['confidence_gap']['gap']:.4f} {t.sub_scores['failure']:6.1f}   {t.score:4d}  {t.grade}     {t.is_blocked}  {t.verdict[:45]}")
print()
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
for m in [LogisticRegression(max_iter=5000), RandomForestClassifier(random_state=0)]:
    m.fit(Xtr, ytr); r = run(m, Xte, yte)
    t = r.trust_score
    print(f"breast_cancer {type(m).__name__}: acc={m.score(Xte, yte):.3f} ECE={r.results['calibration']['ece']:.3f} gap={r.results['failure']['confidence_gap']['gap']} sub={t.sub_scores} score={t.score} grade={t.grade} blocked={t.is_blocked} | {t.verdict[:60]}")
X, y = make_classification(n_samples=4000, n_features=20, n_informative=10, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)
m = LogisticRegression(max_iter=5000).fit(Xtr, ytr); r = run(m, Xte, yte); t = r.trust_score
print(f"make_classification LR: acc={m.score(Xte, yte):.3f} ECE={r.results['calibration']['ece']:.3f} gap={r.results['failure']['confidence_gap']['gap']} sub={t.sub_scores} score={t.score} grade={t.grade} blocked={t.is_blocked} | {t.verdict[:60]}")
