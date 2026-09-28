"""WP1 T2: pipeline wiring (binary vs multiclass calibration, label encoding)."""
import logging, io, contextlib
import numpy as np
from sklearn.metrics import brier_score_loss
from trustlens import analyze
from trustlens.trust_score import compute_trust_score, _calibration_score

logging.disable(logging.CRITICAL)
rng = np.random.default_rng(1)

def run(**kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return analyze(None, None, verbose=False, **kw)

def tryrun(name, **kw):
    try:
        r = run(**kw)
        cal = r.results.get("calibration", {})
        print(f"{name}: brier={cal.get('brier_score')} ece={cal.get('ece')} ts={r.trust_score.score} grade={r.trust_score.grade}")
        return r
    except Exception as e:
        print(f"{name}: RAISES {type(e).__name__}: {str(e)[:140]}")

# --- A: multiclass Brier reference and its scale vs binary ---
K = 3; n = 3000
logits = rng.normal(size=(n, K)) * 1.5
P = np.exp(logits); P /= P.sum(1, keepdims=True)
y = np.array([rng.choice(K, p=row) for row in P])  # perfectly calibrated multiclass
r = run(y_true=y, y_pred=P.argmax(1), y_prob=P)
onehot = np.eye(K)[y]
ref_mb = np.mean(np.sum((P - onehot) ** 2, 1))
print("A1 multiclass Brier:", r.results["calibration"]["brier_score"], "ref:", ref_mb)
# top-label ECE ref
conf = P.max(1); corr = (P.argmax(1) == y).astype(float)
print("A2 top-label ECE:", r.results["calibration"]["ece"], " perfectly calibrated model -> calibration sub-score:", r.trust_score.sub_scores.get("calibration"))
# Uniform (uninformative) multiclass predictions K=10
K = 10; n = 5000
P = np.full((n, K), 1 / K); y = rng.integers(0, K, n)
P2 = P + rng.normal(scale=1e-6, size=P.shape); P2 = np.abs(P2); P2 /= P2.sum(1, keepdims=True)
r = run(y_true=y, y_pred=P2.argmax(1), y_prob=P2)
print("A3 K=10 uniform-probability (perfectly calibrated, uninformative) Brier:", round(r.results["calibration"]["brier_score"], 4),
      "ECE:", round(r.results["calibration"]["ece"], 4), "cal sub-score:", r.trust_score.sub_scores.get("calibration"))
# Same model perf but binary: perfectly calibrated binary with p=0.5
n = 5000; p = np.full(n, 0.5); yb = rng.integers(0, 2, n)
Pb = np.c_[1 - p, p]
r = run(y_true=yb, y_pred=(p > 0.5).astype(int), y_prob=Pb)
print("A4 binary p=0.5 (calibrated, uninformative) Brier:", r.results["calibration"]["brier_score"], "cal sub-score:", r.trust_score.sub_scores.get("calibration"))
# A5: strong, well calibrated multiclass K=5 vs its binary one-vs-rest brier
K = 5; n = 5000
logits = rng.normal(size=(n, K)) * 3
P = np.exp(logits); P /= P.sum(1, keepdims=True)
y = np.array([rng.choice(K, p=row) for row in P])
r = run(y_true=y, y_pred=P.argmax(1), y_prob=P)
print("A5 K=5 calibrated sharp: multiclass Brier", round(r.results["calibration"]["brier_score"], 4),
      " top-label Brier (conf vs correct)", round(float(np.mean((P.max(1) - (P.argmax(1) == y)) ** 2)), 4),
      " ECE", round(r.results["calibration"]["ece"], 4), " cal sub-score", r.trust_score.sub_scores.get("calibration"))

# --- B: label encoding ---
n = 400; p = rng.random(n); yb = (rng.random(n) < p).astype(int)
Pb = np.c_[1 - p, p]
tryrun("B1 binary labels 0/1", y_true=yb, y_pred=(p > .5).astype(int), y_prob=Pb)
tryrun("B2 binary labels {1,2} no class_labels", y_true=yb + 1, y_pred=(p > .5).astype(int) + 1, y_prob=Pb)
tryrun("B3 binary labels {1,2} with class_labels", y_true=yb + 1, y_pred=(p > .5).astype(int) + 1, y_prob=Pb, class_labels=np.array([1, 2]))
lab = np.array(["neg", "pos"])
tryrun("B4 binary string labels no class_labels", y_true=lab[yb], y_pred=lab[(p > .5).astype(int)], y_prob=Pb)
tryrun("B5 binary string labels with class_labels", y_true=lab[yb], y_pred=lab[(p > .5).astype(int)], y_prob=Pb, class_labels=lab)
# B6: bool labels
tryrun("B6 bool labels", y_true=yb.astype(bool), y_pred=(p > .5), y_prob=Pb)
# B7: multiclass labels {10,20,30} with no class_labels
K = 3; n = 300
P = rng.dirichlet(np.ones(K), n); y = np.array([rng.choice(K, p=row) for row in P])
labs = np.array([10, 20, 30])
tryrun("B7 multiclass labels {10,20,30} no class_labels", y_true=labs[y], y_pred=labs[P.argmax(1)], y_prob=P)
tryrun("B8 multiclass labels {10,20,30} with class_labels", y_true=labs[y], y_pred=labs[P.argmax(1)], y_prob=P, class_labels=labs)
# B9: multiclass y_pred NOT argmax (e.g. cost-sensitive decision) -> top-label ECE uses correct=(y==y_pred) but conf=max prob
ypred_alt = np.where(P[:, 2] > 0.2, 2, P.argmax(1))
r = tryrun("B9 multiclass with y_pred != argmax(y_prob)", y_true=y, y_pred=ypred_alt, y_prob=P)
conf = P.max(1); c_argmax = (P.argmax(1) == y).astype(float)
from trustlens.metrics.calibration import expected_calibration_error as ECE
print("   B9 ref top-label ECE (argmax-based):", round(ECE(c_argmax, conf), 4), " pipeline ECE:", round(r.results['calibration']['ece'], 4),
      " frac y_pred!=argmax:", np.mean(ypred_alt != P.argmax(1)))
# B10: binary with only one class present in y_true
tryrun("B10 binary all-positive y_true", y_true=np.ones(100, int), y_pred=np.ones(100, int), y_prob=np.c_[np.full(100, .1), np.full(100, .9)])
# B11: 3-class problem but only 2 classes present in this y_prob? (y_prob has 3 columns) fine; y_prob 2 cols but 3 labels
tryrun("B11 y_prob 2 columns but y_true has 3 classes", y_true=np.array([0, 1, 2] * 20), y_pred=np.array([0, 1, 1] * 20), y_prob=np.tile([[.6, .4], [.3, .7], [.2, .8]], (20, 1)))
# B12: unnormalized rows (each row sums to 0.5)
K = 3; n = 300
P = rng.dirichlet(np.ones(K), n); y = np.array([rng.choice(K, p=row) for row in P])
tryrun("B12 multiclass rows sum to 0.5 (unnormalized, accepted?)", y_true=y, y_pred=P.argmax(1), y_prob=P * 0.5)
tryrun("B13 multiclass rows sum to 1 (same model)", y_true=y, y_pred=P.argmax(1), y_prob=P)
# B14: multiclass with y_prob shape (n,K) where K>2 but label 'int' strings e.g. '0','1','2'
tryrun("B14 multiclass string digit labels no class_labels", y_true=y.astype(str), y_pred=P.argmax(1).astype(str), y_prob=P)
