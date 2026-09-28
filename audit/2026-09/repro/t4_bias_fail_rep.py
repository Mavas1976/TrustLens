"""WP1 T4: bias, failure, representation metrics vs references."""
import warnings
import numpy as np
from sklearn.metrics import silhouette_score, accuracy_score, confusion_matrix
from trustlens.metrics.bias import class_imbalance_report, subgroup_performance, equalized_odds
from trustlens.metrics.failure import confidence_gap, misclassification_summary
from trustlens.metrics.representation import embedding_separability, centered_kernel_alignment
from trustlens.trust_score import compute_trust_score
warnings.simplefilter("ignore")
rng = np.random.default_rng(3)

# ---- equalized odds vs manual Hardt definition ----
maxd = 0
for t in range(300):
    n = rng.integers(10, 200); y = rng.integers(0, 2, n); yp = rng.integers(0, 2, n); g = rng.integers(0, 3, n)
    r = equalized_odds(y, yp, {"g": g})
    tprs, fprs = [], []
    for k in np.unique(g):
        m = g == k
        pos = (y[m] == 1); neg = (y[m] == 0)
        tpr = (yp[m][pos] == 1).mean() if pos.any() else 0.0
        fpr = (yp[m][neg] == 1).mean() if neg.any() else 0.0
        maxd = max(maxd, abs(r["g"][str(k)]["tpr"] - tpr), abs(r["g"][str(k)]["fpr"] - fpr))
        tprs.append(tpr); fprs.append(fpr)
    maxd = max(maxd, abs(r["g"]["__summary__"]["tpr_gap"] - round(max(tprs) - min(tprs), 4)))
print("E1 equalized_odds per-group TPR/FPR & gaps vs manual max|diff|:", maxd)

# group with no positives -> TPR := 0 -> spurious severe gap
y = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]); yp = y.copy()  # PERFECT classifier
g = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])   # group 1 has only negatives
r = equalized_odds(y, yp, {"g": g})
print("E2 perfect classifier, group1 has no positives:", r["g"])
res = {"calibration": {"brier_score": 0.0, "ece": 0.0},
       "failure": {"confidence_gap": {"gap": 0.5}, "misclassification_summary": {"__overall__": {"overall_error_rate": 0.0}}},
       "bias": {"class_imbalance": class_imbalance_report(y), "subgroup_performance": subgroup_performance(y, yp, {"g": g}), "equalized_odds": r}}
ts = compute_trust_score(res)
print("E2 -> trust score:", ts.score, ts.grade, ts.is_blocked, ts.verdict)

# tiny subgroup (n=1) drives subgroup gap -> severe
n = 1000; y = rng.integers(0, 2, n); yp = y.copy(); yp[:50] = 1 - yp[:50]  # 95% acc
g = np.zeros(n, int); g[0] = 1  # single-sample group, misclassified
sp = subgroup_performance(y, yp, {"g": g})
print("E3 subgroup with n=1 (wrong): gap", sp["g"]["__summary__"], sp["g"]["1"])
eo = equalized_odds(y, yp, {"g": g})
print("E3 equalized_odds with n=1 group summary:", eo["g"]["__summary__"])

# threshold boundary semantics: doc says 'moderate' between, 'acceptable' < moderate; gap==0.15 exactly
from trustlens.metrics.bias import _violation_level
print("E4 violation at 0.15:", _violation_level(0.15), " at 0.05:", _violation_level(0.05), " at 0.1500001:", _violation_level(0.1500001))
# float representation: tpr 0.65 - 0.5
print("E4b round(0.65-0.5,4)", round(0.65 - 0.5, 4), _violation_level(round(0.65 - 0.5, 4)))

# equalized odds on string / non-{0,1} binary labels -> recall pos_label
try:
    r = equalized_odds(np.array([2, 2, 1, 1]), np.array([2, 1, 1, 1]), {"g": np.array([0, 0, 1, 1])})
    print("E5 labels {1,2}:", r["g"])
except Exception as e:
    print("E5 labels {1,2}: RAISES", type(e).__name__, str(e)[:120])

# class imbalance
print("E6 imbalance ratio [0,0,0,1]:", class_imbalance_report(np.array([0, 0, 0, 1]))["imbalance_ratio"])
try:
    print("E6b empty:", class_imbalance_report(np.array([])))
except Exception as e:
    print("E6b empty: RAISES", type(e).__name__, str(e)[:80])

# subgroup performance NaN group labels
g = np.array([0.0, np.nan, np.nan, 1.0]); y = np.array([1, 0, 1, 0]); yp = np.array([1, 1, 1, 0])
try:
    sp = subgroup_performance(y, yp, {"g": g})
    print("E7 NaN group labels -> groups:", {k: v for k, v in sp["g"].items()})
except Exception as e:
    print("E7 NaN group labels: RAISES", type(e).__name__, str(e)[:90])

# ---- failure ----
maxd = 0
for t in range(200):
    n = rng.integers(5, 300); K = rng.integers(2, 5)
    P = rng.dirichlet(np.ones(K), n); y = rng.integers(0, K, n); yp = P.argmax(1)
    cg = confidence_gap(y, yp, P)
    c = P.max(1); m = y == yp
    ref = (c[m].mean() - c[~m].mean()) if (m.any() and (~m).any()) else 0.0
    maxd = max(maxd, abs(cg["gap"] - round(ref, 4)))
    ms = misclassification_summary(y, yp, P)
    for k in np.unique(y):
        er = np.mean(yp[y == k] != k)
        maxd = max(maxd, abs(ms[int(k)]["error_rate"] - round(er, 4)))
print("F1 confidence_gap & per-class error_rate vs ref max|diff|:", maxd)
# 1-D binary probs -> confidence of positive class not predicted class
p = np.array([0.1, 0.2, 0.9, 0.8]); y = np.array([0, 0, 1, 0]); yp = (p > .5).astype(int)
print("F2 1-D y_prob gap:", confidence_gap(y, yp, p)["gap"], " 2-col gap:", confidence_gap(y, yp, np.c_[1 - p, p])["gap"])
# upper bound of gap for binary 2-col: conf in [.5,1] -> gap <= .5
print("F3 binary max possible gap (correct conf 1.0, wrong conf .5):", confidence_gap(np.array([1, 0]), np.array([1, 1]), np.array([[0, 1.], [.5, .5]]))["gap"])
# histogram: conf exactly 1.0 included in last bin?
cg = confidence_gap(np.array([1, 1]), np.array([1, 1]), np.array([[0, 1.], [0, 1.]]))
print("F4 conf=1.0 hist last bin:", cg["correct_hist"][-1])

# ---- representation ----
maxd = 0
for t in range(30):
    n = rng.integers(20, 400); E = rng.normal(size=(n, 5)); y = rng.integers(0, 3, n); E[:, 0] += y * 2
    s = embedding_separability(E, y)
    maxd = max(maxd, abs(s["silhouette_score"] - round(silhouette_score(E, y), 4)))
print("R1 silhouette vs sklearn max|diff|:", maxd)
# within-class distance includes self-pairs (distance 0)
n = 30; E = rng.normal(size=(n, 5)); y = np.repeat([0, 1], 15)
s = embedding_separability(E, y)
from scipy.spatial.distance import pdist
ref_within = np.mean(np.r_[pdist(E[y == 0]), pdist(E[y == 1])])
print("R2 within_class_distance:", s["within_class_distance"], " exact mean pairwise (i<j):", round(ref_within, 4))
# with few samples per class
E = np.array([[0, 0], [1, 0], [10, 0], [11, 0.]]); y = np.array([0, 0, 1, 1])
s = embedding_separability(E, y)
print("R3 2 pts per class (true within=1.0, between mean=10):", s["within_class_distance"], s["between_class_distance"])
# single class
try:
    s = embedding_separability(rng.normal(size=(10, 3)), np.zeros(10))
    print("R4 single class:", s)
except Exception as e:
    print("R4 single class RAISES", type(e).__name__, e)
# silhouette with a singleton class -> sklearn raises? n_labels == n_samples
try:
    s = embedding_separability(rng.normal(size=(3, 2)), np.array([0, 1, 2]))
    print("R5 n_labels==n_samples:", s)
except Exception as e:
    print("R5 n_labels==n_samples RAISES", type(e).__name__, str(e)[:100])

# CKA vs reference linear CKA (Kornblith: ||Y^T X||_F^2 / (||X^T X|| ||Y^T Y||) on centered features)
def ref_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return np.linalg.norm(Y.T @ X) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y))
maxd = 0
for t in range(50):
    n = rng.integers(5, 100); X = rng.normal(size=(n, 4)) + 3; Y = X @ rng.normal(size=(4, 6)) + rng.normal(size=(n, 6)) * rng.random()
    maxd = max(maxd, abs(centered_kernel_alignment(X, Y) - ref_cka(X, Y)))
print("R6 CKA vs Kornblith linear CKA max|diff|:", maxd)
X = rng.normal(size=(50, 4))
Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
print("R7 CKA invariances: orthogonal", centered_kernel_alignment(X, X @ Q), " scale 1e-4:", centered_kernel_alignment(X * 1e-4, X * 1e-4), " scale 1e-2:", centered_kernel_alignment(X * 1e-2, X * 1e-2))
print("R7b ref at 1e-4:", ref_cka(X * 1e-4, X * 1e-4))
