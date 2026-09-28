"""WP1 T1: calibration metrics vs independent references."""
import warnings
import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from trustlens.metrics.calibration import (brier_score, expected_calibration_error,
    maximum_calibration_error, reliability_curve)

rng = np.random.default_rng(0)

def ref_bins_uniform(y, p, B):
    # Guo 2017 style: bins (lo, hi], but we emulate [lo,hi) w/ last closed, via digitize
    edges = np.linspace(0, 1, B + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, B - 1)
    w, g = [], []
    for b in range(B):
        m = idx == b
        if m.sum() == 0: continue
        w.append(m.mean()); g.append(abs(y[m].mean() - p[m].mean()))
    return np.array(w), np.array(g)

maxd_b = maxd_e = maxd_m = 0
viol_mce = 0
for t in range(2000):
    n = rng.integers(1, 400)
    p = rng.random(n)
    if t % 5 == 0: p = np.round(p, 1)  # exact bin edges
    if t % 7 == 0: p[: n // 3] = 1.0
    if t % 11 == 0: p[: n // 3] = 0.0
    y = (rng.random(n) < p ** 1.5).astype(float)
    maxd_b = max(maxd_b, abs(brier_score(y, p) - brier_score_loss(y, p)) if len(np.unique(y)) > 0 else 0)
    w, g = ref_bins_uniform(y, p, 10)
    maxd_e = max(maxd_e, abs(expected_calibration_error(y, p) - (w * g).sum()))
    maxd_m = max(maxd_m, abs(maximum_calibration_error(y, p) - g.max()))
    if maximum_calibration_error(y, p) + 1e-12 < expected_calibration_error(y, p): viol_mce += 1
print("T1a brier vs sklearn max|diff|:", maxd_b)
print("T1a ECE(uniform) vs ref max|diff| (incl. edge ties):", maxd_e)
print("T1a MCE(uniform) vs ref max|diff|:", maxd_m, " MCE<ECE violations:", viol_mce)

# Reliability curve vs sklearn calibration_curve
maxd = 0
for t in range(300):
    n = rng.integers(20, 300)
    p = rng.random(n); y = (rng.random(n) < p).astype(int)
    fp, mp, c = reliability_curve(y, p, n_bins=10)
    sfp, smp = calibration_curve(y, p, n_bins=10, strategy="uniform")
    if len(fp) == len(sfp):
        maxd = max(maxd, np.abs(fp - sfp).max(), np.abs(mp - smp).max())
    else:
        maxd = np.inf
print("T1b reliability_curve vs sklearn calibration_curve max|diff|:", maxd)

# Edge-tie consistency: reliability_curve vs ECE binning at exact edges
p = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 0.0]); y = np.array([0, 0, 1, 1, 1, 1, 1, 0.])
fp, mp, c = reliability_curve(y, p)
ece_from_curve = float(np.sum(c / c.sum() * np.abs(fp - mp)))
print("T1c edge ties: ECE", expected_calibration_error(y, p), " ECE rebuilt from reliability_curve", ece_from_curve, "counts", c)
# floating edge: 0.3 vs linspace edge 0.30000000000000004
edges = np.linspace(0, 1, 11)
print("T1c linspace edges repr:", [repr(e) for e in edges[2:5]], " 0.3>=edge3?", 0.3 >= edges[3], " 0.7>=edges[7]?", 0.7 >= edges[7])
pp = np.array([0.3, 0.7, 0.6]); yy = np.array([1., 0., 1.])
_, mp2, c2 = reliability_curve(yy, pp)
print("T1c reliability bins for 0.3/0.7/0.6 -> mean_pred", mp2, "counts", c2)

# quantile strategy
maxd = 0
for t in range(300):
    n = rng.integers(20, 300)
    p = rng.random(n); y = (rng.random(n) < p).astype(int)
    e1 = expected_calibration_error(y, p, strategy="quantile")
    fp, mp, c = reliability_curve(y, p, strategy="quantile")
    e2 = float(np.sum(c / c.sum() * np.abs(fp - mp)))
    maxd = max(maxd, abs(e1 - e2))
print("T1d quantile ECE vs ECE rebuilt from quantile reliability_curve max|diff|:", maxd)

# perfectly calibrated large-n
n = 200000; p = rng.random(n); y = (rng.random(n) < p).astype(float)
print("T1e perfectly calibrated n=2e5: ECE", round(expected_calibration_error(y, p), 4), "MCE", round(maximum_calibration_error(y, p), 4))

# Edge cases
def tryit(name, f):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = f()
        print(f"T1f {name}: {r!r}", ("warnings:" + str([str(x.message)[:60] for x in w])) if w else "")
    except Exception as e:
        print(f"T1f {name}: RAISES {type(e).__name__}: {str(e)[:100]}")

tryit("ECE empty", lambda: expected_calibration_error(np.array([]), np.array([])))
tryit("ECE shape mismatch", lambda: expected_calibration_error(np.array([1, 0, 1]), np.array([0.2, 0.3])))
tryit("ECE n_bins=0", lambda: expected_calibration_error(np.array([1, 0]), np.array([0.2, 0.3]), n_bins=0))
tryit("ECE NaN prob", lambda: expected_calibration_error(np.array([1, 0, 1]), np.array([0.9, np.nan, 0.8])))
tryit("MCE NaN prob", lambda: maximum_calibration_error(np.array([1, 0, 1]), np.array([0.9, np.nan, 0.8])))
tryit("Brier NaN prob", lambda: brier_score(np.array([1, 0, 1]), np.array([0.9, np.nan, 0.8])))
tryit("ECE prob>1 (1.5)", lambda: expected_calibration_error(np.array([1, 1, 0]), np.array([1.5, 1.5, 0.2])))
tryit("ECE prob<0", lambda: expected_calibration_error(np.array([0, 0, 1]), np.array([-0.5, 0.1, 0.9])))
tryit("Brier prob>1", lambda: brier_score(np.array([0, 1]), np.array([2.0, 0.5])))
tryit("Brier labels {1,2}", lambda: brier_score(np.array([1, 2]), np.array([0.5, 0.5])))
tryit("ECE labels {1,2} (no validation)", lambda: expected_calibration_error(np.array([2, 2, 1]), np.array([0.9, 0.8, 0.1])))
tryit("ECE constant pred quantile", lambda: expected_calibration_error(np.array([1, 1, 0, 0]), np.array([0.9] * 4), strategy="quantile"))
tryit("ECE constant pred uniform", lambda: expected_calibration_error(np.array([1, 1, 0, 0]), np.array([0.9] * 4)))
tryit("reliability_curve empty", lambda: reliability_curve(np.array([]), np.array([])))
tryit("reliability_curve NaN", lambda: reliability_curve(np.array([1, 0, 1]), np.array([0.9, np.nan, 0.8])))
tryit("reliability_curve p=1.5", lambda: reliability_curve(np.array([1, 1, 0]), np.array([1.5, 1.5, 0.2])))
tryit("reliability_curve bad strategy", lambda: reliability_curve(np.array([1]), np.array([0.5]), strategy="x"))
tryit("reliability_curve n_bins=0", lambda: reliability_curve(np.array([1, 0]), np.array([0.5, 0.2]), n_bins=0))
tryit("Brier docstring example", lambda: brier_score(np.array([1, 0, 1, 1, 0]), np.array([0.9, 0.1, 0.8, 0.7, 0.3])))
