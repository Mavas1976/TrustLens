"""WP1 T3: Trust Score formula, penalties, redistribution, grades."""
import logging, io, contextlib
import numpy as np
from trustlens import analyze
from trustlens.trust_score import compute_trust_score, _calibration_score, _failure_score, _bias_score, _representation_score

logging.disable(logging.CRITICAL)
rng = np.random.default_rng(2)

def run(**kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return analyze(None, None, verbose=False, **kw)

def show(tag, ts):
    print(f"{tag}: score={ts.score} base={ts.base_score} grade={ts.grade} blocked={ts.is_blocked} sub={ts.sub_scores} pen={ts.penalties_applied} verdict={ts.verdict[:70]!r}")

# --- 1. docstring formula vs implementation ---
cal = {"brier_score": 0.10, "ece": 0.08}
doc = 100 * (1 - np.clip(0.5 * 0.10 + 0.5 * 0.08, 0, 1))
print("1a CalibrationScore docstring(0.5BS+0.5ECE) =", doc, " impl =", _calibration_score(cal))
fail = {"confidence_gap": {"gap": 0.30}, "misclassification_summary": {"__overall__": {"overall_error_rate": 0.10}}}
print("1b FailureScore docstring(100*clip(gap)) =", 100 * 0.30, " impl =", _failure_score(fail))
bias = {"class_imbalance": {"imbalance_ratio": 5.0}, "subgroup_performance": {"g": {"__summary__": {"performance_gap": 0.10}}}}
doc = 100 * (1 - (0.5 * np.clip(5 / 20, 0, 1) + 0.5 * 0.10))
print("1c BiasScore docstring(ratio/20) =", doc, " impl ((ratio-1)/19) =", _bias_score(bias))
print("1d Bias balanced ratio=1: docstring =", 100 * (1 - 0.5 * (1 / 20)), " impl =", _bias_score({"class_imbalance": {"imbalance_ratio": 1.0}}))

# --- 2. well-calibrated realistic binary model: why grade D? ---
n = 2000; p = rng.random(n); y = (rng.random(n) < p).astype(int)
r = run(y_true=y, y_pred=(p > .5).astype(int), y_prob=np.c_[1 - p, p])
show("2a perfectly calibrated binary, p~U(0,1)", r.trust_score)
print("    gap:", r.results["failure"]["confidence_gap"]["gap"], " err:", r.results["failure"]["misclassification_summary"]["__overall__"])
# a strong, well-calibrated model
n = 4000; logit = rng.normal(0, 4, n); p = 1 / (1 + np.exp(-logit)); y = (rng.random(n) < p).astype(int)
r = run(y_true=y, y_pred=(p > .5).astype(int), y_prob=np.c_[1 - p, p])
show("2b strong calibrated binary (logit sd 4)", r.trust_score)
print("    acc:", np.mean((p > .5) == y), " gap:", r.results["failure"]["confidence_gap"]["gap"])
# very strong model: acc ~ 99%, calibrated
n = 4000; logit = rng.normal(0, 12, n); p = 1 / (1 + np.exp(-logit)); y = (rng.random(n) < p).astype(int)
r = run(y_true=y, y_pred=(p > .5).astype(int), y_prob=np.c_[1 - p, p])
show("2c very strong calibrated binary (logit sd 12)", r.trust_score)
print("    acc:", np.mean((p > .5) == y), " gap:", r.results["failure"]["confidence_gap"]["gap"], " n_incorrect:", r.results["failure"]["confidence_gap"]["n_incorrect"])
# perfect model: all correct -> gap defined 0.0
p = np.r_[np.full(500, 0.99), np.full(500, 0.01)]; y = np.r_[np.ones(500, int), np.zeros(500, int)]
r = run(y_true=y, y_pred=(p > .5).astype(int), y_prob=np.c_[1 - p, p])
show("2d all-correct, conf 0.99", r.trust_score)
# ALL WRONG model
r = run(y_true=1 - y, y_pred=(p > .5).astype(int), y_prob=np.c_[1 - p, p])
show("2e all-WRONG, conf 0.99", r.trust_score)

# --- 3. double counting ECE: sub-score + penalty + blocker ---
base = {"failure": fail}
for e in [0.0, 0.05, 0.08, 0.10, 0.1001, 0.15, 0.30]:
    res = {"calibration": {"brier_score": 0.05, "ece": e}, "failure": fail,
           "bias": {"class_imbalance": {"imbalance_ratio": 1.0}}}
    ts = compute_trust_score(res)
    print(f"3 ECE={e:<6}: cal_sub={ts.sub_scores['calibration']:5.1f} pen={ts.penalties_applied} base={ts.base_score} score={ts.score} grade={ts.grade} blocked={ts.is_blocked}")

# --- 4. subgroup gap double count ---
for g in [0.0, 0.05, 0.10, 0.15, 0.1501, 0.30]:
    res = {"calibration": {"brier_score": 0.05, "ece": 0.02}, "failure": fail,
           "bias": {"class_imbalance": {"imbalance_ratio": 1.0}, "subgroup_performance": {"f": {"__summary__": {"performance_gap": g}}}}}
    ts = compute_trust_score(res)
    print(f"4 gap={g:<6}: bias_sub={ts.sub_scores['bias']:5.1f} pen={ts.penalties_applied} score={ts.score} grade={ts.grade} blocked={ts.is_blocked}")

# --- 5. grade boundaries ---
def fake_score(target):
    # single dimension representation: RepScore = 100*clip(.5+.5 sil)
    sil = target / 50 - 1
    return compute_trust_score({"representation": {"separability": {"silhouette_score": sil}}})
for t in [39.4, 39.5, 39.6, 40, 59.5, 60, 79.49, 79.5, 80, 100]:
    ts = fake_score(t)
    print(f"5 raw={t}: score={ts.score} grade={ts.grade}")

# --- 6. redistribution & custom weights ---
res = {"calibration": {"brier_score": 0.05, "ece": 0.02}, "failure": fail}
print("6a two dims:", compute_trust_score(res).weights_used)
print("6b custom weights sum 2:", compute_trust_score(res, weights={"calibration": 1.0, "failure": 1.0}).weights_used)
print("6c negative weight:", compute_trust_score(res, weights={"calibration": -0.5}).weights_used, compute_trust_score(res, weights={"calibration": -0.5}).score)
try:
    ts = compute_trust_score(res, weights={"calibration": 0.0, "failure": 0.0})
    print("6d all-zero weights -> equal fallback:", ts.weights_used, ts.score)
except Exception as e:
    print("6d raises", e)
print("6e unknown weight key 'fairness':", compute_trust_score(res, weights={"fairness": 5.0}).weights_used)
print("6f empty results:", compute_trust_score({}).score, compute_trust_score({}).grade, compute_trust_score({}).verdict)

# --- 7. skipped / degraded modules still scored ---
skipped = {"calibration": {"status": "skipped", "reason": "missing_probabilities"},
           "failure": {"status": "degraded", "misclassification_summary": {"__overall__": {"overall_error_rate": 0.02}}, "confidence_gap": {"gap": 0.0, "status": "skipped"}},
           "bias": {"class_imbalance": {"imbalance_ratio": 1.0}}}
show("7 no y_prob (calibration skipped, failure degraded), 98% acc", compute_trust_score(skipped))
# real pipeline with no y_prob
n = 1000; y = rng.integers(0, 2, n); yp = y.copy(); yp[:20] = 1 - yp[:20]
r = run(y_true=y, y_pred=yp)
show("7b analyze(y_pred only, 98% acc)", r.trust_score)

# --- 8. multiclass Brier > 1 clipped ---
print("8 multiclass Brier 1.2 + ECE 0:", _calibration_score({"brier_score": 1.2, "ece": 0.0}))

# --- 9. imbalance normalisation ---
for ratio in [1, 2, 5, 10, 20, 50, float("inf")]:
    print(f"9 imbalance_ratio={ratio}: bias_sub={_bias_score({'class_imbalance': {'imbalance_ratio': ratio}})}")

# --- 10. missing keys defaults ---
print("10a calibration dict without keys:", _calibration_score({}))
print("10b failure dict without misclassification_summary, gap .5:", _failure_score({"confidence_gap": {"gap": .5}}))
print("10c representation NaN silhouette:", _representation_score({"separability": {"silhouette_score": float('nan')}}))

# --- 11. penalty cap scaling rounding: sum(penalties_applied) vs base-score ---
res = {"calibration": {"brier_score": 0.3, "ece": 0.5}, "failure": {"confidence_gap": {"gap": 0.0}, "misclassification_summary": {"__overall__": {"overall_error_rate": 0.5}}},
       "bias": {"class_imbalance": {"imbalance_ratio": 1.0}, "subgroup_performance": {"f": {"__summary__": {"performance_gap": 0.5}}}}}
ts = compute_trust_score(res); show("11 all penalties max", ts); print("   sum(pen)=", sum(ts.penalties_applied.values()))

# --- 12. equalized-odds severe but NOT subgroup gap: severe fairness flag ---
res = {"calibration": {"brier_score": 0.05, "ece": 0.02}, "failure": fail,
       "bias": {"class_imbalance": {"imbalance_ratio": 1.0}, "equalized_odds": {"f": {"__summary__": {"tpr_violation": "severe"}}}}}
show("12 eq-odds severe only", compute_trust_score(res))
# equalized_odds as a skipped dict (status key string) must not crash
res["bias"]["equalized_odds"] = {"status": "skipped", "reason": "x", "details": "y"}
show("12b eq-odds skipped dict", compute_trust_score(res))
