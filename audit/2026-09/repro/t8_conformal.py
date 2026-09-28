import warnings, logging, io, contextlib
import numpy as np
warnings.simplefilter("ignore"); logging.disable(logging.CRITICAL)
from trustlens.metrics.conformal import conformal_diagnostics, marginal_coverage, class_conditional_coverage, size_stratified_coverage, set_size_summary
from trustlens import analyze
rng = np.random.default_rng(8)
maxd = 0
for t in range(200):
    n = rng.integers(5, 400); K = rng.integers(2, 6)
    S = rng.random((n, K)) < 0.4; y = rng.integers(0, K, n)
    d = conformal_diagnostics(y, S.astype(int), nominal_coverage=0.9)
    cov = S[np.arange(n), y]
    sizes = S.sum(1)
    ref_cc = min(cov[y == k].mean() for k in np.unique(y))
    elig = [cov[sizes == s].mean() for s in np.unique(sizes) if (sizes == s).sum() >= 20]
    ref_ssc = max(0, 0.9 - min(elig)) if elig else None
    eff = min(1, max(0, 1 - (sizes.mean() - 1) / (K - 1)))
    diffs = [abs(d["marginal_coverage"] - cov.mean()), abs(d["worst_class_coverage"] - ref_cc), abs(d["avg_set_size"] - sizes.mean()),
             abs(d["size_efficiency"] - eff), abs(d["coverage_gap"] - (cov.mean() - 0.9))]
    if ref_ssc is None: diffs.append(0 if d["ssc_violation"] is None else 1)
    else: diffs.append(abs(d["ssc_violation"] - ref_ssc))
    maxd = max(maxd, max(diffs))
print("C1 conformal_diagnostics vs manual (marginal, class-cond, SSC, size, efficiency) max|diff|:", maxd)
# pipeline: labels {1,2,3}, class_labels given, sets as matrix over columns (0..2)
K = 3; n = 300
P = rng.dirichlet(np.ones(K), n); yi = np.array([rng.choice(K, p=p) for p in P]); labs = np.array([1, 2, 3])
S = (P >= 0.2).astype(int)
ref = S[np.arange(n), yi].mean()
with contextlib.redirect_stdout(io.StringIO()):
    rep = analyze(None, None, labs[yi], y_pred=labs[P.argmax(1)], y_prob=P, class_labels=labs, y_pred_sets=S, confidence_level=0.9, verbose=False)
c = rep.results["calibration"]["conformal"]
print("C2 labels {1,2,3} + class_labels, matrix sets: pipeline marginal", c.get("marginal_coverage"), " true (column-index) coverage", round(ref, 4), " n_classes", c.get("n_classes"), c.get("status"), c.get("details"))
with contextlib.redirect_stdout(io.StringIO()):
    rep = analyze(None, None, np.array(["a","b","c"])[yi], y_pred=np.array(["a","b","c"])[P.argmax(1)], y_prob=P, class_labels=np.array(["a","b","c"]), y_pred_sets=S, confidence_level=0.9, verbose=False)
print("C3 string labels + class_labels:", rep.results["calibration"]["conformal"])
