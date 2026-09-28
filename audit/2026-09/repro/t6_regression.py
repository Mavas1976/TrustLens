"""WP1 T6: regression metrics + regression trust score vs references."""
import warnings, logging, io, contextlib
import numpy as np
from scipy import stats
warnings.simplefilter("ignore"); logging.disable(logging.CRITICAL)
from trustlens.metrics.regression import (error_distribution, prediction_interval_coverage, multilevel_interval_coverage,
    crps_from_intervals, crps_decomposition, error_variance_correlation)
from trustlens.trust_score import regression_trust_score
from trustlens import analyze
rng = np.random.default_rng(6)

# error_distribution
maxd = 0
for t in range(200):
    n = rng.integers(1, 500); y = rng.normal(size=n) * 10; yp = y + rng.standard_t(3, n)
    d = error_distribution(y, yp); e = np.abs(y - yp)
    maxd = max(maxd, abs(d["median_absolute_error"] - round(np.median(e), 4)), abs(d["rmse"] - round(np.sqrt(np.mean(e**2)), 4)),
               abs(d["p90_absolute_error"] - round(np.percentile(e, 90), 4)), abs(d["mean_absolute_error"] - round(e.mean(), 4)))
print("G1 error_distribution vs numpy (MedAE, RMSE, P90, MAE) max|diff|:", maxd)

# PICP
y = rng.normal(size=10000); lo = np.full_like(y, stats.norm.ppf(.05)); hi = -lo
r = prediction_interval_coverage(y, lo, hi, confidence_level=0.9)
print("G2 PICP Gaussian 90% interval:", r["picp"], r["calibration_error"], r["verdict"])
r = prediction_interval_coverage(np.array([1.0, np.nan, 2.0]), np.array([0, 0, 0.]), np.array([3, 3, 3.]), confidence_level=0.9)
print("G2b PICP with NaN y_true:", r)

# multi-level ICE vs manual
levels = [0.5, 0.8, 0.9, 0.95]
mu = rng.normal(size=5000); y = mu + rng.normal(size=5000)
ivs = {t: (mu - stats.norm.ppf(.5 + t/2), mu + stats.norm.ppf(.5 + t/2)) for t in levels}
r = multilevel_interval_coverage(y, ivs)
emp = [np.mean((y >= ivs[t][0]) & (y <= ivs[t][1])) for t in levels]
print("G3 ICE:", r["ice"], " manual:", round(np.mean(np.abs(np.array(emp) - levels)), 4), " sharpness_skill:", r["sharpness_skill"], r["verdict"])

# CRPS vs Gaussian closed form
def crps_gauss(mu, s, y):
    z = (y - mu) / s
    return s * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
for L in [3, 9, 19, 49]:
    taus = np.linspace(0.05, 0.95, L) if L > 3 else np.array([0.5, 0.8, 0.95])
    s = 1.0
    ivs = {float(t): (mu - s * stats.norm.ppf(.5 + t/2), mu + s * stats.norm.ppf(.5 + t/2)) for t in taus}
    c = crps_from_intervals(y, ivs)
    d = crps_decomposition(y, ivs)
    ref = crps_gauss(mu, s, y).mean()
    print(f"G4 L={L:<3} CRPS(trap)={c['mean_crps']:.4f} CRPS(recon)={d['crps']:.4f} closed-form={ref:.4f} rel.err(trap)={(c['mean_crps']-ref)/ref:+.3f} span={c['quantile_level_span']} | rel={d['reliability']} res={d['resolution']} unc={d['uncertainty']} pot={d['crps_potential']} rel+pot={d['reliability']+d['crps_potential']:.4f}")

# Decomposition sanity: climatological forecast -> resolution 0
taus = np.linspace(0.05, 0.95, 19)
qs = {float(t): (np.full_like(y, np.quantile(y, .5 - t/2)), np.full_like(y, np.quantile(y, .5 + t/2))) for t in taus}
d = crps_decomposition(y, qs)
print("G5 climatology forecast: resolution", d["resolution"], " reliability", d["reliability"])
# Miscalibrated (too narrow) forecast -> reliability grows
ivs_n = {float(t): (mu - .3 * stats.norm.ppf(.5 + t/2), mu + .3 * stats.norm.ppf(.5 + t/2)) for t in taus}
ivs_ok = {float(t): (mu - stats.norm.ppf(.5 + t/2), mu + stats.norm.ppf(.5 + t/2)) for t in taus}
print("G6 reliability too-narrow:", crps_decomposition(y, ivs_n)["reliability"], " vs calibrated:", crps_decomposition(y, ivs_ok)["reliability"])
# Outlier beyond outermost quantile ignored in recon?
yy = np.array([0.0, 100.0]); ivs2 = {float(t): (np.zeros(2) - stats.norm.ppf(.5 + t/2), np.zeros(2) + stats.norm.ppf(.5 + t/2)) for t in taus}
print("G7 obs=100 far outside forecast N(0,1): crps_from_intervals", crps_from_intervals(yy, ivs2)["mean_crps"],
      " decomposition crps", crps_decomposition(yy, ivs2)["crps"], " closed form", crps_gauss(np.zeros(2), 1, yy).mean().round(4))

# error-variance correlation vs scipy
maxd = 0
for t in range(100):
    n = rng.integers(3, 300); v = rng.random(n); v[: n // 4] = v[0]; e = rng.normal(size=n) * v
    r = error_variance_correlation(np.zeros(n), e, v)
    maxd = max(maxd, abs(r["pearson"] - round(stats.pearsonr(v, np.abs(e))[0], 4)), abs(r["spearman"] - round(stats.spearmanr(v, np.abs(e))[0], 4)))
print("G8 error_variance_correlation pearson/spearman vs scipy max|diff|:", maxd)

# ---- regression trust score: rounding / scale dependence ----
def rts(scale, seed=0):
    rg = np.random.default_rng(seed)
    y = rg.normal(size=2000) * scale
    yp = y * 0.0 + rg.normal(size=2000) * scale * 0.001 + y * 0.0  # skill ~ -? predict noise around 0 → R2 ≈ 0
    yp = rg.normal(size=2000) * scale * 1.0                         # independent predictions: R2 ≈ -1 (worse than mean)
    with contextlib.redirect_stdout(io.StringIO()):
        rep = analyze(None, None, y, y_pred=yp, task="regression", verbose=False)
    t = rep.trust_score
    return rep.results["regression"]["error_distribution"]["rmse"], 1 - np.mean((y - yp) ** 2) / np.var(y), t
for sc in [1.0, 1e-2, 1e-3, 1e-4, 1e-5]:
    rmse, true_r2, t = rts(sc)
    print(f"G9 scale={sc:<7} stored rmse={rmse} true R2={true_r2:+.3f} -> sub={t.sub_scores} score={t.score} grade={t.grade} blocked={t.is_blocked}")

# heavy-tail ratio from rounded median/p90 at small scale
y = rng.normal(size=2000) * 1e-4; yp = y + rng.normal(size=2000) * 1e-5
with contextlib.redirect_stdout(io.StringIO()):
    rep = analyze(None, None, y, y_pred=yp, task="regression", verbose=False)
print("G10 small-scale good model: err_dist", {k: rep.results['regression']['error_distribution'][k] for k in ['median_absolute_error','p90_absolute_error','rmse']},
      " true R2", round(1 - np.mean((y-yp)**2)/np.var(y), 4), " score", rep.trust_score.score, rep.trust_score.sub_scores)

# ICE blocker uses worst_calibration_error vs ICE sub-score
mu = rng.normal(size=3000); y = mu + rng.normal(size=3000)
ivs = {0.5: (mu - .6745, mu + .6745), 0.95: (mu - 1.2, mu + 1.2)}
with contextlib.redirect_stdout(io.StringIO()):
    rep = analyze(None, None, y, y_pred=mu, prediction_intervals=ivs, task="regression", verbose=False)
print("G11 multilevel (95% level under-covers):", {k: rep.results['regression']['interval_coverage'][k] for k in ['ice','worst_calibration_error','sharpness_skill','n_calibrated_levels']}, rep.trust_score.score, rep.trust_score.grade, rep.trust_score.verdict[:60])
# single-level PICP: target 0.95 vs interval designed for 0.95
with contextlib.redirect_stdout(io.StringIO()):
    rep = analyze(None, None, y, y_pred=mu, prediction_intervals=(mu - 1.96, mu + 1.96), task="regression", verbose=False)
print("G12 single-level well-calibrated 95%:", rep.trust_score.score, rep.trust_score.grade, rep.trust_score.sub_scores, rep.trust_score.weights_used)
# perfect regressor
with contextlib.redirect_stdout(io.StringIO()):
    rep = analyze(None, None, y, y_pred=y.copy(), task="regression", verbose=False)
print("G13 perfect regressor:", rep.trust_score.score, rep.trust_score.grade, rep.trust_score.sub_scores)
# constant target
yc = np.full(100, 3.3) + np.r_[np.zeros(99), 0.0]
with contextlib.redirect_stdout(io.StringIO()):
    try:
        rep = analyze(None, None, yc, y_pred=yc + 0.1, task="regression", verbose=False); print("G14 constant target, pred off by .1:", rep.trust_score.score, rep.trust_score.grade, rep.trust_score.sub_scores)
    except Exception as e:
        print("G14 raises", e)
# negative informativeness correlation
v = rng.random(3000); yp = mu; y2 = mu + rng.normal(size=3000) * (1.5 - v)
with contextlib.redirect_stdout(io.StringIO()):
    rep = analyze(None, None, y2, y_pred=yp, predicted_variance=v, task="regression", verbose=False)
print("G15 anti-informative variance:", rep.results['regression']['error_variance_correlation'].get('pearson'), rep.trust_score.sub_scores, rep.trust_score.penalties_applied, rep.trust_score.score)
