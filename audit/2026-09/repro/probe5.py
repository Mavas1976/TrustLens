import io, contextlib, logging
logging.basicConfig(level=logging.ERROR)
import numpy as np, matplotlib; matplotlib.use("Agg")
from trustlens import analyze
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import xgboost as xgb, lightgbm as lgb, catboost as cb
X3, y3 = make_classification(n_samples=800, n_features=6, n_informative=4, n_classes=3, random_state=1)
Xb, yb = make_classification(n_samples=800, n_features=6, random_state=2)
def run(name, fn):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = fn()
        c = r.results["calibration"]; e = r.results["failure"]["misclassification_summary"]["__overall__"]
        print(f"[{name}] OK fw={r.framework} score={r.trust_score.score} brier={c['brier_score']:.4f} ece={c['ece']:.4f} err={e['overall_error_rate']}")
    except Exception as ex:
        print(f"[{name}] EXC {type(ex).__name__}: {str(ex)[:180]}")
names = np.array(["cat","dog","eel"])
ys3 = names[y3]; y13 = y3 + 1
ref = LogisticRegression(max_iter=500)
# xgboost
for lab, yy in (("0..2", y3), ("1..3", y13)):
    try:
        mdl = xgb.XGBClassifier(n_estimators=20).fit(X3[:400], yy[:400]); run(f"xgb clf labels {lab}", lambda: analyze(mdl, X3[400:], yy[400:], verbose=False))
    except Exception as ex: print(f"[xgb fit {lab}] EXC {type(ex).__name__}: {str(ex)[:100]}")
bst = xgb.train({"objective":"multi:softprob","num_class":3}, xgb.DMatrix(X3[:400], label=y3[:400]), 20)
run("xgb Booster 0..2", lambda: analyze(bst, X3[400:], y3[400:], verbose=False))
run("xgb Booster string y + class_labels", lambda: analyze(bst, X3[400:], ys3[400:], class_labels=names, verbose=False))
run("xgb Booster string y no class_labels", lambda: analyze(bst, X3[400:], ys3[400:], verbose=False))
run("xgb Booster 1..3 no class_labels", lambda: analyze(bst, X3[400:], y13[400:], verbose=False))
bstb = xgb.train({"objective":"binary:logistic"}, xgb.DMatrix(Xb[:400], label=yb[:400]), 20)
run("xgb Booster binary", lambda: analyze(bstb, Xb[400:], yb[400:], verbose=False))
bstr = xgb.train({"objective":"reg:squarederror"}, xgb.DMatrix(Xb[:400], label=yb[:400]), 5)
run("xgb Booster reg objective, binary y", lambda: analyze(bstr, Xb[400:], yb[400:], verbose=False))
# lightgbm
for lab, yy in (("string", ys3), ("1..3", y13)):
    mdl = lgb.LGBMClassifier(n_estimators=20, verbose=-1).fit(X3[:400], yy[:400]); run(f"lgbm clf labels {lab}", lambda: analyze(mdl, X3[400:], yy[400:], verbose=False))
lb = lgb.train({"objective":"multiclass","num_class":3,"verbose":-1}, lgb.Dataset(X3[:400], label=y3[:400]), 20)
run("lgb Booster 0..2", lambda: analyze(lb, X3[400:], y3[400:], verbose=False))
run("lgb Booster 1..3 no class_labels", lambda: analyze(lb, X3[400:], y13[400:], verbose=False))
lbr = lgb.train({"objective":"regression","verbose":-1}, lgb.Dataset(Xb[:400], label=yb[:400]), 5)
run("lgb Booster regression objective", lambda: analyze(lbr, Xb[400:], yb[400:], verbose=False))
# catboost
for lab, yy in (("string", ys3), ("1..3", y13), ("0..2", y3)):
    mdl = cb.CatBoostClassifier(iterations=20, verbose=0).fit(X3[:400], yy[:400]); run(f"catboost labels {lab}", lambda: analyze(mdl, X3[400:], yy[400:], verbose=False))
mdlb = cb.CatBoostClassifier(iterations=20, verbose=0).fit(Xb[:400], np.where(yb[:400]==1,"pos","neg"))
run("catboost binary string", lambda: analyze(mdlb, Xb[400:], np.where(yb[400:]==1,"pos","neg"), verbose=False))
print("catboost classes_ dtype", mdlb.classes_, type(mdlb.classes_[0]))
