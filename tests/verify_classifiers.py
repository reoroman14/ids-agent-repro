"""Verify registry.py, metrics.py and train_eval.py."""
import copy, json, os, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.data import loader as L
from src.data import preprocessing as P
from src.classifiers import registry as R
from src.classifiers import train_eval as TE
from src.evaluation import metrics as M

ok = True
def check(name, cond, detail=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" â€” {detail}" if detail else ""))
    if not cond: ok = False


# =========================== registry ====================================
print("=== registry ===")
check("six classifiers registered", len(R.available_classifiers()) == 6,
      str(R.available_classifiers()))
check("stub short-name registry present",
      set(R.CLASSIFIER_REGISTRY) == {"rf", "knn", "lr", "dt", "mlp", "svm"})
check("aliases resolve", (R.resolve_name("rf") == "random_forest"
                          and R.resolve_name("LR") == "logistic_regression"
                          and R.resolve_name("dt") == "decision_tree"
                          and R.resolve_name("random_forest") == "random_forest"))
try:
    R.resolve_name("adaboost"); check("unknown name raises", False)
except KeyError as e:
    check("unknown name raises", "adaboost" in str(e))

m = R.build_classifier("rf", {"n_estimators": 7}, seed=42, n_jobs=-1)
check("builds with params", isinstance(m, RandomForestClassifier) and m.n_estimators == 7)
check("seed injected where supported", m.random_state == 42)
check("n_jobs injected where supported", m.n_jobs == -1)
k = R.build_classifier("knn", {"n_neighbors": 3}, seed=42, n_jobs=-1)
check("no random_state forced on knn", not hasattr(k, "random_state"))
check("n_jobs still injected on knn", k.n_jobs == -1)
mlp = R.build_classifier("mlp", {}, seed=42, n_jobs=-1)
check("n_jobs not forced on mlp", not hasattr(mlp, "n_jobs") and mlp.random_state == 42)
explicit = R.build_classifier("rf", {"random_state": 7}, seed=42)
check("explicit params win over injection", explicit.random_state == 7)

try:
    R.build_classifier("rf", {"n_estimator": 200})
    check("typo'd hyperparameter raises", False)
except ValueError as e:
    check("typo'd hyperparameter raises", "n_estimator" in str(e) and "models.yaml" in str(e))

mcfg = TE.load_models_config()
check("training block present in models.yaml",
      mcfg["training"]["max_train_rows"]["svm"] == 50000
      and len(mcfg["training"]["classifiers"]) == 6)
for nm in R.available_classifiers():
    try:
        R.build_from_config(nm, mcfg, seed=42, n_jobs=-1)
    except Exception as e:
        check(f"build_from_config({nm})", False, str(e))
check("all six build from models.yaml as written", True)

# score interface
Xs = np.random.RandomState(0).rand(200, 4)
ys = np.random.RandomState(0).randint(0, 3, 200)
svc_plain = SVC(kernel="rbf").fit(Xs, ys)
svc_proba = SVC(kernel="rbf", probability=True, random_state=0).fit(Xs, ys)
rf = RandomForestClassifier(n_estimators=5, random_state=0).fit(Xs, ys)
check("score_kind: plain SVC falls back", R.score_kind(svc_plain) == "decision_function")
check("score_kind: SVC(probability=True) is proba", R.score_kind(svc_proba) == "proba")
check("score_kind: RF is proba", R.score_kind(rf) == "proba")
check("supports_proba false for plain SVC", not R.supports_proba(svc_plain))
for name, mdl in (("plain SVC", svc_plain), ("SVC proba", svc_proba), ("RF", rf)):
    s = R.predict_scores(mdl, Xs)
    check(f"predict_scores shape/normalized ({name})",
          s.shape == (200, 3) and np.allclose(s.sum(axis=1), 1.0))

# =========================== metrics =====================================
print("\n=== metrics ===")
names = ["Benign", "A", "B"]
y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])
r = M.compute_metrics(y_true, y_pred, class_names=names)
check("per-class covers unobserved class B", set(r["per_class"]) == {"Benign", "A", "B"})
check("B has zero support", r["per_class"]["B"]["support"] == 0)
check("macro over all 3 classes", abs(r["macro_f1"] - np.mean(
      [r["per_class"][c]["f1"] for c in names])) < 1e-12)
check("macro_f1_present_only excludes B",
      abs(r["macro_f1_present_only"] - np.mean(
          [r["per_class"]["Benign"]["f1"], r["per_class"]["A"]["f1"]])) < 1e-12)
check("present/total classes reported",
      r["n_classes_present"] == 2 and r["n_classes_total"] == 3)
check("benign FPR = 1/4", abs(r["benign_false_positive_rate"] - 0.25) < 1e-12)
check("attack detection = 3/4", abs(r["attack_detection_rate"] - 0.75) < 1e-12)
check("confusion matrix is 3x3", np.array(r["confusion_matrix"]).shape == (3, 3))

# zero-day: held-out class B, flagged as attack A -> recall 1.0
zt = np.array([0, 0, 2, 2, 2, 2])
zp = np.array([0, 0, 1, 1, 1, 0])
z = M.compute_metrics(zt, zp, class_names=names, holdout_classes=["B"])
check("zero-day recall counts any attack prediction",
      abs(z["zero_day_recall"] - 0.75) < 1e-12, str(z["zero_day_recall"]))
check("per-class recall for held-out B is 0 (cannot be named)",
      z["per_class"]["B"]["recall"] == 0.0)
check("zero-day per-class reported", z["zero_day"]["per_class"]["B"]["n_samples"] == 4)
try:
    M.compute_metrics(zt, zp, class_names=names, holdout_classes=["Nope"])
    check("unknown holdout class raises", False)
except ValueError as e:
    check("unknown holdout class raises", "Nope" in str(e))
try:
    M.compute_metrics(np.array([0, 1]), np.array([0]), class_names=names)
    check("shape mismatch raises", False)
except ValueError:
    check("shape mismatch raises", True)
check("summarize renders", "macro" in M.summarize(r, title="t"))
check("worst_classes skips zero-support",
      all(c != "B" for c, _, _ in M.worst_classes(r)))

# =========================== subsampling =================================
print("\n=== subsampling ===")
Xb = pd.DataFrame(np.arange(2000).reshape(1000, 2), columns=["a", "b"])
yb = np.array([0] * 900 + [1] * 95 + [2] * 5)
Xs2, ys2 = TE._stratified_subsample(Xb, yb, 100, 42)
check("subsample hits the cap", len(ys2) == 100, str(len(ys2)))
check("subsample keeps every class, incl. the 5-row one",
      set(np.unique(ys2)) == {0, 1, 2}, str(np.bincount(ys2)))
check("subsample roughly preserves proportions", 85 <= int((ys2 == 0).sum()) <= 92,
      str(int((ys2 == 0).sum())))
Xs3, ys3 = TE._stratified_subsample(Xb, yb, 5000, 42)
check("no subsampling when under the cap", len(ys3) == 1000)
a1, b1 = TE._stratified_subsample(Xb, yb, 100, 42)
a2, b2 = TE._stratified_subsample(Xb, yb, 100, 42)
check("subsampling is deterministic", np.array_equal(b1, b2)
      and list(a1.index) == list(a2.index))

# =========================== end-to-end ==================================
print("\n=== end-to-end on an ACI subsample ===")
cfg = L.load_config()
aci = L.load_dataset(L.ACI, cfg, use_cache=False)
rng = np.random.default_rng(0)
keep = []
for cls, grp in aci.groupby("label", observed=True):
    n = min(len(grp), max(5, int(round(len(grp) / len(aci) * 30000))))
    keep.append(grp.iloc[rng.choice(len(grp), size=n, replace=False)])
small = pd.concat(keep)
small.attrs["dataset"] = L.ACI
print(f"subsample: {small.shape}, {small['label'].nunique()} classes")

sp = P.prepare_splits(small, cfg, mcfg)
mcfg_t = copy.deepcopy(mcfg)
mcfg_t["training"]["max_train_rows"] = {"svm": 5000, "knn": None}

t = time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    table = TE.run_all(L.ACI, cfg, mcfg_t, splits=sp, verbose=False,
                       output_dir="experiments/phase1_aci_iot/_verify")
print(f"(sweep took {time.time()-t:.0f}s)\n")
print(table.to_string(index=False))

check("all six classifiers ran", len(table) == 6, str(len(table)))
check("no classifier errored", "error" not in table.columns,
      str(table.get("error", pd.Series(dtype=object)).tolist()))
check("sorted by macro_f1 descending",
      list(table["macro_f1"]) == sorted(table["macro_f1"], reverse=True))
check("svm flagged as subsampled",
      bool(table.loc[table.classifier == "svm", "subsampled"].iloc[0]))
check("svm n_train_used == cap",
      int(table.loc[table.classifier == "svm", "n_train_used"].iloc[0]) == 5000)
check("uncapped classifiers used the full train split",
      int(table.loc[table.classifier == "random_forest", "n_train_used"].iloc[0])
      == len(sp.y_train))
check("benign FPR reported", "benign_false_positive_rate" in table.columns
      and table["benign_false_positive_rate"].notna().all())
check("metrics are finite",
      np.isfinite(table[["accuracy", "macro_f1", "weighted_f1"]].to_numpy()).all())
check("RF is a sane baseline (macro_f1 > 0.5)",
      float(table.loc[table.classifier == "random_forest", "macro_f1"].iloc[0]) > 0.5,
      f"{float(table.loc[table.classifier=='random_forest','macro_f1'].iloc[0]):.4f}")

out = "experiments/phase1_aci_iot/_verify"
check("results.csv written", os.path.exists(f"{out}/results.csv"))
check("per-classifier json written",
      all(os.path.exists(f"{out}/{n}.json") for n in R.available_classifiers()))
run = json.load(open(f"{out}/run.json"))
check("run.json records provenance",
      run["dataset"] == L.ACI and run["scaler"] == "standard"
      and len(run["classes"]) == 12 and "split" in run)
rf_json = json.load(open(f"{out}/random_forest.json"))
check("per-class breakdown persisted", len(rf_json["per_class"]) == 12)
check("confusion matrix persisted",
      np.array(rf_json["confusion_matrix"]).shape == (12, 12))
check("ARP Spoofing has zero test support (train-only)",
      rf_json["per_class"]["ARP Spoofing"]["support"] == 0)

# =========================== zero-day ====================================
print("\n=== zero-day holdout ===")
cfg_z = copy.deepcopy(cfg)
cfg_z["split"]["holdout_classes"] = ["Slowloris"]
sp_z = P.prepare_splits(small, cfg_z, mcfg)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    tz = TE.run_all(L.ACI, cfg_z, mcfg_t, splits=sp_z,
                    classifiers=["random_forest", "decision_tree"], verbose=False)
print(tz.to_string(index=False))
check("zero_day_recall present in the table", "zero_day_recall" in tz.columns)
check("zero_day_recall is a real number in [0,1]",
      bool(tz["zero_day_recall"].between(0, 1).all()), str(tz["zero_day_recall"].tolist()))
det = tz.attrs["details"]["random_forest"]
check("held-out class never predicted by name",
      det["per_class"]["Slowloris"]["recall"] == 0.0)
check("but is scored on attack-flagging", det["zero_day"]["n_samples"] > 0
      and det["zero_day"]["classes"] == ["Slowloris"])
check("held-out class absent from training",
      det["fit_info"]["classes_seen"] == 11, str(det["fit_info"]["classes_seen"]))

# =========================== failure isolation ===========================
print("\n=== failure isolation ===")
bad = copy.deepcopy(mcfg_t)
bad["knn"] = {"n_neighbors": -3}          # invalid at fit time
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    tb = TE.run_all(L.ACI, cfg, bad, splits=sp,
                    classifiers=["decision_tree", "knn"], verbose=False)
check("a failing classifier is recorded, sweep continues",
      len(tb) == 2 and tb["error"].notna().sum() == 1
      and tb.loc[tb.classifier == "decision_tree", "macro_f1"].notna().all())

import shutil
shutil.rmtree(out, ignore_errors=True)
print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
