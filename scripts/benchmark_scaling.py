"""
Measure how each classifier's fit/predict time grows with training rows, then
extrapolate to the real full-dataset sizes.

Fits t = a * n^b on log-log for each (dataset, classifier, phase). The exponent
b is the interesting number: ~1 means linear, ~2 means quadratic.
"""
import copy, os, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore")

from src.data import loader as L
from src.data import preprocessing as P
from src.classifiers import train_eval as TE
from src.classifiers import registry as R

cfg = L.load_config()
mcfg = TE.load_models_config()
CAPS = mcfg["training"]["max_train_rows"]
NJOBS = mcfg["training"]["n_jobs"]
SEED = cfg["split"]["seed"]

SIZES = {"default": [5000, 10000, 20000, 40000], "svm": [2500, 5000, 10000, 20000]}
N_TEST_BENCH = 10000

# Real full-scale shapes, measured in earlier runs.
FULL = {
    "aci_iot_2023": {"n_train": 861989, "n_test": 184711, "n_features": 75, "n_classes": 12},
    "cic_iot_2023": {"n_train": 901577, "n_test": 193196, "n_features": 39, "n_classes": 34},
}


def get_splits(name, file_glob=None):
    c = copy.deepcopy(cfg)
    if file_glob:
        c[name]["file_glob"] = file_glob
        c[name]["n_files_expected"] = None
        c[name]["n_rows_expected"] = None
    df = L.load_dataset(name, c, use_cache=False)
    return P.prepare_splits(df, c, mcfg)


def bench(sp, dataset):
    Xte = sp.X_test.iloc[:N_TEST_BENCH]
    yte = sp.y_test[:N_TEST_BENCH]
    rows = []
    for clf in R.available_classifiers():
        sizes = SIZES["svm"] if clf == "svm" else SIZES["default"]
        for n in sizes:
            if n > len(sp.y_train):
                continue
            Xs, ys = TE._stratified_subsample(sp.X_train, sp.y_train, n, SEED)
            model = R.build_classifier(clf, mcfg.get(clf, {}), seed=SEED, n_jobs=NJOBS)
            t0 = time.time(); model.fit(Xs, ys); fit_s = time.time() - t0
            t0 = time.time(); model.predict(Xte); pred_s = time.time() - t0
            rows.append({"dataset": dataset, "classifier": clf, "n": len(ys),
                         "fit_s": fit_s, "pred_s": pred_s})
            print(f"  {dataset:14} {clf:20} n={len(ys):>6,}  fit {fit_s:8.2f}s  "
                  f"pred/{N_TEST_BENCH//1000}k {pred_s:7.2f}s", flush=True)
    return pd.DataFrame(rows)


def power_fit(n, t):
    """Return (a, b) for t = a * n^b, guarding against zero timings."""
    t = np.maximum(np.asarray(t, dtype=float), 1e-4)
    b, loga = np.polyfit(np.log(np.asarray(n, dtype=float)), np.log(t), 1)
    return float(np.exp(loga)), float(b)


print("=== loading ===", flush=True)
t0 = time.time()
sp_aci = get_splits("aci_iot_2023")
print(f"ACI splits ready ({time.time()-t0:.0f}s), train={len(sp_aci.y_train):,}", flush=True)
t0 = time.time()
sp_cic = get_splits("cic_iot_2023", "Merged0[1-6].csv")
print(f"CIC splits ready ({time.time()-t0:.0f}s), train={len(sp_cic.y_train):,}", flush=True)

print("\n=== benchmarking ===", flush=True)
data = pd.concat([bench(sp_aci, "aci_iot_2023"), bench(sp_cic, "cic_iot_2023")])
data.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "scaling_raw.csv"), index=False)

print("\n=== extrapolation to full scale ===\n", flush=True)
out = []
for dataset, grp in data.groupby("dataset"):
    full = FULL[dataset]
    for clf, g in grp.groupby("classifier"):
        cap = CAPS.get(clf)
        n_train = min(full["n_train"], cap) if cap else full["n_train"]

        a_f, b_f = power_fit(g["n"], g["fit_s"])
        fit_est = a_f * n_train ** b_f

        a_p, b_p = power_fit(g["n"], g["pred_s"])
        # predict measured on N_TEST_BENCH rows; scale linearly in n_test
        pred_est = a_p * n_train ** b_p * (full["n_test"] / N_TEST_BENCH)

        out.append({
            "dataset": dataset, "classifier": clf,
            "n_train": n_train, "capped": bool(cap and full["n_train"] > cap),
            "fit_exp": round(b_f, 2), "fit_min": fit_est / 60,
            "pred_exp": round(b_p, 2), "pred_min": pred_est / 60,
            "total_min": (fit_est + pred_est) / 60,
        })

est = pd.DataFrame(out).sort_values(["dataset", "total_min"], ascending=[True, False])
pd.set_option("display.width", 200)
for dataset, g in est.groupby("dataset"):
    print(f"--- {dataset} (train={FULL[dataset]['n_train']:,}, "
          f"test={FULL[dataset]['n_test']:,}, "
          f"{FULL[dataset]['n_features']} features, "
          f"{FULL[dataset]['n_classes']} classes) ---")
    show = g[["classifier", "n_train", "capped", "fit_exp", "fit_min",
              "pred_exp", "pred_min", "total_min"]].copy()
    for c in ("fit_min", "pred_min", "total_min"):
        show[c] = show[c].round(1)
    print(show.to_string(index=False))
    print(f"  SWEEP TOTAL: {g['total_min'].sum():.0f} min\n")

print(f"GRAND TOTAL (both datasets): {est['total_min'].sum():.0f} min "
      f"+ ~5 min CIC load")
est.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "scaling_estimates.csv"), index=False)
