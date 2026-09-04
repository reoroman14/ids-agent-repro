"""
Decide the MLP cap on evidence: how many iterations does it actually use on
CIC, and what do the cheaper variants cost in accuracy?
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
from src.evaluation import metrics as M

cfg = L.load_config()
mcfg = TE.load_models_config()
SEED = cfg["split"]["seed"]

c = copy.deepcopy(cfg)
c["cic_iot_2023"]["file_glob"] = "Merged0[1-8].csv"
c["cic_iot_2023"]["n_files_expected"] = None
c["cic_iot_2023"]["n_rows_expected"] = None

t0 = time.time()
df = L.load_dataset("cic_iot_2023", c, use_cache=False)
sp = P.prepare_splits(df, c, mcfg)
print(f"CIC subset ready in {time.time()-t0:.0f}s, train={len(sp.y_train):,}", flush=True)

N = 60000
X, y = TE._stratified_subsample(sp.X_train, sp.y_train, N, SEED)
Xte, yte = sp.X_test, sp.y_test
print(f"benchmark train={len(y):,}  test={len(yte):,}  classes={len(np.unique(y))}\n",
      flush=True)

base = dict(mcfg["mlp"])
VARIANTS = {
    "current (max_iter=300)": base,
    "early_stopping=True": {**base, "early_stopping": True},
    "max_iter=100": {**base, "max_iter": 100},
    "max_iter=50": {**base, "max_iter": 50},
}

rows = []
for label, params in VARIANTS.items():
    model = R.build_classifier("mlp", params, seed=SEED)
    t0 = time.time()
    model.fit(X, y)
    fit_s = time.time() - t0
    pred = model.predict(Xte)
    m = M.compute_metrics(pred if False else yte, pred, class_names=sp.classes_,
                          include_confusion=False)
    rows.append({
        "variant": label,
        "n_iter_": int(getattr(model, "n_iter_", -1)),
        "fit_s": round(fit_s, 1),
        "weighted_f1": round(m["weighted_f1"], 4),
        "macro_f1": round(m["macro_f1"], 4),
        "accuracy": round(m["accuracy"], 4),
    })
    print(f"  {label:24} n_iter={rows[-1]['n_iter_']:>4}  fit {fit_s:7.1f}s  "
          f"wF1 {m['weighted_f1']:.4f}  mF1 {m['macro_f1']:.4f}", flush=True)

t = pd.DataFrame(rows)
print("\n" + t.to_string(index=False))

full = 901577
base_row = t.iloc[0]
print(f"\nExtrapolated MLP fit time on the full CIC train split ({full:,} rows),")
print("assuming per-epoch cost is linear in rows:")
for _, r in t.iterrows():
    est = r["fit_s"] * (full / N) / 60
    delta_w = r["weighted_f1"] - base_row["weighted_f1"]
    delta_m = r["macro_f1"] - base_row["macro_f1"]
    print(f"  {r['variant']:24} ~{est:5.1f} min   Î”wF1 {delta_w:+.4f}   Î”mF1 {delta_m:+.4f}")
