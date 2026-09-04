"""
Re-run the full zero-day sweep reporting AUC alongside the fixed-threshold
recall, so the two can be compared directly on identical models.
"""
import copy, json, os, sys, time, warnings
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore")

from src.data import loader as L
from src.data import preprocessing as P
from src.classifiers import train_eval as TE
from src.classifiers import registry as R
from src.evaluation import metrics as M

CLF, OUT = "random_forest", "experiments/zero_day"
os.makedirs(OUT, exist_ok=True)
cfg = L.load_config(); mcfg = TE.load_models_config()
SEED, NJOBS = cfg["split"]["seed"], mcfg["training"]["n_jobs"]
rows, curves = [], {}


def sweep(dataset, df, holdouts, regime):
    for label, classes in holdouts:
        c = copy.deepcopy(cfg); c["split"]["holdout_classes"] = list(classes)
        t0 = time.time()
        try:
            sp = P.prepare_splits(df, c, mcfg)
            model = TE.train_classifier(CLF, sp.X_train, sp.y_train, mcfg[CLF],
                                        seed=SEED, n_jobs=NJOBS)
            y_pred = model.predict(sp.X_test)
            fixed = M.compute_metrics(sp.y_test, y_pred, class_names=sp.classes_,
                                      holdout_classes=list(classes),
                                      include_confusion=False)
            scores = R.predict_scores(model, sp.X_test)
            cv = M.zero_day_tradeoff(sp.y_test, scores, class_names=sp.classes_,
                                     holdout_classes=list(classes),
                                     model_classes=model.classes_)
        except Exception as exc:
            print(f"  [{regime}] {label}: SKIPPED â€” {type(exc).__name__}: {exc}", flush=True)
            continue
        ops = M.tradeoff_operating_points(cv)
        key = f"{dataset}::{regime}::{label}"
        curves[key] = {k: v for k, v in cv.items() if k != "thresholds"}
        rows.append({
            "dataset": dataset, "regime": regime, "holdout": label,
            "n_classes_held": len(classes), "n_held_rows": cv["n_zero_day"],
            "auc": cv["auc"],
            "fixed_threshold_recall": fixed["zero_day_recall"],
            **{k: v for k, v in ops.items()},
            "fpr_for_recall_0.61": M.fpr_at_recall(cv, 0.61),
            "benign_fpr_default": fixed.get("benign_false_positive_rate"),
            "seconds": round(time.time() - t0),
        })
        print(f"  [{regime}] {label:26} AUC {cv['auc']:.4f}   "
              f"fixed {fixed['zero_day_recall']:.4f}   "
              f"r@1%FPR {ops['recall_at_fpr_0.01']:.4f}   "
              f"({rows[-1]['seconds']}s)", flush=True)


print("=== ACI ===", flush=True)
aci = L.load_dataset(L.ACI, cfg, use_cache=False)
g = cfg[L.ACI]["label_groups"]; cnt = cfg[L.ACI]["class_counts"]
attacks = [c for c, gr in g.items() if gr != "Benign" and cnt[c] >= 100]
sweep(L.ACI, aci, [(c, [c]) for c in sorted(attacks)], "leave-one-class-out")
fams = {}
for c, gr in g.items():
    if gr != "Benign" and c in attacks: fams.setdefault(gr, []).append(c)
sweep(L.ACI, aci, [(f"{k} ({len(v)})", v) for k, v in sorted(fams.items())],
      "leave-one-family-out")
del aci

print("\n=== CIC ===", flush=True)
cic = L.load_dataset(L.CIC, cfg, use_cache=True)
g = cfg[L.CIC]["label_groups"]; cnt = cfg[L.CIC]["class_counts"]
fams = {}
for c, gr in g.items():
    if gr != "Benign": fams.setdefault(gr, []).append(c)
reps = [(max(v, key=lambda x: cnt[x]), k) for k, v in sorted(fams.items())]
sweep(L.CIC, cic, [(f"{c}  [{k}]", [c]) for c, k in reps], "leave-one-class-out")
sweep(L.CIC, cic, [(f"{k} ({len(v)})", v) for k, v in sorted(fams.items())],
      "leave-one-family-out")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "zero_day_auc_results.csv"), index=False)
json.dump(curves, open(os.path.join(OUT, "zero_day_auc_curves.json"), "w"), indent=2)

pd.set_option("display.width", 240)
print("\n================ AUC vs FIXED THRESHOLD ================")
show = res[["dataset", "regime", "holdout", "auc", "fixed_threshold_recall",
            "recall_at_fpr_0.01", "recall_at_fpr_0.05", "fpr_for_recall_0.61"]].round(4)
print(show.to_string(index=False))

print("\n---- summary ----")
s = res.groupby(["dataset", "regime"])[["auc", "fixed_threshold_recall"]].agg(
    ["count", "mean", "median", "min", "max"]).round(4)
print(s.to_string())
gap = res["auc"] - res["fixed_threshold_recall"]
print(f"\nAUC exceeds fixed-threshold recall in {int((gap > 0).sum())}/{len(res)} runs; "
      f"median gap {gap.median():.4f}, max {gap.max():.4f}")
r = np.corrcoef(res["auc"], res["fixed_threshold_recall"])[0, 1]
print(f"correlation between AUC and fixed-threshold recall: r = {r:.3f}")
