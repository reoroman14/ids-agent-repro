"""Build zero-day trade-off curves for representative holdouts on both datasets."""
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

OUT = "experiments/zero_day"
os.makedirs(OUT, exist_ok=True)
cfg = L.load_config(); mcfg = TE.load_models_config()
CLF = "random_forest"

# --- unit checks on synthetic scores --------------------------------------
names = ["Benign", "A", "B"]
yt = np.array([0, 0, 0, 0, 2, 2, 2, 2])
sc = np.array([[.9,.05,.05],[.8,.1,.1],[.7,.2,.1],[.4,.3,.3],
               [.3,.4,.3],[.2,.4,.4],[.1,.5,.4],[.05,.5,.45]])
cv = M.zero_day_tradeoff(yt, sc, class_names=names, holdout_classes=["B"])
ok = True
def check(n, c, d=""):
    global ok
    print(f"[{'PASS' if c else 'FAIL'}] {n}" + (f" â€” {d}" if d else ""))
    if not c: ok = False

check("perfectly separable -> AUC 1.0", abs(cv["auc"] - 1.0) < 1e-9, f"{cv['auc']:.4f}")
check("counts right", cv["n_benign"] == 4 and cv["n_zero_day"] == 4)
check("recall@0 FPR is 1.0 when separable", abs(M.recall_at_fpr(cv, 0.0) - 1.0) < 1e-9)
check("fpr_at_recall(0.61) ~ 0 when separable", M.fpr_at_recall(cv, 0.61) < 1e-9)
rnd = M.zero_day_tradeoff(yt, np.tile([[1/3,1/3,1/3]], (8,1)),
                          class_names=names, holdout_classes=["B"])
check("uninformative scores -> AUC 0.5", abs(rnd["auc"] - 0.5) < 1e-9, f"{rnd['auc']:.4f}")
try:
    M.zero_day_tradeoff(yt, sc[:, :2], class_names=names, holdout_classes=["B"])
    check("shape mismatch raises", False)
except ValueError: check("shape mismatch raises", True)
try:
    M.zero_day_tradeoff(np.zeros(8, int), sc, class_names=names, holdout_classes=["B"])
    check("no zero-day rows raises", False)
except ValueError: check("no zero-day rows raises", True)

# a model trained without the held-out class emits one column fewer
sc2 = sc[:, [0, 1]]                       # trained on {Benign, A} only
cv2 = M.zero_day_tradeoff(yt, sc2, class_names=names, holdout_classes=["B"],
                          model_classes=[0, 1])
check("model_classes maps the Benign column correctly",
      abs(cv2["auc"] - 1.0) < 1e-9, f"{cv2['auc']:.4f}")
try:
    M.zero_day_tradeoff(yt, sc2, class_names=names, holdout_classes=["B"])
    check("misaligned columns raise without model_classes", False)
except ValueError as e:
    check("misaligned columns raise without model_classes", "model_classes" in str(e))
try:
    M.zero_day_tradeoff(yt, sc[:, [1, 2]], class_names=names,
                        holdout_classes=["B"], model_classes=[1, 2])
    check("model that never saw Benign raises", False)
except ValueError as e:
    check("model that never saw Benign raises", "never trained on the Benign" in str(e))
print()

# --- real curves -----------------------------------------------------------
CASES = [
    (L.ACI, "Slowloris [one class]", ["Slowloris"]),
    (L.ACI, "DoS family [5 classes]",
     ["DNS Flood", "ICMP Flood", "SYN Flood", "Slowloris", "UDP Flood"]),
    (L.CIC, "DDoS-ICMP Flood [one class]", ["DDoS-ICMP Flood"]),
    (L.CIC, "Recon family [5 classes]",
     ["Recon-Host Discovery", "Recon-OS Scan", "Recon-Port Scan",
      "Recon-Ping Sweep", "Vulnerability Scan"]),
]

rows, curves = [], {}
loaded = {}
for ds, label, held in CASES:
    if ds not in loaded:
        loaded[ds] = L.load_dataset(ds, cfg, use_cache=(ds == L.CIC))
    c = copy.deepcopy(cfg); c["split"]["holdout_classes"] = held
    t0 = time.time()
    sp = P.prepare_splits(loaded[ds], c, mcfg)
    model = TE.train_classifier(CLF, sp.X_train, sp.y_train, mcfg[CLF],
                                seed=c["split"]["seed"],
                                n_jobs=mcfg["training"]["n_jobs"])
    scores = R.predict_scores(model, sp.X_test)
    cv = M.zero_day_tradeoff(sp.y_test, scores, class_names=sp.classes_,
                             holdout_classes=held, model_classes=model.classes_)
    curves[f"{ds}::{label}"] = cv
    ops = M.tradeoff_operating_points(cv)
    rows.append({"dataset": ds, "holdout": label, "auc": round(cv["auc"], 4),
                 **{k: round(v, 4) for k, v in ops.items()},
                 "fpr_for_recall_0.61": round(M.fpr_at_recall(cv, 0.61), 4),
                 "seconds": round(time.time() - t0)})
    print(M.summarize_tradeoff(cv, title=f"--- {ds} / {label} ---"), flush=True)
    print()

t = pd.DataFrame(rows)
t.to_csv(os.path.join(OUT, "tradeoff_summary.csv"), index=False)
json.dump({k: {kk: vv for kk, vv in v.items() if kk != "thresholds"}
           for k, v in curves.items()},
          open(os.path.join(OUT, "tradeoff_curves.json"), "w"), indent=2)
pd.set_option("display.width", 220)
print("================ TRADE-OFF SUMMARY ================")
print(t.to_string(index=False))
print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
