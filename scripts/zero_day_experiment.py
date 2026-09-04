"""
Zero-day experiment.

Design question: what counts as a zero-day attack?

  (a) LEAVE-ONE-CLASS-OUT. Withhold one attack class, leaving its siblings in
      training. Withholding DDoS-ICMP Flood while 11 other DDoS classes remain
      is an easy task â€” the model has a near-identical neighbour to fall back
      on and should still flag the traffic as an attack.

  (b) LEAVE-ONE-FAMILY-OUT. Withhold an entire semantic group, so no related
      class remains. This is the honest "attack type nobody has seen" test.

The paper reports a single zero-day recall of 0.61 without saying which regime
it used, so both are measured here. Reporting a distribution across held-out
classes is more informative than one number either way.

Recall is "flagged as any non-benign class", not "named correctly" â€” a model
cannot emit a label it never trained on.
"""
import copy, json, os, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore")

from src.data import loader as L
from src.data import preprocessing as P
from src.classifiers import train_eval as TE

CLF = "random_forest"          # best baseline model on both datasets
OUT = "experiments/zero_day"
os.makedirs(OUT, exist_ok=True)

cfg = L.load_config()
mcfg = TE.load_models_config()
rows = []


def sweep(dataset, df, holdouts, regime):
    for label, classes in holdouts:
        c = copy.deepcopy(cfg)
        c["split"]["holdout_classes"] = list(classes)
        t0 = time.time()
        try:
            sp = P.prepare_splits(df, c, mcfg)
            tbl = TE.run_all(dataset, c, mcfg, splits=sp, classifiers=[CLF],
                             verbose=False)
        except Exception as exc:
            print(f"  [{regime}] {label}: SKIPPED â€” {type(exc).__name__}: {exc}",
                  flush=True)
            continue
        det = tbl.attrs["details"][CLF]
        zd = det["zero_day"]
        rec = {
            "dataset": dataset, "regime": regime, "holdout": label,
            "n_classes_held": len(classes),
            "n_held_rows": zd["n_samples"],
            "n_train": int(det["fit_info"]["n_train_used"]),
            "zero_day_recall": zd["recall"],
            "weighted_f1_rest": det["weighted_f1"],
            "benign_fpr": det.get("benign_false_positive_rate"),
            "seconds": round(time.time() - t0, 1),
        }
        rows.append(rec)
        print(f"  [{regime}] {label:26} held {zd['n_samples']:>7,} rows  "
              f"recall {zd['recall']:.4f}  ({rec['seconds']:.0f}s)", flush=True)


# =============================== ACI =====================================
print("=== ACI ===", flush=True)
aci = L.load_dataset(L.ACI, cfg, use_cache=False)
aci_groups = cfg[L.ACI]["label_groups"]
# ARP Spoofing (5 rows) is already routed train-only, so holding it out is a
# contradiction the splitter rejects by design. Excluded here deliberately.
aci_attacks = [c for c, g in aci_groups.items()
               if g != "Benign" and cfg[L.ACI]["class_counts"][c] >= 100]
sweep(L.ACI, aci, [(c, [c]) for c in sorted(aci_attacks)], "leave-one-class-out")

aci_fams = {}
for c, g in aci_groups.items():
    if g != "Benign" and c in aci_attacks:
        aci_fams.setdefault(g, []).append(c)
sweep(L.ACI, aci, [(f"{g} ({len(v)})", v) for g, v in sorted(aci_fams.items())],
      "leave-one-family-out")
del aci

# =============================== CIC =====================================
print("\n=== CIC ===", flush=True)
cic = L.load_dataset(L.CIC, cfg, use_cache=True)
cic_groups = cfg[L.CIC]["label_groups"]
counts = cfg[L.CIC]["class_counts"]

fams = {}
for c, g in cic_groups.items():
    if g != "Benign":
        fams.setdefault(g, []).append(c)

# One representative per family: the largest member, so the per-class and
# per-family numbers are directly comparable.
reps = [(max(v, key=lambda c: counts[c]), g) for g, v in sorted(fams.items())]
sweep(L.CIC, cic, [(f"{c}  [{g}]", [c]) for c, g in reps], "leave-one-class-out")
sweep(L.CIC, cic, [(f"{g} ({len(v)})", v) for g, v in sorted(fams.items())],
      "leave-one-family-out")

# =============================== report ==================================
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "zero_day_results.csv"), index=False)
json.dump(rows, open(os.path.join(OUT, "zero_day_results.json"), "w"), indent=2)

pd.set_option("display.width", 220)
print("\n================ ZERO-DAY RESULTS ================")
print(res.to_string(index=False))

print("\n---- summary by dataset and regime ----")
summary = (res.groupby(["dataset", "regime"])["zero_day_recall"]
             .agg(["count", "mean", "median", "min", "max"]).round(4))
print(summary.to_string())
print(f"\nPaper reports zero-day recall ~ 0.61")
