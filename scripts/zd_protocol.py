"""Redesign the zero-day protocol by behavioural similarity, and validate it."""
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
from src.evaluation import zero_day_protocol as Z

OUT = "experiments/zero_day"
cfg = L.load_config(); mcfg = TE.load_models_config()
SEED, NJOBS, CLF = cfg["split"]["seed"], mcfg["training"]["n_jobs"], "random_forest"
pd.set_option("display.width", 230)

hist = pd.read_csv(os.path.join(OUT, "zero_day_auc_results.csv"))
spaces, splits, all_rows = {}, {}, []


def taxonomic_holdouts(ds):
    g = cfg[ds]["label_groups"]; cnt = cfg[ds]["class_counts"]
    attacks = [c for c, gr in g.items() if gr != "Benign" and cnt[c] >= 100]
    fams = {}
    for c, gr in g.items():
        if gr != "Benign" and c in attacks:
            fams.setdefault(gr, []).append(c)
    if ds == L.ACI:
        singles = [(c, [c]) for c in sorted(attacks)]
    else:
        singles = [(f"{max(v, key=lambda x: cnt[x])}  [{k}]",
                    [max(v, key=lambda x: cnt[x])]) for k, v in sorted(fams.items())]
    return singles, [(f"{k} ({len(v)})", v) for k, v in sorted(fams.items())]


# ---------- build behavioural spaces ---------------------------------------
for ds in (L.ACI, L.CIC):
    df = L.load_dataset(ds, cfg, use_cache=(ds == L.CIC))
    sp = P.prepare_splits(df, cfg, mcfg)
    splits[ds] = sp
    spaces[ds] = Z.build_behavioural_space(sp.X_train, sp.y_train, sp.classes_)
    del df
    print(f"\n=== {ds}: behavioural space over {len(spaces[ds].class_names)} classes ===")
    print("nearest behavioural neighbours (and distance to Benign):")
    for c in spaces[ds].attack_classes:
        nn = spaces[ds].nearest_neighbours(c, 2)
        print(f"  {c:26} -> " + ", ".join(f"{n} ({d:.1f})" for n, d in nn)
              + f"   | benign {spaces[ds].distance_to_benign(c):.1f}")

# ---------- behavioural vs taxonomic clusters ------------------------------
CLUSTERS = {}
for ds, k in ((L.ACI, 4), (L.CIC, 5)):
    CLUSTERS[ds] = Z.behavioural_clusters(spaces[ds], k)
    print(f"\n=== {ds}: behavioural clusters (k={k}) vs taxonomy ===")
    for name, members in CLUSTERS[ds].items():
        tax = sorted({cfg[ds]["label_groups"][m] for m in members})
        print(f"  {name}: {members}")
        print(f"      taxonomic families spanned: {tax}")

# ---------- validate: does distance predict measured AUC? ------------------
print("\n=== validation against the 27 measured runs ===")
recs = []
for ds in (L.ACI, L.CIC):
    singles, fams = taxonomic_holdouts(ds)
    for regime, items in (("leave-one-class-out", singles),
                          ("leave-one-family-out", fams)):
        for label, classes in items:
            row = hist[(hist.dataset == ds) & (hist.regime == regime)
                       & (hist.holdout == label)]
            if row.empty:
                continue
            d = Z.nearest_remaining_distance(spaces[ds], classes)
            if d is None:
                continue
            recs.append({"dataset": ds, "regime": regime, "holdout": label,
                         "n_classes": len(classes),
                         "nearest_remaining_distance": d,
                         "auc": float(row.iloc[0]["auc"]),
                         "fixed": float(row.iloc[0]["fixed_threshold_recall"])})

val = pd.DataFrame(recs)
print(val.sort_values("nearest_remaining_distance")[
    ["dataset", "regime", "holdout", "nearest_remaining_distance", "auc"]
].round(3).to_string(index=False))

stats = Z.validate_distance_predicts_difficulty(recs)
print(f"\nAcross {stats['n']} measured holdouts:")
print(f"  Pearson  r(distance, AUC) = {stats['pearson']:+.3f}")
print(f"  Spearman rho             = {stats['spearman']:+.3f}")
print(f"  slope                    = {stats['slope']:+.5f} AUC per unit distance")
for ds in (L.ACI, L.CIC):
    s = Z.validate_distance_predicts_difficulty([r for r in recs if r["dataset"] == ds])
    print(f"  {ds}: n={s['n']}  pearson {s['pearson']:+.3f}  spearman {s['spearman']:+.3f}")

# ---------- run the new behavioural protocol -------------------------------
print("\n=== running behavioural-cluster holdouts ===")
new_rows = []
for ds in (L.ACI, L.CIC):
    sp_full = splits[ds]
    df = L.load_dataset(ds, cfg, use_cache=(ds == L.CIC))
    for name, members in CLUSTERS[ds].items():
        d = Z.nearest_remaining_distance(spaces[ds], members)
        if d is None:
            print(f"  {ds} {name}: skipped (no attack class would remain)")
            continue
        c = copy.deepcopy(cfg); c["split"]["holdout_classes"] = list(members)
        t0 = time.time()
        try:
            sp = P.prepare_splits(df, c, mcfg)
            model = TE.train_classifier(CLF, sp.X_train, sp.y_train, mcfg[CLF],
                                        seed=SEED, n_jobs=NJOBS)
            scores = R.predict_scores(model, sp.X_test)
            cv = M.zero_day_tradeoff(sp.y_test, scores, class_names=sp.classes_,
                                     holdout_classes=list(members),
                                     model_classes=model.classes_)
            fixed = M.compute_metrics(sp.y_test, model.predict(sp.X_test),
                                      class_names=sp.classes_,
                                      holdout_classes=list(members),
                                      include_confusion=False)
        except Exception as exc:
            print(f"  {ds} {name}: SKIPPED â€” {type(exc).__name__}: {exc}")
            continue
        new_rows.append({"dataset": ds, "regime": "behavioural-cluster",
                         "holdout": f"{name} ({len(members)})",
                         "members": ", ".join(members), "n_classes": len(members),
                         "nearest_remaining_distance": d, "auc": cv["auc"],
                         "fixed_threshold_recall": fixed["zero_day_recall"],
                         "recall_at_fpr_0.01": M.recall_at_fpr(cv, 0.01),
                         "seconds": round(time.time() - t0)})
        print(f"  {ds:14} {name:5} d={d:6.1f}  AUC {cv['auc']:.4f}  "
              f"fixed {fixed['zero_day_recall']:.4f}  ({new_rows[-1]['seconds']}s)",
              flush=True)
    del df

new = pd.DataFrame(new_rows)
new.to_csv(os.path.join(OUT, "behavioural_protocol_results.csv"), index=False)
json.dump({"clusters": {k: v for k, v in CLUSTERS.items()},
           "validation": stats},
          open(os.path.join(OUT, "behavioural_protocol.json"), "w"), indent=2)

print("\n================ BEHAVIOURAL PROTOCOL ================")
print(new[["dataset", "holdout", "members", "nearest_remaining_distance",
           "auc", "fixed_threshold_recall"]].round(4).to_string(index=False))

combined = recs + [{"dataset": r["dataset"], "regime": r["regime"],
                    "nearest_remaining_distance": r["nearest_remaining_distance"],
                    "auc": r["auc"]} for r in new_rows]
final = Z.validate_distance_predicts_difficulty(combined)
print(f"\nAll {final['n']} holdouts (taxonomic + behavioural):")
print(f"  Pearson r = {final['pearson']:+.3f}   Spearman rho = {final['spearman']:+.3f}")
print(f"  AUC range {final['score_range'][0]:.3f}-{final['score_range'][1]:.3f} "
      f"over distance {final['distance_range'][0]:.1f}-{final['distance_range'][1]:.1f}")
