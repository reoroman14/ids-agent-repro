"""Verify preprocessing.py against the real data."""
import copy, os, sys, time, warnings
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import loader as L
from src.data import preprocessing as P
from src.data import feature_schema as fs

ok = True


def check(name, cond, detail=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" â€” {detail}" if detail else ""))
    if not cond:
        ok = False


cfg = L.load_config()
mcfg = yaml.safe_load(open("config/models.yaml", encoding="utf-8"))
check("models.yaml has preprocessing.scaler",
      mcfg.get("preprocessing", {}).get("scaler") == "standard")

t = time.time()
aci = L.load_dataset(L.ACI, cfg, use_cache=False)
print(f"(ACI loaded in {time.time()-t:.0f}s, {aci.shape})\n")

# --- clean ----------------------------------------------------------------
cleaned = P.clean(aci, cfg, dataset=L.ACI)
check("clean drops identifier columns",
      not ({"Flow ID", "Src IP", "Dst IP", "Timestamp"} & set(cleaned.columns)))
check("clean keeps the ports kept by decision",
      {"Src Port", "Dst Port"} <= set(cleaned.columns))
check("clean keeps all three label columns",
      set(P.LABEL_COLUMNS) <= set(cleaned.columns))
check("clean preserves row count", len(cleaned) == len(aci))
check("clean leaves no infinities",
      not np.isinf(cleaned.select_dtypes(include=[np.number]).to_numpy()).any())

# --- split: rare-class routing (the headline behaviour) -------------------
tr, va, te = P.train_val_test_split(cleaned, cfg)
rep = tr.attrs["split_report"]
print("\n" + rep.summary() + "\n")

check("ARP Spoofing routed to rare bucket", rep.rare_classes == {"ARP Spoofing": 5},
      str(rep.rare_classes))
check("all 5 ARP Spoofing rows in train", int((tr["label"] == "ARP Spoofing").sum()) == 5)
check("ARP Spoofing absent from val", int((va["label"] == "ARP Spoofing").sum()) == 0)
check("ARP Spoofing absent from test", int((te["label"] == "ARP Spoofing").sum()) == 0)
check("reason recorded in report", "ARP Spoofing" in rep.classes_absent_from_test)
check("train has 12 classes", tr["label"].nunique() == 12)
check("test has 11 classes", te["label"].nunique() == 11, str(te["label"].nunique()))

# --- split invariants -----------------------------------------------------
check("splits are disjoint",
      not (set(tr.index) & set(va.index)) and not (set(tr.index) & set(te.index))
      and not (set(va.index) & set(te.index)))
check("no rows lost", len(tr) + len(va) + len(te) == len(cleaned),
      f"{len(tr)+len(va)+len(te):,} vs {len(cleaned):,}")

n = len(cleaned)
check("ratios ~70/15/15", abs(len(tr)/n - 0.70) < 0.005 and abs(len(va)/n - 0.15) < 0.005
      and abs(len(te)/n - 0.15) < 0.005,
      f"{len(tr)/n:.4f}/{len(va)/n:.4f}/{len(te)/n:.4f}")

# stratification: per-class proportions should match between train and test
p_tr = tr[tr["label"] != "ARP Spoofing"]["label"].value_counts(normalize=True)
p_te = te["label"].value_counts(normalize=True)
check("stratification preserved (max class-proportion drift < 0.002)",
      (p_tr - p_te).abs().max() < 0.002, f"{(p_tr - p_te).abs().max():.5f}")

# --- determinism ----------------------------------------------------------
tr2, va2, te2 = P.train_val_test_split(cleaned, cfg)
check("split is deterministic at a fixed seed",
      list(tr.index) == list(tr2.index) and list(te.index) == list(te2.index))

cfg99 = copy.deepcopy(cfg); cfg99["split"]["seed"] = 99
tr3, _, _ = P.train_val_test_split(cleaned, cfg99)
check("a different seed gives a different split", list(tr.index) != list(tr3.index))

# --- holdout_classes (zero-day) -------------------------------------------
cfg_h = copy.deepcopy(cfg)
cfg_h["split"]["holdout_classes"] = ["Slowloris"]
trh, vah, teh = P.train_val_test_split(cleaned, cfg_h)
check("holdout class absent from train", int((trh["label"] == "Slowloris").sum()) == 0)
check("holdout class absent from val", int((vah["label"] == "Slowloris").sum()) == 0)
check("all holdout rows in test",
      int((teh["label"] == "Slowloris").sum()) == 18643,
      str(int((teh["label"] == "Slowloris").sum())))
check("holdout still conserves rows", len(trh) + len(vah) + len(teh) == len(cleaned))
check("ARP Spoofing still train-only alongside a holdout",
      int((trh["label"] == "ARP Spoofing").sum()) == 5)

# --- the conflict case must raise -----------------------------------------
cfg_c = copy.deepcopy(cfg)
cfg_c["split"]["holdout_classes"] = ["ARP Spoofing"]
try:
    P.train_val_test_split(cleaned, cfg_c)
    check("holdout/min_rows conflict raises", False)
except ValueError as e:
    check("holdout/min_rows conflict raises",
          "ARP Spoofing" in str(e) and "min_rows_for_split" in str(e))

# --- misspelled holdout class ---------------------------------------------
cfg_m = copy.deepcopy(cfg)
cfg_m["split"]["holdout_classes"] = ["Slowlorris"]
try:
    P.train_val_test_split(cleaned, cfg_m)
    check("misspelled holdout class raises", False)
except ValueError as e:
    check("misspelled holdout class raises", "Slowlorris" in str(e))

# --- other rare_class_policy values ---------------------------------------
cfg_d = copy.deepcopy(cfg); cfg_d["split"]["rare_class_policy"] = "drop"
trd, vad, ted = P.train_val_test_split(cleaned, cfg_d)
check("policy 'drop' removes the class everywhere",
      int((trd["label"] == "ARP Spoofing").sum()) == 0
      and len(trd) + len(vad) + len(ted) == len(cleaned) - 5)

cfg_e = copy.deepcopy(cfg); cfg_e["split"]["rare_class_policy"] = "error"
try:
    P.train_val_test_split(cleaned, cfg_e)
    check("policy 'error' raises", False)
except ValueError as e:
    check("policy 'error' raises", "ARP Spoofing" in str(e))

# --- bad ratios -----------------------------------------------------------
cfg_r = copy.deepcopy(cfg); cfg_r["split"]["test_ratio"] = 0.25
try:
    P.train_val_test_split(cleaned, cfg_r)
    check("ratios that do not sum to 1 raise", False)
except ValueError as e:
    check("ratios that do not sum to 1 raise", "sum to 1.0" in str(e))

# --- ratios are honoured, not hardcoded -----------------------------------
cfg_802 = copy.deepcopy(cfg)
cfg_802["split"].update({"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1})
t8, v8, s8 = P.train_val_test_split(cleaned, cfg_802)
m = len(cleaned)
check("80/10/10 config is honoured",
      abs(len(t8)/m - 0.8) < 0.005 and abs(len(v8)/m - 0.1) < 0.005
      and abs(len(s8)/m - 0.1) < 0.005,
      f"{len(t8)/m:.3f}/{len(v8)/m:.3f}/{len(s8)/m:.3f}")

# --- full pipeline --------------------------------------------------------
sp = P.prepare_splits(aci, cfg, mcfg)
print("\n" + sp.summary() + "\n")
check("no NaN in X_train", not sp.X_train.isna().to_numpy().any())
check("no NaN in X_val", not sp.X_val.isna().to_numpy().any())
check("no NaN in X_test", not sp.X_test.isna().to_numpy().any())
check("no inf in X_train", not np.isinf(sp.X_train.to_numpy()).any())
check("same feature columns across splits",
      list(sp.X_train.columns) == list(sp.X_val.columns) == list(sp.X_test.columns))
check("Connection Type one-hot to 2 columns",
      sum(c.startswith("Connection Type=") for c in sp.feature_names) == 2,
      str([c for c in sp.feature_names if c.startswith("Connection Type=")]))
check("one-hot columns left unscaled (still 0/1)",
      set(np.unique(sp.X_train["Connection Type=wired"])) <= {0, 1})
check("critical-leakage columns absent from X",
      not ({"Src IP", "Dst IP", "Flow ID", "Timestamp"} & set(sp.feature_names)))
check("ports present by decision", {"Src Port", "Dst Port"} <= set(sp.feature_names))

# scaler fitted on train only
scaled = [c for c in sp.numeric_transform.columns if c in sp.X_train.columns]
tr_mean = np.abs(sp.X_train[scaled].to_numpy().mean(axis=0)).max()
te_mean = np.abs(sp.X_test[scaled].to_numpy().mean(axis=0)).max()
check("train is standardized (max |mean| ~ 0)", tr_mean < 1e-6, f"{tr_mean:.2e}")
check("test NOT re-standardized (proves fit-on-train-only)", te_mean > 1e-6,
      f"max |mean| on test = {te_mean:.2e}")

check("y encoded via frozen class_order, Benign == 0",
      sp.classes_[0] == "Benign" and sp.classes_ == L.class_order(L.ACI, cfg))
check("y_train covers 12 classes", len(np.unique(sp.y_train)) == 12)
check("y_test covers 11 classes", len(np.unique(sp.y_test)) == 11)
check("decode_labels round-trips",
      list(P.decode_labels(sp.y_train[:50], sp.classes_))
      == list(sp.X_train.index.to_series().map(aci["label"])[:50]))

# --- target variants share the same split ---------------------------------
sp_bin = P.prepare_splits(aci, cfg, mcfg, target="label_binary")
check("label_binary uses the same row partition as label",
      list(sp_bin.X_train.index) == list(sp.X_train.index))
check("label_binary y has 2 classes", len(np.unique(sp_bin.y_train)) == 2)
sp_grp = P.prepare_splits(aci, cfg, mcfg, target="label_group")
check("label_group uses the same row partition",
      list(sp_grp.X_train.index) == list(sp.X_train.index))
check("label_group y has 5 classes", len(np.unique(sp_grp.y_train)) == 5,
      str(len(np.unique(sp_grp.y_train))))

# --- include_high_leakage toggle ------------------------------------------
sp_lo = P.prepare_splits(aci, cfg, mcfg, include_high_leakage=False)
check("include_high_leakage=False drops ports and Idle*",
      not ({"Src Port", "Dst Port", "Idle Mean", "Idle Max"} & set(sp_lo.feature_names)))

# --- scaler config is read from models.yaml -------------------------------
m_rob = copy.deepcopy(mcfg); m_rob["preprocessing"]["scaler"] = "robust"
sp_rob = P.prepare_splits(aci, cfg, m_rob)
check("scaler kind comes from models.yaml", sp_rob.numeric_transform.scaler_kind == "robust")
m_bad = copy.deepcopy(mcfg); m_bad["preprocessing"]["scaler"] = "quantile"
try:
    P.prepare_splits(aci, cfg, m_bad)
    check("unknown scaler raises", False)
except ValueError as e:
    check("unknown scaler raises", "quantile" in str(e))

# --- unseen categorical value at transform time ---------------------------
enc_train, cats = P.encode_categoricals(tr, cfg, dataset=L.ACI)
odd = va.copy()
odd.loc[odd.index[:3], "Connection Type"] = "satellite"
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    enc_odd, _ = P.encode_categoricals(odd, cfg, dataset=L.ACI, categories=cats)
check("unseen category warns, does not add a column",
      any("unseen" in str(x.message) for x in w)
      and list(enc_odd.columns) == list(enc_train.columns))
check("unseen category encodes as all-zero dummies",
      int(enc_odd.loc[enc_odd.index[:3], P.one_hot_columns(cats)].to_numpy().sum()) == 0)

# --- CIC subset through the same interface --------------------------------
sub = copy.deepcopy(cfg)
sub[L.CIC]["file_glob"] = "Merged0[1-4].csv"
sub[L.CIC]["n_files_expected"] = 4
sub[L.CIC]["n_rows_expected"] = None
cic = L.load_cic_iot(sub)
sp_c = P.prepare_splits(cic, sub, mcfg)
print("\n" + sp_c.report.summary())
print("\n" + sp_c.summary())
# On this 4-file subset Uploading Attack holds only 84 of its 1,196 rows, so it
# legitimately falls under min_rows_for_split â€” a useful proof that the policy
# is driven by observed counts and is not ACI-specific. The full-file behaviour
# (all 34 classes split normally) is verified separately in full_cic_prep.py.
check("CIC: sub-threshold class routed train-only, same as ACI",
      sp_c.report.rare_classes == {"Uploading Attack": 84}
      and int((sp_c.X_train.index.to_series().map(cic["label"]) == "Uploading Attack").sum()) == 84,
      str(sp_c.report.rare_classes))
check("CIC: train holds 34 classes, test holds 33 on this subset",
      len(np.unique(sp_c.y_train)) == 34 and len(np.unique(sp_c.y_test)) == 33,
      f"{len(np.unique(sp_c.y_train))}/{len(np.unique(sp_c.y_test))}")
check("CIC: no NaN in X", not sp_c.X_train.isna().to_numpy().any()
      and not sp_c.X_test.isna().to_numpy().any())
check("CIC: no categorical columns", sp_c.categories == {})
check("CIC: rows conserved",
      len(sp_c.X_train) + len(sp_c.X_val) + len(sp_c.X_test) == len(cic))

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
