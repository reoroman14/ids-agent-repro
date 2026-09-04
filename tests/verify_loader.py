"""Verify the new data layer against the real files."""
import copy, os, sys, time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import feature_schema as fs
from src.data import loader as L

ok = True


def check(name, cond, detail=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" â€” {detail}" if detail else ""))
    if not cond:
        ok = False


cfg = L.load_config()

# --- 1. schema <-> real headers -------------------------------------------
aci_hdr = list(pd.read_csv("data/raw/aci_iot_2023/ACI-IoT-2023.csv", nrows=0).columns)
cic_hdr = list(pd.read_csv("data/raw/cic_iot_2023/Merged01.csv", nrows=0).columns)
check("ACI schema == file header", fs.all_columns(L.ACI) == aci_hdr,
      f"{len(fs.all_columns(L.ACI))} vs {len(aci_hdr)}")
check("CIC schema == file header", fs.all_columns(L.CIC) == cic_hdr,
      f"{len(fs.all_columns(L.CIC))} vs {len(cic_hdr)}")

# --- 2. config internal consistency ---------------------------------------
for ds in (L.ACI, L.CIC):
    d = cfg[ds]
    lm, lg, co, cc = d["label_map"], d["label_groups"], d["class_order"], d["class_counts"]
    canon = set(lm.values())
    check(f"{ds}: class_order == label_map values", set(co) == canon,
          f"{len(co)} order / {len(canon)} canonical")
    check(f"{ds}: label_groups covers all classes", canon <= set(lg))
    check(f"{ds}: class_counts covers all classes", canon == set(cc))
    check(f"{ds}: class_counts sums to n_rows_expected",
          sum(cc.values()) == d["n_rows_expected"] - d.get("n_malformed_rows_expected", 0),
          f"{sum(cc.values()):,} vs {d['n_rows_expected']:,}")
    check(f"{ds}: label_column matches schema", d["label_column"] == fs.label_column(ds))
    check(f"{ds}: class_order() accessor works", L.class_order(ds, cfg) == co)

# --- 3. label normalization ------------------------------------------------
s = pd.Series(["BENIGN", "Benign", " benign ", "DDoS-ICMP_Flood"])
n = L.normalize_labels(s, cfg[L.CIC]["label_map"], dataset=L.CIC)
check("case/whitespace normalization", list(n[:3]) == ["Benign"] * 3, str(list(n[:3])))
check("mixed-case original CIC spelling maps", n[3] == "DDoS-ICMP Flood", str(n[3]))
try:
    L.normalize_labels(pd.Series(["NOT_A_CLASS"]), cfg[L.CIC]["label_map"], dataset=L.CIC)
    check("unknown label raises", False)
except ValueError as e:
    check("unknown label raises", "NOT_A_CLASS" in str(e))

# --- 4. header mismatch is refused ----------------------------------------
try:
    L._validate_header("data/raw/cic_iot_2023/Merged01.csv", fs.all_columns(L.ACI), "x")
    check("header mismatch refused", False)
except ValueError:
    check("header mismatch refused", True)

# --- 5. real ACI load ------------------------------------------------------
t = time.time()
aci = L.load_dataset(L.ACI, cfg, use_cache=False)
rep = aci.attrs["load_report"]
print("\n" + rep.summary())
print(f"  elapsed: {time.time()-t:.1f}s   shape: {aci.shape}")
check("ACI rows == expected", rep.rows_read == cfg[L.ACI]["n_rows_expected"])
check("ACI returns all 12 classes", len(rep.class_counts_returned) == 12)
check("ACI no count drift", not rep.count_drift, str(rep.count_drift))
check("ACI label columns present",
      {"label", "label_group", "label_binary"} <= set(aci.columns))
check("ACI raw Label column removed", "Label" not in aci.columns)
check("ACI ARP Spoofing survives", rep.class_counts_returned.get("ARP Spoofing") == 5)
check("ACI zero-variance cols dropped",
      not (set(fs.columns_with(L.ACI, "zero_variance")) & set(aci.columns)))
check("ACI identifiers still present (dropped at X-build, not load)",
      "Src IP" in aci.columns)
check("ACI binary labels", set(aci["label_binary"].unique()) == {"Benign", "Attack"})
check("ACI groups", set(aci["label_group"].unique()) ==
      {"Benign", "DoS", "Recon", "Spoofing", "Brute Force"},
      str(sorted(aci["label_group"].unique())))
import numpy as np
check("ACI inf replaced with NaN",
      not np.isinf(aci[["Flow Bytes/s", "Flow Packets/s"]].to_numpy()).any())
check("feature_columns excludes critical leakage",
      not ({"Src IP", "Dst IP", "Flow ID", "Timestamp"} & set(fs.feature_columns(L.ACI))))
check("feature_columns keeps ports by decision",
      {"Src Port", "Dst Port"} <= set(fs.feature_columns(L.ACI)))
check("feature_columns can exclude high-risk",
      not ({"Src Port", "Dst Port"} & set(fs.feature_columns(L.ACI, include_high_leakage=False))))

# --- 6. CIC smoke load on 2 files -----------------------------------------
sub = copy.deepcopy(cfg)
sub[L.CIC]["file_glob"] = "Merged0[12].csv"
sub[L.CIC]["n_files_expected"] = 2
sub[L.CIC]["n_rows_expected"] = None
t = time.time()
cic = L.load_cic_iot(sub, chunksize=500_000)
rep2 = cic.attrs["load_report"]
print("\n" + rep2.summary())
print(f"  elapsed: {time.time()-t:.1f}s   shape: {cic.shape}")
check("CIC read 2 files", rep2.files_read == 2)
check("CIC sampled", rep2.sampled and rep2.per_class_cap == 50000)
check("CIC label columns", {"label", "label_group", "label_binary"} <= set(cic.columns))
check("CIC float32 downcast", str(cic["Rate"].dtype) == "float32", str(cic["Rate"].dtype))
check("CIC groups valid",
      set(cic["label_group"].unique()) <= {"Benign", "DDoS", "DoS", "Mirai", "Recon",
                                           "Spoofing", "Web", "Brute Force"},
      str(sorted(cic["label_group"].unique())))

# --- 7. sample is independent of chunksize --------------------------------
cic_b = L.load_cic_iot(sub, chunksize=137_000)
same = (len(cic) == len(cic_b)
        and cic["label"].value_counts().equals(cic_b["label"].value_counts()))
check("CIC sample independent of chunksize", same,
      f"{len(cic):,} vs {len(cic_b):,}")

# --- 8. parquet cache round-trip ------------------------------------------
tmp = "data/processed/_verify_tmp.parquet"
sub[L.CIC]["processed_path"] = tmp
if os.path.exists(tmp):
    os.remove(tmp)
a = L.load_dataset(L.CIC, sub, use_cache=True, rebuild_cache=True)
b = L.load_dataset(L.CIC, sub, use_cache=True)
check("parquet cache written", os.path.exists(tmp))
check("cache round-trips identically", a.shape == b.shape and
      a["label"].value_counts().equals(b["label"].value_counts()))
check("cache-hit report is honest", any("cache" in w for w in b.attrs["load_report"].warnings))
os.remove(tmp)

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
