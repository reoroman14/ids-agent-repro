"""Verify baseline_report.py, including on a real run."""
import copy, os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import loader as L
from src.data import preprocessing as P
from src.classifiers import train_eval as TE
from src.evaluation import baseline_report as BR

ok = True
def check(name, cond, detail=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" â€” {detail}" if detail else ""))
    if not cond: ok = False


# --- synthetic tables, to exercise the comparison logic exactly ------------
aci_tbl = pd.DataFrame([
    {"classifier": "random_forest", "n_train_used": 21009, "subsampled": False,
     "accuracy": 0.9953, "macro_f1": 0.8272, "macro_f1_present_only": 0.9024,
     "weighted_f1": 0.9953},
    {"classifier": "svm", "n_train_used": 5000, "subsampled": True,
     "accuracy": 0.9017, "macro_f1": 0.6316, "macro_f1_present_only": 0.6890,
     "weighted_f1": 0.8890},
])
cic_tbl = pd.DataFrame([
    {"classifier": "random_forest", "n_train_used": 900000, "subsampled": False,
     "accuracy": 0.80, "macro_f1": 0.70, "macro_f1_present_only": 0.70,
     "weighted_f1": 0.7550},
])
res = {"aci_iot_2023": aci_tbl, "cic_iot_2023": cic_tbl}

cmps = BR.compare_to_paper(res)
by = {c.dataset: c for c in cmps}
check("one comparison per dataset plus zero-day",
      set(by) == {"aci_iot_2023", "cic_iot_2023", "zero-day"}, str(sorted(by)))
check("ACI compares weighted_f1 by default", by["aci_iot_2023"].metric == "weighted_f1"
      and abs(by["aci_iot_2023"].our_value - 0.9953) < 1e-9)
check("best classifier chosen on the compared metric",
      by["aci_iot_2023"].our_classifier == "random_forest")
check("paper targets loaded", by["aci_iot_2023"].paper_value == 0.97
      and by["cic_iot_2023"].paper_value == 0.75)
check("delta computed", abs(by["aci_iot_2023"].delta - 0.0253) < 1e-4)
check("CIC verdict 'matched' within tolerance",
      by["cic_iot_2023"].verdict() == "matched",
      f"{by['cic_iot_2023'].delta:.4f} -> {by['cic_iot_2023'].verdict()}")
check("ACI verdict 'above paper' outside tolerance",
      by["aci_iot_2023"].verdict() == "above paper")
check("zero-day reported as not run when no holdout",
      by["zero-day"].our_value is None and by["zero-day"].verdict() == "not run"
      and "holdout_classes" in by["zero-day"].note)
check("all four metric variants captured",
      set(by["aci_iot_2023"].variants) == set(BR.METRIC_VARIANTS),
      str(sorted(by["aci_iot_2023"].variants)))

# choosing macro instead flips the verdict â€” the whole point of the report
cm = {c.dataset: c for c in BR.compare_to_paper(res, metric="macro_f1")}
check("macro_f1 comparison flips ACI to 'below paper'",
      cm["aci_iot_2023"].verdict() == "below paper",
      f"{cm['aci_iot_2023'].our_value:.4f} vs 0.97")

# --- markdown rendering ----------------------------------------------------
md = BR.generate_report(res)
check("headline table rendered", "| Benchmark | Paper | Ours |".split("|")[1].strip()
      in md and "aci_iot_2023" in md)
check("variant table rendered", "scored four ways" in md and "`macro_f1`" in md)
check("averaging caveat present when unconfirmed",
      "averaging method is inferred" in md)
check("subsampling caveat names the classifier",
      "`svm` on aci_iot_2023" in md and "5,000 rows" in md)
check("per-dataset tables rendered without tabulate",
      md.count("| classifier |") == 2)
check("no NaN leaking into the markdown", "nan" not in md.lower())

conf = copy.deepcopy(BR.PAPER_BASELINE)
conf["averaging_confirmed"] = True
md2 = BR.generate_report(res, baseline=conf)
check("caveat disappears once averaging is confirmed",
      "averaging method is inferred" not in md2)

# --- failed classifier -----------------------------------------------------
with_err = pd.concat([aci_tbl, pd.DataFrame([{"classifier": "knn",
                                              "error": "ValueError: bad param"}])],
                     ignore_index=True)
md3 = BR.generate_report({"aci_iot_2023": with_err})
check("failed classifier surfaced in caveats",
      "failed to run" in md3 and "bad param" in md3)
c3 = BR.compare_to_paper({"aci_iot_2023": with_err})[0]
check("failed classifier excluded from 'best'", c3.our_classifier == "random_forest")

# --- writing to disk -------------------------------------------------------
out = "experiments/phase1_aci_iot/_verify_report/baseline.md"
BR.generate_report(res, out)
check("report written to disk", os.path.exists(out))
check("file matches returned text",
      open(out, encoding="utf-8").read() == BR.generate_report(res))
check("summarize_comparisons renders", "paper" in BR.summarize_comparisons(cmps))

try:
    BR.generate_report({"x": 42})
    check("bad input type raises", False)
except TypeError as e:
    check("bad input type raises", "DataFrame" in str(e))

# --- real end-to-end run with a zero-day holdout --------------------------
print("\n=== real run ===")
cfg = L.load_config()
mcfg = TE.load_models_config()
cfg["split"]["holdout_classes"] = ["Slowloris"]

aci = L.load_dataset(L.ACI, cfg, use_cache=False)
rng = np.random.default_rng(0)
keep = []
for cls, grp in aci.groupby("label", observed=True):
    n = min(len(grp), max(5, int(round(len(grp) / len(aci) * 30000))))
    keep.append(grp.iloc[rng.choice(len(grp), size=n, replace=False)])
small = pd.concat(keep); small.attrs["dataset"] = L.ACI

sp = P.prepare_splits(small, cfg, mcfg)
mt = copy.deepcopy(mcfg); mt["training"]["max_train_rows"] = {"svm": 5000, "knn": None}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    tbl = TE.run_all(L.ACI, cfg, mt, splits=sp,
                     classifiers=["random_forest", "decision_tree", "svm"],
                     verbose=False)

real = BR.generate_report({"aci_iot_2023": tbl},
                          "experiments/phase1_aci_iot/_verify_report/real.md")
cr = {c.dataset: c for c in BR.compare_to_paper({"aci_iot_2023": tbl})}
check("real run produces a zero-day comparison",
      cr["zero-day"].our_value is not None and cr["zero-day"].paper_value == 0.61)
check("zero-day classifier label includes the dataset",
      "on aci_iot_2023" in (cr["zero-day"].our_classifier or ""))
check("real report renders end to end", "Headline comparison" in real
      and "scored four ways" in real)
print("\n----- report excerpt -----")
print("\n".join(real.splitlines()[:28]))

import shutil
shutil.rmtree("experiments/phase1_aci_iot/_verify_report", ignore_errors=True)
print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
