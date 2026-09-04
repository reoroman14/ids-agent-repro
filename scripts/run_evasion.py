"""Phase 2, evasion: can a budget-limited attacker flip the detector?"""
import json, os, sys, time, warnings
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore")

from src.data import loader as L
from src.data import preprocessing as P
from src.data import feature_schema as fs
from src.classifiers import train_eval as TE
from src.attacks.base_attack import AttackBudget, summarise_results
from src.attacks.evasion import GreedyEvasion, RandomNoiseEvasion, build_scorer

OUT = "experiments/phase2_evasion"
os.makedirs(OUT, exist_ok=True)
N_FLOWS = int(os.environ.get("N_FLOWS", "200"))

ok = True
def check(n, c, d=""):
    global ok
    print(f"[{'PASS' if c else 'FAIL'}] {n}" + (f" â€” {d}" if d else ""))
    if not c: ok = False

print("=== attack surface ===")
for ds in (L.ACI, L.CIC):
    low = fs.attacker_controllable_columns(ds, max_cost="low")
    med = fs.attacker_controllable_columns(ds, max_cost="medium")
    high = fs.attacker_controllable_columns(ds, max_cost="high")
    print(f"  {ds}: {len(low)} low-cost, {len(med)} <=medium, {len(high)} <=high")
check("low is a subset of medium is a subset of high",
      set(fs.attacker_controllable_columns(L.CIC, max_cost="low"))
      <= set(fs.attacker_controllable_columns(L.CIC, max_cost="medium"))
      <= set(fs.attacker_controllable_columns(L.CIC, max_cost="high")))
check("SYN flags are high cost (definitional for a SYN flood)",
      fs.manipulation_cost(L.CIC, "syn_flag_number") == "high"
      and fs.manipulation_cost(L.CIC, "syn_count") == "high")
check("padding is low cost", fs.manipulation_cost(L.CIC, "Tot size") == "low")
check("protocol is high cost", fs.manipulation_cost(L.CIC, "Protocol Type") == "high")
check("Connection Type is not attacker-controllable (it is the testbed link)",
      not fs.schema(L.ACI)["Connection Type"]["attacker_controllable"])
check("identifiers never enter the surface",
      not ({"Src IP", "Flow ID"} & set(fs.attacker_controllable_columns(L.ACI,
                                                                       max_cost="high"))))
try:
    fs.attacker_controllable_columns(L.CIC, max_cost="free")
    check("bad max_cost raises", False)
except ValueError:
    check("bad max_cost raises", True)

print("\n=== fitting the detector ===", flush=True)
cfg = L.load_config(); mcfg = TE.load_models_config()
cic = L.load_dataset(L.CIC, cfg, use_cache=True)
sp = P.prepare_splits(cic, cfg, mcfg)
model = TE.train_classifier("random_forest", sp.X_train, sp.y_train,
                            mcfg["random_forest"], seed=42,
                            n_jobs=mcfg["training"]["n_jobs"])
scorer = build_scorer(model)
print(f"random_forest fitted on {len(sp.y_train):,} rows", flush=True)

# Attack only flows the detector currently gets RIGHT: evading a flow that was
# already misclassified is not an evasion.
test_lbl = np.array([sp.classes_[i] for i in sp.y_test])
preds = np.array([sp.classes_[i] for i in model.predict(sp.X_test)])
correct = np.flatnonzero((preds == test_lbl) & (test_lbl != "Benign"))
rng = np.random.default_rng(0)
picks = rng.choice(correct, size=min(N_FLOWS, len(correct)), replace=False)
print(f"attacking {len(picks)} correctly-classified attack flows "
      f"(of {len(correct):,} available)\n", flush=True)

rows = []
for cost in ("low", "medium"):
    for eps in (0.25, 0.5, 1.0):
        budget = AttackBudget(epsilon=eps, max_cost=cost, max_queries=120,
                              max_features=8)
        for Attack in (RandomNoiseEvasion, GreedyEvasion):
            atk = Attack(scorer, sp.classes_, sp.feature_names,
                         dataset=L.CIC, budget=budget, seed=42)
            t0 = time.time()
            results = [atk.attack(sp.X_test.iloc[i], test_lbl[i]) for i in picks]
            wins = [r for r in results if r.success]
            benign = [r for r in wins if r.evaded_to_benign]
            rows.append({
                "attack": atk.name, "max_cost": cost, "epsilon": eps,
                "n": len(results),
                "evasion_rate": len(wins) / len(results),
                "to_benign_rate": len(benign) / len(results),
                "median_features": float(np.median(
                    [r.n_features_changed for r in wins])) if wins else 0.0,
                "median_queries": float(np.median(
                    [r.queries_used for r in wins])) if wins else 0.0,
                "seconds": round(time.time() - t0, 1),
            })
            print(f"  cost<={cost:6} eps={eps:<5} {atk.name:13} "
                  f"evasion {rows[-1]['evasion_rate']:6.1%}  "
                  f"to-benign {rows[-1]['to_benign_rate']:6.1%}  "
                  f"({rows[-1]['seconds']}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "evasion_results.csv"), index=False)
print("\n=== EVASION SUMMARY ===")
print(df.to_string(index=False))

print("\n=== greedy vs random (does the search do any work?) ===")
for (cost, eps), g in df.groupby(["max_cost", "epsilon"]):
    r = float(g[g.attack == "random_noise"]["evasion_rate"].iloc[0])
    gr = float(g[g.attack == "greedy"]["evasion_rate"].iloc[0])
    print(f"  cost<={cost:6} eps={eps:<5} random {r:6.1%}  greedy {gr:6.1%}  "
          f"lift {gr - r:+.1%}")

check("greedy beats random at every setting",
      all(float(g[g.attack == 'greedy']['evasion_rate'].iloc[0])
          >= float(g[g.attack == 'random_noise']['evasion_rate'].iloc[0])
          for _, g in df.groupby(["max_cost", "epsilon"])))
check("evasion rises with the budget",
      df[df.attack == "greedy"].sort_values("epsilon")
        .groupby("max_cost")["evasion_rate"].apply(
            lambda s: bool((s.diff().dropna() >= -1e-9).all())).all())
print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
print(f"artifacts -> {OUT}")
