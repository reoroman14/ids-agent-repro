"""
Agent vs. classifiers on CIC-IoT-2023, where there is actually headroom:
the best classifier reaches 0.7408 weighted F1, so a difference can show.
"""
import json, os, sys, time, warnings
from collections import Counter
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore")

from src.data import loader as L
from src.data import preprocessing as P
from src.classifiers import train_eval as TE
from src.tools.base_tool import ToolRegistry
from src.tools.classification_tool import ClassificationTool
from src.tools.aggregation_tool import AggregationTool
from src.tools.retrieval_tool import RetrievalTool
from src.memory.session_memory import CrossSessionMemory, SessionMemory, CombinedMemory
from src.agent.llm_client import build_llm_client, load_llm_config, LLMError
from src.agent.react_loop import ReactAgent

N_FLOWS = int(os.environ.get("N_FLOWS", "68"))
MAX_STEPS = 5
OUT = "experiments/phase1_cic_iot/agent_run"
os.makedirs(OUT, exist_ok=True)

llm_cfg = load_llm_config()
client = build_llm_client(llm_cfg)
print(f"model={llm_cfg[llm_cfg['backend']]['model']}", flush=True)
t0 = time.time()
try:
    client.complete([{"role": "user", "content": "Reply with the single word: ready"}])
except LLMError as exc:
    print(f"FATAL: {exc}"); sys.exit(1)
print(f"warm-up {time.time()-t0:.1f}s\n", flush=True)

cfg = L.load_config(); mcfg = TE.load_models_config()
t0 = time.time()
cic = L.load_dataset(L.CIC, cfg, use_cache=True)
sp = P.prepare_splits(cic, cfg, mcfg)
print(f"loaded+split in {time.time()-t0:.0f}s  train={len(sp.y_train):,} "
      f"test={len(sp.y_test):,} classes={len(sp.classes_)}", flush=True)

# random_forest and decision_tree are the two strongest cheap models; knn adds a
# genuinely different inductive bias for ~0 fit cost. logistic_regression is
# skipped here only because it costs 6.6 min to fit and ranks below all three.
names = ["random_forest", "decision_tree", "knn"]
caps = mcfg["training"]["max_train_rows"]
models = {}
for n in names:
    t0 = time.time()
    models[n] = TE.train_classifier(n, sp.X_train, sp.y_train, mcfg[n], seed=42,
                                    n_jobs=mcfg["training"]["n_jobs"],
                                    max_train_rows=caps.get(n))
    print(f"  fitted {n} in {time.time()-t0:.0f}s", flush=True)
hints = {"random_forest": 0.7408, "decision_tree": 0.7021, "knn": 0.6888}

cross = CrossSessionMemory(backend="tfidf")
train_lbl = np.array([sp.classes_[i] for i in sp.y_train])
for cls in sorted(set(train_lbl)):
    mask = train_lbl == cls
    centroid = sp.X_train.loc[mask].mean(axis=0)
    top = centroid.abs().nlargest(4).index.tolist()
    desc = ", ".join(f"{f} {'high' if centroid[f] > 0 else 'low'}" for f in top)
    cross.remember(f"{cls}: flows of this type typically show {desc}.",
                   verdict=cls, kind="class_profile")
print(f"memory pre-loaded with {len(cross)} class profiles\n", flush=True)

test_lbl = np.array([sp.classes_[i] for i in sp.y_test])
present = sorted(set(test_lbl))
rng = np.random.default_rng(0)
per = max(1, N_FLOWS // len(present))
picks = []
for cls in present:
    idx = np.flatnonzero(test_lbl == cls)
    picks.extend(rng.choice(idx, size=min(per, len(idx)), replace=False).tolist())
picks = picks[:N_FLOWS]
print(f"evaluating {len(picks)} flows across {len(present)} classes\n", flush=True)

rows, started = [], time.time()
for n, i in enumerate(picks, 1):
    row = sp.X_test.iloc[i].to_dict()
    truth = test_lbl[i]
    ct = ClassificationTool(models, sp.feature_names, sp.classes_, accuracy_hints=hints)
    ag = AggregationTool(models, sp.feature_names, sp.classes_, accuracy_hints=hints)
    rt = RetrievalTool(cross)
    agent = ReactAgent(client, ToolRegistry([ct, ag, rt]),
                       memory=CombinedMemory(SessionMemory(), cross),
                       class_names=sp.classes_, max_steps=MAX_STEPS)
    t0 = time.time()
    trace = agent.run(row)
    elapsed = time.time() - t0

    frame = sp.X_test.iloc[[i]]
    preds = {k: sp.classes_[m.predict(frame)[0]] for k, m in models.items()}
    ag.set_flow(row); a = ag.run()
    agg = a.data["verdict"] if a.ok else ""

    rows.append({"i": int(i), "truth": truth, "agent": trace.verdict,
                 "agent_raw": trace.verdict_raw, "agent_ok": trace.ok,
                 "known_class": trace.verdict_is_known_class,
                 **{f"clf_{k}": v for k, v in preds.items()},
                 "aggregate": agg, "steps": trace.n_steps,
                 "tools": ",".join(trace.tools_called),
                 "untrusted_fraction": round(trace.untrusted_fraction, 4),
                 "seconds": round(elapsed, 1), "error": trace.error})
    mark = "OK " if trace.verdict == truth else "MISS"
    print(f"[{n:2}/{len(picks)}] {mark} truth={truth:26} agent={trace.verdict or '(none)':26} "
          f"rf={preds['random_forest']:26} {elapsed:5.1f}s"
          + ("" if trace.ok else f"  ERR {trace.error[:50]}"), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "agent_vs_classifiers.csv"), index=False)
    if n == 1:
        json.dump({"messages": trace.messages},
                  open(os.path.join(OUT, "example_transcript.json"), "w"), indent=2)

df = pd.DataFrame(rows)
print(f"\n=== {len(df)} flows in {(time.time()-started)/60:.1f} min "
      f"({df.seconds.mean():.1f}s/flow) ===")
print(f"runs completed            : {int(df.agent_ok.sum())}/{len(df)}")
print(f"verdicts a valid class    : {int(df.known_class.sum())}/{len(df)}")
print(f"mean steps                : {df.steps.mean():.2f}")
print(f"mean untrusted context    : {df.untrusted_fraction.mean():.1%}")
c = Counter(t for r in df.tools.fillna("") for t in r.split(",") if t)
print("tool usage:", dict(c.most_common()))

print("\n=== ACCURACY ON IDENTICAL FLOWS ===")
cols = [("agent", "agent"), ("aggregate", "aggregate")] + \
       [(k, f"clf_{k}") for k in models]
scores = {}
for label, col in cols:
    acc = float((df[col] == df.truth).mean())
    scores[label] = acc
    print(f"  {label:16} {acc:.3f}  ({int((df[col] == df.truth).sum())}/{len(df)})")

best_clf = max(models, key=lambda k: scores[k])
print(f"\nagent {scores['agent']:.3f} vs best single classifier "
      f"({best_clf}) {scores[best_clf]:.3f}  -> "
      f"{'AGENT BETTER' if scores['agent'] > scores[best_clf] else 'AGENT NOT BETTER'}")
print(f"agent {scores['agent']:.3f} vs aggregate {scores['aggregate']:.3f}  -> "
      f"{'AGENT BETTER' if scores['agent'] > scores['aggregate'] else 'AGENT NOT BETTER'}")

agree = float((df.agent == df.aggregate).mean())
print(f"\nagent agreed with the aggregate on {agree:.1%} of flows")
diff = df[df.agent != df.aggregate]
if len(diff):
    won = int(((diff.agent == diff.truth)).sum())
    lost = int(((diff.aggregate == diff.truth)).sum())
    print(f"where they differed ({len(diff)} flows): agent right {won}, aggregate right {lost}")

print("\nper-class (truth: agent correct / n):")
for cls, g in df.groupby("truth"):
    print(f"  {cls:26} {int((g.agent==g.truth).sum())}/{len(g)}"
          f"   agent said: {', '.join(sorted(set(g.agent.fillna('(none)'))))}")
df.to_csv(os.path.join(OUT, "agent_vs_classifiers.csv"), index=False)
print(f"\nartifacts -> {OUT}")
