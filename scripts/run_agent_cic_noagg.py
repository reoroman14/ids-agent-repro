"""
Same CIC evaluation, with the aggregation tool withheld.

The previous run showed the agent agreeing with `aggregate_classifiers` on 68
flows out of 68 â€” a pass-through. This removes that shortcut: the agent can
still call each classifier individually, but nothing will combine them for it.
Everything else (flows, seed, models, prompt) is unchanged, so the two runs are
directly comparable.

The aggregate is still COMPUTED for scoring; it is simply not offered as a tool.
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
OUT = "experiments/phase1_cic_iot/agent_run_no_aggregation"
os.makedirs(OUT, exist_ok=True)

llm_cfg = load_llm_config()
client = build_llm_client(llm_cfg)
print(f"model={llm_cfg[llm_cfg['backend']]['model']}  (aggregation tool WITHHELD)",
      flush=True)
try:
    client.complete([{"role": "user", "content": "Reply with the single word: ready"}])
except LLMError as exc:
    print(f"FATAL: {exc}"); sys.exit(1)

cfg = L.load_config(); mcfg = TE.load_models_config()
cic = L.load_dataset(L.CIC, cfg, use_cache=True)
sp = P.prepare_splits(cic, cfg, mcfg)
names = ["random_forest", "decision_tree", "knn"]
caps = mcfg["training"]["max_train_rows"]
models = {}
for n in names:
    models[n] = TE.train_classifier(n, sp.X_train, sp.y_train, mcfg[n], seed=42,
                                    n_jobs=mcfg["training"]["n_jobs"],
                                    max_train_rows=caps.get(n))
    print(f"  fitted {n}", flush=True)
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

test_lbl = np.array([sp.classes_[i] for i in sp.y_test])
present = sorted(set(test_lbl))
rng = np.random.default_rng(0)          # identical sample to the previous run
per = max(1, N_FLOWS // len(present))
picks = []
for cls in present:
    idx = np.flatnonzero(test_lbl == cls)
    picks.extend(rng.choice(idx, size=min(per, len(idx)), replace=False).tolist())
picks = picks[:N_FLOWS]
print(f"\nevaluating {len(picks)} flows across {len(present)} classes\n", flush=True)

rows, started = [], time.time()
for n, i in enumerate(picks, 1):
    row = sp.X_test.iloc[i].to_dict()
    truth = test_lbl[i]
    ct = ClassificationTool(models, sp.feature_names, sp.classes_, accuracy_hints=hints)
    rt = RetrievalTool(cross)
    # aggregation deliberately NOT registered
    agent = ReactAgent(client, ToolRegistry([ct, rt]),
                       memory=CombinedMemory(SessionMemory(), cross),
                       class_names=sp.classes_, max_steps=MAX_STEPS)
    t0 = time.time()
    trace = agent.run(row)
    elapsed = time.time() - t0

    frame = sp.X_test.iloc[[i]]
    preds = {k: sp.classes_[m.predict(frame)[0]] for k, m in models.items()}
    hidden = AggregationTool(models, sp.feature_names, sp.classes_,
                             accuracy_hints=hints)
    hidden.set_flow(row); a = hidden.run()
    agg = a.data["verdict"] if a.ok else ""

    # which classifiers did it actually consult?
    consulted = sorted({s.action_input.get("classifier", "(default)")
                        for s in trace.steps if s.action == "classify_flow"})
    rows.append({"i": int(i), "truth": truth, "agent": trace.verdict,
                 "agent_raw": trace.verdict_raw, "agent_ok": trace.ok,
                 "known_class": trace.verdict_is_known_class,
                 **{f"clf_{k}": v for k, v in preds.items()},
                 "hidden_aggregate": agg,
                 "n_classify_calls": sum(1 for s in trace.steps
                                         if s.action == "classify_flow"),
                 "classifiers_consulted": ",".join(consulted),
                 "steps": trace.n_steps, "tools": ",".join(trace.tools_called),
                 "untrusted_fraction": round(trace.untrusted_fraction, 4),
                 "seconds": round(elapsed, 1), "error": trace.error})
    mark = "OK " if trace.verdict == truth else "MISS"
    print(f"[{n:2}/{len(picks)}] {mark} truth={truth:26} agent={trace.verdict or '(none)':26} "
          f"agg={agg:26} calls={rows[-1]['n_classify_calls']} {elapsed:5.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "agent_vs_classifiers.csv"), index=False)
    if n == 1:
        json.dump({"messages": trace.messages},
                  open(os.path.join(OUT, "example_transcript.json"), "w"), indent=2)

df = pd.DataFrame(rows)
print(f"\n=== {len(df)} flows in {(time.time()-started)/60:.1f} min "
      f"({df['seconds'].mean():.1f}s/flow) ===")
print(f"runs completed         : {int(df['agent_ok'].sum())}/{len(df)}")
print(f"valid class verdicts   : {int(df['known_class'].sum())}/{len(df)}")
print(f"mean steps             : {df['steps'].mean():.2f}")
print(f"mean classify calls    : {df['n_classify_calls'].mean():.2f}")
print(f"distinct classifiers/flow: "
      f"{df['classifiers_consulted'].str.count(',').add(1).mean():.2f}")
c = Counter(t for r in df['tools'].fillna('') for t in r.split(',') if t)
print("tool usage:", dict(c.most_common()))

METHODS = {"agent": "agent", "hidden_aggregate": "hidden_aggregate",
           "random_forest": "clf_random_forest",
           "decision_tree": "clf_decision_tree", "knn": "clf_knn"}
print("\n=== ACCURACY ===")
acc = {}
for k, col in METHODS.items():
    acc[k] = float((df[col] == df['truth']).mean())
    print(f"  {k:18} {acc[k]:.3f}  ({int((df[col] == df['truth']).sum())}/{len(df)})")

print("\n=== does it still echo the (hidden) aggregate? ===")
for k, col in METHODS.items():
    if k == "agent":
        continue
    print(f"  agent == {k:18} {float((df['agent'] == df[col]).mean()):.1%}")

d = df[df['agent'] != df['hidden_aggregate']]
print(f"\nagent departed from the hidden aggregate on {len(d)}/{len(df)} flows")
if len(d):
    print(f"  agent right     : {int((d['agent'] == d['truth']).sum())}")
    print(f"  aggregate right : {int((d['hidden_aggregate'] == d['truth']).sum())}")
    print(f"  both wrong      : "
          f"{int(((d['agent'] != d['truth']) & (d['hidden_aggregate'] != d['truth'])).sum())}")
    print(d[["truth", "agent", "hidden_aggregate", "clf_random_forest",
             "clf_decision_tree", "clf_knn"]].to_string(index=False))
df.to_csv(os.path.join(OUT, "agent_vs_classifiers.csv"), index=False)
print(f"\nartifacts -> {OUT}")
