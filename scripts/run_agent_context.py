"""
Final Phase 1 investigation: does giving the agent the flow in words â€” protocol
names, flags, byte sizes â€” let it beat the classifiers it calls?

The classifiers consume a standardised vector. The agent has, until now, been
shown that same vector, in which "Protocol Type" is -0.31. This run additionally
gives it the raw values described in language. The tools still receive the
scaled record, so nothing about the classifiers changes and the comparison with
the previous qwen2.5 runs is like for like.
"""
import json, os, sys, time, warnings
from collections import Counter
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore")

from src.data import loader as L
from src.data import preprocessing as P
from src.data import feature_schema as fs
from src.classifiers import train_eval as TE
from src.tools.base_tool import ToolRegistry
from src.tools.classification_tool import ClassificationTool
from src.tools.aggregation_tool import AggregationTool
from src.tools.retrieval_tool import RetrievalTool
from src.memory.session_memory import CrossSessionMemory, SessionMemory, CombinedMemory
from src.agent.llm_client import build_llm_client, load_llm_config, LLMError
from src.agent.react_loop import ReactAgent

MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
N_FLOWS = int(os.environ.get("N_FLOWS", "68"))
MAX_STEPS = 5
OUT = "experiments/phase1_cic_iot/agent_run_with_context"
os.makedirs(OUT, exist_ok=True)

client = build_llm_client(load_llm_config(), model=MODEL)
print(f"model={MODEL}  (flow context ENABLED)", flush=True)
try:
    client.complete([{"role": "user", "content": "Reply with the single word: ready"}])
except LLMError as exc:
    print(f"FATAL: {exc}"); sys.exit(1)

cfg = L.load_config(); mcfg = TE.load_models_config()
cic = L.load_dataset(L.CIC, cfg, use_cache=True)
sp = P.prepare_splits(cic, cfg, mcfg)
# Same cleaning, before scaling â€” indices match the splits, so a test row's
# raw values are recoverable by index.
raw = P.clean(cic, cfg, dataset=L.CIC)

names = ["random_forest", "decision_tree", "knn"]
caps = mcfg["training"]["max_train_rows"]
models = {n: TE.train_classifier(n, sp.X_train, sp.y_train, mcfg[n], seed=42,
                                 n_jobs=mcfg["training"]["n_jobs"],
                                 max_train_rows=caps.get(n)) for n in names}
hints = {"random_forest": 0.7408, "decision_tree": 0.7021, "knn": 0.6888}
print("classifiers fitted", flush=True)

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
rng = np.random.default_rng(0)          # identical sample to every earlier run
per = max(1, N_FLOWS // len(present))
picks = []
for cls in present:
    idx = np.flatnonzero(test_lbl == cls)
    picks.extend(rng.choice(idx, size=min(per, len(idx)), replace=False).tolist())
picks = picks[:N_FLOWS]

print("\n--- example of the context the agent now receives ---")
print(fs.describe_flow(L.CIC, raw.loc[sp.X_test.index[picks[0]]].to_dict()))
print("-----------------------------------------------------\n", flush=True)


def run_config(with_aggregation: bool):
    tag = "with_aggregation" if with_aggregation else "no_aggregation"
    print(f"\n{'='*70}\n=== {MODEL} + flow context â€” {tag}\n{'='*70}", flush=True)
    rows, started = [], time.time()
    for n, i in enumerate(picks, 1):
        row = sp.X_test.iloc[i].to_dict()
        truth = test_lbl[i]
        context = fs.describe_flow(L.CIC, raw.loc[sp.X_test.index[i]].to_dict())

        ct = ClassificationTool(models, sp.feature_names, sp.classes_, accuracy_hints=hints)
        tools = [ct, RetrievalTool(cross)]
        if with_aggregation:
            tools.insert(1, AggregationTool(models, sp.feature_names, sp.classes_,
                                            accuracy_hints=hints))
        agent = ReactAgent(client, ToolRegistry(tools),
                           memory=CombinedMemory(SessionMemory(), cross),
                           class_names=sp.classes_, max_steps=MAX_STEPS)
        t0 = time.time()
        trace = agent.run(row, context=context)
        elapsed = time.time() - t0

        frame = sp.X_test.iloc[[i]]
        preds = {k: sp.classes_[m.predict(frame)[0]] for k, m in models.items()}
        hidden = AggregationTool(models, sp.feature_names, sp.classes_,
                                 accuracy_hints=hints)
        hidden.set_flow(row); a = hidden.run()
        agg = a.data["verdict"] if a.ok else ""

        rows.append({"config": tag, "i": int(i), "truth": truth,
                     "agent": trace.verdict, "agent_ok": trace.ok,
                     "known_class": trace.verdict_is_known_class,
                     **{f"clf_{k}": v for k, v in preds.items()},
                     "aggregate": agg, "steps": trace.n_steps,
                     "n_classify_calls": sum(1 for s in trace.steps
                                             if s.action == "classify_flow"),
                     "tools": ",".join(trace.tools_called),
                     "untrusted_fraction": round(trace.untrusted_fraction, 4),
                     "seconds": round(elapsed, 1), "error": trace.error})
        mark = "OK " if trace.verdict == truth else "MISS"
        print(f"[{n:2}/{len(picks)}] {mark} truth={truth:26} "
              f"agent={trace.verdict or '(none)':26} agg={agg:26} {elapsed:5.1f}s",
              flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{tag}.csv"), index=False)
        if n == 1:
            json.dump({"messages": trace.messages},
                      open(os.path.join(OUT, f"{tag}_transcript.json"), "w"), indent=2)

    df = pd.DataFrame(rows)
    print(f"\n--- {tag}: {len(df)} flows in {(time.time()-started)/60:.1f} min "
          f"({df['seconds'].mean():.1f}s/flow) ---")
    print(f"  completed {int(df['agent_ok'].sum())}/{len(df)}   "
          f"valid verdicts {int(df['known_class'].sum())}/{len(df)}   "
          f"mean steps {df['steps'].mean():.2f}   "
          f"untrusted {df['untrusted_fraction'].mean():.1%}")
    for k, col in (("agent", "agent"), ("aggregate", "aggregate"),
                   ("random_forest", "clf_random_forest"),
                   ("decision_tree", "clf_decision_tree"), ("knn", "clf_knn")):
        print(f"  {k:16} {float((df[col] == df['truth']).mean()):.3f}  "
              f"({int((df[col] == df['truth']).sum())}/{len(df)})")
    d = df[df['agent'] != df['aggregate']]
    print(f"  departed from the aggregate on {len(d)}/{len(df)} flows"
          + (f" -> agent right {int((d['agent'] == d['truth']).sum())}, "
             f"aggregate right {int((d['aggregate'] == d['truth']).sum())}, "
             f"both wrong {int(((d['agent'] != d['truth']) & (d['aggregate'] != d['truth'])).sum())}"
             if len(d) else ""))
    return df


both = pd.concat([run_config(True), run_config(False)], ignore_index=True)
both.to_csv(os.path.join(OUT, "both_configs.csv"), index=False)

print(f"\n{'='*78}\n=== FINAL PHASE 1 TABLE â€” agent accuracy on the same 68 CIC flows\n{'='*78}")
prior = {("with_aggregation", "llama3"): 0.574, ("no_aggregation", "llama3"): 0.500,
         ("with_aggregation", "qwen"): 0.588, ("no_aggregation", "qwen"): 0.574}
print(f"{'config':20} {'llama3':>8} {'qwen':>8} {'qwen+context':>14} {'aggregate':>11} {'departures':>12}")
for tag, g in both.groupby("config"):
    a = float((g['agent'] == g['truth']).mean())
    ag = float((g['aggregate'] == g['truth']).mean())
    dep = int((g['agent'] != g['aggregate']).sum())
    print(f"{tag:20} {prior[(tag,'llama3')]:8.3f} {prior[(tag,'qwen')]:8.3f} "
          f"{a:14.3f} {ag:11.3f} {dep:9}/{len(g)}")
print(f"\nartifacts -> {OUT}")
