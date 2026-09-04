"""
Re-run the CIC agent evaluation with a stronger model, in both configurations.

llama3 8B scored 0.574 with the aggregation tool (a pure pass-through, zero
deviations) and 0.500 without it. This repeats both with a different model so
the negative result can be attributed either to the model or to the approach.

Identical flows, seed, models and prompts throughout â€” only the LLM changes.
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

MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
N_FLOWS = int(os.environ.get("N_FLOWS", "68"))
MAX_STEPS = 5
OUT = f"experiments/phase1_cic_iot/agent_run_{MODEL.replace(':', '_').replace('.', '')}"
os.makedirs(OUT, exist_ok=True)

# Override the model without touching the committed config default.
client = build_llm_client(load_llm_config(), model=MODEL)
print(f"model={MODEL}", flush=True)
t0 = time.time()
try:
    r = client.complete([{"role": "user", "content": "Reply with the single word: ready"}])
except LLMError as exc:
    print(f"FATAL: {exc}"); sys.exit(1)
print(f"warm-up {time.time()-t0:.1f}s -> {r.text.strip()[:40]!r}\n", flush=True)

cfg = L.load_config(); mcfg = TE.load_models_config()
cic = L.load_dataset(L.CIC, cfg, use_cache=True)
sp = P.prepare_splits(cic, cfg, mcfg)
names = ["random_forest", "decision_tree", "knn"]
caps = mcfg["training"]["max_train_rows"]
models = {n: TE.train_classifier(n, sp.X_train, sp.y_train, mcfg[n], seed=42,
                                 n_jobs=mcfg["training"]["n_jobs"],
                                 max_train_rows=caps.get(n)) for n in names}
hints = {"random_forest": 0.7408, "decision_tree": 0.7021, "knn": 0.6888}
print("classifiers fitted\n", flush=True)

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


def run_config(with_aggregation: bool):
    tag = "with_aggregation" if with_aggregation else "no_aggregation"
    print(f"\n{'='*70}\n=== {MODEL} â€” {tag} â€” {len(picks)} flows\n{'='*70}", flush=True)
    rows, started = [], time.time()
    for n, i in enumerate(picks, 1):
        row = sp.X_test.iloc[i].to_dict()
        truth = test_lbl[i]
        ct = ClassificationTool(models, sp.feature_names, sp.classes_,
                                accuracy_hints=hints)
        rt = RetrievalTool(cross)
        tools = [ct, rt]
        if with_aggregation:
            tools.insert(1, AggregationTool(models, sp.feature_names, sp.classes_,
                                            accuracy_hints=hints))
        agent = ReactAgent(client, ToolRegistry(tools),
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

        rows.append({"config": tag, "i": int(i), "truth": truth,
                     "agent": trace.verdict, "agent_raw": trace.verdict_raw,
                     "agent_ok": trace.ok, "known_class": trace.verdict_is_known_class,
                     **{f"clf_{k}": v for k, v in preds.items()},
                     "aggregate": agg,
                     "n_classify_calls": sum(1 for s in trace.steps
                                             if s.action == "classify_flow"),
                     "steps": trace.n_steps, "tools": ",".join(trace.tools_called),
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
          f"classify calls {df['n_classify_calls'].mean():.2f}")
    c = Counter(t for r in df['tools'].fillna('') for t in r.split(',') if t)
    print(f"  tools: {dict(c.most_common())}")
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

print(f"\n{'='*70}\n=== SUMMARY: {MODEL} vs llama3 ===\n{'='*70}")
print(f"{'config':20} {'agent':>8} {'aggregate':>10} {'departures':>11} {'llama3 agent':>13}")
prior = {"with_aggregation": 0.574, "no_aggregation": 0.500}
for tag, g in both.groupby("config"):
    a = float((g['agent'] == g['truth']).mean())
    ag = float((g['aggregate'] == g['truth']).mean())
    dep = int((g['agent'] != g['aggregate']).sum())
    print(f"{tag:20} {a:8.3f} {ag:10.3f} {dep:8}/{len(g)} {prior[tag]:13.3f}")
print(f"\nartifacts -> {OUT}")
