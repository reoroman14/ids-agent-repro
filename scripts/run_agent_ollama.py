"""
Run the ReAct agent against a real Ollama model on ACI flows, and compare it
with the classifiers it has access to.

This is the paper's central claim under test: does the agent beat the
classifiers it calls? Every flow is scored three ways on identical rows â€”
agent verdict, best single classifier, and the deterministic aggregate â€” so
the comparison is like for like.
"""
import json, os, sys, time, warnings
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore")

from src.data import loader as L
from src.data import preprocessing as P
from src.classifiers import train_eval as TE
from src.classifiers import registry as R
from src.tools.base_tool import ToolRegistry
from src.tools.classification_tool import ClassificationTool
from src.tools.aggregation_tool import AggregationTool
from src.tools.retrieval_tool import RetrievalTool
from src.memory.session_memory import CrossSessionMemory, SessionMemory, CombinedMemory
from src.agent.llm_client import build_llm_client, load_llm_config, LLMError
from src.agent.react_loop import ReactAgent

N_FLOWS = int(os.environ.get("N_FLOWS", "20"))
MAX_STEPS = 5
OUT = "experiments/phase1_aci_iot/agent_run"
os.makedirs(OUT, exist_ok=True)

llm_cfg = load_llm_config()
print(f"backend={llm_cfg['backend']}  model={llm_cfg[llm_cfg['backend']]['model']}", flush=True)
client = build_llm_client(llm_cfg)

print("warming up the model (first call loads weights)...", flush=True)
t0 = time.time()
try:
    warm = client.complete([{"role": "user", "content": "Reply with the single word: ready"}])
except LLMError as exc:
    print(f"FATAL: {exc}"); sys.exit(1)
print(f"  warm-up {time.time()-t0:.1f}s -> {warm.text.strip()[:60]!r}\n", flush=True)

# ---- data and models ------------------------------------------------------
cfg = L.load_config(); mcfg = TE.load_models_config()
aci = L.load_dataset(L.ACI, cfg, use_cache=False)
rng = np.random.default_rng(0)
small = pd.concat([g.iloc[rng.choice(len(g), size=min(len(g),
                   max(5, int(round(len(g)/len(aci)*40000)))), replace=False)]
                   for _, g in aci.groupby("label", observed=True)])
small.attrs["dataset"] = L.ACI
sp = P.prepare_splits(small, cfg, mcfg)
print(f"train={len(sp.y_train):,}  test={len(sp.y_test):,}  classes={len(sp.classes_)}",
      flush=True)

names = ["random_forest", "decision_tree", "logistic_regression"]
models = {n: TE.train_classifier(n, sp.X_train, sp.y_train, mcfg[n], seed=42, n_jobs=-1)
          for n in names}
hints = {"random_forest": 0.9995, "decision_tree": 0.9994, "logistic_regression": 0.9826}

# ---- pre-populate cross-session memory with per-class descriptions ---------
# Derived from TRAIN only. Gives the retrieval tool something real to find.
cross = CrossSessionMemory(backend="tfidf")
train_lbl = np.array([sp.classes_[i] for i in sp.y_train])
for cls in sorted(set(train_lbl)):
    mask = train_lbl == cls
    centroid = sp.X_train.loc[mask].mean(axis=0)
    top = centroid.abs().nlargest(4).index.tolist()
    desc = ", ".join(f"{f} {'high' if centroid[f] > 0 else 'low'}" for f in top)
    cross.remember(f"{cls}: flows of this type typically show {desc}. "
                   f"Seen {int(mask.sum())} times in past analyses.",
                   verdict=cls, kind="class_profile")
print(f"memory pre-loaded with {len(cross)} class profiles\n", flush=True)

# ---- stratified sample of test flows --------------------------------------
test_lbl = np.array([sp.classes_[i] for i in sp.y_test])
present = sorted(set(test_lbl))
per = max(1, N_FLOWS // len(present))
picks = []
for cls in present:
    idx = np.flatnonzero(test_lbl == cls)
    picks.extend(rng.choice(idx, size=min(per, len(idx)), replace=False).tolist())
picks = picks[:N_FLOWS]
print(f"evaluating {len(picks)} flows across {len(present)} classes\n", flush=True)

# ---- run ------------------------------------------------------------------
rows = []
started = time.time()
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
    rf = sp.classes_[models["random_forest"].predict(frame)[0]]
    ag.set_flow(row); agg_res = ag.run()
    agg = agg_res.data["verdict"] if agg_res.ok else ""

    rec = {"i": int(i), "truth": truth, "agent": trace.verdict,
           "agent_ok": trace.ok, "known_class": trace.verdict_is_known_class,
           "rf": rf, "aggregate": agg, "steps": trace.n_steps,
           "tools": ",".join(trace.tools_called),
           "untrusted_fraction": round(trace.untrusted_fraction, 4),
           "seconds": round(elapsed, 1), "error": trace.error,
           "confidence": trace.confidence}
    rows.append(rec)
    mark = "OK " if trace.verdict == truth else "MISS"
    print(f"[{n:2}/{len(picks)}] {mark} truth={truth:20} agent={trace.verdict or '(none)':20} "
          f"rf={rf:20} steps={trace.n_steps} {elapsed:5.1f}s"
          + ("" if trace.ok else f"  ERROR: {trace.error[:60]}"), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "agent_vs_classifiers.csv"), index=False)
    if n == 1:
        json.dump({"messages": trace.messages,
                   "steps": [{"thought": s.thought, "action": s.action,
                              "action_input": s.action_input,
                              "observation": s.observation[:2000]} for s in trace.steps]},
                  open(os.path.join(OUT, "example_transcript.json"), "w"), indent=2)

# ---- results --------------------------------------------------------------
df = pd.DataFrame(rows)
total = time.time() - started
print(f"\n=== {len(df)} flows in {total/60:.1f} min "
      f"({df.seconds.mean():.1f}s/flow) ===")
print(f"agent runs that completed : {int(df.agent_ok.sum())}/{len(df)}")
print(f"verdicts that were a real class: {int(df.known_class.sum())}/{len(df)}")
print(f"mean steps                : {df.steps.mean():.2f}")
print(f"mean untrusted context    : {df.untrusted_fraction.mean():.1%}")
print("\ntool usage:")
from collections import Counter
c = Counter(t for row in df.tools.fillna("") for t in row.split(",") if t)
for k, v in c.most_common(): print(f"  {k}: {v}")

print("\n=== ACCURACY ON IDENTICAL FLOWS ===")
for label, col in (("agent", "agent"), ("random_forest", "rf"), ("aggregate", "aggregate")):
    acc = float((df[col] == df.truth).mean())
    print(f"  {label:14} {acc:.3f}  ({int((df[col] == df.truth).sum())}/{len(df)})")

print("\nper-class agent results:")
for cls, g in df.groupby("truth"):
    hit = int((g.agent == g.truth).sum())
    print(f"  {cls:22} {hit}/{len(g)}   agent said: "
          + ", ".join(sorted(set(g.agent.fillna('(none)')))))
if (~df.agent_ok).any():
    print("\nfailures:")
    for _, r in df[~df.agent_ok].iterrows():
        print(f"  flow {r.i}: {r.error[:120]}")
df.to_csv(os.path.join(OUT, "agent_vs_classifiers.csv"), index=False)
print(f"\nartifacts -> {OUT}")
