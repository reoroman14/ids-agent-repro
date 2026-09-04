"""Verify memory store, session memory, retrieval + aggregation tools, end to end."""
import copy, os, sys, tempfile, warnings
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore")

from src.memory.store import MemoryStore, TfidfMemoryStore, build_store
from src.memory.session_memory import (CombinedMemory, CrossSessionMemory,
                                       SessionMemory, summarize_record)
from src.tools.base_tool import Provenance, ToolRegistry
from src.tools.retrieval_tool import RetrievalTool
from src.tools.aggregation_tool import AggregationTool
from src.tools.classification_tool import ClassificationTool
from src.agent import prompts as PR
from src.agent.llm_client import EchoClient
from src.agent.react_loop import ReactAgent
from src.data import loader as L
from src.data import preprocessing as P
from src.classifiers import train_eval as TE

ok = True
def check(n, c, d=""):
    global ok
    print(f"[{'PASS' if c else 'FAIL'}] {n}" + (f" â€” {d}" if d else ""))
    if not c: ok = False

# =========================== store =========================================
print("=== memory store ===")
s = build_store("tfidf")
check("build_store returns a MemoryStore", isinstance(s, TfidfMemoryStore))
ids = s.add(["SYN flood from 10.0.0.5 against port 80",
             "Port scan sweeping many destination ports",
             "Benign HTTPS session to a CDN"],
            [{"verdict": "SYN Flood"}, {"verdict": "Port Scan"}, {"verdict": "Benign"}])
check("add returns one id per text", len(ids) == 3 and len(set(ids)) == 3)
check("len reflects corpus", len(s) == 3)
hits = s.query("flood attack against port 80", k=2)
check("query ranks the relevant note first",
      hits and hits[0].metadata["verdict"] == "SYN Flood", hits[0].text if hits else "")
check("scores descend", len(hits) < 2 or hits[0].score >= hits[1].score)
check("query on an empty-ish term returns nothing", s.query("zzzz nonexistent", k=3) == [])
check("delete removes", s.delete([ids[0]]) == 1 and len(s) == 2)
check("delete of a missing id is a no-op", s.delete(["nope"]) == 0)
check("empty store queries safely", build_store("tfidf").query("anything") == [])

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "mem.json")
    s.save(path)
    s2 = build_store("tfidf"); n = s2.load(path)
    check("save/load round-trips", n == 2 and len(s2) == 2)
    check("ids survive the round trip", {r.id for r in s2.all_records()} == {ids[1], ids[2]})
    check("loaded store is queryable",
          bool(s2.query("scan across ports")))
check("clear empties", s.clear() == 2 and len(s) == 0)
try:
    build_store("chroma"); check("chroma raises NotImplementedError", False)
except NotImplementedError: check("chroma raises NotImplementedError", True)
try:
    build_store("pinecone"); check("unknown backend raises", False)
except ValueError: check("unknown backend raises", True)
try:
    from src.memory.store import FaissMemoryStore
    FaissMemoryStore(None); check("faiss without embed_fn raises", False)
except ValueError as e: check("faiss without embed_fn raises", "embed_fn" in str(e))

# =========================== session memory ================================
print("\n=== session vs cross-session ===")
sm = SessionMemory()
sm.remember("random_forest says Port Scan with 0.98")
sm.remember("decision_tree agrees")
check("session memory recalls its notes", "random_forest" in sm.recall({}))
check("session memory counts", len(sm) == 2)
sm.clear(); check("session memory clears", len(sm) == 0 and sm.recall({}) == "")
sm.remember("   "); check("empty notes ignored", len(sm) == 0)

cs = CrossSessionMemory(backend="tfidf")
rec = {"Src Port": 443, "Dst Port": 80, "Protocol": 6}
check("empty cross-session recalls nothing", cs.recall(rec) == "")
cs.remember_verdict(rec, "Port Scan", reasoning="Many ports touched.",
                    session_id="run-1")
check("verdict remembered", len(cs) == 1)
out = cs.recall(rec)
check("similar flow recalls the note", "Port Scan" in out, out)
check("recall reports similarity", "similarity" in out)
check("empty text is not stored", cs.remember("") is None and len(cs) == 1)

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "cross.json")
    cs.persist(path)
    revived = CrossSessionMemory(backend="tfidf", path=path)
    check("cross-session memory survives a restart",
          len(revived) == 1 and "Port Scan" in revived.recall(rec))
try:
    CrossSessionMemory(backend="tfidf").persist()
    check("persist without a path raises", False)
except ValueError: check("persist without a path raises", True)

combined = CombinedMemory(SessionMemory(), cs)
combined.session.remember("this run: two classifiers agreed")
text = combined.recall(rec)
check("combined memory labels both sources",
      "From this session:" in text and "From earlier sessions:" in text)
check("the two memories stay separate objects",
      len(combined.session) == 1 and len(combined.cross_session) == 1)

# =========================== retrieval tool ================================
print("\n=== retrieval tool ===")
rt = RetrievalTool(cs)
check("defaults to untrusted provenance", rt.provenance == Provenance.MEMORY)
r = rt.run(query="port scan")
check("retrieval succeeds", r.ok and r.data["n_hits"] >= 1)
check("result is untrusted", r.is_untrusted)
check("previous verdict surfaced in the text", "previous verdict" in r.rendered)
rt.set_flow(rec)
check("set_flow gives a default query", rt.run().ok)
check("no query and no flow -> failed result",
      not RetrievalTool(cs).run().ok)
check("no hits renders cleanly",
      "No stored notes matched" in rt.run(query="quantum teleportation").rendered)
check("k is clamped", len(rt.run(query="port scan", k=999).data["hits"]) <= 10)
check("junk k falls back", rt.run(query="port scan", k="lots").ok)
kt = RetrievalTool(cs, provenance=Provenance.KNOWLEDGE)
check("knowledge provenance is trusted", not kt.run(query="port scan").is_untrusted)

# =========================== aggregation tool ==============================
print("\n=== aggregation tool ===")
cfg = L.load_config(); mcfg = TE.load_models_config()
aci = L.load_dataset(L.ACI, cfg, use_cache=False)
rng = np.random.default_rng(0)
small = pd.concat([g.iloc[rng.choice(len(g), size=min(len(g),
                   max(5, int(round(len(g)/len(aci)*30000)))), replace=False)]
                   for _, g in aci.groupby("label", observed=True)])
small.attrs["dataset"] = L.ACI
sp = P.prepare_splits(small, cfg, mcfg)
models = {n: TE.train_classifier(n, sp.X_train, sp.y_train, mcfg[n], seed=42, n_jobs=-1)
          for n in ("random_forest", "decision_tree", "logistic_regression")}
hints = {"random_forest": 0.9995, "decision_tree": 0.9994, "logistic_regression": 0.9826}

agg = AggregationTool(models, sp.feature_names, sp.classes_, accuracy_hints=hints)
row = sp.X_test.iloc[0].to_dict(); truth = sp.classes_[sp.y_test[0]]
agg.set_flow(row)
a = agg.run()
check("aggregation succeeds", a.ok, a.error or "")
check("verdict is a known class", a.data["verdict"] in sp.classes_)
check("verdict is correct here", a.data["verdict"] == truth,
      f"{a.data['verdict']} vs {truth}")
check("all three classifiers voted", a.data["n_classifiers"] == 3)
check("agreement reported", 0 < a.data["agreement"] <= 1)
check("individual votes rendered", "individual votes:" in a.rendered)
check("weights come from accuracy hints",
      any(abs(m.get("weight", 0) - 0.9995) < 1e-9 for m in a.data["per_classifier"]))
for method in ("majority", "mean_score", "weighted_mean"):
    m = agg.run(method=method)
    check(f"method {method} works", m.ok and m.data["method"] == method)
check("unknown method -> failed result", not agg.run(method="magic").ok)
check("no flow -> failed result",
      not AggregationTool(models, sp.feature_names, sp.classes_).run().ok)

class Broken:
    classes_ = np.arange(len(sp.classes_))
    def predict_proba(self, X): raise RuntimeError("broken model")
mixed = AggregationTool({**models, "broken": Broken()}, sp.feature_names, sp.classes_)
mixed.set_flow(row)
mb = mixed.run()
check("one broken model does not sink the aggregate",
      mb.ok and mb.data["n_classifiers"] == 3
      and any("error" in m for m in mb.data["per_classifier"]))

# holdout-trained model: score columns misalign with the frozen class list
cfg_h = copy.deepcopy(cfg); cfg_h["split"]["holdout_classes"] = ["Slowloris"]
sph = P.prepare_splits(small, cfg_h, mcfg)
mh = TE.train_classifier("random_forest", sph.X_train, sph.y_train,
                         mcfg["random_forest"], seed=42, n_jobs=-1)
ah = AggregationTool({"rf": mh}, sph.feature_names, sph.classes_)
ah.set_flow(sph.X_test.iloc[0].to_dict())
rh = ah.run()
check("holdout-trained model aligned correctly",
      rh.ok and rh.data["verdict"] != "Slowloris", rh.data.get("verdict", rh.error))

# =========================== full Phase 1 agent ============================
print("\n=== full agent: all four tools + memory ===")
ct = ClassificationTool(models, sp.feature_names, sp.classes_, accuracy_hints=hints)
mem = CombinedMemory(SessionMemory(), cs)
tools = ToolRegistry([ct, agg, rt])
agent = ReactAgent(EchoClient([
    'THOUGHT: consult a model\nACTION: classify_flow\nACTION_INPUT: {"classifier": "random_forest"}',
    'THOUGHT: check prior context\nACTION: search_memory\nACTION_INPUT: {}',
    'THOUGHT: settle it\nACTION: aggregate_classifiers\nACTION_INPUT: {"method": "weighted_mean"}',
    f"VERDICT: {truth}\nCONFIDENCE: high\nREASONING: Classifiers agreed and memory concurred."]),
    tools, memory=mem, class_names=sp.classes_, max_steps=5)
t = agent.run(row)
check("four-step run completes", t.ok, t.error)
check("all three tools were used",
      t.tools_called == ["classify_flow", "search_memory", "aggregate_classifiers"],
      str(t.tools_called))
check("verdict correct and known", t.verdict == truth and t.verdict_is_known_class)
transcript = "\n".join(m["content"] for m in t.messages)
check("aggregate reasoning reached the context", "aggregate verdict:" in transcript)
check("memory content is fenced as untrusted",
      any("similarity" in s for s in PR.untrusted_spans(transcript)))
check("classifier output is NOT fenced",
      not any("aggregate verdict:" in s for s in PR.untrusted_spans(transcript)))
check("exposure measured", 0 < t.untrusted_fraction < 1,
      f"{t.untrusted_fraction:.3f}")
print("\n" + t.summary())

# memory written back, as a later session would see it
mem.cross_session.remember_verdict(row, t.verdict, reasoning=t.reasoning,
                                   session_id="run-2")
check("verdict written back to cross-session memory", len(cs) == 2)
later = CombinedMemory(SessionMemory(), cs).recall(row)
check("a later session recalls it", "From earlier sessions:" in later
      and t.verdict in later)

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
