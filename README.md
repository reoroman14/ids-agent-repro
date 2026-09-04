# ids-agent-repro — LLM-Agent NIDS: Reproduction & Adversarial Studies

![Phase 1: Reproduction](https://img.shields.io/badge/Phase%201%20Reproduction-complete-brightgreen)
![Phase 2: Attacks](https://img.shields.io/badge/Phase%202%20Attacks-complete-brightgreen)
![Phase 3: Memory Poisoning](https://img.shields.io/badge/Phase%203%20Memory%20Poisoning-complete-brightgreen)

Simplified reproduction of IDS-Agent (Li, Xiang, Bastian, Song, Bo Li —
NeurIPS 2024 Workshop on Open-World Agents) as the base system for a
Q1 SCIE-targeted paper on adversarial robustness of LLM-based network
intrusion-detection agents. All three phases are complete: Phase 1 reproduces
the baseline, Phase 2 studies feature-space evasion and telemetry prompt
injection, and Phase 3 studies cross-session memory poisoning. See
`PROGRESS.md` for the full decision log and results.

## Phase 1 goal
Reproduce a baseline matching the original paper's reported numbers:
- ACI-IoT'23: F1 ~ 0.97
- CIC-IoT'23: F1 ~ 0.75
- Zero-day attack recall ~ 0.61

## Structure
- `config/` — dataset, model, and LLM backend configuration
- `data/` — raw / processed / split data (raw is gitignored)
- `src/data/` — loading, preprocessing, feature schema
- `src/classifiers/` — RF/KNN/LR/DT/MLP/SVM registry + train/eval
- `src/memory/` — vector store + session vs cross-session memory split
- `src/tools/` — agent tools (classification, retrieval, aggregation)
- `src/agent/` — manual ReAct loop, prompts, LLM client (Ollama/OpenAI)
- `src/evaluation/` — metrics + baseline comparison report
- `experiments/` — per-dataset run logs and results
- `tests/` — unit and end-to-end tests

## Notes for future phases
- `src/data/feature_schema.py` and `src/memory/session_memory.py`
  already establish structural boundaries needed for Phase 2
  (telemetry injection) and Phase 3 (cross-session memory poisoning).
  No refactor of the data/memory layers should be needed when those
  phases start.
- `src/agent/react_loop.py` keeps an explicit boundary between raw
  telemetry data and text inserted into the LLM prompt — this is the
  future injection point, do not collapse it.

## Datasets
Both are loaded through one interface, `load_dataset(name, config)`. They share
no column names, so each has its own entry in `src/data/feature_schema.py` and
models are trained per dataset. Every figure below was measured from the local
raw files.

| | ACI-IoT'23 | CIC-IoT'23 |
|---|---|---|
| Files | 1 | 63 (`Merged01..63.csv`, headers byte-identical) |
| Rows | 1,231,411 | 45,019,243 |
| Columns | 85 (83 numeric + `Label` + `Connection Type`) | 40 (39 numeric + `Label`) |
| Classes | 12 | 34 |
| Imbalance | 441,282 : 5 | 6,893,259 : 1,196 |

Labels are normalized to an uppercase lookup key and mapped through an explicit
table in `config/datasets.yaml`; an unmapped label raises rather than passing
through. Each load yields three label columns — `label` (fine-grained),
`label_group` (Benign / DoS / DDoS / Mirai / Recon / Spoofing / Web / Brute
Force) and `label_binary` — so the 34-, 8- and 2-class settings all come free.

### Loading decisions
- **CIC is streamed, never concatenated.** 45M rows would need ~14 GB as
  float64. The loader samples during a single pass, taking each class with
  probability `min(1, cap / count)` from the counts frozen in the config. With
  `per_class_cap: 50000` that returns ~1.288M rows across all 34 classes in
  ~215 s (254 MB, float32). Set the cap to `null` to stream all 45M.
- **Rare classes are capped, not dropped or oversampled.** The 10 CIC classes
  below the cap are taken whole. They are the Web/BruteForce/Backdoor classes
  the zero-day evaluation holds out, so dropping them would put the ~0.61
  recall target out of reach; synthesising them would contaminate Phase 2's
  threat model, whose premise is attacker-controlled telemetry.
- **ACI is not capped** — at 1.23M rows it has neither the memory nor the
  imbalance problem, and it keeps float64 because `Idle*` holds values near
  1.7e15.
- **Leakage is tagged, not assumed.** `Flow ID`, `Src IP`, `Dst IP` and
  `Timestamp` are `role: identifier` and never enter X — in a single-testbed
  capture `Src IP` alone nearly predicts the label. `Src Port` / `Dst Port` and
  the `Idle*` group are kept but marked `leakage_risk: high`; exclude them with
  `feature_columns(..., include_high_leakage=False)`.
- Sampling is deterministic given the seed and independent of `chunksize`.

### Known data quirks
- ACI `Idle Mean/Max/Min` hold absolute epoch-microsecond timestamps, not
  durations — a dataset defect, and capture-time correlated. Flagged `suspect`.
- 6 ACI columns are constant across the full file and are dropped. Four more
  look constant in a 300k-row head sample but are not; sample evidence is not
  enough here.
- 9 CIC rows (one at the end of each of files 42, 44, 46–52) are truncated.
  Verified against the source: the files are otherwise complete. Pandas pads
  short rows with NaN, so these surface as null labels and are dropped and
  counted, not silently skipped.
- Infinities are real (rate columns over a zero-length window): ACI
  `Flow Bytes/s` 915, `Flow Packets/s` 1,924; CIC `Rate`. Converted to NaN at
  load; median imputation is fit on train only.

## Setup
```bash
pip install -r requirements.txt
```
`pyarrow` is required for the `data/processed/*.parquet` cache. The cache is
keyed only by path — pass `rebuild_cache=True` after changing `per_class_cap`,
the seed, or a label map.

## Pipeline
One call from raw CSV to scored classifiers:

```python
from src.data import loader, preprocessing
from src.classifiers import train_eval

cfg   = loader.load_config()                       # config/datasets.yaml
mcfg  = train_eval.load_models_config()            # config/models.yaml
table = train_eval.run_all("aci_iot_2023", cfg, mcfg,
                           output_dir="experiments/phase1_aci_iot")
```

`run_all` loads, splits, fits and scores all six classifiers, and returns a
table sorted by macro F1. Or drive the stages directly:

```python
df     = loader.load_dataset("cic_iot_2023", cfg)  # streams 63 files
splits = preprocessing.prepare_splits(df, cfg, mcfg)
print(splits.report.summary())                     # what went where, and why
```

Every stage returns a report — `LoadReport`, `SplitReport`, and the results
table's `n_train_used` / `subsampled` columns — so sampling, routing and
capping are visible in the output rather than buried in the code.

## Split policy
Set in `config/datasets.yaml`; each class falls into exactly one bucket.

| Bucket | Rule | Routing |
|---|---|---|
| holdout | listed in `holdout_classes` | 100% to test, unseen in training |
| rare | fewer than `min_rows_for_split` (100) rows | per `rare_class_policy` (`train_only`) |
| normal | everything else | stratified 70/15/15 |

On ACI this catches `ARP Spoofing` (5 rows) — it trains but is never tested, so
**ACI trains on 12 classes and is scored on 11**. On CIC nothing is caught; the
rarest class, `Uploading Attack` at 1,196 rows, splits 837/180/179. A class that
is both held out and rare raises, rather than one policy silently winning.

## Classifiers
Six, behind one interface (`src/classifiers/registry.py`): Random Forest, KNN,
Logistic Regression, Decision Tree, MLP, SVM. Hyperparameters live in
`config/models.yaml`, where an unknown key raises instead of being ignored.

**Two are capped, and the cap is always reported.** Kernel SVC is
quadratic-to-cubic in training rows and single-threaded; KNN compares every test
row against every stored training row, with no useful index at 75 features.
Neither finishes on ~900k rows, so `training.max_train_rows` caps `svm` at 50k
and `knn` at 200k via stratified subsampling on the split seed. Every results
row carries `n_train_used` and a `subsampled` flag so a capped score is never
mistaken for a comparable one.

## Reading the scores
Averaging method matters more than usual here, because a train-only class
contributes a structural zero to the macro average. On a 30k ACI subsample the
same Random Forest scores:

| Measure | Value |
|---|---|
| Weighted F1 | 0.995 |
| Macro F1, all 12 classes | 0.827 |
| Macro F1, 11 scored classes | 0.902 |

The paper's ACI ≈ 0.97 is therefore almost certainly weighted or micro F1, not
macro. All three are reported side by side so the comparison cannot be made by
accident.

**Zero-day recall measures detection, not naming.** A model cannot emit a class
it never trained on, so a held-out class's own recall is 0 by construction.
`zero_day_recall` is the fraction of held-out rows flagged as *any* non-benign
class — which is what the paper's ≈ 0.61 describes.

## Status
**Phase 1 complete** — every component built and verified against the real
datasets. See `PROGRESS.md` for the full decision log, including two
corrections and one negative result.

Done:
- `config/datasets.yaml`, `config/models.yaml` — dataset, split, preprocessing
  and training policy; every count measured from the raw files.
- `src/data/feature_schema.py` — all 85 ACI and 40 CIC columns tagged with
  role, source, leakage risk, and the Phase 2 `attacker_controllable` flag.
- `src/data/loader.py` — streaming loaders, label normalization, sampling,
  parquet cache, `LoadReport`.
- `src/data/preprocessing.py` — cleaning, split policy, train-only fitting of
  imputer and scaler, `prepare_splits`.
- `src/classifiers/registry.py`, `train_eval.py`, `src/evaluation/metrics.py` —
  six classifiers, the sweep, and shared scoring.

Verified against real data throughout: 33 checks on the data layer, 62 on
preprocessing, 55 on the classifiers, 26 on the comparison report, 51 on the
agent foundation, 43 on the reasoning loop, 62 on memory and the remaining
tools. Full CIC load reads 45,019,243 rows and returns 1,287,968 in ~4 minutes
at 254 MB.

- `src/evaluation/` — shared scoring, the paper-comparison report, the zero-day
  trade-off curve and the behavioural holdout protocol.
- `src/agent/` — LLM client (Ollama / OpenAI / scripted), prompt templates, and
  the hand-written ReAct loop.
- `src/tools/` — classification, retrieval and aggregation, behind one
  interface with provenance on every result.
- `src/memory/` — note store plus the session / cross-session split.

Full baselines run: ACI **0.9995** weighted F1 (paper 0.97), CIC **0.7408**
(paper 0.75). Zero-day is reported as AUC per holdout rather than a single
figure — see `PROGRESS.md` for why.

Next:
- Point the agent at a real Ollama model; all agent testing so far uses the
  scripted backend that keeps tests deterministic.
- Establish an agent-vs-classifiers baseline.
- Phase 2 (telemetry injection) and Phase 3 (memory poisoning).
