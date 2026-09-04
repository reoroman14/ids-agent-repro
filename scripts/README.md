# Experiment runners

Each script regenerates a set of committed results under `experiments/`. They
resolve the repo root from their own location, so they run from any checkout:

```bash
python scripts/run_evasion.py
```

They import from `src/`, read the configs in `config/`, and (for the agent and
attack scripts) expect a local Ollama server with the model pulled — `llama3`
or `qwen2.5:14b`. Long runs write results to their output CSV incrementally, so
a killed run loses only the row in progress; `run_injection_scaled.py` resumes
from its CSV. Several honour environment overrides: `N_FLOWS`, `OLLAMA_MODEL`,
and `N_STAGE_B` for the memory-poisoning script.

The exact figures, dates and the reasoning behind each result are in
`PROGRESS.md` (section 9 is the consolidated summary).

## Phase 1 — reproduction

| Script | Regenerates |
|---|---|
| `run_aci_baseline.py` | `experiments/phase1_aci_iot/` — six-classifier ACI baseline |
| `run_cic_baseline.py` | `experiments/phase1_cic_iot/` — six-classifier CIC baseline |
| `benchmark_scaling.py` | classifier runtime-scaling estimates (pre-run timing) |
| `mlp_cap_experiment.py` | the `max_iter` cost/accuracy trade-off behind the MLP cap |

## Phase 1 — zero-day

| Script | Regenerates |
|---|---|
| `zero_day_experiment.py` | `experiments/zero_day/` — 27 leave-out runs, fixed-threshold recall |
| `zero_day_auc_sweep.py` | the same 27 runs re-scored as AUC |
| `run_tradeoff.py` | detection trade-off curves (recall vs benign false-alarm rate) |
| `zd_protocol.py` | behavioural-clustering holdout protocol + the rejected distance measure |

## Phase 1 — the agent

| Script | Regenerates |
|---|---|
| `run_agent_ollama.py` | `.../agent_run/` — agent vs classifiers on ACI (llama3) |
| `run_agent_cic.py` | `.../agent_run/` — agent vs classifiers on CIC (llama3) |
| `run_agent_cic_noagg.py` | `.../agent_run_no_aggregation/` — aggregation tool withheld |
| `run_agent_model_compare.py` | `.../agent_run_qwen25_14b/` — both configs, qwen2.5 14B |
| `run_agent_context.py` | `.../agent_run_with_context/` — human-readable flow context added |

## Phase 2 — attacks

| Script | Regenerates |
|---|---|
| `run_evasion.py` | `experiments/phase2_evasion/` — budgeted feature-space evasion |
| `run_injection.py` | `experiments/phase2_injection/` — injection pilot (12 flows, superseded) |
| `run_injection_scaled.py` | `experiments/phase2_injection_scaled/` — 300-run injection with CIs (resumable) |

## Phase 3 — memory poisoning

| Script | Regenerates |
|---|---|
| `run_memory_poisoning.py` | `experiments/phase3_memory_poisoning/` — retrieval (Stage A) + verdict impact (Stage B) |
| `run_tool_imitation.py` | `.../tool_imitation_verdicts.csv` — the combined retrieval + forged-tool note |

## Note on the simulated field

`run_injection*.py` and `run_tool_imitation.py` add a **simulated** HTTP
`User-Agent` field that neither dataset contains, to carry text payloads.
Values follow real protocol conventions. See the module docstring in
`src/attacks/injection.py` and PROGRESS.md — no injection result is a finding
about the published datasets.
