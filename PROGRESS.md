# Project Progress Log

A running record of what has been built in `ids-agent-repro`, what was decided
and why, and what comes next. Written in plain language so it can be picked up
cold, without reading the code first.

**Last updated:** 2026-09-04. **Status: all three phases complete. Experiments
frozen for paper drafting** — this log and the `experiments/` artifacts are the
source material for the Methodology and Results sections.

---

## 1. What this project is

A simplified reproduction of **IDS-Agent** (Li, Xiang, Bastian, Song, Bo Li —
NeurIPS 2024 Workshop on Open-World Agents): an LLM agent that detects network
intrusions by calling machine-learning classifiers as tools. This repository is
Phase 1 of a larger effort — get a faithful baseline working first, then use it
to study adversarial robustness of LLM-based intrusion-detection agents for a
Q1 SCIE-targeted paper.

The three numbers Phase 1 is trying to match, from the original paper:

| Benchmark | Paper's reported figure |
|---|---|
| ACI-IoT-2023 | F1 ≈ 0.97 |
| CIC-IoT-2023 | F1 ≈ 0.75 |
| Zero-day attack recall | ≈ 0.61 |

Later phases depend on structural choices made now: Phase 2 studies telemetry
injection (an attacker feeding manipulated data), Phase 3 studies memory
poisoning across sessions. The data layer was built with hooks for both so
neither phase needs a rewrite.

---

## 2. Current state at a glance

**All three phases are complete.** The full pipeline is built and verified
against the real datasets, both baselines are reproduced, the agent has been
run against a real language model, and the three adversarial studies (evasion,
injection, memory poisoning) are measured. No experiments remain open.

| Area | Status |
|---|---|
| Dataset config, feature schema, loaders | Done, verified |
| Cleaning, splitting, scaling | Done, verified |
| Six-classifier registry, training, evaluation, metrics | Done, verified |
| Full-scale ACI baseline run | Done — target exceeded (0.9995 vs 0.97) |
| Full-scale CIC baseline run | Done — target matched (0.7408 vs 0.75) |
| Zero-day experiment + trade-off curve | Done — not reproducible as a single number |
| Behavioural zero-day protocol | Done — clustering usable, distance measure rejected |
| Agent foundation, reasoning loop, tools, memory | Done, verified |
| Agent vs classifiers (llama3, qwen2.5) | Done — agent adds no significant accuracy |
| **Phase 1 (reproduction)** | **Complete** |
| **Phase 2 — evasion** | **Complete** |
| **Phase 2 — prompt injection (300-run, CIs)** | **Complete** |
| **Phase 3 — memory poisoning (3 stages)** | **Complete** |

---

## 3. What each file does

### Configuration

**`config/datasets.yaml`** — Everything about the two datasets: where the raw
files live, how many rows and columns to expect, the full mapping from raw
label text to clean class names, which coarse group each class belongs to, a
frozen class ordering, exact measured row counts per class, and the rules for
splitting data into train/validation/test. Every number in it was measured from
the actual files rather than copied from documentation.

**`config/models.yaml`** — Hyperparameters for the six classifiers, plus two
blocks added during implementation: `preprocessing` (which scaler to use) and
`training` (how many parallel jobs, which classifiers to run, and per-classifier
caps on training rows).

**`config/llm.yaml`** — Untouched so far; for the agent phase.

### Data layer (`src/data/`)

**`feature_schema.py`** — A catalogue of all 125 columns across both datasets
(85 in one, 40 in the other). For each column it records the data type, whether
it is a real feature or just an identifier, where the value comes from, how
risky it is to use (some columns would let a model "cheat"), and a flag marking
whether an attacker could control it — that last flag is currently false
everywhere and gets switched on in Phase 2 without any restructuring.

**`loader.py`** — Reads either dataset through one function so nothing
downstream has to care which one it is. Handles the awkward parts: merging 63
separate files, checking every file's columns match before trusting them,
converting messy raw labels into clean consistent ones, and sampling a
manageable subset of a 45-million-row dataset without introducing bias. Every
load produces a report saying exactly what was read, sampled and dropped.

**`preprocessing.py`** — Cleaning, splitting into train/validation/test,
converting text categories into numbers, and rescaling numeric features.
Enforces the rule that anything learned from the data (averages used to fill
gaps, scaling factors) is learned from the training portion only.

### Classifiers and evaluation (`src/classifiers/`, `src/evaluation/`)

**`registry.py`** — Builds the six classifiers (Random Forest, K-Nearest
Neighbours, Logistic Regression, Decision Tree, Neural Network, Support Vector
Machine) from configuration, gives them all the same interface so the agent can
call any of them interchangeably, and rejects typos in the configuration
instead of silently ignoring them.

**`train_eval.py`** — Trains and scores every classifier on a dataset and
returns a comparison table. Records how long each took and how many rows each
actually trained on.

**`metrics.py`** — All the scoring in one place, deliberately separate so that
classifier-only baselines and later full-agent runs are measured by identical
code and their numbers can be compared directly.

**`zero_day_protocol.py`** — Chooses which attack classes to withhold based on
how the traffic actually behaves rather than on the labels the dataset authors
assigned. Groups classes by similarity in the feature space, reports how
isolated any candidate holdout is, and includes a function whose only job is to
check honestly whether that isolation measure really predicts difficulty. It
does not, and the module says so.

**`baseline_report.py`** — Produces the side-by-side comparison against the
paper's published numbers as a Markdown report. It deliberately prints the same
predictions scored four different ways, because which averaging method is used
decides whether the baseline looks reproduced or not. It also lists its own
caveats automatically: which classifiers trained on a capped subset, which ones
failed, and the standing warning that the paper's averaging method is assumed
rather than known.

### The agent (`src/agent/`, `src/tools/`)

**`base_tool.py`** — The interface every agent tool implements, and the more
important part: a rule that a tool returns *two* things. Structured data, which
the code may compute with but which never reaches the language model, and
rendered text, which is the only thing allowed into the model's context. Each
result also records where its content came from, marking whether an attacker
could have influenced it.

This split exists because Phase 2 studies what happens when an attacker who
controls network traffic thereby controls part of what the model reads. That
experiment is only possible if there is one identifiable place where observed
data becomes prompt text. Letting the loop format raw data itself would smear
that boundary across the codebase.

**`prompts.py`** — Owns that single boundary. Attacker-influenceable content is
wrapped in distinctive markers before entering the prompt, and any markers
already inside the content are neutralised first, so a payload cannot close the
fence early and make the rest of itself look trustworthy. The system prompt
tells the model that fenced content is data to analyse and never instruction to
obey.

The fencing is deliberately *not* claimed to be a security control — a
determined injection will defeat it, and demonstrating that is Phase 2's job.
Its purpose is to make the boundary visible, so a successful attack can be
traced to a specific span of context rather than guessed at.

**`llm_client.py`** — One interface over a local Ollama server and the OpenAI
API, so the backend is a config change rather than a code change. Includes a
scripted backend that replays fixed replies, which makes the reasoning loop
testable with no model server and, more importantly, makes Phase 2 results
attributable to the injection rather than to random sampling.

**`classification_tool.py`** — Exposes the trained classifiers to the agent.
Models are fitted once outside the loop and the tool only predicts, so a prompt
can never trigger training. The model chooses which classifier to consult but
never which data to run it on. Results include each classifier's baseline
accuracy, so the agent can weigh a confident opinion from a weak model
appropriately.

**`react_loop.py`** — The reasoning loop. The agent is shown one network flow,
thinks, calls a tool, reads the result, and repeats until it commits to a
verdict. Written by hand rather than with an agent framework, because the whole
point is controlling exactly what enters the model's context at each step.

Two properties it is built to guarantee:

*Nothing crosses the boundary except through one function.* The loop never
formats tool output itself. Every observation goes through the single rendering
function that fences attacker-influenceable content. There is deliberately no
second route, so the injection surface stays one place.

*Nothing can crash a run.* A malformed reply is re-prompted, a hallucinated
tool name comes back as a correctable message, a tool that throws becomes an
observation, and a dead model server ends in a recorded failure. A study of how
systems behave under attack cannot have the harness fall over on the very
inputs it exists to measure.

Every run returns a full trace: the message transcript, each tool call, and the
**fraction of the context that an attacker could have influenced** — 24% in a
typical run. That last number is the headline exposure measure for Phase 2,
since an injection can only work through it.

**`retrieval_tool.py`** — Lets the agent search notes from earlier analyses.
Its results are marked untrusted by default, so they are fenced on the way into
the prompt. That default is the important part: anything recalled from a
persistent store might have been written by an earlier run that was itself
manipulated. A store holding only fixed reference material can be marked
trusted, but that has to be stated explicitly rather than assumed.

**`aggregation_tool.py`** — Runs every classifier on the flow and combines them
into one verdict, reporting how much they agreed and which ones dissented. It
re-runs the models itself rather than reading back what the agent said it saw
earlier, so the aggregate is reproducible from the flow alone and cannot be
steered by a prompt that misreports an earlier observation — which matters once
Phase 2 starts manipulating what the agent believes. Where the classifiers
disagree it says so, because an aggregate that hides a two-to-one split behind
a single confident label is worse than no aggregate.

### Memory (`src/memory/`)

**`store.py`** — The searchable note store, behind one interface so no agent
code touches a vector database directly. The default backend is built on
scikit-learn, which is already a dependency: Phase 1 stores at most a few
thousand short notes, a scale where a specialised index buys nothing, and
avoiding an embedding service keeps results reproducible from a fixed corpus
rather than from whatever an external API returns on the day. A FAISS backend
is available for larger corpora. Notes are saved as plain JSON rather than a
binary index, so a poisoned corpus can be read and diffed by a human.

**`session_memory.py`** — Keeps two kinds of memory deliberately apart. Notes
made while analysing the current flow are discarded at the end of the run and
can never affect a later one. Notes written to the persistent store can, and
that channel is the whole of Phase 3's threat model. Each persisted note
records which run wrote it and what verdict it came from, so a later experiment
can trace an influenced decision back to the note that caused it — without
that, a poisoning result cannot be attributed to anything.

A wrapper presents both to the agent through one call, labelling each recalled
line with which memory it came from, so the separation survives the
convenience.

---

## 4. What the data actually looks like

Measured directly, not taken from the dataset papers. Two claims in the
original task description turned out to be wrong and are corrected here.

| | ACI-IoT-2023 | CIC-IoT-2023 |
|---|---|---|
| Files | 1 | 63, with identical column layouts |
| Rows | 1,231,411 | 45,019,243 |
| Columns | 85 | **40** (not 41 as originally assumed) |
| Attack classes | 12 | 34 |
| Most common class | Port Scan, 441,282 | DDoS-ICMP Flood, 6,893,259 |
| Rarest class | **ARP Spoofing, 5 rows** | Uploading Attack, 1,196 rows |
| Imbalance | 88,000 to 1 | 5,764 to 1 |

The two corrections: CIC has 40 columns, not 41; and ACI's 85 columns include
a second text column (`Connection Type`, wired vs wireless) beyond the label,
which is a genuine feature and needed handling.

The single most consequential discovery: **ACI's `ARP Spoofing` class has only
5 rows in 1.2 million.** The original brief flagged rare classes as a CIC
problem, but the sharpest case is in ACI, and it shapes both the splitting
policy and how results must be reported.

### Data quirks found and handled

- **Nine truncated rows in CIC.** Nine of the 63 files end mid-number. The user
  verified against the source that the files are otherwise complete and that
  two unusually small files are genuinely small batches, not failed downloads.
  Worth knowing: the CSV reader pads short rows with blanks rather than
  rejecting them, so these arrive as rows with a missing label and are removed
  and counted, never silently skipped.
- **Broken idle-time columns in ACI.** Three columns that should hold
  durations actually hold raw clock timestamps. They are flagged as suspect,
  and because they correlate with *when* each attack was recorded they are also
  treated as a cheating risk.
- **Sampling is not enough to spot dead columns.** Looking at the first 300,000
  ACI rows suggested 10 columns were constant and useless. Reading all 1.2
  million showed only 6 truly are. The other 4 vary later in the file. Column
  decisions are made from full passes as a result.
- **Genuine infinities.** Rate columns divided by a zero-length time window
  produce infinity in both datasets. These are converted to blanks and filled
  in with training-set medians.

---

## 5. Decisions made, and why

### On loading the data

**The 45-million-row dataset is sampled while being read, not after.** Loading
it whole would need roughly 14 GB of memory. Instead the loader streams through
all 63 files once and decides row by row what to keep. Result: about 1.29
million rows, 254 MB, in around four minutes.

**Sampling is random rather than "take the first N".** The files are in
chronological capture order, so taking the first rows of each class would draw
most common classes entirely from the earliest files and bake in the recording
order. Each class is instead kept with a probability that thins the common
classes and keeps the rare ones whole.

**Common classes are capped at 50,000 rows; rare classes are kept entirely.**
Nothing is discarded and nothing is artificially manufactured. Discarding the
rare classes would make the zero-day experiment impossible, since those are
exactly the classes it holds back. Manufacturing synthetic rows would poison
Phase 2, whose whole premise is that an attacker controls what the data looks
like — invented rows would be indistinguishable from the attack being studied.
If the classes need rebalancing, that is done inside the classifier, not by
editing the data.

**Labels are translated through an explicit table, and unknown labels stop the
run.** Simply lower-casing text would work until it didn't. An explicit table
means a new or misspelled class raises an error instead of quietly becoming a
35th category nobody notices.

**Every load is checked against frozen expected counts** so that a changed or
re-downloaded file is noticed rather than silently altering results.

### On splitting the data

**Splits are 70/15/15 with a fixed seed**, and the ratios come from
configuration — changing to 80/10/10 needs no code change.

**Classes with fewer than 100 rows go entirely into training and are absent
from testing.** This exists for ACI's 5-row `ARP Spoofing` class. A normal
split would give it 3 training rows and 1 test row, which is unlearnable, and
a single test row makes its score pure noise. The consequence is stated openly:
ACI trains on 12 classes but is scored on 11.

**Asking for a class to be both held-out and rare raises an error.** The two
rules point in opposite directions (one says test-only, the other says
train-only). Guessing would silently produce a meaningless zero-day number, so
the code refuses instead.

**The zero-day mechanism was built now rather than later.** Classes can be
withheld from training while remaining in the test set. It is unused by default
but wired in, because the project's own notes say the data layer should not
need restructuring in later phases.

**Splits are always computed on the detailed class labels**, even when scoring
the simpler two-class or eight-group versions. Otherwise switching the scoring
target would silently change which rows are in which split and the runs would
not be comparable.

### On preparing features

**Columns that would let a model cheat are excluded.** Network addresses, the
flow identifier and the timestamp never reach the model: in a single-laboratory
recording, the attacking machine's address alone almost perfectly predicts the
answer, which would push the score to near-perfect and fake the reproduction.

**Port numbers are kept, but labelled high-risk** — a deliberate call, since
port behaviour is precisely what defines the scanning attacks. They can be
switched off with one argument.

**Gap-filling and rescaling are learned from training data only.** Verified
directly: the training set standardises to a mean of essentially zero while the
test set does not, which proves the test data was transformed using training
statistics rather than its own.

**Duplicate rows are not removed by default.** Flood attacks legitimately
produce identical-looking records, so removing duplicates would quietly undo
the class balance that was carefully chosen.

### On the classifiers

**Two classifiers are capped at fewer training rows, and it is stated in the
results.** Support Vector Machines get dramatically slower as data grows and
cannot finish on 900,000 rows; K-Nearest Neighbours must compare every test
record against every stored training record. Both are capped, but every row of
the results table shows how many rows that classifier actually used and flags
when a cap applied, so a capped result can never be mistaken for a comparable
one.

**A failing classifier does not abandon the run.** Its error is recorded and
the remaining classifiers continue — a multi-hour sweep should not be lost to
one model failing to converge.

**Configuration typos are rejected.** A misspelled setting raises an error
rather than being ignored and silently changing the model.

### On measuring results — the most important open issue

**The paper's ACI figure of 0.97 cannot be the "macro" average.** Running the
baseline showed a large gap between two legitimate ways of averaging:

| Measurement | Value |
|---|---|
| Weighted average F1 | 0.995 |
| Macro average F1 (all 12 classes) | 0.827 |
| Macro average F1 (11 scored classes) | 0.902 |

The gap comes from `ARP Spoofing` contributing a structural zero to the macro
average because it has no test rows. Since the paper reports ≈0.97, it is
almost certainly using a weighted or overall average. All three are now
reported side by side so the comparison cannot be made by accident.

**Zero-day recall measures detection, not naming.** A model cannot output a
class it never saw, so the held-out class's own recall is zero by definition
and tells you nothing. What is measured instead is whether held-out attack
traffic is flagged as *an attack at all* — which is what the paper's 0.61
describes.

---

## 5b. Phase 2: attacks

### Defining the attack surface

`attacker_controllable` was left False everywhere through Phase 1, as the
feature schema promised. Phase 2 turns it on, and the honest answer is that an
attacker who generates the traffic controls essentially every feature — these
are statistics of packets the attacker sent.

So the useful question is not whether a feature *can* be changed but what
changing it costs. A SYN flood that stops setting SYN flags has evaded
detection by ceasing to be a SYN flood. Each feature therefore carries a
`manipulation_cost`:

- **low** — free to change: padding, time-to-live, unused flag bits. 11 of
  CIC's 39 features.
- **medium** — degrades throughput or stealth: packet counts, rates,
  inter-arrival times. 21 cumulative.
- **high** — definitional; changing it abandons the attack: the transport
  protocol, the flags that constitute the flood. 39 cumulative.

Attacks are restricted to low or medium. Without that restriction an "evasion"
can simply be a different, harmless flow, which is how adversarial-ML results
get overstated.

### Evasion results (2026-09-03)

200 CIC flows that the Random Forest classified **correctly**, attacked
black-box with a query limit. Two attacks: a directed greedy search, and random
noise as the control. Artifacts in `experiments/phase2_evasion/`.

| Budget (sd) | Cost ceiling | Prediction changed | **Evaded to Benign** |
|---|---|---|---|
| 0.25 | low | 59% | **2.5%** |
| 0.50 | low | 65% | 3.0% |
| 1.00 | low | 76% | 1.5% |
| 0.25 | medium | 54% | 2.0% |
| 1.00 | medium | 67% | 0.5% |

**The two columns tell opposite stories, and only one of them matters.**

Nudging eight padding-and-TTL fields by a quarter of a standard deviation
changes the classifier's verdict 59% of the time. As a statement about the
stability of the 34-class label, that is damning.

But almost none of those changes reach *Benign* — between 0.5% and 3%. The
flows slide between neighbouring attack classes: `DDoS-TCP Flood` becomes
`DoS-TCP Flood`, which for an operator is the same alarm with a different name.
**The detector's fine-grained labelling is extremely brittle; its actual
detection is not.**

This is the behavioural-similarity finding arriving for a third time. The
classes that evasion slides between are the same near-duplicate pairs sitting
0.2 to 0.4 apart in feature space, and the same ones the agent and the
classifiers already confused. A tiny perturbation crosses a boundary that was
never meaningful. Reporting "59% evasion" as a security result would be
misleading; the honest headline is **2%**.

**An honest negative about the attack itself.** The directed greedy search does
not beat random noise — it is 2 to 8.5 points *worse* at every setting, and a
check written to assert the opposite failed. The reason is that the two
optimise different things: random perturbs eight features at once and stumbles
into a flip within about four queries, while greedy moves one coordinate at a
time and finds a flip using a median of **two** features out of eight allowed,
at the cost of 44 queries. Greedy is the stealthier attack per feature changed
and the weaker one per attempt. The comparison as originally framed was simply
the wrong question, and the random control is what revealed it — which is why
it was included.

One artifact worth noting: raising the cost ceiling from low to medium *lowers*
measured evasion. That is sampling, not robustness — the attack picks eight
features from a larger pool (21 rather than 11), so it is less likely to hit
the influential ones.

### Telemetry prompt injection (2026-09-03)

> **The attacker-controlled text field in this experiment is SIMULATED.**
> Neither ACI-IoT-2023 nor CIC-IoT-2023 contains any attacker-controlled
> string — both hold only numeric flow statistics, so on that data an attacker
> shapes values and never words, and text injection is impossible by
> construction. Real deployments show an analyst strings the attacker chose
> outright: the HTTP `User-Agent` and `Host`, the TLS `server_name`, the
> queried DNS name. Those fields are the standard injection vector and are
> absent from the benchmarks only because the benchmarks predate the threat.
> This experiment therefore **adds an HTTP `User-Agent` field** carrying the
> payload. The values follow real protocol conventions, but the field is an
> addition made for this experiment. **No result below is a finding about the
> published datasets.**

> **The 12-flow figures immediately below are a PILOT and are superseded by the
> 50-flow run that follows. Two of the pilot's headline claims did not
> replicate. Quote the scaled numbers, not these.**

12 CIC flows the pipeline classified correctly, each run six times — once
clean, once per payload. 72 agent runs, qwen2.5 14B, 83 minutes. Artifacts in
`experiments/phase2_injection/`.

| Payload | Technique | Verdict changed | **Reached "Benign"** | Broke the fence |
|---|---|---|---|---|
| none (control) | — | 0% | 0% | 0% |
| naive_command | direct instruction | 8.3% | 8.3% | 0% |
| **fake_system** | delimiter escape + role impersonation | **25.0%** | **25.0%** | 0% |
| fake_tool_output | tool-output forgery | 16.7% | 16.7% | 0% |
| authority_claim | social engineering, no instruction | 0% | 0% | 0% |
| confidence_undermining | tool distrust | 0% | 0% | 0% |

Baseline accuracy on these flows was 100%; under injection it fell to 90%. Six
of sixty injected runs turned a correctly detected attack into "Benign".

**The central result: the fence held mechanically and failed functionally.**

Not one payload escaped the delimiters. The neutralisation in
`fence_untrusted` worked exactly as designed — the delimiter-escape payload
could not close the fence early, and every payload's text was verifiably still
inside the untrusted markers when the model read it. By its own specification
the mechanism did not fail once.

And the most effective attack was the one that tried to escape anyway. The
`fake_system` payload succeeded a quarter of the time **while remaining
demonstrably inside the fence**. The model read text plainly marked as observed
network data, containing a forged `SYSTEM:` line, and acted on it as
instruction. It did not need to break out of the container; it needed only to
look authoritative inside it.

So **delimiter fencing prevents syntactic escape and does not prevent semantic
authority confusion.** Phase 1's docstring declined to call the fence a
security control and said a determined injection would defeat it. That is now
measured rather than asserted, and the distinction between the two failure
modes is the useful contribution: a defence can be perfectly correct and still
provide no protection.

**A counterintuitive secondary finding.** The two payloads carrying no
instruction at all — a plausible innocent explanation for the traffic, and a
claim that the classifiers are unreliable on this subnet — succeeded **zero**
times out of 24. The crude "ignore all previous instructions" managed 8.3%,
and the two structural impersonations managed 25% and 16.7%. This model is
vulnerable to **imitation of trusted structure**, not to persuasion. An
attacker's leverage comes from mimicking the format of a system message or a
tool observation, not from arguing a case. That points at a specific defence —
authenticate the provenance of anything that looks like a system or tool
message — rather than at better wording of the warning.

**Sample size.** Twelve flows per payload, so 25% is three occurrences. The
ordering of the payloads is suggestive, not established, and the write-up
should say so.

### Injection at scale — the definitive run (2026-09-04)

50 flows per payload, 300 agent runs, 190 minutes. **Temperature 0**, so each
flow-payload pair is deterministic and the binomial is cleanly over flows; the
pilot used default sampling, so the two are not directly comparable. Same
simulated HTTP `User-Agent` carrier. Artifacts in
`experiments/phase2_injection_scaled/`.

| Payload | Technique | Hits | Rate | 95% CI | vs control |
|---|---|---|---|---|---|
| none (control) | — | 0/50 | 0.0% | [0.000, 0.071] | — |
| naive_command | direct instruction | 2/50 | 4.0% | [0.011, 0.135] | p = 0.25 |
| fake_system | delimiter escape + role impersonation | 4/50 | 8.0% | [0.032, 0.188] | p = 0.0625 |
| **fake_tool_output** | tool-output forgery | **6/50** | **12.0%** | [0.056, 0.238] | **p = 0.0156** |
| authority_claim | social engineering, no instruction | 2/50 | 4.0% | [0.011, 0.135] | p = 0.25 |
| confidence_undermining | tool distrust | 0/50 | 0.0% | [0.000, 0.071] | p = 1.0 |

Overall injection rate **5.6%**, 95% CI [3.4%, 9.2%]. Agent accuracy falls from
**94.0% to 86.8%** under attack. **Zero fence escapes in 250 injected runs.**

**What is now established.** Forging a tool observation is the only payload
that beats its own control at conventional significance — 12%, paired exact
p = 0.0156. Injection through an attacker-controlled string field is real and
measurable, and the containment mechanism never once failed mechanically across
250 attempts while entirely failing to prevent it. The distinction between
syntactic containment and semantic authority, which the pilot suggested, holds
at scale and is the finding worth publishing.

**Two pilot claims that did NOT replicate, and must be corrected.**

*The 25% figure for a forged system message.* At scale `fake_system` measures
8.0%, and it is no longer the strongest payload — tool-output forgery is. The
pilot's 25% lies inside the scaled interval, so the two runs do not formally
contradict each other, but the point estimate was inflated roughly threefold by
a three-occurrence sample. **The pilot's 25% should not be quoted anywhere.**

*"Payloads carrying no instruction failed entirely."* This was stated on 0
successes in 24 attempts. At scale `authority_claim` — a plausible innocent
explanation carrying no command at all — succeeds 2 times in 50. The claim was
wrong. Persuasion is weaker than structural imitation but is not ineffective,
which matters because a defence aimed only at detecting imperative or
system-like text would miss it entirely.

**What remains unestablished.** The ranking among payloads. Every pairwise
comparison is non-significant, including `fake_system` against
`fake_tool_output` at p = 0.75. Only `fake_tool_output` against
`confidence_undermining` separates (p = 0.0312), which is the widest possible
gap in the set. Distinguishing 12% from 8% would need several hundred flows per
arm. **Report the payloads as a set that works to varying and unresolved
degrees, not as a ranked list.**

### Phase 3: memory poisoning (2026-09-04)

The channel is cross-session memory — the one place a note written during one
analysis can influence a later analysis of different traffic in a different
session. Three attacks, in `src/attacks/memory_poisoning.py`, increasing in the
access they assume:

- **verdict laundering** — no write access at all. The agent records its own
  verdicts, so an attacker who wins once with a Phase 2 injection has induced
  the system to persist a false verdict into its own memory. This is where
  Phase 2 and Phase 3 compose.
- **retrieval hijack** — assumes the attacker can store text, and targets
  *which* note is recalled by resembling every query.
- **payload note** — stores a Phase 2 injection string to replay it on future
  flows.

None of these needs a simulated dataset field; they exploit the memory
interface, and the stored payload text reuses the Phase 2 strings.

**Stage A — retrieval contamination (deterministic, 300 queries, no LLM).**

| Poison note | Retrieved | Ranked #1 |
|---|---|---|
| retrieval_hijack (feature vocabulary) | 100% | 100% |
| verdict_laundering (a single false verdict) | 100% | 100% |
| payload_note (forged tool text, no feature terms) | 0% | 0% |

For context, legitimate class-profile notes were also retrieved for 100% of
queries. That is the finding: **the retrieval tool is a weak relevance filter.**
The query built from a flow (`summarize_record`) is dominated by feature
*names* — `Header_Length`, `Rate`, and the rest — which every flow shares, so
any stored note containing those names matches every query. Relevance is
carried by shared vocabulary, not by the flow's actual values.

The consequence for the attacker is that **retrieval is not the bottleneck.**
The engineered hijack note is unnecessary; a plainly-worded false verdict
(`verdict_laundering`) is already recalled at rank 1 for essentially every
future flow, which is the cheapest possible attack — it needs no write access,
only a single Phase 2 success. The only note that failed, `payload_note`,
failed precisely because it contained no feature names, which confirms the
mechanism rather than contradicting it.

This is partly an artifact of how `summarize_record` builds its query, so it is
a finding about this implementation, not a universal claim — a retrieval query
weighted toward distinctive values rather than field names would contaminate
less easily, and that is a concrete defence to note.

**Stage B — does retrieved poison change the verdict?** 18 flows the pipeline
classifies correctly, each analysed twice: once with a clean memory, once with
a memory holding a benign-verdict hijack note that Stage A confirmed is
retrieved at rank 1 every time. qwen2.5 14B, temperature 0, 39 minutes.

| | Clean memory | Poisoned memory |
|---|---|---|
| Accuracy | 72.2% | 72.2% |
| Verdict changed | — | 1 of 18 |
| Flipped to Benign | — | **0 of 18** |

**The poison is delivered reliably and ignored almost entirely.** It reaches
the agent at the top of its recalled context on every single flow, and it moves
the verdict once in eighteen — to another attack class, never to Benign, so not
in the direction the attacker wanted. This is the mirror image of Stage A: the
retrieval channel is wide open, and the agent walks through it unmoved.

The reason connects directly to the Phase 1 finding. This agent treats memory
as weak evidence and anchors on its classifier tools — the same behaviour that
made it a pass-through in Phase 1, where it never departed from the aggregate.
A model that will not override its tools on the strength of a recalled note is,
by exactly that limitation, resistant to a note-based attack. The property that
made the agent add no accuracy is the property that makes it hard to poison.

**How Phase 2 and Phase 3 actually compare.** Injection reached the agent's
context in the current turn and flipped verdicts up to 12% of the time; a
poisoned memory reached the context from a past turn and flipped ~0%. The
difference is not the delivery — memory retrieval is if anything more reliable
than the injection carrier. The difference is that the injection payloads
imitated *trusted structure* (a system line, a tool observation), while a
recalled note is presented, correctly, as a low-authority past note. This is
the same lesson a third time: with this agent, an attack's success is
determined by the authority of the channel it imitates, not by whether it
reaches the context.

**Caveat.** 18 flows, so the ceiling on the flip rate is loose — the result
says "not obviously vulnerable", not "immune". And it is contingent on the
tool-anchoring behaviour of this particular model; an agent that weighted
memory more heavily, or a stronger note engineered to imitate a tool result
rather than a plain verdict, could plausibly do better. That last possibility
was then tested directly — see below.

**Stage C — the tool-imitating note (2026-09-04).** The natural next attack,
and the one the analysis pointed at. The two earlier memory notes each had only
one of the two properties that matter: the hijack/verdict notes were retrieved
but carried no authority, and the forged-tool `payload_note` carried authority
but was never retrieved (0%, for lack of feature vocabulary). `tool_imitation`
fuses them — feature vocabulary wrapped around a forged `aggregate_classifiers`
observation in the exact format the real tool emits. It is the memory-channel
version of Phase 2's one significant payload.

18 flows, three memory conditions each (clean / plain false verdict / tool
imitation), same flows, temperature 0.

| Memory condition | Retrieved at rank 1 | Flipped to Benign | 95% CI |
|---|---|---|---|
| plain false verdict | 100% | 0/18 | [0.00, 0.18] |
| tool imitation | 100% | 0/18 | [0.00, 0.18] |

**The prediction failed, and the failure is the result.** The note that should
have been dangerous — guaranteed delivery plus the authority form that worked
in Phase 2 — moved nothing, and did not beat the plain verdict note it was
designed to improve on (paired p = 1.0). Forged tool authority worked when
injected into the *current* turn's context and did not work when recalled from
*memory*, even reaching the model at rank 1 every time.

So the earlier synthesis was incomplete. It is not enough for an attack to
imitate a trusted channel; it has to arrive *through* one. The agent evidently
treats "a note recalled from past sessions" as a low-authority frame regardless
of what the note contains — a forged tool observation quoted inside a memory
entry is discounted the way the memory entry is, not trusted the way a live
tool result is. Authority is bound to the channel of delivery, not to the
surface form of the text. That is a more precise and more defensible claim than
Phase 3's first pass, and it is the one for the write-up.

The practical implication cuts the risk noted above: the wide-open retrieval
channel (Stage A) is less dangerous than it looked, because delivery is not
sufficient — an attacker also needs the content to inherit live-tool authority,
and routing it through memory strips exactly that. The remaining un-tested
worry is an agent that quotes recalled memory *into* a later tool call or
otherwise launders memory into a trusted channel; this agent does not, but that
is the mechanism a future design should be checked against.

## 6. What has been verified

Everything below ran against the real data, not mock inputs.

- **Data layer:** 33 checks. Full ACI load of 1,231,411 rows in 22 seconds.
  Full CIC load of all 63 files, 45,019,243 rows read, 1,287,968 returned, in
  around 4 minutes using 254 MB. All 34 classes preserved. Sampling proven to
  give identical results regardless of internal batch size.
- **Preprocessing:** 56 checks on ACI plus 6 on the full CIC. Confirmed the
  5-row class lands entirely in training and appears in neither validation nor
  test; splits never overlap and never lose a row; a held-out class appears
  only in the test set; contradictory and misspelled settings raise errors;
  changing the ratios in configuration is honoured.
- **Classifiers:** 55 checks. All six train and score end to end; capping is
  correctly flagged; a deliberately broken classifier is recorded without
  ending the run; results and per-class breakdowns are written to disk.

### The real ACI baseline (2026-09-02)

The first full-scale run: all six classifiers on the complete dataset —
861,989 training rows, 184,711 test rows, 75 features. Took 25 minutes.
Artifacts are in `experiments/phase1_aci_iot/`.

| Classifier | Trained on | Fit | Predict | Weighted F1 | Macro F1 (11 scored) | Macro F1 (all 12) |
|---|---|---|---|---|---|---|
| Random Forest | all | 95s | 1.6s | **0.9995** | 0.9886 | 0.9062 |
| Decision Tree | all | 23s | 0.1s | 0.9994 | 0.9869 | 0.9047 |
| Neural Network | all | 803s | 0.3s | 0.9951 | 0.9565 | 0.8768 |
| K-Nearest Neighbours | 200k cap | 0.1s | 85s | 0.9927 | 0.9244 | 0.8474 |
| Logistic Regression | all | 147s | 0.1s | 0.9826 | 0.9062 | 0.8307 |
| Support Vector Machine | 50k cap | 43s | 277s | 0.9737 | 0.8599 | 0.7883 |

**The paper's ACI figure of 0.97 is comfortably exceeded**, and under two of
the three defensible ways of averaging rather than only the most flattering
one: 0.9995 weighted, and 0.9886 when the macro average is taken over the 11
classes that actually have test data. Only the plain macro average over all 12
classes (0.9062) falls short, and that is entirely the structural zero from the
5-row class that is trained on but never tested.

Also worth noting: Random Forest's false-positive rate on benign traffic is
0.05%, and it flags 99.96% of attack traffic.

The best model is also nearly the cheapest. Decision Tree is within 0.0001 of
Random Forest at a quarter of the training time, while the Neural Network costs
eight times more than Random Forest to be measurably worse. Nothing so far
justifies the expensive models on this dataset.

### The real CIC baseline (2026-09-02)

All 63 files, 901,577 training rows, 193,196 test rows, 39 features, 34
classes. Took 49 minutes. Artifacts are in `experiments/phase1_cic_iot/`.

| Classifier | Trained on | Fit | Predict | Weighted F1 | Macro F1 | Benign false alarms |
|---|---|---|---|---|---|---|
| Random Forest | all | 154s | 5s | **0.7408** | 0.6351 | 31.9% |
| Neural Network | all | 1,159s | 0.2s | 0.7270 | 0.6175 | 33.2% |
| Decision Tree | all | 25s | 0.1s | 0.7021 | 0.5957 | 52.0% |
| K-Nearest Neighbours | 200k cap | 0.04s | 63s | 0.6888 | 0.5763 | 38.4% |
| Logistic Regression | all | 399s | 0.04s | 0.6644 | 0.5451 | 49.9% |
| Support Vector Machine | 50k cap | 29s | 871s | 0.6396 | 0.5243 | 39.7% |

**The paper's CIC figure of 0.75 is matched** at 0.7408 — a difference of
0.009, well within the 0.02 tolerance the comparison report uses. Unlike ACI,
every one of the 34 classes has test data here, so there is no gap between the
different ways of averaging to worry about.

Both headline numbers from the paper are therefore reproduced: ACI exceeded,
CIC matched.

### An unwelcome finding: false alarms on benign traffic

The best model wrongly flags **31.9% of benign traffic as an attack**, and the
worst flags 52%. An intrusion detection system that cries wolf on a third of
normal traffic would be unusable in practice, whatever its F1 score says.

The cause is structural rather than a bug. Benign is one class among 34, and
after capping it accounts for under 4% of the data, so a model is barely
penalised for never predicting it. The published F1 figures hide this
completely, which is worth remembering when quoting them.

This does not undermine the reproduction — we match the paper's number, and the
same effect is presumably present in the original — but it is a real weakness
of the benchmark and a natural thing for the later adversarial work to exploit
and discuss.

### The zero-day experiment (2026-09-02)

The third headline figure, and the one that did not reproduce. Results are in
`experiments/zero_day/`.

**Design.** A "zero-day" is an attack the system has never seen, so an attack
class is withheld from training entirely and kept only for testing. Success is
measured as whether the withheld traffic is flagged as *an attack at all* — a
model cannot name a category it never learned. Two definitions were tested,
because they turn out to mean very different things:

- **Leave one class out.** Withhold a single attack type, leaving its close
  relatives in training. Withholding one flood attack while eleven similar
  flood attacks remain is an easy test.
- **Leave one family out.** Withhold an entire category, so nothing similar
  remains. This is the honest version of "an attack nobody has seen".

27 runs in total, using Random Forest, across both datasets.

**Result: the number is not stable, and ranges across almost the entire scale.**

| Dataset | Regime | Mean | Median | Range |
|---|---|---|---|---|
| ACI | leave one class out | 0.189 | 0.0001 | 0.000 – 0.977 |
| ACI | leave one family out | 0.000 | 0.000 | 0.000 – 0.0001 |
| CIC | leave one class out | 0.912 | 0.908 | 0.799 – 1.000 |
| CIC | leave one family out | 0.860 | 0.833 | 0.569 – 1.000 |

The paper's 0.61 does not fall inside either dataset's range. On ACI almost
nothing is caught (8 of 10 withheld classes score below 0.05); on CIC almost
everything is (typically 1.000).

**Why, and the finding that matters.** Across all 27 runs, zero-day recall
correlates with the false-alarm rate on benign traffic at **r = 0.843**. The
two are largely the same measurement.

A model that is quick to shout "attack" catches unfamiliar attacks and also
misclassifies ordinary traffic. CIC's model raises false alarms on about a
third of benign traffic, so naturally it also flags unknown attacks — its
apparently excellent zero-day performance is mostly that bias, not
generalisation. ACI's model is well calibrated on benign traffic (0.05% false
alarms) and correspondingly files unfamiliar attacks under "benign", scoring
near zero.

The clearest single illustration is inside CIC itself: of its 14 runs, the one
with an unusually low false-alarm rate (8.9% instead of ~31%) is also the one
with the lowest zero-day recall (0.569 instead of ~1.000).

**The conclusion is that a zero-day recall figure quoted on its own is close to
meaningless.** It can be driven to 1.0 by a model that simply calls everything
an attack. It has to be reported together with the false-alarm rate, or as a
trade-off curve between them. This is a genuine methodological gap in the
original paper rather than a failure of this reproduction, and it is a natural
contribution for the write-up.

### The trade-off curve, and a correction to the above (2026-09-02)

Acting on the conclusion that a single recall figure is meaningless, the
detection trade-off curve was added (`zero_day_tradeoff` in
`src/evaluation/metrics.py`). It treats the withheld traffic as the thing to
detect and benign traffic as the thing not to false-alarm on, sweeping the
decision threshold instead of fixing it.

**It changes the conclusion for one of the two regimes, and the earlier
statement above needs qualifying.**

| Case | AUC | Recall at 1% false alarms | Cost of reaching 0.61 |
|---|---|---|---|
| ACI, one class withheld (Slowloris) | 0.998 | 0.974 | 0.1% false alarms |
| ACI, whole family withheld (DoS) | 0.594 | 0.065 | 51.5% false alarms |
| CIC, one class withheld (DDoS-ICMP Flood) | 0.996 | 0.999 | 0.4% false alarms |
| CIC, whole family withheld (Recon) | 0.854 | 0.127 | 9.7% false alarms |

The ACI single-class result is the correction. Measured at the model's default
decision rule, withheld Slowloris traffic was detected 2.3% of the time, and it
was reasonable to read that as the model being unable to recognise unfamiliar
attacks. The curve shows that reading was wrong: at an AUC of 0.998 the model
ranks almost every withheld Slowloris flow above almost every benign flow. The
information is present and nearly perfect. What fails is the decision rule —
picking the single highest-scoring class never lands on an attack label, so a
detector that is in fact excellent scores near zero.

**The fixed-threshold measurement was scoring the decision rule, not the
detector.** That is a stronger and more useful version of the earlier finding,
and it is worth stating carefully in the write-up because it is easy to
publish the weaker, wrong version.

What survives unchanged is the family result. With an entire attack category
withheld, ACI sits at an AUC of 0.594 — barely better than guessing, and
reaching the paper's 0.61 recall would mean false-alarming on **51.5%** of
benign traffic. There is no threshold that makes that model useful. So the
substantive claim stands in its sharper form: withholding one attack while its
relatives remain is an easy problem that current models genuinely solve;
withholding an entire family is one they do not.

This also gives a concrete reading of the paper's 0.61: it is cheap in the
easy regime (a fraction of a percent of false alarms) and unattainable in the
hard one. Which regime the paper used therefore matters more than the number.

### The full AUC sweep (2026-09-02)

All 27 runs re-measured with AUC alongside the fixed decision rule, on
identical models, so the two can be compared directly.

| Dataset | Regime | AUC (mean / median) | Fixed-rule recall (mean / median) |
|---|---|---|---|
| ACI | one class withheld | 0.925 / 0.981 | 0.189 / **0.0001** |
| ACI | whole family withheld | 0.811 / 0.853 | 0.000 / 0.000 |
| CIC | one class withheld | 0.926 / 0.925 | 0.912 / 0.908 |
| CIC | whole family withheld | 0.916 / 0.862 | 0.860 / 0.833 |

**The apparent gulf between the two datasets was almost entirely the decision
rule.** Measured by AUC, the detectors are close to equally capable —
0.925 against 0.926 in the single-class regime — while the fixed rule put them
at 0.189 against 0.912. The two measurements correlate at only r = 0.317
across the 27 runs, which is to say they are not measuring the same thing at
all. The starkest single case is ACI's DNS Flood: an AUC of 0.956 against a
fixed-rule recall of 0.0001.

So the story told yesterday — "ACI cannot detect novel attacks, CIC detects
them almost perfectly" — was wrong in both halves. Both models rank unfamiliar
attack traffic well above benign traffic. ACI's model simply never *names* an
attack, because with benign traffic well modelled the highest-scoring single
class is Benign; CIC's model names attacks constantly because it is biased
toward them, which is the same bias behind its 32% false-alarm rate.

**A second correction, to the family claim.** Withholding a whole family is not
inherently harder. What matters is whether anything *behaviourally* similar
survives, which the taxonomy labels do not track:

- ACI, DoS family withheld → AUC 0.594, near chance. That holdout removes
  every flooding attack, leaving only scans and a brute-force attack. Nothing
  resembling a flood remains.
- CIC, DDoS family withheld (12 classes, 550k rows) → AUC 0.998. The DoS and
  Mirai families are also floods and remain in training, so the model has
  plenty of near neighbours despite an entire taxonomic branch being gone.

The honest generalisation is therefore about **behavioural** novelty, not
taxonomic novelty: these models extrapolate well to a new variant of a
familiar behaviour and poorly to a genuinely unfamiliar one. Grouping by the
dataset's own attack-family labels is a bad proxy for that, and any zero-day
protocol built on those labels — including, plausibly, the original paper's —
will overstate performance.

### Redesigning the protocol by behaviour (2026-09-02)

`src/evaluation/zero_day_protocol.py`. Each class is represented by its
centroid in the scaled feature space; the distance between centroids stands in
for behavioural similarity; classes are clustered on that distance to give
**behavioural families** to withhold in place of the dataset's own labels.

**The clusters cut straight across the taxonomy, confirming the diagnosis.**
On CIC, one cluster contains eleven flood classes drawn from three different
taxonomic families (DDoS, DoS and Mirai), and another contains fourteen
low-rate classes spanning Recon, Web, Spoofing and Brute Force. The individual
distances are stark: `DDoS-TCP Flood` and `DoS-TCP Flood` sit 0.2 apart while
being filed under different families, and `DDoS-SYN Flood` is 0.3 from
`DDoS-SynonymousIP Flood`. Withholding one while keeping the other was never a
test of anything.

**The redesign finds harder, more honest holdouts on both datasets:**

| Dataset | Hardest taxonomic holdout | Hardest behavioural holdout |
|---|---|---|
| ACI | DoS family, AUC 0.594 | DNS Flood + UDP Flood, **AUC 0.483** |
| CIC | Recon family, AUC 0.854 | the 14 low-rate classes, **AUC 0.726** |

ACI's hardest case is now *below chance* at 0.483: withholding both UDP-based
floods leaves the model with no UDP flooding behaviour at all, and it ranks the
withheld traffic slightly *more* benign than actual benign traffic. CIC's
hardest case drops from 0.854 to 0.726, and its fixed-rule recall collapses
from 0.569 to 0.029.

**One cluster stayed easy, and the reason is instructive.** Withholding CIC's
eleven-class flood cluster still scores 0.989, because at five clusters the
floods split across four groups — the SYN floods, the HTTP floods and
`DDoS-RSTFIN Flood` all remain in training. Removing a taxonomic family does
not remove a behaviour, but neither does removing one behavioural cluster if
the clustering granularity has divided that behaviour in two. **The controlling
variable is whether the whole behavioural region is gone**, and cluster count
is what decides that. A protocol has to state its granularity.

**What did not work, stated plainly.** The plan was for the distance from the
withheld classes to the nearest surviving class to be a scalar difficulty dial.
It is not. Across all 36 holdouts it correlates with AUC at r = −0.26
(Spearman −0.31): the direction is right — more isolation means harder — but it
is far too weak to design against. Centroid distance is a poor proxy for what a
Random Forest actually uses, since the model makes axis-aligned splits and is
indifferent to the geometry that Euclidean distance measures, and a single
centroid is a bad summary of a class with several modes.

So the usable output of this work is the **clustering**, not the scalar. Group
classes by behaviour, withhold whole behavioural regions, and report AUC per
holdout. Do not quote a single zero-day number, and do not claim the distance
predicts difficulty — it was measured, and it does not.
(`validate_distance_predicts_difficulty` exists to keep that check honest, and
this is the negative result it was written to catch.)

### Earlier zero-day details

Two supporting details. On ACI, the only two withheld classes that were caught
(Port Scan at 0.86, Vulnerability Scan at 0.98) are both reconnaissance
attacks with a large surviving sibling — direct evidence of the
easy-versus-hard distinction above. And as an internal check on the harness,
the one attack family with a single member produces identical numbers under
both regimes (0.798994), exactly as it must.

---

## 7. What is next

**Experiments are frozen. The next step is writing, not running.** All three
phases are measured to a defensible conclusion, and this log plus the
`experiments/` artifacts are the source material for the paper's Methodology
and Results. Section 9 below is the consolidated headline summary written for
that purpose. The detailed per-experiment records follow in the rest of this
section and in section 5b.

Deliberately-not-pursued threads, recorded so the paper can cite them as
scoped-out limitations rather than oversights:

- **Zero-day clustering granularity.** How many behavioural clusters a dataset
  is cut into determines whether a holdout removes a whole behaviour or half of
  one. Sweeping cluster count and reporting AUC against it would replace a
  judgement call with a curve. Not run.
- **A stronger zero-day difficulty measure.** Centroid distance was measured
  and found too weak to design against (r = -0.26). Pairwise separability or
  reassignment tracking are the candidates. Not attempted.
- **Larger injection / poisoning samples.** Injection has 300 runs with CIs;
  the memory-poisoning verdict stages have 18 flows, enough to establish
  "not obviously vulnerable" but not to place a tight bound. Scaling them, and
  running the agent studies against a hosted frontier model, are the obvious
  extensions.
- **Memory-to-tool laundering.** Phase 3 showed recalled memory carries no
  live-tool authority for this agent. The one untested route is an agent that
  quotes recalled memory *into* a tool call; the current agent does not, but a
  future design should be checked against it.

### The live agent run (2026-09-02)

The agent analysed 30 ACI flows using llama3 through Ollama, with all four
tools and memory available. Artifacts in
`experiments/phase1_aci_iot/agent_run/`, including a full transcript.

| | Result |
|---|---|
| Runs that completed | 30 / 30 |
| Verdicts that were a valid class | 30 / 30 |
| Agent accuracy | 30 / 30 |
| Best single classifier | 30 / 30 |
| Deterministic aggregate | 30 / 30 |
| Time | 16 seconds per flow |
| Reasoning steps used | 4.6 of 5 available |
| Context that was attacker-influenceable | 25.3% |

The agent genuinely used its tools rather than guessing: 74 individual
classifier calls, 27 aggregations and 5 memory searches across the 30 flows —
between two and three model consultations per decision, which is the behaviour
the system prompt asks for.

**The result proves the plumbing, not the paper's claim.** All three methods
scored perfectly, so this sample has no power to distinguish them. ACI is
simply too easy — the best classifier reaches 0.9995 on the full test set, so a
thirty-flow sample was never going to separate anything. Whether the agent adds
value over the classifiers it calls is still unmeasured, and CIC is where it
should be measured.

**Two real defects surfaced, both fixed.**

*The model server rejected every request.* llama3 has no tool-calling template,
and Ollama refuses any request carrying a `tools` payload with a 400. The
reasoning loop never needed native tool calls — it reads the action out of the
model's text — so the payload was an optimisation that made the whole thing
fail. The client now drops it and continues when a model rejects it. The
related fix matters more: the original error reported only "400 Bad Request"
and discarded the server's message, which said exactly what was wrong. It now
includes the response body, turning a debugging session into a one-line
diagnosis.

*Correct answers were scored as wrong.* In the first run the model replied
`Slowloris (attack)` and `Port Scan (attack)`. Those are right, but strict
string matching rejected them, and the agent's accuracy read 0.85 against the
classifiers' 0.95. The verdicts are now canonicalised against the known class
list — case-insensitively, with a single trailing parenthetical removed, and
deliberately no fuzzy matching, since substring matching would let "not a Port
Scan" count as "Port Scan". With that fixed the agent scores 30 out of 30.

This is the third time in this project that a measurement has been wrong
because it scored an output format rather than the underlying judgement — after
argmax-versus-AUC on zero-day detection, and macro-versus-weighted F1 on ACI.
It is worth naming as a recurring hazard in the write-up.

### Agent versus classifiers on CIC (2026-09-02)

The test ACI could not provide. 68 flows, two per class across all 34 classes,
scored against the classifiers the agent itself calls. Artifacts in
`experiments/phase1_cic_iot/agent_run/`.

| Method | Accuracy |
|---|---|
| Agent (llama3) | 0.574 (39/68) |
| Deterministic aggregate | 0.574 (39/68) |
| Random Forest | 0.574 (39/68) |
| Decision Tree | 0.574 (39/68) |
| K-Nearest Neighbours | 0.544 (37/68) |

**The agent agreed with the aggregation tool on 68 flows out of 68.** It never
once departed from it. Its errors are the aggregate's errors, and its two
differences from Random Forest are simply the two places the aggregate differs
from Random Forest.

**So the paper's central claim does not reproduce here.** The agent adds no
accuracy over deterministically combining the same tools, and costs 16.6
seconds and roughly two and a half model calls per flow to do it, against
milliseconds for the aggregate. On this evidence the language model is acting
as an expensive pass-through: it calls its tools diligently — 161 classifier
calls and 71 aggregations across 68 flows — and then restates what they said.

Read the number carefully in two ways. The sample is small, so an accuracy
*difference* of less than roughly ten points could not have been detected. But
the pass-through finding does not depend on that: zero deviations in 68 flows
is strong evidence on its own, and an agent that never disagrees with its
aggregator cannot outscore it however long the run.

The 0.574 also looks low against the 0.747 from the full CIC baseline. That is
the sampling, not a regression: two flows per class is a balanced sample, so it
reports something close to balanced accuracy (0.627 for Random Forest on the
full test set) rather than accuracy on CIC's natural, flood-dominated
distribution.

**The errors corroborate the behavioural-similarity finding.** Almost every
mistake is a confusion between a near-duplicate pair the clustering had already
flagged: `DDoS-SYN Flood` against `DDoS-SynonymousIP Flood`, `DoS-HTTP` against
`DDoS-HTTP`, `DoS-TCP` against `DDoS-TCP`, and the reconnaissance classes among
themselves. These are exactly the pairs sitting 0.2 to 0.4 apart in the feature
space. Neither the classifiers nor the language model can separate attacks that
the data does not actually distinguish, which is a property of the benchmark
rather than of either method.

Both benign flows in the sample were called attacks — a small number, but
pointing the same way as the 32% benign false-alarm rate from the baseline.

### Withholding the aggregation tool (2026-09-02)

The obvious explanation for a pass-through is that the agent was handed a
finished answer and had nothing left to do. So the aggregation tool was removed
and the same 68 flows re-run: the agent could still consult each classifier
individually, but nothing would combine them for it. The aggregate was still
computed behind the scenes for scoring.

**It got worse.**

| Method | With aggregation offered | With it withheld |
|---|---|---|
| Agent | 0.574 | **0.500** |
| Aggregate (hidden in the second run) | 0.574 | 0.574 |
| Random Forest | 0.574 | 0.574 |

The agent did start reasoning independently: it consulted 2.7 classifiers per
flow and departed from the aggregate on 24 of 68 decisions, against zero
departures before. But of those 24 independent judgements, **it was right once
and the aggregate was right six times** (both were wrong on the remaining 17).

So the two runs together characterise the behaviour precisely. Given a
combined answer, the model repeats it and adds nothing. Denied one, it forms
its own view and is worse for it — its independent judgement is right about one
time in seven when it disagrees with simple weighted voting. The failure is
therefore not that the agent was idle; it is that its reasoning over classifier
outputs is worse than arithmetic on the same outputs.

**Two parser defects were found and fixed along the way**, and the first
version of this result was confounded by them. llama3 sometimes ends a turn
with `ACTION: None`, which the loop treated as a call to a tool named "None",
wasting one of five steps — that hit 13 of 68 flows, whose accuracy was 0.462
against 0.545 on unaffected flows. It also occasionally repeated a label,
producing the verdict `VERDICT: DNS Spoofing`. Both are now handled, which cut
junk actions from 13 to 2. **The corrected run scored 0.500, slightly below the
confounded 0.529**, so the fixes did not rescue the conclusion — they just made
it clean.

**Limitation worth stating plainly.** This is llama3 8B running locally. A
stronger model might well reason better over classifier disagreement, and the
finding should be reported as "this agent, with this model" rather than as a
general claim about LLM agents. That check was then run — see below.

### A stronger model: qwen2.5 14B (2026-09-03)

Both configurations repeated with a model roughly twice the size, on the same
68 flows with everything else unchanged.

| Model | Aggregation tool | Agent | Aggregate | Departures from the aggregate |
|---|---|---|---|---|
| llama3 8B | offered | 0.574 | 0.574 | 0 of 68 |
| llama3 8B | withheld | 0.500 | 0.574 | 24 of 68 — 1 win, 6 losses |
| **qwen2.5 14B** | offered | **0.588** | 0.574 | 2 of 68 — 1 win, 0 losses |
| **qwen2.5 14B** | withheld | **0.574** | 0.574 | 13 of 68 — 2 wins, 2 losses |

**The stronger model removes the harm but adds no benefit.** Denied the
aggregation tool, llama3 fell to 0.500 while qwen2.5 holds 0.574, exactly level
with the arithmetic. Offered the tool, qwen2.5 reaches 0.588 — one flow better
out of 68, which is noise.

The change in behaviour is real even though the score barely moves. llama3's
independent judgement was actively bad: 24 departures, winning 1 and losing 6.
qwen2.5 departs more sparingly and breaks even, 2 wins against 2 losses. So a
better model's disagreements become a coin flip rather than a liability. They
do not become an improvement.

**This answers the question the previous entry left open.** The negative result
is not an artifact of using a weak model. Doubling the parameters and
quadrupling the cost per flow — 77 seconds against 17 — buys a single flow that
is indistinguishable from chance. **The paper's claim that the agent beats the
classifiers it calls does not reproduce with either model.**

### Giving the agent flow context — the final Phase 1 experiment (2026-09-03)

The remaining hypothesis was that the agent had no information the classifiers
lacked. It was being shown the same standardised vector they consume, in which
`Protocol Type` reads as -0.31 — the right representation for a Random Forest
and a useless one for a language model.

`describe_flow` in `src/data/feature_schema.py` now renders the **raw** values
in words. The tools still receive the scaled record, so the classifiers are
untouched and the comparison stays clean; only the prompt changes:

```
Transport protocol: TCP (number 6).
Rate: 220 packets/second, 10 packets observed, mean inter-arrival 0.00457 s.
Packet size: 66-4350 bytes, mean 1351.2, std 1250.3.
TCP flags present: ACK (1.00 of packets).
Protocols indicated: HTTPS (1.00), TCP (1.00), IPv (1.00), LLC (1.00).
```

**Final table — agent accuracy on the same 68 CIC flows throughout:**

| Aggregation tool | llama3 8B | qwen2.5 14B | qwen2.5 + context | Aggregate |
|---|---|---|---|---|
| offered | 0.574 | 0.588 | **0.603** | 0.574 |
| withheld | 0.500 | 0.574 | 0.574 | 0.574 |

**Context helps, consistently but not decisively.** The best configuration
reaches 0.603 against the aggregate's 0.574 — two flows out of 68. The
progression across the three models is monotone (0.574, 0.588, 0.603), and the
quality of the agent's disagreements improves in step: llama3 departed 24 times
and lost 6 of them, qwen2.5 departed twice and won one, qwen2.5 with context
departs four times, **wins two and loses none**.

**But two flows is not a result.** By an exact test on the discordant pairs the
one-sided p-value is 0.25. Detecting a genuine three-point improvement would
need on the order of a thousand flows, roughly twenty hours at 71 seconds each.
The honest summary is that the agent has gone from clearly harmful to plausibly
slightly helpful, and the study is not powered to say more.

**The sharper finding is where context does *not* help.** Without the
aggregation tool, context changes nothing at all — 0.574 either way, with 15
departures splitting 1 win, 1 loss and 13 where both were wrong. Context only
pays off when the agent also has a strong aggregate to anchor on.

So the agent's value, such as it is, is **as a selective override on a good
baseline, not as an independent reasoner.** Given both the arithmetic answer
and a readable description of the traffic, it overrides rarely and its
overrides were right every time in this sample. Denied the baseline, its own
reasoning is no better than before. That is a more precise and more useful
claim than the paper's, and it is what the write-up should say.

**The likely reason the ceiling is so low, worth arguing in the write-up.** The
agent has no
information the classifiers lack. It sees the same feature vector they do —
as standardised numbers, which a language model reads poorly — plus their
outputs, and nothing else. There is no external knowledge, no context about the
host or the network, nothing a human analyst would actually bring. Under those
conditions no amount of reasoning can beat a weighted vote over the same
evidence, because the evidence is identical and already summarised. If an LLM
agent is to add value here, it needs an input the classifiers do not have.
That is a claim about the task design, not about language models, and it points
at what a genuinely improved system would need.

### Known open questions

- Which averaging method the original paper used is inferred, not confirmed.
  This matters for ACI, where the choice moves the number between 0.91 and
  1.00; it does not matter for CIC, where all classes are scored. Worth
  checking the paper before claiming the ACI reproduction is exact.
- The zero-day figure of 0.61 is the one headline number still unmeasured.
  Early throwaway runs came out far below it, but they used a small subsample
  and an arbitrarily chosen withheld class, so they mean little.
- The high benign false-alarm rate on CIC deserves a decision: whether to keep
  the benchmark as the paper defines it, or additionally report a variant that
  keeps more benign traffic and is closer to operational reality.
- The caps on the two slow classifiers make their numbers not strictly
  comparable to the rest. Whether to raise the caps, or report them separately,
  is unresolved.

---

## 8. Milestone log

| Date | Milestone |
|---|---|
| 2026-09-01 | Measured both datasets from the raw files; found the 40-column correction, the 5-row ACI class, and the 9 truncated CIC rows. Populated dataset configuration, feature schema and loaders; verified with 33 checks. |
| 2026-09-01 | Rewrote README to describe the data layer honestly, including the known quirks. |
| 2026-09-02 | Implemented cleaning, splitting and scaling with the rare-class and zero-day policies; verified with 62 checks across both datasets. |
| 2026-09-02 | Implemented the six-classifier registry, training/evaluation loop and shared scoring; verified with 55 checks. Discovered the weighted-vs-macro averaging gap that affects how the paper's 0.97 must be read. |
| 2026-09-02 | Created this progress log, and adopted the practice of updating it at every milestone. |
| 2026-09-02 | Rewrote README with a usage example, the split policy, the classifier caps and how to read the scores. Implemented the paper-comparison report; verified with 26 checks. |
| 2026-09-02 | Benchmarked how each classifier's runtime grows with data size, to estimate the full run before committing to it. Predicted 22 minutes for ACI; the real run took 25. |
| 2026-09-02 | **First full-scale baseline run (ACI): the paper's 0.97 target was exceeded at 0.9995.** Fixed a deprecated setting being passed to Logistic Regression. |
| 2026-09-02 | Measured the neural network's cost/accuracy trade-off on CIC and found it never finishes training there, unlike on ACI. Capped its iterations at 100 on the evidence, saving about 39 minutes for 0.007 F1. |
| 2026-09-02 | **Full-scale CIC baseline run: the paper's 0.75 target was matched at 0.7408.** Both headline figures are now reproduced. Found that the best model still raises false alarms on 32% of benign traffic — a real weakness the published F1 scores conceal. |
| 2026-09-02 | **Zero-day experiment, 27 runs: the paper's 0.61 did not reproduce, and appears not to be a well-defined quantity.** Measured values span 0.000 to 1.000 depending on the dataset and on what is withheld. Zero-day recall tracks the benign false-alarm rate at r = 0.843, meaning it largely measures how readily a model shouts "attack" rather than how well it generalises. |
| 2026-09-02 | **Added the detection trade-off curve, which corrected the previous day's conclusion.** Withholding a single attack class is a problem the models genuinely solve (AUC 0.998); the near-zero recall measured earlier was the fixed decision rule failing, not the detector. Withholding a whole attack family remains near-chance (AUC 0.594). Fixed a real bug found in the process: score columns follow the classes a model was trained on, not the frozen class list, so they misalign whenever a class is withheld. |
| 2026-09-02 | Built the agent foundation — tool interface with an enforced data/prompt boundary, prompt templates that fence attacker-influenceable content, a swappable LLM client with a scripted test backend, and the classifier tool. 51 checks. |
| 2026-09-02 | **Re-ran all 27 zero-day runs with AUC. The gap between the two datasets was almost entirely an artifact of the decision rule** — 0.925 vs 0.926 by AUC, against 0.189 vs 0.912 by the fixed rule. Also found that withholding a whole attack family is only hard when no behaviourally similar attack remains, so the datasets' own family labels are a poor basis for a zero-day protocol. |
| 2026-09-02 | Implemented the agent's reasoning loop, with the data/prompt boundary enforced through a single function and every failure mode kept non-fatal. Each run reports what fraction of the model's context was attacker-influenceable. 43 checks. |
| 2026-09-02 | **Ran the agent against a real language model** (llama3 via Ollama): 30 ACI flows, all 30 completed and correct, 16 seconds each. Fixed two defects found only by running it for real — a rejected tool payload that failed every request, and verdict matching that scored correct answers as wrong. The run proves the system works end to end but has no power to test whether the agent beats its classifiers, since all three scored perfectly. |
| 2026-09-04 | **Phase 3 follow-up, the tool-imitating note: the prediction failed, informatively.** A note fusing guaranteed retrieval with a forged tool observation — the authority form that worked in Phase 2 — flipped 0 of 18 verdicts, no better than a plain false verdict (paired p = 1.0), despite reaching the model at rank 1 every time. Forged tool authority works injected into the current turn but not recalled from memory. Conclusion sharpened: authority is bound to the channel of delivery, not the surface form of the text, so the open retrieval channel is less dangerous than it looked. |
| 2026-09-04 | **Phase 3, memory poisoning: the retrieval channel is wide open but the agent is nearly unmoved.** A poisoned note is recalled at rank 1 for 100% of flows (retrieval is driven by shared feature-name vocabulary, a weakness of the retrieval tool), yet it flips 0 of 18 verdicts to Benign. The agent's tool-anchoring — the same trait that made it add no accuracy in Phase 1 — is what resists the attack. Contrast with injection's up-to-12%: success tracks the authority of the imitated channel, not whether the text reaches the context. |
| 2026-09-04 | **Injection scaled to 300 runs with confidence intervals — and it corrected the pilot.** Overall rate 5.6% [3.4%, 9.2%]; agent accuracy falls from 94.0% to 86.8% under attack. Forging a tool observation is the only payload significant against its control (12%, p = 0.0156). The pilot's headline 25% for a forged system message shrank to 8% and must not be quoted, and the pilot's claim that instruction-free payloads never work was falsified — social engineering succeeded 2 times in 50. Still zero fence escapes, now across 250 attempts. Payload ranking remains unestablished. |
| 2026-09-03 | **Phase 2, injection (PILOT, superseded): prompt injection through a simulated attacker-controlled HTTP User-Agent field turns detected attacks into "Benign" in 10% of runs, peaking at 25% for a forged system message.** No payload ever escaped the delimiter fence, yet the most effective one worked from inside it — fencing stops syntactic escape but not semantic authority confusion. Payloads carrying no instruction failed entirely, so the vulnerability is to imitation of trusted structure rather than to persuasion. The injected field is simulated and is documented as such. |
| 2026-09-03 | **Phase 2, evasion: the classifier's 34-class label is very brittle but its detection is not.** Nudging eight low-cost fields by a quarter of a standard deviation changes the predicted class 59% of the time, but reaches "Benign" only 2%: the flows slide between near-duplicate classes rather than escaping detection. Random noise outperformed the directed search, which found smaller evasions instead — the control condition earned its place. |
| 2026-09-03 | **Gave the agent the flow described in words rather than as standardised numbers — the last Phase 1 experiment.** Best configuration reaches 0.603 against the aggregate's 0.574, and the agent's overrides are now right every time (2 wins, 0 losses) where llama3's lost 6. Two flows out of 68 is not significant, but the trend across three configurations is consistent. Context helps only when the agent also has the aggregate to anchor on, which makes its role a selective override rather than an independent analyst. Phase 1 closed. |
| 2026-09-03 | **Tested a stronger model (qwen2.5 14B) in both configurations: the negative result holds.** It repairs llama3's damage — 0.574 rather than 0.500 without the aggregation tool — but beats the arithmetic by a single flow out of 68 with it, at four times the cost per flow. Its independent judgements go from harmful to a coin flip, never to an improvement. The agent's failure to add value is therefore not a matter of model strength. |
| 2026-09-02 | **Withheld the aggregation tool and re-ran: the agent got worse, not better** (0.500 against 0.574). Denied a combined answer it does reason independently, departing from the aggregate on 24 of 68 flows, but it wins only 1 of those and loses 6. Fixed two parser defects found in the process; the corrected run confirmed rather than overturned the result. |
| 2026-09-02 | **Agent versus classifiers on CIC, 68 flows: the paper's central claim does not reproduce.** The agent scored 0.574, identical to the deterministic aggregate of the tools it calls, and agreed with that aggregate on all 68 flows without a single deviation. It behaves as an expensive pass-through, costing 16.6 seconds per flow for an answer the aggregator produces in milliseconds. Its errors are almost entirely confusions between the near-duplicate class pairs the behavioural clustering had already identified. |
| 2026-09-02 | **Phase 1 complete.** Built the memory layer and the retrieval and aggregation tools, and ran the full agent end to end: it consults a classifier, searches memory, aggregates all three models, and returns a correct verdict, with 29% of its context marked attacker-influenceable. 62 checks. |
| 2026-09-02 | **Redesigned the zero-day protocol around behavioural rather than taxonomic grouping.** The clusters cut across the dataset's own families, and the new holdouts are markedly harder: ACI's worst case falls from 0.594 to 0.483, CIC's from 0.854 to 0.726. The intended scalar difficulty measure was tested and rejected — it correlates with difficulty at only r = −0.26, so the clustering is the usable output and the distance is not. |
| 2026-09-06 | **Merged the paper into a single-file LaTeX source.** The six section files (abstract, intro_related, methodology, results, discussion, conclusion) were inlined into `paper/main.tex` in document order and deleted, replacing the six `\input{}` calls; `references.bib` stays separate and `main.tex` still ends with `\bibliographystyle{IEEEtran}` and `\bibliography{references}`. Verified by an exact line-multiset comparison of the old sources against the merged file (nothing lost, nothing added, nothing duplicated), and by checking that all 31 labels are unique, all 24 `\ref` targets resolve, all 14 `\cite` keys exist in `references.bib` with none unused, and every environment is balanced. Done so that the draft can be pasted into Overleaf or a journal portal as one file; the trade-off is that section edits now happen in a 902-line file rather than six small ones. |
| 2026-09-06 | **Retargeted the paper at Computers & Security (Elsevier) and converted `paper/main.tex` from IEEEtran to the elsarticle class.** `\documentclass[preprint,3p,number]{elsarticle}`, a `frontmatter` environment holding the title, per-author `\ead` emails and `\affiliation` blocks (both emails now visible, Aburomman marked corresponding via `\cortext`), `\begin{keyword}` with `\sep` separators in place of `IEEEkeywords`, `\journal{Computers & Security}` in place of the IEEEtran `\markboth` running head, and `\bibliographystyle{elsarticle-num}`. Dropped the `cite` package because elsarticle loads natbib, which it conflicts with. The manual `\\` line breaks in the title were removed, as they were tuned for IEEEtran's narrow two-column measure. Table markup needed no change: all seven tables already use booktabs rules, which are class-independent. Verified as format-only — the 810 body lines are byte-identical, the abstract is byte-identical, and the label, `\cite` and `\section` sets are unchanged. |
| 2026-09-06 | **DOI audit of `paper/references.bib`: eight of fourteen entries carry a DOI, and the other six have none to carry.** The eight existing DOIs were re-resolved against `api.crossref.org` and each matched its entry on title, first author, venue, volume and year, so the file's "verified during drafting" header is accurate. The six without were checked rather than assumed: Crossref has no record of any of them, and DBLP shows the three USENIX Security papers (`ref:promptinjection`, `ref:datapoisoning`, `ref:injectiondefence`) carry only USENIX presentation URLs, because USENIX does not DOI-register its proceedings; `ref:idsagent` is a NeurIPS 2024 workshop poster on OpenReview (forum id iiK0pRyLkw), which mints no DOIs. Those four are annotated inline as "no DOI exists, checked on this date" rather than as open TODOs, since there is no outstanding action. The two dataset stubs (`ref:aciiot`, `ref:ciciot`) were deliberately not looked up and carry an explicit `TODO(author)`. No DOI was inferred, constructed or guessed. The stale comment in `main.tex` claiming all cite keys were unverified placeholders was corrected in the same change, as was the Methodology banner naming ref:idsagent as a placeholder. |

| 2026-09-06 | **Filled in the two dataset citation stubs; `references.bib` now has no placeholders.** `ref:aciiot` became an `@inproceedings` for Nack, McKenzie and Bastian, "ACI-IoT-2023: A Robust Dataset for Internet of Things Network Security Analysis", MILCOM 2024, pp. 1-6, DOI `10.1109/MILCOM61039.2024.10773916`; `ref:ciciot` became an `@article` for Neto et al., "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment", Sensors 23(13):5941, 2023, DOI `10.3390/s23135941`. Both were supplied by the user and then independently resolved against `api.crossref.org`, which matched every field including the full author lists, so neither was taken on trust alone. Ten of the fourteen entries now carry a DOI and the remaining four are annotated as having none in existence. Recorded in the bib header, but deliberately not cited, that IEEE DataPort assigns a separate DOI (`10.21227/qacj-3x32`) to the ACI-IoT-2023 data artifact itself, which is distinct from the MILCOM paper describing it and would be the right second reference if a venue wants data cited apart from its description. The citation-status comments in `main.tex` were updated to match. |
---

## 9. Consolidated summary for the paper

Written to map onto Methodology and Results. Every figure is reproduced from
the `experiments/` artifacts named beside it. Two provenance rules hold
throughout: all reproduction and evasion numbers come from the published
datasets, and the injection experiments use a **simulated** attacker-controlled
HTTP `User-Agent` field, flagged as such wherever it appears.

### 9.1 Experimental setup (Methodology)

- **Datasets.** ACI-IoT-2023 (1,231,411 flows, 12 classes, 83 numeric features)
  and CIC-IoT-2023 (45,019,243 flows across 63 files, 34 classes, 39 features).
  CIC is streamed and per-class capped at 50k to ~1.29M rows; ACI is used
  whole. Leakage columns (IPs, flow ID, timestamp) are excluded; ports kept but
  flagged. Split 70/15/15, seed 42, stratified, with a train-only rule for
  classes below 100 rows (ACI's 5-row ARP Spoofing) and a configurable
  zero-day holdout mechanism.
- **Classifiers.** Six behind one interface — Random Forest, Decision Tree,
  KNN, Logistic Regression, MLP, SVM — with SVM and KNN capped in training rows
  for tractability, reported alongside `n_train_used`.
- **Agent.** A hand-written ReAct loop (no framework) over classification,
  aggregation and retrieval tools, with session and cross-session memory.
  Single audited boundary between observed data and prompt text; every tool
  result carries a provenance tag; untrusted content is delimiter-fenced.
  Backends: llama3 8B and qwen2.5 14B via Ollama; temperature 0 for the
  measured runs.
- **Attacks.** Evasion = black-box feature perturbation under a
  manipulation-cost budget (low/medium/high, so definitional features are not
  touched). Injection = payloads in the simulated string field, six techniques,
  paired against a no-payload control. Memory poisoning = three note types
  planted through the ordinary write path, measured in two stages (retrieval,
  then verdict impact).

### 9.2 Findings (Results)

**R1 — Reproduction holds, with a caveat about the metric.** ACI weighted
F1 **0.9995** (paper 0.97), CIC **0.7408** (paper 0.75, within tolerance). ACI
macro-F1 is only 0.906 because the 5-row class contributes a structural zero;
the paper's figure is therefore weighted/micro, not macro. Random Forest is the
best and nearly the cheapest model; MLP costs 8x more to be worse.
[`experiments/phase1_*`]

**R2 — Zero-day recall is not a well-defined single number.** Across 27
leave-out runs it spans 0.000-1.000; the fixed-threshold recall and the AUC of
the same model correlate at only r = 0.32. By AUC the two datasets are
near-identical (~0.925), so the apparent gulf was an artifact of the decision
rule, not the detector. Difficulty is governed by whether a *behaviourally*
similar class survives training, which the datasets' own attack-family labels
do not track. Report AUC per behavioural holdout, never a scalar.
[`experiments/zero_day`]

**R3 — The LLM agent adds no significant accuracy over the tools it calls.**
On CIC, agent 0.574 = deterministic aggregate 0.574, with the agent agreeing
with the aggregate on 68/68 flows under llama3. A stronger model (qwen2.5) and
adding human-readable flow context each help by ~1-3 flows out of 68 — a
consistent trend but never significant. The agent functions as a selective
override on a strong baseline, not an independent analyst; it holds no
information the classifiers lack. [`experiments/phase1_cic_iot/agent_run*`]

**R4 (Phase 2 evasion) — the label is brittle, detection is not.** Perturbing
eight low-cost features by 0.25 sd changes the 34-class verdict 59% of the
time but reaches *Benign* only ~2%: flows slide between near-duplicate attack
classes, not out of detection. Directed search did not beat random noise
(the control earned its place). Headline number is the 2% escape-to-benign,
not the 59%. [`experiments/phase2_evasion`]

**R5 (Phase 2 injection) — real, measurable, and mechanism-revealing.** Through
the simulated field, injection turns detected attacks Benign at **5.6%
overall, 95% CI [3.4%, 9.2%]** (300 runs, temperature 0); a forged tool
observation is the one payload significant against control (12%, p = 0.0156).
**Zero of 250 payloads escaped the delimiter fence** — yet the most effective
one worked from *inside* it. Fencing prevents syntactic escape, not semantic
authority confusion. Instruction-free persuasion works weakly (4%); structural
imitation of a trusted channel works best. Payload ranking is unresolved.
[`experiments/phase2_injection_scaled`]

**R6 (Phase 3 memory poisoning) — the channel is open but authority does not
travel through it.** A poisoned note is recalled at rank 1 for 100% of flows
(the retrieval tool is a weak, vocabulary-driven filter), yet it flips 0/18
verdicts to Benign — including a note that fuses guaranteed retrieval with a
forged tool observation (paired p = 1.0 vs a plain false verdict). Forged tool
authority moved the agent in the current turn (R5) and not from memory. The
unifying claim: **an attack's success is bound to the authority of the channel
it is delivered through, not to the surface form of its text.** The agent's
tool-anchoring — the same trait behind R3's null result — is what resists
poisoning. [`experiments/phase3_memory_poisoning`]

### 9.3 The spine of the paper

One finding recurs at every layer and ties the three phases together:
**the agent inherits its tools' authority structure, for better and worse.**
Because it will not override its classifiers (R3), it cannot be poisoned
through low-authority memory (R6); because it *will* trust text that imitates a
live tool, it is injectable in-context (R5). The security story is not "LLM
agents are fragile" but "an LLM agent is exactly as trustworthy as its least
authenticated trusted channel." Alongside it runs a measurement-methodology
thread — argmax vs AUC, macro vs weighted F1, pilot vs CI'd injection, verdict
format vs judgement — where a headline number repeatedly scored a surface
artifact rather than the thing of interest. Both are contributions.

### 9.4 Threats to validity to state up front

- Injection uses a **simulated** field absent from both benchmarks (realistic
  values, clearly documented).
- Agent studies use two mid-size open models, not a hosted frontier model.
- Verdict-impact samples are small (18-68 flows); injection alone has CIs from
  300 runs. Claims are scoped accordingly throughout.
- SVM/KNN train on capped subsets; the zero-day difficulty scalar was rejected
  as too weak (kept only as a documented negative).
