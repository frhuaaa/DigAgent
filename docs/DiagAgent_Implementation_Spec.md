# DiagAgent Implementation Specification

## Version 1.0 — End-to-End Round-0-to-Test Workflow

This document is the detailed implementation specification for the runnable DiagAgent repository at `C:\Users\Andy\Desktop\DigAgent_WWW`. Read the root `AGENTS.md` first; when this document is incomplete or conflicts with it, `AGENTS.md` is authoritative. DiagAgent uses the installed Qlib package and the Qlib-format data in this repository; it is not implemented in a separate Qlib source repository.

All existing source code is explanatory reference material, not a frozen
implementation. Codex may rewrite, reorganize, replace, or remove it and may
change interfaces as needed to produce a clean, tested, end-to-end runnable
system. Existing code should be inspected for intended logic, not preserved for
compatibility. This freedom does not extend to immutable files under `data/` or
to the frozen behavioral and isolation contracts in `AGENTS.md`.

### Current validation-split amendment

The executable contract now uses `train_valid` and `agent_valid` instead of a
single `valid` split. Per-epoch checkpoint selection and early stopping use
only `train_valid` (default metric `train_valid.ic`). Train frozen seeds
`[0,1,2]` concurrently and select a separate best checkpoint for each seed.
Evaluate all three on `agent_valid`, daily cross-sectionally winsorize/z-score
each member, equal-weight average, and renormalize the ensemble; its
`artifacts/agent_valid_alpha.csv`, signal metrics, portfolio result, and
`result.json` drive CLEM, FAMA, RAPA, the Validation Gate, and memory. Do not
compute per-epoch `agent_valid` metrics. For now the two periods may overlap or
be identical, but both must remain after training and before isolated test.
Where older prose below says generic “validation,” interpret adaptive outcome
evidence as `agent_valid` and checkpoint/learning-curve evidence as
`train_valid`; `AGENTS.md` remains authoritative.

### Current prediction-mode amendment

`task.prediction_mode` is the only mode selector. `c2c` remains the default for
legacy submissions; `o2o` selects the open-to-open label and execution data.
Configuration materialization binds the label, return panel, directional masks,
execution price, return definition, and slippage as one preset. The run and
shared-RASS namespace is
`runs/<instruments>_<agent-name-or-provider>_<prediction-mode>/`; C2C and O2O
must never share RASS evidence or a frozen factor cache. This amendment and
`AGENTS.md` override older C2C-only wording below.

## 1. Objective and non-objective

DiagAgent iteratively improves the adaptive portions of one explicitly selected
`./submit/*.json` using Qlib-based experiments. Every formal launch requires
`--submit submit/<file>.json`; there is no implicit default submit file. The
selected file is the authoritative seed for that trajectory, and `task.name`
must scope its state and outputs so different submissions cannot collide.

The adaptive configuration is divided into:

```text
z_alpha      factor/signal layer       -> RASS
z_model      learning/model layer      -> FAMA
z_portfolio  portfolio layer           -> RAPA
```

The implementation must create a diagnosis-driven experimental loop:

```text
Baseline
  -> validation evidence
  -> CLEM layer diagnosis
  -> one specialist intervention
  -> deterministic contract validation
  -> dependency-aware execution
  -> deterministic validation gate
  -> accepted-state update or rollback
  -> shared memory write
  -> next round or stop
  -> frozen final configuration
  -> final reporting/test stage
```

This is not an unrestricted AutoML system. Never enumerate configurations, optimize directly against test results, or keep a candidate merely because it has the largest observed validation Sharpe.

## 2. System invariants

These rules must be enforced in code, not only described in prompts:

1. Adaptive components can read only train/validation-derived information.
2. Each adaptive round starts from the latest accepted configuration.
3. Each round changes one layer and one declared parameter group under one falsifiable hypothesis.
4. Contract-invalid proposals never reach the executor.
5. `SUPPORTED` and `PARTIALLY_SUPPORTED` candidates become the accepted state.
   RASS uses this same Gate and has no unconditional structural promotion.
6. Every experiment remains reproducible and auditable even after rollback.
7. Test artifacts are physically outside every adaptive read allowlist.
8. All formal-run policies are frozen before the trajectory begins.

## 3. Complete experiment pipeline

A full candidate experiment consists of:

```text
select/build alpha factors
  -> Qlib model training
  -> best checkpoint selected by one frozen validation metric
  -> validation predictions
  -> portfolio optimization
  -> validation backtest
  -> validation metrics and artifacts
```

Frozen repository-relative data locations:

```text
Qlib provider_uri:  ./data/cn_data
portfolio returns:  ./data/portfolio/c_2_c_1D.csv
limit-up mask:       ./data/portfolio/mask_limit_up_1D.csv
limit-down mask:     ./data/portfolio/mask_limit_down_1D.csv

O2O mode instead uses:

```text
portfolio returns:  ./data/portfolio/o_2_o_1D.csv
limit-up mask:       ./data/portfolio/mask_limit_up_open_1D.csv
limit-down mask:     ./data/portfolio/mask_limit_down_open_1D.csv
```
```

All relative paths are resolved from the repository root. Files under `data/`
are immutable inputs and must not be rewritten by experiments or tests.

### 3.1 One-time Round 0 configuration materialization

Treat the JSON selected by the required `--submit` argument as the authoritative
seed template for that trajectory. Persist its repository-relative path and
SHA256 and never silently substitute another file. Its compact pipeline
contract, for example `pipeline_contract=qlib_ts_lstm_v1`, selects the fixed Dataset/Handler,
preprocessing, data-key, loader, and derived-value behavior specified below;
these details are intentionally not duplicated in JSON. Seed fields such as
batch size, worker count, and label normalization remain explicit. Before
`EXP_000`, Codex may derive only additional runtime values genuinely absent from
both sources. The pipeline must not depend on any other unstated Python, Qlib,
or example-code default.

Validate all run-path components, derive the shared namespace as
`runs/<task.instruments>_<agent.name-or-provider>_<task.prediction_mode>/` and the trajectory root as
`<namespace>/<task.run_id>/<z_model.model_name>/<launch-MMDDHHMM>/`. Treat the
four-digit `task.run_id` as the evidence-year identifier and always freeze a
separate launch minute in the controller environment for every worker. Use
`--run-timestamp` to resume the same root. Store only
`rass_train_evidence.json` directly under `<namespace>/<task.run_id>/` for reuse
across backbones of the same year. Never share an Agent-selected factor set
across trajectories. Use `rass_evidence_context_hash`, containing only the
evidence-relevant universe, year, mode, periods, anchors, label, data, catalog,
and evidence-code identity; never use a full backbone/model configuration hash
as shared evidence identity. Write the
fully expanded configuration once to `<run-root>/configs/initial.json`. Alongside
it, persist validation-safe provenance containing all inserted paths and values
with their source or rationale, dependency versions, relevant data hashes, and
the final configuration SHA256. Schema, path, split, feature, label, and data-
coverage checks must pass before Round 0 starts, and materialization must not
read test results or test-derived artifacts.

Once `EXP_000` begins, `<run-root>/configs/initial.json` is immutable for
that formal trajectory. Initialize `<run-root>/configs/current.json` from
it byte-for-byte or from a
canonical semantically identical serialization. A later dependency or default
change requires a new formal trajectory, not regeneration of the existing
baseline.

All mutable state and outputs are scoped under that run root. An existing run
may resume only when its recorded submit path and SHA256 match; otherwise fail
without overwriting or mixing submissions.

Every trajectory executes EXP_000 with the five submitted anchors and
`d_feat=5`, then makes a fresh RASS Agent selection using the shared deterministic
evidence. Accepted and rejected factor sets remain local to the trajectory and
are never cached or replayed. EXP_001 executes the newly selected eight-factor
configuration with otherwise default submitted parameters. The normal Gate may
reject it. RASS may test at most three distinct
three-factor additions, always from the accepted five-factor parent. Acceptance
sets `alpha_frozen=true` and `d_feat=8`; two complete failures instead freeze
the original five factors with `d_feat=5`. Either outcome permanently disables
later RASS routing.
Only deterministic RASS evidence is shared; factor selections, models,
predictions, results, portfolio artifacts, accepted state, and memory remain
isolated per run.

Dependency-aware reruns:

| Selected agent | Recompute | Reuse |
|---|---|---|
| RASS | factors, model, predictions, portfolio, validation | immutable source data |
| FAMA | model, predictions, portfolio, validation | selected factors/source data |
| RAPA | portfolio and validation backtest | factors, trained model, predictions |

The executor must verify parent artifact identity/hash before reuse. RAPA reuses
the latest accepted parent's `agent_valid_alpha.csv`, agent-valid signal metrics, and selected
checkpoint. Interrupted RAPA recovery validates that checkpoint against the
accepted parent's model-config hash, not against the portfolio-only candidate
hash. It must not silently reuse stale outputs.

### 3.2 Canonical prediction-to-portfolio interface

`artifacts/agent_valid_alpha.csv` and any equivalent agent-valid prediction passed to the
portfolio runner must be a UTF-8 comma-separated wide panel physically
compatible with the return panel selected by `task.prediction_mode`:

```text
,000001_XSHE,000002_XSHE,...,600000_XSHG,...
20240102,<alpha>,<alpha>,...
20240103,<alpha>,<alpha>,...
```

Contract:

1. The first column has no header and contains dates formatted `YYYYMMDD`.
2. Asset columns exactly match the return panel's complete header set and order.
3. Convert Qlib instruments deterministically: `SZ000001 -> 000001_XSHE` and
   `SH600000 -> 600000_XSHG`; reject unknown exchange prefixes or collisions.
4. Sort rows by date and include only the requested split after deterministic
   label-boundary purging.
5. Preserve unavailable predictions as empty/NaN. Never fill them with zero.
6. Do not write a second header row, index name, MultiIndex columns, or metadata.
7. After writing, reload and validate the file's index, header, duplicate status,
   split dates, and compatibility with return/mask panels.

The Qlib adapter may use a long `(datetime, instrument) -> score` representation
internally, but the portfolio adapter boundary is always the wide format above.
Before execution it must verify that the return panel contains every required
realized-return date and both masks contain every required T+1 trade date. A
coverage failure is fatal; never truncate or alter the frozen split silently.

### 3.3 Target backbone, trainer, metrics, and checkpoint ownership

Select the existing architecture example using frozen `z_model.model_name` from
`backbone/lstm_example.py`, `backbone/gru_example.py`, or
`backbone/alstm_example.py`; shared layers live in `backbone/modules.py`. These
examples may be fully rewritten. In the runnable implementation, keep each
resulting backbone module
limited to pure PyTorch model architecture and place the Qlib adapter, training,
metrics, checkpointing, split access, prediction, and logging responsibilities
in separate modules. The target ownership is:

```text
backbone/*_example.py              network modules and forward pass only
backbone/modules.py                shared network components only
training/model_trainer.py          loss, optimizer, epochs, deterministic loaders
evaluation/prediction_metrics.py   IC, ICIR, RankIC, RankICIR
adapters/qlib_runner.py            Qlib init, Dataset construction, split access
```

For `pipeline_contract=qlib_ts_lstm_v1`, build `TSDatasetH` over
`DataHandlerLP(PTYPE_A)` and `QlibDataLoader`. Apply feature
processors in the declared order `ProcessInf`, train-fit `ZScoreNorm`, then
`Fillna(0)`; apply `DropnaLabel` then daily-cross-sectional `CSZScoreNorm` to the
learning label; retain sampler `fillna_type=ffill+bfill`. Use `batch_size=2048`
and `n_jobs=8` for adaptive training and train/validation evaluation. The
nested isolated researcher test evaluator uses `n_jobs=0` so it never launches
a second multiprocessing DataLoader inside the Slurm training worker. The
training loader shuffles and drops the last incomplete batch; train-evaluation,
validation, and isolated test-evaluation loaders do neither. Derive `d_feat`
from the selected-feature count (5 for `EXP_000`, 8
after RASS). Train and validation learning views use `DK_L`; metrics join
predictions to the separately retained raw `DK_R` label.

The trainer must support `z_model.base_params.metric` in exactly:

```text
ic | icir | rank_ic | rank_icir
```

Default is `ic`; all four are maximized on validation only. The chosen metric is
frozen for the formal trajectory and is not FAMA-adaptive. A generic prediction
method requires an explicit `train`, `valid`, or `test` split. Adaptive callers
cannot request `test`; only the isolated researcher runner has that capability.

The raw label expression is
`Ref($close, -2) / Ref($close, -1) - 1`. Preserve it separately from the model-
training label. On each date, standardize the finite raw-label cross-section and
train with MSE against that standardized target. Signal evaluation always pairs
predictions with the raw, unstandardized forward-return label. Do not use the
future realized label mean or standard deviation to convert predictions back to
return units.

For each date, compute IC as cross-sectional Pearson correlation and RankIC as
cross-sectional Spearman correlation after dropping non-finite pairs. Aggregate
by mean across dates. Define ICIR/RankICIR as mean divided by sample standard
deviation (`ddof=1`) of the corresponding daily correlation series, without
annualization. Exclude and count dates with fewer than two valid assets or a
constant vector. Evaluation loaders must use `shuffle=false`,
`drop_last=false`.

The model predicts a relative alpha score. At the portfolio boundary, for each
date independently, drop non-finite prediction values, clip the remaining
cross-section at its 1st and 99th percentiles, and calculate a new z-score with
the winsorized cross-section's mean and population standard deviation
(`ddof=0`). A non-finite or `<= 1e-12` standard deviation maps all available
scores for that date to zero; an entirely missing cross-section is reported as
a missing-signal date. Perform this normalization before ranking candidates,
then multiply the normalized scores by `z_portfolio.alpha_scale` and pass the
result to the optimizer. This operation is signal scaling, not inverse label
normalization. Realized portfolio returns come unchanged from
`data/portfolio/c_2_c_1D.csv`.

Universe and benchmark selection are independent explicit task fields. Qlib
alpha generation uses exactly `task.instruments`. The portfolio evaluator loads
the index instrument named by `task.benchmark` from `task.provider_uri` and
computes the aligned benchmark return for signal date `T` as
`close[T+2] / close[T+1] - 1`. Do not infer one field from the other and do not
substitute an equal-weight return of stocks for the configured index benchmark.

Training logging is physically split while preserving one run/checkpoint ID:

```text
logs/training_metrics.jsonl    per-epoch train + valid IC/ICIR/RankIC/RankICIR
test/epoch_metrics.jsonl       per-epoch train + valid + test copies of those metrics
test/result.json               selected-checkpoint full 27-field test result
test/test_alpha.csv            selected-checkpoint test predictions
test/portfolio_daily_diagnostics.csv
                               test portfolio/evaluator audit detail
```

Start one persistent physically isolated researcher evaluator per experiment;
it loads the test Dataset exactly once. After every epoch checkpoint is written,
send it a request that immediately appends that epoch's train/valid/test metrics to
`test/epoch_metrics.jsonl`; the trainer receives no test values. The remaining
test files are produced only after validation checkpoint selection and
adaptive-output sealing. `test/result.json` uses the same exact schema and
three-decimal serialization as validation `result.json`, but its location and
access policy identify it as test-only. Test metrics never enter the normal log
or any adaptive context.

Determinism uses `task.seed` for Python, NumPy, PyTorch CPU/CUDA, DataLoader
generators, and workers. Enable deterministic algorithms, disable cuDNN
benchmarking, persist split indices/sample order, and fail on unsupported
nondeterministic operations. Round 0 reproducibility is evaluated under the same
frozen data, software versions, hardware class, config, and seed.

`task.GPU` is a requested zero-based CUDA index. Resolve it at process startup:
use `cuda:<task.GPU>` only if CUDA is available and that index exists; otherwise
use CPU and emit a warning. Persist requested/resolved device, fallback reason,
and PyTorch/CUDA versions in initial provenance, then freeze the resolved device
for the trajectory. Do not silently fall back after a CUDA OOM or runtime device
failure; such a mid-run change is fatal to trajectory comparability.

## 4. Round 0

Round 0 creates `<run-root>/experiments/EXP_000` from the submit JSON explicitly selected
for the current trajectory.

Procedure:

1. Load and validate the initial JSON.
2. Freeze the initial contract and record a hash of it.
3. Run the full Qlib alpha -> model -> prediction -> portfolio -> validation pipeline.
4. Write `config.json`, the exact validation `result.json`, validation artifacts, and logs.
5. Mark `EXP_000` as the initial accepted experiment.
6. Write the baseline memory item.

## 4.1 Bounded initial RASS search

The current target baseline begins with five ordered features. Immediately after
`EXP_000`, while `alpha_frozen=false`, CLEM must use the observed configuration
count as evidence of insufficient initial feature breadth and route RASS. This
priority continues only until one candidate is accepted or two complete
candidates fail.

RASS is an LLM-backed specialist. It follows `agents/RASS/SYSTEM.md` and
`agents/RASS/SKILL.md`, explores the frozen catalog using a deterministic
three-complete-calendar-year development evidence table derived from `task.run_id`, preserves all anchors in their original order,
and selects exactly three additions. Its only legal diff is the three-item
append that produces exactly eight features. It then reruns alpha construction,
model training, validation prediction, portfolio optimization, and validation
evaluation.

Every legal proposal reruns the complete model, prediction, portfolio, and
agent-valid evaluation path. `SUPPORTED` or `PARTIALLY_SUPPORTED` promotes and
freezes the eight-factor alpha. `UNCERTAIN` or `FALSIFIED` rolls back to
`EXP_000`; RASS then receives the validation-safe failure outcome and may choose
another three-factor set. Partial overlap is allowed, but the full unordered
set may not repeat. After two complete rejected candidates, freeze the
original five-factor alpha and route only FAMA or RAPA. Contract, API, data, or
runtime failures are not factor-performance evidence and do not count toward
the two failures.

For factor-quality evidence under year ID `Y`, use the three complete calendar
years `Y-3`, `Y-2`, and `Y-1`. Purge the final two trading signal dates of every
year so the T+2 realized label remains inside that period; derive the last usable
signal date from the Qlib calendar. Report
period-level and aggregate IC, ICIR, RankIC, RankICIR, cross-period sign
stability, worst-period behavior, and shortlist anchor-residual metrics. These quantities
inform the Agent; they do not define a fixed weighted score or mechanical Top-K
selector. Test data remains unavailable to RASS.

Round 0 has no CLEM decision, specialist proposal, config diff, or rollback decision.

## 5. Adaptive round state machine

For round `t >= 1`:

```text
LOAD_ACCEPTED_PARENT
  -> BUILD_VALIDATION_EVIDENCE
  -> CLEM_DECISION
       -> RASS/FAMA/RAPA
  -> RETRIEVE_SPECIALIST_MEMORY
  -> SPECIALIST_PROPOSAL
  -> CONTRACT_VALIDATE
       -> REJECTED -> record invalid proposal; accepted state unchanged
       -> PASSED
  -> MATERIALIZE_CANDIDATE_CONFIG
  -> EXECUTE_MINIMUM_DEPENDENCY_PATH
  -> WRITE_VALIDATION_RESULT_AND_ARTIFACTS
  -> VALIDATION_GATE
       -> SUPPORTED -> promote candidate
       -> PARTIALLY_SUPPORTED -> promote candidate
       -> UNCERTAIN/FALSIFIED -> retain experiment and rollback to parent
  -> WRITE_SHARED_MEMORY
  -> CHECK_STOPPING_RULES
  -> NEXT_ROUND
```

Every state transition should be explicit, logged, restartable, and idempotent. A crashed experiment must resume from the last completed deterministic stage rather than creating a second experiment ID.

## 6. Agent inputs and outputs

### 6.1 CLEM — WHERE + WHY

CLEM receives only:

- current accepted validation evidence;
- accepted config diff history from Round 0;
- compact recent global state;
- the complete validation-safe Memory Bank in deterministic order;
- the frozen CLEM instruction and output schema.

CLEM returns RASS while the five-factor alpha remains unresolved and fewer than
two complete RASS candidates have failed. After acceptance or fallback freezes
alpha, it returns exactly one of `FAMA` or `RAPA`, plus:

- selected layer;
- failure summary;
- supporting evidence;
- why the other layers are less supported;
- intervention goal;
- confidence.

CLEM must not name a parameter, direction, magnitude, or new value.

The Contract Validator also enforces a deterministic one-round routing
cooldown. If the trailing three post-bootstrap Memory Bank records all selected
the same FAMA/RAPA layer and all have `accepted=false`, that layer is ineligible
in the next round. Contract-rejected attempts count because they are persisted
with `accepted=false`. CLEM must select and re-diagnose the other of FAMA or
RAPA; that forced-switch round breaks the consecutive streak, so the prior
layer is eligible again afterward. This is a routing-diversity safeguard, not
evidence that the forced layer will be accepted.

Absolute validation Sharpe on strategy net daily returns is the sole adaptive
acceptance objective. Its sign is not a routing gate: positive Sharpe does not
prove portfolio translation is efficient. Excess annual return, information ratio, and every other
benchmark-relative metric are diagnostic-only. They may help explain a
portfolio mechanism, but they must never independently trigger an intervention
or RAPA routing. A RAPA route
must include `primary_objective_assessment` with the exact accepted validation
`sharpe_ratio`; deterministic validation rejects a mismatch or
`failure_supported=false`. If only excess performance is weak, CLEM must choose
the best-supported eligible non-portfolio layer.

### 6.2 Specialists — WHAT + HOW

The selected specialist receives:

- CLEM's diagnosis and goal;
- the current accepted adaptive config;
- current validation evidence relevant to its layer;
- relevant specialist-local memories;
- its frozen intervention space and output schema.

It returns:

- local diagnosis;
- exactly one selected parameter group;
- one legal diff (or no change);
- reasoning tied to observed evidence;
- expected observable signature;
- falsification condition;
- confidence.

FAMA first resolves one model-specific space from frozen `z_model.model_name`
through `agents/FAMA/intervention_space.yaml`. `lstm`, `gru`, and `alstm` are
supported; any other model is rejected without fallback. It may change at most
two tightly coupled parameters inside exactly one group from the resolved file.
Spaces and groups are never unioned.

```text
all models train_params = {lr, hidden_size, dropout, weight_decay}
lstm/gru model_params = {model_layer, use_pe, use_bn, use_ln}
alstm model_params = {model_layer, attention_hidden_size, use_pe, use_bn}
```

Inclusive FAMA bounds and types:

```text
lr                     float    [0.0001, 0.01], log scale
hidden_size            integer  16..128, step 8
dropout                 float    0.0..0.5, step 0.05
weight_decay            float    0 or [0.000001, 0.001], log scale
model_layer             integer  {1, 2, 3, 4}
attention_hidden_size   integer  16..128, step 16 (ALSTM only)
use_pe/use_bn/use_ln    boolean  where declared by the selected model file
```

The selected model YAML is authoritative for exact membership and types.

RAPA changes exactly one of:

```text
risk_aversion | turnover_penalty
```

Inclusive RAPA bounds are:

```text
risk_aversion     float [0.1, 2.0]
turnover_penalty  float [0.1, 2.0]
```

`alpha_scale` is frozen. Each adaptive RAPA parameter may be evaluated at most
four times per trajectory. Its first trial is a local absolute step of at most
0.1. After an accepted trial, continue in the same direction with a
non-shrinking step. After a rejected trial, roll back and reverse direction
with a local step of at most 0.1. A previously evaluated numeric value may not
be repeated, and there is no fifth refinement trial. Validation Gate promotion
and rollback retain the best accepted value.

There is also no target, preferred band, or reference value for realized
turnover. Each RAPA round may probe one direction only. Increasing
`turnover_penalty` must declare an expected turnover decrease; decreasing it
must declare an expected turnover increase. Net validation Sharpe, which
already includes costs, controls acceptance.

RASS has only the append-only `initial_feature_bootstrap` group during its
bounded initial search.
Bootstrap preserves the initial ordered features and, as an LLM-backed
specialist, performs two fixed-development-evidence Agent stages. The first reviews every
deterministically eligible catalog factor and selects exactly twelve candidates.
Code then computes shortlist-only daily cross-sectional redundancy and
information remaining after linear residualization against the anchors.
The second Agent stage selects exactly enough factors from that shortlist,
reaching eight total. It cannot
remove, replace, reorder, or edit an anchor; add duplicates; or invent
expressions outside the pool. Resolve, persist, and hash the complete pool
catalog and three-calendar-year rolling evidence table before the formal trajectory so later
Qlib/library changes cannot alter the available evidence. Deterministic code
must validate the Agent's proposal but must not select or rank factors for it.
An accepted eight-factor candidate is frozen. If all three candidates fail,
the original five-factor alpha is frozen instead. RASS is then no longer an
eligible route.

Eligibility removes only factors with inadequate coverage or valid IC dates,
mostly constant daily cross-sections, or duplication of an anchor; it is not a
predictive ranking. Both selection stages receive the same three frozen
year-scoped factor-evidence views.

Use the aligned rows from calendar years `Y-3` through `Y-1` for IC, RankIC,
coverage, and temporal stability. To keep
initialization bounded, Stage-1 all-candidate redundancy uses a deterministic
evenly spaced sample capped at 20,000 rows. The twelve-factor shortlist stage
uses full development-period daily cross-sections for its joint evidence.

The machine-readable versions are the three
`agents/*/intervention_space.yaml` files. Contract validation must read them;
do not duplicate numeric bounds in application code.

An intervention route requires CLEM confidence >= 0.5. This is comparative
routing confidence; every in-budget adaptive round must select a legal layer.

## 7. Deterministic Contract Validator

Validation order:

1. JSON/schema validity.
2. The proposal targets the layer selected by CLEM.
3. All changed paths belong to the selected specialist.
4. Exactly one parameter group is touched.
5. Change-count rules are satisfied.
6. Old values equal the accepted parent values.
7. New values satisfy type, enum, and numeric bounds.
8. Relative-change limits are satisfied.
9. Frozen fields are unchanged.
10. Hypothesis, expected signature, and falsification condition are present.
11. No path, reference, or evidence points to `test/`.

Return `PASSED` or `REJECTED` with deterministic reason codes. Do not ask an LLM whether a proposal is legal.

The same accepted parent may not retry a parameter whose earlier executed
candidate was `FALSIFIED` or `UNCERTAIN`; an accepted-state change resets this
eligibility. A contract-invalid specialist response receives exactly one
corrective call in the same round. If the second response is still invalid,
write a `CONTRACT_REJECTED` shared-memory record and consume that round.

## 8. Qlib execution adapter

Keep DiagAgent orchestration independent of project-specific Qlib code. Implement an adapter with behavior equivalent to:

```text
load_config(path) -> normalized config
run_alpha(config, split) -> alpha artifacts
train_model(config, alpha_artifacts) -> checkpoints + train/train_valid metric history
select_checkpoint(checkpoints, metric=config.z_model.base_params.metric, mode=max) -> checkpoint
predict(checkpoint, split=agent_valid) -> agent_valid_alpha.csv
run_portfolio(config, agent_valid_alpha.csv, split=agent_valid) -> account/order artifacts
evaluate(split=agent_valid) -> exact result metrics + diagnostic evidence
```

Before implementation, inspect the actual Qlib runner and every current
`submit/*.json`; map their existing keys rather than creating a parallel
configuration format. At runtime, operate on exactly the submit file selected
by `--submit`. DiagAgent's `config.json` is an experiment envelope around the
effective Qlib config, not a replacement for Qlib semantics.

## 9. Validation Gate and accepted state

The Validation Gate compares the candidate with its accepted parent using
validation evidence only. Absolute strategy Sharpe is the sole adaptive
primary metric; excess metrics are diagnostic-only. The original ordered
Sharpe/mechanism/trade-off Gate remains active with fixed
`epsilon_sharpe = 0.002`:

```text
fatal contract/data/execution checks
  -> Sharpe state versus fixed 0.002 epsilon
  -> expected mechanism signature
  -> frozen trade-off guardrails
```

Verdicts:

- `SUPPORTED`: Sharpe improves above `0.002`, every declared mechanism metric
  matches, and no guardrail materially degrades. Promote.
- `PARTIALLY_SUPPORTED`: Sharpe improves above `0.002` without a complete match,
  or Sharpe is within `[-0.002, 0.002]` while mechanism state is `MATCH` or
  `PARTIAL`; no opposite mechanism or material trade-off may exist. Promote.
- `UNCERTAIN`: unchanged Sharpe without mechanism support, or required evidence
  is unavailable. Do not promote.
- `FALSIFIED`: Sharpe worsens below `-0.002`, a mechanism is opposite, or a
  guardrail materially degrades. Do not promote.

There are currently no hard performance thresholds, including no fixed maximum
drawdown limit. Contract, data-integrity, isolation, and execution-legality
failures remain fatal.

`epsilon_sharpe` is the frozen constant `0.002`. The paired moving-block
bootstrap remains in use for mechanism and guardrail metrics: align validation
daily observations and resample 2,000 moving-block samples of 20 trading days
using seed 0. Define
`numerical_tolerance = max(1e-12, 1e-10 * max(abs(parent_statistic),
abs(candidate_statistic)))`, canonicalize observed deltas at or below that
tolerance to zero. For each mechanism and guardrail metric, set its epsilon to
`max(1.96 * std(bootstrap_delta, ddof=1), numerical_tolerance)`. If insufficient data
prevents a required mechanism or guardrail calculation, return `UNCERTAIN`.

Apply this exact verdict order:

```text
1. Sharpe < -0.002                              -> FALSIFIED
2. any mechanism metric materially opposite    -> FALSIFIED
3. any guardrail materially degrades            -> FALSIFIED
4. required evidence unavailable                -> UNCERTAIN
5. Sharpe > +0.002 and full match                -> SUPPORTED
6. Sharpe > +0.002 but incomplete match          -> PARTIALLY_SUPPORTED
7. Sharpe within [-0.002, 0.002] and match/partial -> PARTIALLY_SUPPORTED
8. otherwise                                     -> UNCERTAIN
```

Equality at `+0.002` belongs to the unchanged band. Contract,
data, access-control, and execution failures are fatal before Gate evaluation.
Store daily Gate inputs in `artifacts/portfolio_daily_diagnostics.csv` and the
complete deterministic comparison in `validation_gate.json`; do not add these
diagnostics to the exact top-level `result.json` contract.

Both `SUPPORTED` and `PARTIALLY_SUPPORTED` promote. This applies to RASS as
well: its portfolio-based Gate verdict controls promotion or rollback.

FAMA has one additional explicit promotion override. Compare the candidate's
full-precision ensemble mean daily validation IC with the accepted parent's
value. Promote only when the ordinary verdict is `UNCERTAIN`, ensemble
`delta_valid_ic > 0.001`, at least two of three seed IC deltas are positive,
and their median exceeds `0.001`. Never override `FALSIFIED`, worse-Sharpe,
opposite-mechanism, or material-trade-off outcomes. Preserve the ordinary verdict and
all Sharpe, mechanism, and trade-off evidence in `validation_gate.json`; record
`promotion_reason = fama_validation_ic_override` and
`gate_verdict_usage = overridden_by_fama_validation_ic`. Equality at `0.001`
does not trigger the override. RASS and RAPA never receive this exception, and
test evidence remains forbidden.

The accepted-state update must be atomic:

```text
<run-root>/configs/current.json -> candidate config
accepted_experiment_id -> candidate EXP_xxx
```

If promotion fails midway, restore both pointers to the parent. Never delete a rejected experiment.

## 10. Validation evidence

`result.json` is intentionally small and has the exact keys defined in `AGENTS.md`. Build richer internal evidence separately for CLEM, specialists, and the Validation Gate. It may include:

- alpha/model diagnostics: IC, RankIC, ICIR, RankICIR, checkpoint epoch, learning behavior, generalization gap when available;
- portfolio diagnostics: strategy and arithmetic-excess ARR, volatility,
  drawdown, Sharpe/IR, Calmar, Sortino, win rate, buy/sell/one-way turnover,
  holdings, universe size, and invested weight;
- RAPA objective diagnostics over the complete validation period, not a single day: alpha term, risk term, turnover term, and their ratios;
- constraints and deltas versus the accepted parent.

Never add these richer structures as extra keys in `result.json`. Store them as artifacts or deterministic in-memory evidence, then compress only admissible summaries into Memory Bank entries.

Standard Sharpe:

```text
mean(net_daily_return) / std(net_daily_return, ddof=1) * sqrt(252)
```

The complete strategy/excess metric formulas and exact 27-field result contract
are authoritative in `AGENTS.md` and `schemas/result.schema.json`. Daily excess
return is strategy net return minus the aligned `task.benchmark` return;
`information_ratio` is the annualized Sharpe of that excess-return series.

## 11. A-share alignment and execution

For signal date `T`:

```text
alpha[T]
  -> rebalance using T+1 close and T+1 directional masks
  -> realize close(T+2) / close(T+1) - 1
```

Use two separate masks with `1 = limit event`, `0 = normal`:

```text
mask_limit_up_1D.csv    -> cannot buy/increase
mask_limit_down_1D.csv  -> cannot sell/decrease
```

Encode them as per-asset lower/upper weight bounds relative to old weight:

```text
limit-up:   upper_bound <= old_weight
limit-down: lower_bound >= old_weight
```

Missing mask values conservatively block the relevant direction. Apply masks on the trade date, not the alpha date or realized-return date. Unit tests must use tiny hand-computable dates to detect off-by-one leakage.

For each signal date, Qlib alpha generation uses exactly the universe named by
`task.instruments`. Form the common ranking universe from stocks in that
configured universe with finite processed alpha scores; do not apply another
constituent-membership filter at the portfolio boundary. On the T+1 trade date,
remove names whose limit-up mask is positive or missing before ranking; these
names cannot consume a new-entry candidate slot. Limit-down names remain
buyable. Rank the remaining finite-alpha universe and take the Top 100 as new-entry
candidates. The Mean-Variance optimizer universe is those Top-100 new entries
union every existing holding, including holdings that are no longer eligible
for new entry, so the optimizer-universe and final holding counts are not
capped at 100. Before ranking, restrict the executable universe to finite-alpha
assets whose columns are physically present in the return, limit-up, and
limit-down panels. Ignore predicted assets missing from any panel header and
persist their codes/counts for audit; this is not a fatal data-coverage error.
After column-level admission, treat a missing mask value on an individual date
under the frozen conservative directional rule.

The first successful Mean-Variance portfolio build uses an effective turnover
penalty of zero because no prior portfolio exists. Normal commissions, tax, and
slippage still apply to that build. Every later rebalance uses the configured
`turnover_penalty`; the zero-penalty initialization is deterministic execution
semantics, not a RAPA parameter change.

The TopK comparison is Top100/Drop10. It must receive the same finite-alpha
ranking universe, Top-100 new-candidate count, signal and return dates,
directional limit masks, and transaction-cost assumptions as the Mean-Variance
path. Portfolio construction and allocation are the comparison variable; do
not give either baseline a broader stock pool or different execution timing.

The Mean-Variance path targets 100% invested weight. If frozen directional
bounds, `max_weight`, or the buyable universe make that target mathematically
infeasible on a date, set the target to the maximum feasible invested weight and
hold the remainder as zero-return cash. Do not relax bounds and do not fail the
experiment solely for this condition. Persist the target, feasible minimum and
maximum, actual invested weight, shortfall, feasibility flag, and deterministic
reason code in `portfolio_daily_diagnostics.csv`; shortfall remains a Validation
Gate trade-off guardrail.

## 12. Test isolation

Every epoch receives immediate test-model evaluation through one persistent
researcher process after that epoch checkpoint is written. The process owns the
test Dataset for the full experiment, loads it once, and appends
only to `test/epoch_metrics.jsonl` and returns no test values to the trainer.
The selected checkpoint receives the complete test prediction, portfolio, and
27-field evaluation only after the adaptive outputs for that round are sealed.
Failure of either researcher job is reported independently and must not alter
the adaptive verdict or state. The adaptive process must use an explicit read
allowlist such as:

```text
allowed: repository configs/ plus active <run-root>/configs/, decision.json,
         result.json, artifacts/validation-safe files, and memory/
denied:  **/test/**, test metrics, test predictions, test summaries
```

Researcher startup and request/response waits have finite frozen timeouts. If
the persistent worker times out or fails, terminate it, disable further
per-epoch dispatches for that experiment, record only a generic failure status,
and continue adaptive training. The final isolated one-shot test evaluation may
be attempted once with a separate finite timeout; its failure cannot block or
change the adaptive trajectory.

The researcher-side evaluator writes only to
`<run-root>/experiments/EXP_xxx/test/`. It writes the per-epoch model
metric curve to `test/epoch_metrics.jsonl`, the selected-checkpoint metric
object to `test/result.json`, and may retain `test/test_alpha.csv` and
`test/portfolio_daily_diagnostics.csv` for researcher audit. It must not update
current config, checkpoint selection, verdicts, memory, routing, stopping,
prompts, or summaries.

Training shuffle reproducibility is defined by one permutation generated from
the fixed `task.seed` with `torch.randperm` and reused identically for every
epoch. Store the fixed seed, epoch, `fixed_across_epochs` policy, algorithm, and
sample count in `logs/training_metrics.jsonl`; do not persist full permutation
arrays.

After finalization, a separate researcher subprocess writes
`<run-root>/test/researcher_outputs_audit.json`. This audit reports only
the presence and completion status of required per-round researcher artifacts,
contains no test metrics, and cannot affect the adaptive state.

After the adaptive trajectory stops, copy the final accepted configuration to
`<run-root>/configs/final_frozen.json` and record its hash. Any official
final test/reporting stage consumes only this frozen configuration.

## 13. Memory Bank

Use one append-only `<run-root>/memory/experiments.jsonl` plus materialized
projections for that trajectory:

```text
cross_layer -> CLEM
alpha       -> RASS
model       -> FAMA
portfolio   -> RAPA
```

Each memory item records:

```text
state before
-> CLEM diagnosis
-> selected layer/group
-> candidate diff
-> expected signature
-> observed validation signature
-> verdict
-> accepted true/false
```

Write memory after every executed candidate, including rolled-back candidates. Memory is evidence about consequences, not a best-configuration table. Never write test-derived content.

Every round retrieves the complete validation-safe Memory Bank. Do not use
top-k, similarity ranking, or truncation. Order all records by ascending round
and then ascending experiment ID. Role-specific projections may hide irrelevant
fields but may not omit stored experiments or verdicts. If the complete bank
cannot fit in the configured model context, stop explicitly rather than
silently truncating it.

## 14. Required project layout

```text
DiagAgent/
  AGENTS.md
  docs/
    DiagAgent_Implementation_Spec.md
    Codex_Implementation_Handoff.md
  submit/
    *.json                              # explicit trajectory seed candidates
  configs/
    frozen_contract.json
    agent_models.json
  agents/
    CLEM/{SYSTEM.md,decision.schema.json}
    RASS/{SYSTEM.md,intervention.schema.json,intervention_space.yaml}
    FAMA/{SYSTEM.md,intervention.schema.json,intervention_space.yaml,intervention_spaces/}
    RAPA/{SYSTEM.md,intervention.schema.json,intervention_space.yaml}
  core/
    orchestrator.py
    contract_validator.py
    executor.py
    validation_gate.py
    evidence_builder.py
    state_store.py
    split_guard.py
  adapters/
    qlib_runner.py
    portfolio_runner.py
    llm_client.py
  training/
    lstm_trainer.py
  memory/
    memory_store.py
  runs/
    <task.name>/
      configs/{initial,current,final_frozen}.json
      memory/{experiments.jsonl,summaries/}
      experiments/{EXP_000,EXP_001,...}/
  schemas/
  scripts/
    run_round0.py
    run_single_round.py
    run_adaptation.py
    run_researcher_test.py
  tests/
  examples/                              # optional curated examples
```

The layout above is the target layout for this repository. Inspect the existing
Qlib-compatible data, all current `submit/*.json`, `backbone/`, and the
portfolio example to
understand the intended logic before scaffolding implementation files. Reuse
code only when it remains the clearest reliable option; compatibility with the
example runners is not required, and they may be replaced completely.

`configs/agent_models.json` is only the provider/API-key/model catalog. The
active Agent base model is selected by `agent.model` and
`agent.reasoning_effort` in the chosen `submit/*.json`. Resolve environment
variable references at runtime, validate the selection against the catalog,
and never persist or log the resolved API key. Additional OpenAI-compatible
providers and models may be added to the catalog without changing submit
schema. Freeze the resolved selection into the trajectory's initial config.

## 15. Stop conditions

Normal adaptive stopping occurs only when the experiment budget is exhausted.
Weak evidence, diminishing returns, or repeated falsified/uncertain outcomes
require cross-layer reassessment and the best-supported eligible bounded
experiment; they do not stop the trajectory. A deterministic safety,
data-integrity, infrastructure, or required-evidence failure terminates as an
explicit failure rather than a successful early stop.

`task.trials = 10` means ten adaptive rounds after Round 0
(`EXP_001` through `EXP_010`). The consecutive `UNCERTAIN`/`FALSIFIED` limit is
also ten, so it does not create an earlier stop by itself; cross-layer
reassessment remains mandatory after each such verdict.

Stopping must not depend on test results or on whether unexplored parameters remain.

## 16. Minimum acceptance tests

Before running a formal trajectory, tests must prove:

1. Round 0 has no adaptive decision.
2. Test paths cannot enter any adaptive context.
3. Exact `result.json` keys and three-decimal serialization are enforced.
4. FAMA cannot mix groups or change more than two parameters.
5. RAPA cannot change more than one parameter.
6. Frozen fields cannot change.
7. Rejected contracts never invoke Qlib.
8. RAPA does not retrain the model; FAMA/RASS rerun their required downstream stages.
9. The configured validation-only metric selects the checkpoint; default IC and
   all four allowed metrics are tested.
10. T/T+1/T+2 alignment and both directional masks are correct.
11. `SUPPORTED` and `PARTIALLY_SUPPORTED` update accepted state. Verify that
    RASS uses the same Gate, retries only with a non-identical three-factor set,
    and after two complete failures freezes the original five factors.
    Verify that every later CLEM decision is restricted to FAMA or RAPA.
    Verify separately that only FAMA promotes through
    `fama_validation_ic_override` when `delta_valid_ic > 0.001`, while the
    ordinary Gate verdict remains unchanged for audit.
12. Rejected candidates remain auditable and retrievable as memory.
13. CLEM anti-repetition and budget-only normal stopping work.
14. Test evaluation cannot mutate memory, accepted config, or stopping state.

## 17. Recommended implementation order

1. Config normalization, frozen contract, schema validation, and state store.
2. Qlib/portfolio adapters and a reproducible Round 0.
3. Evidence builder and exact result writer.
4. Contract Validator plus unit tests.
5. Validation Gate and atomic promotion/rollback.
6. Shared Memory Bank and retrieval views.
7. CLEM and specialist prompt/schema integration.
8. End-to-end adaptive orchestrator and restartability.
9. Physically isolated researcher test process.
10. Formal dry run with synthetic data before real Qlib data.

The first milestone is not “all agents run.” It is: **the same frozen Round 0 configuration produces the same validation artifacts and exact result object twice, with no test path accessible.**
