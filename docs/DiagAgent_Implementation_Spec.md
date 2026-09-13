# DiagAgent Implementation Specification

## Version 1.0 — End-to-End Round-0-to-Test Workflow

This document is the detailed implementation specification for the runnable DiagAgent repository at `C:\Users\Andy\Desktop\DigAgent_WWW`. Read the root `AGENTS.md` first; when this document is incomplete or conflicts with it, `AGENTS.md` is authoritative. DiagAgent uses the installed Qlib package and the Qlib-format data in this repository; it is not implemented in a separate Qlib source repository.

All existing source code is explanatory reference material, not a frozen
implementation. Codex may rewrite, reorganize, replace, or remove it and may
change interfaces as needed to produce a clean, tested, end-to-end runnable
system. Existing code should be inspected for intended logic, not preserved for
compatibility. This freedom does not extend to immutable files under `data/` or
to the frozen behavioral and isolation contracts in `AGENTS.md`.

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
5. `SUPPORTED` and `PARTIALLY_SUPPORTED` candidates become the accepted state,
   except that the mandatory RASS structural initialization in `EXP_001`
   promotes after legal successful execution regardless of its audit verdict.
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

Validate `task.name` against `[A-Za-z0-9][A-Za-z0-9_-]*`, derive the trajectory
root `runs/<task.name>/`, and write the fully expanded configuration once to
`runs/<task.name>/configs/initial.json`. Alongside
it, persist validation-safe provenance containing all inserted paths and values
with their source or rationale, dependency versions, relevant data hashes, and
the final configuration SHA256. Schema, path, split, feature, label, and data-
coverage checks must pass before Round 0 starts, and materialization must not
read test results or test-derived artifacts.

Once `EXP_000` begins, `runs/<task.name>/configs/initial.json` is immutable for
that formal trajectory. Initialize `runs/<task.name>/configs/current.json` from
it byte-for-byte or from a
canonical semantically identical serialization. A later dependency or default
change requires a new formal trajectory, not regeneration of the existing
baseline.

All mutable state and outputs are scoped under that run root. An existing run
may resume only when its recorded submit path and SHA256 match; otherwise fail
without overwriting or mixing submissions.

Dependency-aware reruns:

| Selected agent | Recompute | Reuse |
|---|---|---|
| RASS | factors, model, predictions, portfolio, validation | immutable source data |
| FAMA | model, predictions, portfolio, validation | selected factors/source data |
| RAPA | portfolio and validation backtest | factors, trained model, predictions |

The executor must verify parent artifact identity/hash before reuse. RAPA reuses
the latest accepted parent's `valid_alpha.csv`, signal metrics, and selected
checkpoint. Interrupted RAPA recovery validates that checkpoint against the
accepted parent's model-config hash, not against the portfolio-only candidate
hash. It must not silently reuse stale outputs.

### 3.2 Canonical prediction-to-portfolio interface

`artifacts/valid_alpha.csv` and any equivalent split prediction passed to the
portfolio runner must be a UTF-8 comma-separated wide panel physically
compatible with `data/portfolio/c_2_c_1D.csv`:

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
and `n_jobs=0`. The training loader shuffles and drops the last incomplete
batch; train-evaluation, validation, and isolated test-evaluation loaders do
neither. Derive `d_feat` from the selected-feature count (6 for `EXP_000`, 10
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

After every epoch checkpoint is written, a separate researcher-side evaluator
immediately appends that epoch's train/valid/test metrics to
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

Round 0 creates `runs/<task.name>/experiments/EXP_000` from the submit JSON explicitly selected
for the current trajectory.

Procedure:

1. Load and validate the initial JSON.
2. Freeze the initial contract and record a hash of it.
3. Run the full Qlib alpha -> model -> prediction -> portfolio -> validation pipeline.
4. Write `config.json`, the exact validation `result.json`, validation artifacts, and logs.
5. Mark `EXP_000` as the initial accepted experiment.
6. Write the baseline memory item.

## 4.1 Mandatory Round 1 RASS initialization

The current baseline begins with six ordered features. Immediately after
`EXP_000`, while `alpha_frozen=false`, CLEM must use the observed configuration
count as evidence of insufficient initial feature breadth and route `EXP_001`
to RASS. This is a frozen first-round priority, not an open-ended preference.

RASS is an LLM-backed specialist. It follows `agents/RASS/SYSTEM.md` and
`agents/RASS/SKILL.md`, explores the frozen catalog using a deterministic
train-only evidence table, preserves all six anchors in their original order,
and selects exactly four additions. Its only legal diff is the four-item
append that produces exactly ten features. It then reruns alpha construction,
model training, validation prediction, portfolio optimization, and validation
evaluation.

After a legal proposal and successful complete execution, promote `EXP_001`
regardless of its normal Validation Gate verdict and set `alpha_frozen=true`.
The Gate verdict remains one of `SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNCERTAIN`,
or `FALSIFIED` and is retained solely for audit and memory; record
`mandatory_rass_structural_initialization` separately as the acceptance reason.
Do not add this reason to `result.json` and do not create a fifth verdict. If
the Agent cannot propose four legal, train-evidence-supported additions,
or execution has a fatal failure, stop explicitly and do not promote a partial
feature set. After success, CLEM can no longer route to RASS in this trajectory.

Round 0 has no CLEM decision, specialist proposal, config diff, or rollback decision.

## 5. Adaptive round state machine

For round `t >= 1`:

```text
LOAD_ACCEPTED_PARENT
  -> BUILD_VALIDATION_EVIDENCE
  -> CLEM_DECISION
       -> NO_INTERVENTION -> STOP
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

CLEM returns one of `RASS`, `FAMA`, `RAPA`, or `NO_INTERVENTION`, plus:

- selected layer;
- failure summary;
- supporting evidence;
- why the other layers are less supported;
- intervention goal;
- confidence.

CLEM must not name a parameter, direction, magnitude, or new value.

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
alpha_scale | risk_aversion | turnover_penalty
```

Inclusive RAPA bounds are:

```text
risk_aversion     float [0.1, 2.0]
turnover_penalty  float [0.1, 2.0]
alpha_scale       float [0.0008, 0.0015]
```

There is no additional per-round relative-change cap inside these ranges.

RASS has one append-only `initial_feature_bootstrap` group and is legal only in
`EXP_001`. It preserves the six initial ordered features and, as an LLM-backed
specialist, performs two train-only Agent stages. The first reviews every
deterministically eligible catalog factor and selects exactly twelve candidates.
Code then computes shortlist-only daily cross-sectional redundancy and
information remaining after linear residualization against the six anchors.
The second Agent stage selects exactly four factors from that shortlist,
reaching ten total. It cannot
remove, replace, reorder, or edit an anchor; add duplicates; or invent
expressions outside the pool. Resolve, persist, and hash the complete pool
catalog and train-only evidence table before the formal trajectory so later
Qlib/library changes cannot alter the available evidence. Deterministic code
must validate the Agent's proposal but must not select or rank factors for it.
After successful execution, alpha is frozen for all remaining rounds.

Eligibility removes only factors with inadequate coverage or valid IC dates,
mostly constant daily cross-sections, or duplication of an anchor; it is not a
predictive ranking. Neither selection stage receives formal validation
evidence. Validation evaluates the sealed intervention only after execution and
cannot trigger RASS reselection.

Use all train rows for IC, RankIC, coverage, and temporal stability. To keep
initialization bounded, Stage-1 all-candidate redundancy uses a deterministic
evenly spaced sample capped at 20,000 rows. The twelve-factor shortlist stage
uses full train daily cross-sections for its joint evidence.

The machine-readable versions are the three
`agents/*/intervention_space.yaml` files. Contract validation must read them;
do not duplicate numeric bounds in application code.

An intervention route requires CLEM confidence >= 0.5. Lower confidence must
produce `NO_INTERVENTION`.

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
train_model(config, alpha_artifacts) -> checkpoints + train/valid metric history
select_checkpoint(checkpoints, metric=config.z_model.base_params.metric, mode=max) -> checkpoint
predict(checkpoint, split=valid) -> valid_alpha.csv
run_portfolio(config, valid_alpha.csv, split=valid) -> account/order artifacts
evaluate(split=valid) -> exact result metrics + diagnostic evidence
```

Before implementation, inspect the actual Qlib runner and every current
`submit/*.json`; map their existing keys rather than creating a parallel
configuration format. At runtime, operate on exactly the submit file selected
by `--submit`. DiagAgent's `config.json` is an experiment envelope around the
effective Qlib config, not a replacement for Qlib semantics.

## 9. Validation Gate and accepted state

The Validation Gate compares the candidate with its accepted parent using
validation evidence only. Sharpe is primary; mechanism evidence and frozen
trade-off guardrails provide the additional ordered checks below:

```text
fatal contract/data/execution checks
  -> Sharpe change versus frozen noise tolerance
  -> expected mechanism signature
  -> frozen trade-off guardrails
```

Verdicts:

- `SUPPORTED`: Sharpe improves by more than `epsilon_sharpe`, every declared
  mechanism metric materially matches, and no guardrail materially degrades.
  Promote the candidate.
- `PARTIALLY_SUPPORTED`: Sharpe materially improves without a complete
  mechanism match, or Sharpe is unchanged while the mechanism state is
  `MATCH`/`PARTIAL`; no mechanism may be `OPPOSITE` and no guardrail may
  materially degrade. Promote the candidate.
- `UNCERTAIN`: Sharpe is unchanged without material mechanism support, or
  required Sharpe/mechanism/guardrail evidence cannot be computed. Do not
  promote.
- `FALSIFIED`: Sharpe materially worsens, a mechanism metric is materially
  opposite, or a guardrail materially degrades. Do not promote.

There are currently no hard performance thresholds, including no fixed maximum
drawdown limit. Contract, data-integrity, isolation, and execution-legality
failures remain fatal.

`epsilon_sharpe` is computed for each candidate-parent comparison by the frozen
paired moving-block bootstrap algorithm: align validation daily net returns,
resample paired rows with 2,000 moving-block bootstrap samples of 20 trading
days using seed 0 and compute delta Sharpe for every sample. Define
`numerical_tolerance = max(1e-12, 1e-10 * max(abs(parent_statistic),
abs(candidate_statistic)))`, canonicalize observed deltas at or below that
tolerance to zero, and set `epsilon_sharpe = max(1.96 *
std(bootstrap_delta_sharpe, ddof=1), numerical_tolerance)`. Apply the same
numerical floor to every mechanism and guardrail epsilon. If insufficient data
prevents the bootstrap calculation, return `UNCERTAIN`.

Apply this exact verdict order:

```text
1. Sharpe < -epsilon                         -> FALSIFIED
2. any mechanism metric materially opposite -> FALSIFIED
3. any guardrail materially degrades         -> FALSIFIED
4. required evidence unavailable             -> UNCERTAIN
5. Sharpe > +epsilon and full match           -> SUPPORTED
6. Sharpe > +epsilon but incomplete match     -> PARTIALLY_SUPPORTED
7. Sharpe unchanged and match/partial match   -> PARTIALLY_SUPPORTED
8. otherwise                                  -> UNCERTAIN
```

Equality is inside the unchanged/noise band: `abs(delta) <= epsilon`. Contract,
data, access-control, and execution failures are fatal before Gate evaluation.
Store daily Gate inputs in `artifacts/portfolio_daily_diagnostics.csv` and the
complete deterministic comparison in `validation_gate.json`; do not add these
diagnostics to the exact top-level `result.json` contract.

Except for the explicitly defined mandatory RASS `EXP_001` structural
initialization, both `SUPPORTED` and `PARTIALLY_SUPPORTED` promote a candidate.
Secondary metrics never compensate for materially worse Sharpe. For `EXP_001`, compute
the same Gate verdict for audit, but successful legal execution—not that
verdict—controls its structural promotion.

The accepted-state update must be atomic:

```text
runs/<task.name>/configs/current.json -> candidate config
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
constituent-membership filter at the portfolio boundary. Rank this universe and
take the Top 100 as new-entry
candidates. The Mean-Variance optimizer universe is those Top-100 new entries
union every existing holding, including holdings that are no longer eligible
for new entry, so the optimizer-universe and final holding counts are not
capped at 100. Return and mask panels supply realized returns and directional
constraints; they do not define candidate membership. Ignore extra mask
columns after deterministic reindexing, and treat missing mask values under the
frozen conservative directional rule.

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

Every epoch receives immediate test-model evaluation through a separate
researcher process after that epoch checkpoint is written. The process appends
only to `test/epoch_metrics.jsonl` and returns no test values to the trainer.
The selected checkpoint receives the complete test prediction, portfolio, and
27-field evaluation only after the adaptive outputs for that round are sealed.
Failure of either researcher job is reported independently and must not alter
the adaptive verdict or state. The adaptive process must use an explicit read
allowlist such as:

```text
allowed: repository configs/ plus active runs/<task.name>/configs/, decision.json,
         result.json, artifacts/validation-safe files, and memory/
denied:  **/test/**, test metrics, test predictions, test summaries
```

The researcher-side evaluator writes only to
`runs/<task.name>/experiments/EXP_xxx/test/`. It writes the per-epoch model
metric curve to `test/epoch_metrics.jsonl`, the selected-checkpoint metric
object to `test/result.json`, and may retain `test/test_alpha.csv` and
`test/portfolio_daily_diagnostics.csv` for researcher audit. It must not update
current config, checkpoint selection, verdicts, memory, routing, stopping,
prompts, or summaries.

After finalization, a separate researcher subprocess writes
`runs/<task.name>/test/researcher_outputs_audit.json`. This audit reports only
the presence and completion status of required per-round researcher artifacts,
contains no test metrics, and cannot affect the adaptive state.

After the adaptive trajectory stops, copy the final accepted configuration to
`runs/<task.name>/configs/final_frozen.json` and record its hash. Any official
final test/reporting stage consumes only this frozen configuration.

## 13. Memory Bank

Use one append-only `runs/<task.name>/memory/experiments.jsonl` plus materialized
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

Stop when any frozen rule triggers:

- CLEM returns `NO_INTERVENTION`;
- experiment budget is exhausted;
- no material unresolved failure remains;
- repeated interventions are falsified/uncertain and cross-layer reassessment finds no supported action;
- expected improvement is within validation noise/diminishing returns;
- a deterministic safety or data-integrity condition fails.

`task.trials = 10` means at most ten adaptive rounds after Round 0
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
11. `SUPPORTED` and `PARTIALLY_SUPPORTED` normally update accepted state; also
    verify the sole structural exception that a legal, successfully executed RASS `EXP_001` is promoted
    for structural initialization regardless of its audit-only Gate verdict,
    followed by `alpha_frozen=true` and permanent denial of later RASS routes.
12. Rejected candidates remain auditable and retrievable as memory.
13. CLEM anti-repetition and `NO_INTERVENTION` work.
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
