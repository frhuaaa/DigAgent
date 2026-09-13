# DiagAgent authoritative implementation contract

This file is the concise, authoritative contract for coding agents working in
this repository. `C:\Users\Andy\Desktop\DigAgent_WWW` is the real, runnable
DiagAgent implementation repository; do not search for or require a separate
Qlib source repository. Read this file before changing code. Also consult
`docs/DiagAgent_Implementation_Spec.md` and the current-contract examples under
`agents/`, `backbone/`, `submit/`, and `examples/`, but treat them as supporting
material. Never use `legacy_unsafe_examples/` as implementation or Agent input.

If this file conflicts with the long specification, an example, a prompt, or
existing code, **this file wins**. Do not silently resolve a genuine ambiguity:
record it and request an explicit contract change.

## Status of the existing code

All code currently present in this repository is **reference/example code**
whose purpose is to help Codex understand the intended data flow, model,
portfolio logic, file formats, and experiment behavior. It is not a frozen API,
not a compatibility target, and not required to be preserved.

When implementing the runnable system, Codex is explicitly authorized to edit,
refactor, replace, move, rename, or delete existing code; change module and class
boundaries; and create new files as needed. Prefer the clearest tested
implementation that runs end to end over preserving an example's structure or
historical behavior. Existing imports, function signatures, runners, and model
wrappers may all change. Before replacing useful logic, first inspect it and
carry forward any behavior that is required by this contract.

This permission applies to code, not to the frozen experimental contract or
source data. `AGENTS.md` remains authoritative; the required schemas,
intervention rules, validation-only adaptation, test isolation, execution
alignment, and other frozen decisions may not be weakened merely to make the
code run. Files under `data/` are immutable inputs. Generated experiment output
may be recreated only through the implemented pipeline.

## Goal and non-goals

DiagAgent is a diagnosis-driven, cross-layer experiment system for quantitative
trading under noisy validation feedback. This repository uses the installed
Qlib package as its data, factor, dataset, training, and prediction framework;
Qlib is a dependency, not a second implementation location. Every formal
trajectory must be launched with an explicit `--submit submit/<file>.json`;
there is no implicit or hard-coded default submit file. The selected submit
JSON is that trajectory's authoritative seed, and its adaptive sections are
`z_alpha`, `z_model`, and `z_portfolio`. Use `task.name` to keep state and
outputs from different submit files from colliding.

Repository-relative input locations are frozen as follows:

```text
data/cn_data/                         Qlib provider_uri
data/portfolio/c_2_c_1D.csv          close-to-close return panel
data/portfolio/mask_limit_up_1D.csv   directional limit-up mask
data/portfolio/mask_limit_down_1D.csv directional limit-down mask
```

Resolve these paths from the repository root, not the caller's working
directory. Treat source data as immutable input: implementation and tests must
not rewrite files under `data/`.

### Initial configuration materialization

The JSON selected by the required `--submit` argument is the authoritative seed
template for that trajectory. Record its repository-relative path and SHA256 in
the materialized provenance; never silently substitute another submit file.
It names a fixed pipeline contract such as `qlib_ts_lstm_v1` and keeps only
experiment-facing settings such as batch size, worker count, and label
normalization. Low-level
Dataset/Handler construction, processor order, data keys, loader behavior, and
`d_feat` derivation belong to that versioned pipeline contract below and are
implemented as fixed behavior rather than repeated as verbose JSON. Before the
first `EXP_000` run, Codex may supply only genuinely required runtime fields
that are absent from both the seed and this contract. It must not reinterpret an
explicit seed or pipeline-contract value because example or library defaults
differ.

Derive the run root as `runs/<task.name>/`, after validating `task.name` against
`[A-Za-z0-9][A-Za-z0-9_-]*`. Materialize the fully expanded result exactly once
as `runs/<task.name>/configs/initial.json`
before executing Round 0. The materialization step must be deterministic and
must record, in validation-safe provenance, every supplied field and its source
or rationale, the Qlib/Python dependency versions, relevant input-data hashes,
and the SHA256 of the completed configuration. Validate that no test-derived
observation was used. Do not start `EXP_000` until this file is complete and
passes schema and data-coverage validation.

After materialization, `runs/<task.name>/configs/initial.json` is the frozen
Round 0 baseline:
never regenerate or overwrite it during that formal trajectory, never consult
library or code defaults for an omitted runtime value, and never reinterpret it
after a dependency change. `runs/<task.name>/configs/current.json` initially
copies this exact
configuration. Every experiment must be reproducible from the materialized
configuration rather than from implicit defaults in example code.

All mutable trajectory state belongs under this run root. If it already exists,
resume only when its recorded submit path and SHA256 match the selected submit;
otherwise fail instead of overwriting or mixing trajectories.

Core claim: bounded, falsifiable interventions selected through cross-layer
diagnosis and experimental memory should generalize better than aggressive
validation-set search.

Required loop:

```text
baseline -> evidence -> diagnosis -> bounded intervention -> falsification
         -> memory -> next round or stop
```

Do not turn the system into grid search, random search, Bayesian optimization,
multi-parameter search, or "try configurations and retain the best validation
Sharpe". Memory is causal experimental evidence, not a parameter leaderboard.

## Fixed architecture and ownership

```text
CLEM -> exactly one of RASS / FAMA / RAPA
     -> deterministic Contract Validator
     -> Executor
     -> deterministic Validation Gate
     -> shared Memory Bank
     -> next CLEM round
```

- CLEM owns **WHERE + WHY**. It compares all three layers and returns `RASS`,
  `FAMA`, `RAPA`, or `NO_INTERVENTION`, together with the dominant unresolved
  failure, validation evidence, why alternatives are weaker, the intervention
  goal, and confidence. It must not choose parameter names, values, direction,
  or magnitude.
- RASS, FAMA, and RAPA own **WHAT + HOW** only within their frozen layer and
  intervention space. Each proposal must state a coherent hypothesis, expected
  observable signature, and falsification condition.
- Contract Validator and Validation Gate are deterministic. Do not delegate
  legality, acceptance, rollback, or stopping to an LLM.
- Executor applies only the validated diff and reruns the minimum dependency
  path: RAPA reuses predictions; FAMA retrains model then prediction and
  portfolio; RASS reruns alpha then model, prediction, and portfolio.
- RAPA must reuse `valid_alpha.csv`, signal metrics, and the selected model
  checkpoint from the latest accepted parent. Interrupted RAPA recovery checks
  the checkpoint against that parent's model-config hash, not against the
  portfolio-only candidate hash.

### Agent model runtime

- `configs/agent_models.json` contains only provider connection settings,
  API-key references, and the allowed Agent model catalog. Do not store Agent
  role behavior, experiment policy, or the active model selection there.
- Each `submit/*.json` selects the trajectory's Agent base model through
  `agent.model` and `agent.reasoning_effort`. The model name must exist in the
  catalog and the requested effort must be allowed by that catalog entry.
- `agent.model` applies to the LLM-backed CLEM, RASS, FAMA, and RAPA roles. Only CLEM
  and the specialist it selects are invoked in an adaptive round.
- RASS factor selection is Agent-driven under `agents/RASS/SYSTEM.md` and
  `agents/RASS/SKILL.md`. Deterministic code constructs train-only evidence and
  validates the proposal, but must not use a fixed threshold, greedy ranking,
  or rule-based selector to choose factors for RASS.
- Resolve `${ENV_VAR}` API-key references at runtime and never write resolved
  secrets to configs, decisions, logs, artifacts, memory, or prompts. Additional
  providers/models may be added to the catalog without changing submit schema.
- Materialize and freeze the selected model and reasoning effort with the
  initial configuration before Round 0. Changing either starts a new trajectory.
  Do not silently fall back to another model.

### Model/training separation

- Select the architecture example from frozen `z_model.model_name`:
  `backbone/lstm_example.py`, `backbone/gru_example.py`, or
  `backbone/alstm_example.py`, with shared modules in `backbone/modules.py`.
  These files contain model architecture only, with no Qlib dataset preparation,
  optimization loop, split selection, metric calculation, checkpoint choice,
  prediction export, or experiment logging.
- Put training orchestration in a separate trainer and call it through the Qlib
  adapter. Prediction APIs must require an explicit split; never hard-code
  `test` inside a generic `predict()` method.
- Evaluation loaders for train/valid/test metric calculation use
  `shuffle=false` and `drop_last=false`, so metrics cover every sample.
- Implement `pipeline_contract=qlib_ts_lstm_v1` as fixed behavior:
  `TSDatasetH`, `DataHandlerLP` with `PTYPE_A`, and
  `QlibDataLoader`; feature processors `ProcessInf -> ZScoreNorm(train-fit) ->
  Fillna(0)`; label processors `DropnaLabel -> CSZScoreNorm`; sampler
  `fillna_type=ffill+bfill`; training loader `shuffle=true, drop_last=true`;
  evaluation loaders `shuffle=false, drop_last=false`; `batch_size=2048`; and
  `n_jobs=0`. `d_feat` equals `len(z_alpha.selected_features)`: six in Round 0
  and ten after the mandatory RASS Round 1. Use `DK_L` for train/valid learning
  views and a separately preserved `DK_R` raw label for signal metrics. Test
  `DK_L` access remains restricted to the isolated researcher runner.

## Frozen adaptive protocol

`EXP_000` is Round 0: run the untouched initial configuration through the full
alpha -> model -> validation prediction -> portfolio -> validation backtest
pipeline. Round 0 has no CLEM decision, specialist proposal, or candidate diff.
It becomes the initial accepted state and first Memory Bank reference.

Every later round obeys:

```text
one round = one layer + one declared parameter group
          + one coherent, falsifiable hypothesis
```

Cross-layer and cross-group diffs are forbidden. Multi-parameter diffs are legal
only where this contract explicitly permits them and the changes are tightly
coupled to the same mechanism.

- FAMA resolves exactly one model-specific space from frozen
  `z_model.model_name` through `agents/FAMA/intervention_space.yaml`. Supported
  names are `lstm`, `gru`, and `alstm`; an unsupported name is rejected without
  fallback. FAMA selects exactly one group from the resolved file and may change
  at most two tightly coupled parameters in it; groups and spaces must never be
  unioned. Every supported model has `train_params = {lr, hidden_size, dropout,
  weight_decay}` with `lr` log-scaled in `[0.0001,0.01]`, `hidden_size` on the
  integer grid 16..128 step 8, `dropout` on the grid 0..0.5 step 0.05, and
  `weight_decay` either zero or log-scaled in `[0.000001,0.001]`. Model-specific
  `model_params` come only from the selected YAML: LSTM/GRU expose
  `{model_layer,use_pe,use_bn,use_ln}`; ALSTM exposes
  `{model_layer,attention_hidden_size,use_pe,use_bn}`, with
  `attention_hidden_size` on 16..128 step 16. Loss, checkpoint metric, splits,
  evaluator, labels, factors, and portfolio settings are frozen.
- RAPA changes exactly one of `alpha_scale`, `risk_aversion`, or
  `turnover_penalty` per round. Inclusive bounds are
  `alpha_scale=[0.0008,0.0015]`, `risk_aversion=[0.1,2.0]`, and
  `turnover_penalty=[0.1,2.0]`; there is no additional per-round percentage cap.
  Its objective is:
  `max_w alpha_scale * alpha' w - risk_aversion * w' Sigma w - turnover_penalty * ||w-w_old||^2`.
- RASS is a one-time Agent-driven initial feature-bootstrap specialist. The
  current `EXP_000` baseline has six ordered anchor features. In `EXP_001`,
  preserve those anchors unchanged. Deterministic train-only evidence marks
  basic data-quality eligibility without ranking factors. A first Agent stage
  selects exactly twelve eligible candidates from the complete frozen catalog;
  code then computes shortlist-only daily cross-sectional redundancy and
  information remaining after linear anchor residualization. A second Agent
  stage selects exactly four factors from that shortlist and appends them,
  reaching ten features.
  Removal, replacement, reordering, duplicates, and expressions outside the
  frozen pool catalog are forbidden. Resolve and hash the catalog and its
  deterministic train-only evidence table before the formal trajectory. RASS
  never receives validation evidence in either selection stage; validation
  evaluates the sealed four-factor intervention only after execution. The
  Contract Validator checks provenance, membership, count, scope, and leakage;
  it does not reproduce or override the Agent's ranking.
  Full train rows are used for IC, RankIC, coverage, and stability. Stage-1
  redundancy estimates use a deterministic evenly spaced sample capped at
  20,000 rows; the twelve-factor second stage uses full daily cross-sections.

The machine-readable router `agents/FAMA/intervention_space.yaml`, its selected
file under `agents/FAMA/intervention_spaces/`, and the RAPA/RASS
`intervention_space.yaml` files are authoritative for Contract Validator bounds
and types.

CLEM reassesses all layers every round. Repeating a layer requires new evidence,
an unresolved failure, and stronger support than alternatives. Repeated
`FALSIFIED` or `UNCERTAIN` results require cross-layer reassessment. Return
`NO_INTERVENTION` when no dominant supported failure remains; stopping is better
than unsupported experimentation.

The deterministic anti-repetition floor forbids retrying a parameter that has
already produced `FALSIFIED` or `UNCERTAIN` from the same accepted parent. An
accepted-state change resets eligibility. A contract-invalid specialist output
gets one corrective Agent call in the same round; if both attempts fail, record
`CONTRACT_REJECTED` in shared memory and consume the round.

There is one frozen routing priority for the current trajectory. Immediately
after `EXP_000`, if the accepted configuration contains the expected six
features and `alpha_frozen` is false, CLEM must diagnose insufficient initial
feature breadth and route `EXP_001` to RASS. Configuration feature count is the
supporting evidence for this route. After successful `EXP_001`, set
`alpha_frozen=true`; CLEM must not select RASS again and later rounds may route
only to FAMA, RAPA, or `NO_INTERVENTION`.

An intervention requires CLEM confidence `>= 0.5`; below this threshold CLEM
must return `NO_INTERVENTION`. `task.trials = 10` is the maximum adaptive-round
budget after Round 0, so the trajectory may create `EXP_001` through `EXP_010`.
There is no separate earlier stop based only on consecutive `UNCERTAIN` or
`FALSIFIED` verdicts: their limit is also 10, while cross-layer reassessment is
still required after each such outcome.

Each candidate starts from the latest accepted configuration, never a rejected
candidate. Outside the single `EXP_001` exception below, both `SUPPORTED` and
`PARTIALLY_SUPPORTED` promote it. `UNCERTAIN` and `FALSIFIED` retain artifacts
and memory but roll back to the accepted parent.

The sole exception is the mandatory RASS structural initialization in
`EXP_001`. After its proposal passes the Contract Validator, appends exactly
four Agent-selected factors to the six untouched anchors, and the full dependency
path executes without a fatal contract/data/runtime failure, promote `EXP_001`
unconditionally and freeze the resulting ten-feature alpha. The Validation Gate
still computes one of its normal four verdicts and stores it for audit and
memory, but that verdict does not control `EXP_001` promotion. Do not invent a
fifth Gate verdict. Record the separate acceptance reason as
`mandatory_rass_structural_initialization` outside `result.json`. If RASS cannot
propose exactly four legal additions from the frozen catalog with train-only support,
or execution fails, do not fabricate factors or promote a partial result; fail
the trajectory explicitly.

## Validation-only adaptation and test isolation

All adaptive evidence must come from training or validation. CLEM, specialists,
the Contract Validator, Executor-facing context, Validation Gate, retrieval,
memory, accepted-state logic, rollback, and stopping must be physically unable
to read test files, metrics, predictions, curves, summaries, or derived data.
Enforce this with filesystem/API allowlists and context builders, not prompts.

During training, every completed epoch checkpoint triggers a separate
researcher-side evaluator immediately. It computes test IC, ICIR, RankIC, and
RankICIR and appends a train/valid/test record to
`runs/<task.name>/experiments/EXP_xxx/test/epoch_metrics.jsonl`. The trainer may
dispatch the checkpoint identity but must receive no test metric or read path
back. After validation checkpoint selection and adaptive-output sealing, the
researcher runner also writes `test/result.json` using the same exact 27-field
metric schema and three-decimal serialization as validation `result.json`, plus
`test/test_alpha.csv` and `test/portfolio_daily_diagnostics.csv` when those raw
outputs are produced. These are researcher-only test artifacts, not adaptive
experiment outputs. Never
copy test-derived content into `result.json`, `decision.json`, agent-visible
artifacts or logs, or `runs/<task.name>/memory/`. Test observations can never
change routing, acceptance, rollback, stopping, or the final configuration.
Failure of the researcher test job is reported separately and does not change
the already sealed adaptive verdict or state.

After trajectory finalization, an isolated researcher process writes
`runs/<task.name>/test/researcher_outputs_audit.json`. It reports only whether
every successfully executed round has its required researcher artifacts; it
contains no test metrics and cannot affect adaptive state.

Before a formal trajectory, freeze prompts, intervention spaces, data splits,
evaluator, costs, alignment, constraints, gate thresholds, retrieval, stopping,
seeds/budget policy, and checkpoint rule. Do not tune the frozen contract using
test results.

MSE is the training loss. Select a checkpoint by maximizing exactly one frozen
validation metric from `{ic, icir, rank_ic, rank_icir}`. The configuration field
`z_model.base_params.metric` selects it and defaults to `ic`; FAMA cannot change
it during a formal trajectory. Never use train or test metrics for checkpoint
selection. Validation Sharpe remains the primary adaptive objective. Expected
mechanism signatures and frozen trade-off guardrails also affect the
Validation Gate exactly as specified below; secondary improvements never
compensate for materially worse Sharpe.

### Label, prediction, and portfolio-alpha contract

The raw label is permanently:

```text
Ref($close, -2) / Ref($close, -1) - 1
```

For model fitting, transform that raw forward return into a daily cross-
sectional z-score and optimize MSE against the standardized training target.
Keep the raw label as a separate evaluation field. Compute IC, ICIR, RankIC, and
RankICIR against the raw, unstandardized forward return, not against the model-
training label field.

The model output is a relative alpha score, not a forecast expressed in return
units. Never invert the training-label normalization using realized or future
cross-sectional label statistics. For each signal date, preprocess the available
prediction cross-section before candidate ranking and portfolio optimization:

```text
model score
  -> replace non-finite values with missing and exclude them
  -> clip to that date's 1st and 99th prediction percentiles
  -> subtract that date's prediction mean
  -> divide by that date's population standard deviation (ddof=0)
  -> multiply by z_portfolio.alpha_scale
  -> portfolio optimizer alpha vector
```

If no finite prediction exists, the date is an explicit missing-signal case; do
not invent an alpha. If the winsorized cross-section has non-finite or at most
`1e-12` standard deviation, map every available score on that date to zero.
This second z-score normalizes model predictions for the optimizer; it is not an
inverse transform back to raw returns. The portfolio's realized P&L continues to
come from the unstandardized return panel `data/portfolio/c_2_c_1D.csv`.

Metric definitions are fixed: for each date, calculate cross-sectional Pearson
correlation for IC and Spearman correlation for RankIC after dropping non-finite
prediction/label pairs. `ic` and `rank_ic` are means across valid dates. `icir`
and `rank_icir` are the corresponding mean divided by the daily series sample
standard deviation (`ddof=1`) and are not annualized. Dates with fewer than two
valid assets or a constant prediction/label vector are excluded and counted.

Every training run records all four metrics for train and validation at every
epoch in an agent-visible training log. After each epoch checkpoint is written,
an isolated researcher evaluator immediately computes the same four test
metrics and appends a record containing that epoch's train, validation, and test
metrics to `test/epoch_metrics.jsonl`. The adaptive trainer receives no test
values. This isolated test log is part of the same auditable training record for
researchers, but it is physically inaccessible to adaptive agents, memory,
routing, checkpoint selection, acceptance, rollback, and stopping.

Use `task.seed` as the single random-seed source. The future trainer must seed
Python, NumPy, PyTorch CPU, every CUDA device, DataLoader generators, and workers;
enable deterministic PyTorch algorithms; disable cuDNN benchmarking; persist
split indices and sample order; and fail rather than silently use a known
nondeterministic operation. Reproducibility means identical artifacts and
metrics for the same config, data, dependency versions, hardware class, and
seed; cross-platform bitwise identity is not assumed.

Treat `task.GPU` as a requested zero-based CUDA device index. At startup, use
`cuda:<task.GPU>` only when PyTorch reports CUDA available and that index exists;
otherwise deterministically resolve to `cpu` and emit a warning. Record the
requested device, resolved device, fallback reason (if any), and PyTorch/CUDA
versions in initial provenance and freeze that resolved device for the entire
trajectory. A CUDA OOM or device failure after startup is fatal; do not silently
switch hardware mid-trajectory.

Use daily net returns and risk-free rate zero:

```text
Sharpe = mean(net_daily_return) / std(net_daily_return, ddof=1) * sqrt(252)
```

Never compute Sharpe as CAGR divided by annualized volatility. Freeze practical
Sharpe tolerance and all trade-off thresholds before the formal run.

Sharpe noise tolerance uses the frozen deterministic algorithm in
`configs/frozen_contract.json`: align candidate and parent validation daily net
returns, resample paired rows with a moving-block bootstrap (`B=2000`, block
length 20 trading days, seed 0), compute bootstrap delta Sharpe, and set
`epsilon_sharpe = max(1.96 * std(delta_sharpe_bootstrap, ddof=1),
numerical_equivalence_tolerance)`. The numerical tolerance is
`max(1e-12, 1e-10 * max(abs(parent_statistic), abs(candidate_statistic)))`;
deltas at or below it are canonicalized to zero. If the bootstrap cannot be
computed, the primary-metric verdict is `UNCERTAIN`, never `SUPPORTED`.

There are currently no hard performance thresholds, including no maximum-
drawdown threshold. Schema, access-control, data-integrity, and execution-
legality failures remain fatal.

The deterministic Validation Gate matrix is frozen in
`configs/frozen_contract.json`:

- `delta_sharpe > epsilon_sharpe` is materially improved;
- `abs(delta_sharpe) <= epsilon_sharpe` is practically unchanged;
- `delta_sharpe < -epsilon_sharpe` is materially worse and always
  `FALSIFIED`, even if secondary metrics improve;
- material Sharpe improvement plus complete mechanism match and no material
  guardrail degradation is `SUPPORTED` and promotes the candidate;
- material Sharpe improvement without complete mechanism match is
  `PARTIALLY_SUPPORTED` and also promotes, unless a mechanism moves materially
  opposite or a guardrail materially degrades;
- practically unchanged Sharpe with `MATCH` or `PARTIAL` mechanism evidence is
  `PARTIALLY_SUPPORTED` and also promotes;
- materially worse Sharpe, materially opposite mechanism evidence, or material
  guardrail degradation is `FALSIFIED` and rolls back;
- otherwise the verdict is `UNCERTAIN` and the candidate rolls back.

For each declared mechanism or guardrail metric, materiality uses the same
paired moving-block bootstrap and
`epsilon_metric = max(1.96 * std(bootstrap_metric_delta, ddof=1),
numerical_equivalence_tolerance)`. Apply the same numerical-equivalence rule to
Sharpe, mechanism metrics, and guardrails. Orient guardrail
deltas so positive means worse. Frozen guardrails are drawdown magnitude,
annual volatility, one-way turnover, mean daily transaction-cost drag, mean
portfolio concentration `sum(w_i^2)`, and invested-weight shortfall. Missing
required Sharpe, mechanism, or guardrail evidence yields `UNCERTAIN`.

Equality belongs to the unchanged/noise band: `abs(delta) <= epsilon`. Contract,
data, access-control, and execution failures are fatal before Gate evaluation
and are not converted into a performance verdict.

## A-share execution and portfolio baseline

Frozen alignment:

```text
alpha[T] -> rebalance at T+1 close
         -> realize close(T+2) / close(T+1) - 1
```

Apply the two masks on the T+1 trade date as directional bounds:

- limit-up: cannot buy or increase weight; selling/decreasing is allowed;
- limit-down: cannot sell or decrease weight; buying/increasing is allowed;
- mask semantics are `1 = limit event`, `0 = normal`;
- missing mask values conservatively block the relevant direction.

Use `mask_limit_up_1D.csv` and `mask_limit_down_1D.csv`; never collapse them into
one symmetric tradability flag. Purge split-boundary samples deterministically
when their T+1/T+2 labels would cross into another split.

### Canonical alpha panel format

Every prediction artifact consumed by the portfolio layer, including
`artifacts/valid_alpha.csv`, must use the same physical panel format as
`data/portfolio/c_2_c_1D.csv`:

- comma-separated UTF-8 wide table;
- first column is an unnamed date index serialized exactly as `YYYYMMDD`;
- one subsequent column per stock;
- stock column names, complete column set, and column order exactly match the
  header of `c_2_c_1D.csv`;
- Qlib instruments are mapped deterministically: `SZ000001 -> 000001_XSHE` and
  `SH600000 -> 600000_XSHG`; reject unknown prefixes and duplicate mapped names;
- rows are sorted ascending and contain only the requested split's trading
  dates after boundary purging;
- unavailable predictions are empty/NaN, never zero-filled, because zero is a
  valid alpha value;
- no extra index name, MultiIndex header, metadata row, or unnamed data column
  may be written.

The writer must pivot Qlib `(datetime, instrument)` predictions into this format
and validate the written artifact against the reference header. Before a run,
also verify that return and mask panels cover every required T+1/T+2 date; fail
with a data-coverage error rather than silently shortening or changing a split.

Settled Mean-Variance baseline assumptions:

- long-only with a 100% target invested weight. If directional trade bounds,
  per-stock caps, or an insufficient buyable universe make 100% mathematically
  infeasible on a date, invest the maximum feasible weight and retain the
  remainder as zero-return cash. This feasibility fallback is not an execution
  failure and must not relax any asset bound;
- Qlib alpha generation uses exactly the universe named by
  `task.instruments`. The daily ranking universe is exactly the stocks from
  that configured universe with a finite processed alpha score on the signal
  date; do not apply a second constituent-membership filter at the portfolio
  boundary;
- load the benchmark instrument named by `task.benchmark` from the Qlib
  provider named by `task.provider_uri`. Compute its aligned benchmark return
  as `close[T+2] / close[T+1] - 1`; never infer the benchmark from
  `task.instruments` and never replace it with an equal-weight stock return;
- `candidate_count = 100` applies to new-entry candidates: rank the eligible
  new-entry universe by processed alpha and retain its Top 100;
- optimizer universe = Top-100 new entries union all existing holdings. Existing
  holdings remain present so constraints and exits can be handled even when
  they are no longer eligible as new entries; holdings or optimizer-universe
  size may therefore exceed 100;
- `max_weight = 0.02` per stock;
- rolling diagonal risk model by default;
- starting values: `alpha_scale = 0.001`, `risk_aversion = 1.0`,
  `turnover_penalty = 1.0`;
- buy cost = 6 bp and sell cost = 11 bp;
- external TopK baseline = Top100/Drop10. For comparison, TopK and
  Mean-Variance/RAPA must use the same finite-alpha ranking universe, Top-100
  new-candidate count, signal dates, return alignment, directional masks, and
  transaction-cost assumptions. Their portfolio construction/allocation rule
  is the intended difference;
- `hold_thresh = 1` belongs only to TopK, not Mean-Variance/RAPA.

For every portfolio date, record `target_invested_weight`,
`feasible_min_invested_weight`, `feasible_max_invested_weight`,
`full_position_feasible`, `invested_weight`, and
`invested_weight_shortfall = 1 - invested_weight` in
`artifacts/portfolio_daily_diagnostics.csv`. When full investment is infeasible,
also record a deterministic reason code. Cash earns zero return. The Validation
Gate continues to treat invested-weight shortfall as a trade-off guardrail.

Costs, masks, alignment, candidate construction, max weight, risk-model mode,
and dust/drop rules are not adaptive unless a future frozen contract says so.
Return-panel and mask columns do not define candidate membership. Reindex them
to the alpha/return contract: ignore extra mask columns, require every finite-
alpha asset to be representable in the return panel, and apply missing mask
values conservatively as the directional rules above require.

## Files and experiment records

Target convention:

```text
configs/{frozen_contract,agent_models}.json       # repository-wide inputs
agents/{CLEM,RASS,FAMA,RAPA}/
  SYSTEM.md
  decision.schema.json or intervention.schema.json
  intervention_space.yaml                 # specialists only
runs/<task.name>/
  configs/{initial,current,final_frozen}.json
  experiments/EXP_000/
    config.json
    result.json
    artifacts/valid_alpha.csv
    artifacts/portfolio_daily_diagnostics.csv
    logs/
    test/                                 # isolated, researcher-only
      epoch_metrics.jsonl                 # per-epoch train/valid/test model metrics
      result.json                         # same 27-field schema, test split
      test_alpha.csv
      portfolio_daily_diagnostics.csv
  experiments/EXP_001/
    config.json
    decision.json
    result.json
    validation_gate.json
    artifacts/valid_alpha.csv
    artifacts/portfolio_daily_diagnostics.csv
    logs/
    test/                                 # isolated, researcher-only
      epoch_metrics.jsonl                 # per-epoch train/valid/test model metrics
      result.json                         # same 27-field schema, test split
      test_alpha.csv
      portfolio_daily_diagnostics.csv
  memory/
    experiments.jsonl
    summaries/{cross_layer,alpha,model,portfolio}.json
```

Unless explicitly shown otherwise, experiment and memory paths below are
relative to the active `runs/<task.name>/` root.

- `config.json` must reproduce the run and recover its accepted parent and diff.
- `decision.json` records pre-execution CLEM diagnosis/routing, alternatives,
  goal and evidence; specialist group/diff/reasoning/signature/falsification;
  Contract Validator status; and, when required, both contract-correction
  attempts. It must not contain observed result metrics.
- `result.json` contains only the exact validation result object below. Deltas,
  verdicts, diagnostics, and provenance belong in deterministic validation or
  memory records and raw artifacts.
- `artifacts/portfolio_daily_diagnostics.csv` is the deterministic Gate source
  for daily strategy `net_return`, aligned benchmark and excess returns,
  transaction-cost drag, portfolio concentration, one-way turnover, and
  invested weight. `validation_gate.json` records aligned
  parent/candidate deltas, bootstrap epsilons, Sharpe/mechanism/trade-off states,
  missing-evidence checks, final verdict, and promotion decision. Neither file
  may contain test-derived values.
- `artifacts/` contains reproducible raw outputs. `valid_alpha.csv` is the
  canonical validation prediction. Reuse or hash parent outputs when a layer is
  not rerun. Debug logs are not automatically admissible agent evidence.

For the finite aligned validation dates, let `r_t` be strategy net return,
`b_t` the configured benchmark return, and `e_t = r_t - b_t` the arithmetic
daily excess return. Use these frozen definitions for both strategy and excess
metrics:

```text
NAV(x)       = cumulative product of (1 + x_t)
ARR(x)       = NAV(x)[-1] ** (252 / n) - 1
Vol(x)       = std(x_t, ddof=1) * sqrt(252)
MDD(x)       = min(NAV(x) / cumulative_max(NAV(x)) - 1)
Sharpe(x)    = mean(x_t) / std(x_t, ddof=1) * sqrt(252)
Calmar(x)    = ARR(x) / abs(MDD(x))
Sortino(x)   = mean(x_t) / sqrt(mean(min(x_t, 0) ** 2)) * sqrt(252)
WinRate(x)   = count(x_t > 0) / count(finite x_t)
```

Risk-free rate and minimum acceptable return are zero. `information_ratio` is
`Sharpe(e)`, not a second strategy Sharpe. `excess_annual_return`,
`excess_annual_volatility`, `excess_max_drawdown`, `excess_calmar_ratio`,
`excess_sortino_ratio`, and `excess_win_rate` are computed from `e_t` and its
excess NAV. A zero or unavailable denominator produces `null`, not infinity or
zero. `one_way_turnover_mean = mean((buy_turn + sell_turn) / 2)`.

The authoritative `result.json` has 26 metric keys plus integer `n_periods` (27
top-level keys). Do not add, remove, or rename keys:

```json
{
  "ic": null,
  "icir": null,
  "rank_ic": null,
  "rank_icir": null,
  "final_nav": null,
  "total_return": null,
  "annual_return": null,
  "annual_volatility": null,
  "max_drawdown": null,
  "sharpe_ratio": null,
  "calmar_ratio": null,
  "sortino_ratio": null,
  "win_rate": null,
  "excess_annual_return": null,
  "excess_annual_volatility": null,
  "excess_max_drawdown": null,
  "information_ratio": null,
  "excess_calmar_ratio": null,
  "excess_sortino_ratio": null,
  "excess_win_rate": null,
  "sell_turn_mean": null,
  "buy_turn_mean": null,
  "one_way_turnover_mean": null,
  "holding_count_mean": null,
  "optimizer_universe_count_mean": null,
  "invested_weight_mean": null,
  "n_periods": null
}
```

Keep full numeric precision internally. Serialize every finite float in
`result.json` with exactly three digits after the decimal point. Preserve `null`
and keep `n_periods` integral.

## Shared Memory Bank

Use one physical experiment memory with role-specific projections:

- Every round retrieves the entire validation-safe Memory Bank; `top_k` ranking
  and truncation are disabled.
- Records are ordered deterministically by ascending round and then ascending
  experiment ID.
- CLEM receives the cross-layer projection. RASS/FAMA/RAPA may receive their
  role-specific projection, but it must cover every stored record and verdict.

Each memory record compresses:

```text
state -> CLEM diagnosis -> layer/group -> intervention
      -> expected signature -> observed signature -> verdict
```

Never write test-derived information to memory and never create independent
per-agent memory stores. If the full bank later exceeds a model context limit,
stop with an explicit capacity error; do not silently reintroduce top-k.

## Implementation and verification priorities

Keep contracts, patches, gates, state transitions, and access boundaries
machine-validated. Preserve useful existing examples while migrating them toward
the conventions above; do not treat an outdated example as authority.

Add automated tests for: physical test-path denial; exact result keys, nulls,
types and three-decimal serialization; Round 0; layer/group legality; FAMA's
two-parameter cap; RAPA's one-parameter rule; frozen fields; T/T+1/T+2 alignment;
directional limit masks; dependency-aware reruns; accepted-state promotion and
rollback; the one-time mandatory RASS `EXP_001` promotion and subsequent alpha
freeze; anti-repetition; and `NO_INTERVENTION` stopping.

Do not redesign the architecture or widen an intervention space unless an actual
inconsistency requires an explicit contract revision.
