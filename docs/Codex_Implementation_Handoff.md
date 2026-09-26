# Codex implementation handoff

Use this file when asking a coding agent to build DiagAgent directly in
`C:\Users\Andy\Desktop\DigAgent_WWW`. This directory is the authoritative,
runnable project. Qlib is the underlying data/training dependency; do not search
for a separate implementation repository.

## Authority to change the existing code

Treat every existing code file in this repository as a worked example for
understanding the intended logic, not as a frozen implementation or public API.
You may edit, refactor, relocate, rename, replace, or delete existing code and
change its interfaces whenever that produces a clearer, tested, end-to-end
runnable system. You may also create any missing packages, modules, scripts, and
tests. Inspect the examples before replacing them so required domain behavior is
not lost, but do not preserve historical structure merely for compatibility.

This authority does not permit changing the frozen protocol to accommodate the
examples. `AGENTS.md` wins over all existing code. Preserve test isolation,
validation-only adaptation, schemas, execution alignment, intervention rules,
and deterministic state transitions. Do not modify source inputs under `data/`.

## Current validation-split amendment

New submissions use explicit `train_valid_*` and `agent_valid_*` task fields.
Only per-epoch `train_valid` metrics may select or early-stop a checkpoint.
Run frozen seeds `[0,1,2]` in parallel, select one checkpoint independently per
seed by `train_valid.ic`, then evaluate all three checkpoints on `agent_valid`.
Daily cross-sectionally winsorize/z-score each prediction, equal-weight average,
and renormalize the mean. Write the ensemble as
`artifacts/agent_valid_alpha.csv` and use its signal/portfolio result for all
adaptive agents, Gate decisions, and memory. Never use per-epoch agent-valid
metrics for checkpoint selection. The two periods may currently overlap or be
identical, but neither may overlap training or isolated test. This amendment
and `AGENTS.md` override older generic `valid` wording below.

## Prediction-mode amendment

`task.prediction_mode` is the single frozen switch. Supported values are
`c2c` (legacy default) and `o2o`. Materialization deterministically binds the
target expression, return panel, directional masks, execution price, return
definition, and slippage from this value; stale duplicated values in a submit
must not create a mixed-mode run. Run and shared-RASS namespaces are
`runs/<instruments>_<agent-name-or-provider>_<prediction-mode>/`, so evidence
and caches can never cross between C2C and O2O.

## Required reading

Read, in order:

1. `./AGENTS.md`
2. `./docs/DiagAgent_Implementation_Spec.md`
3. every current `./submit/*.json` (the runtime must require one explicit
   `--submit` selection; there is no default submit file)
4. `./configs/frozen_contract.json`
5. `./agents/FAMA/intervention_space.yaml` and every mapped file under
   `./agents/FAMA/intervention_spaces/`
6. `./agents/RAPA/intervention_space.yaml`
7. `./agents/RASS/intervention_space.yaml`, `./agents/RASS/SYSTEM.md`, and
   `./agents/RASS/SKILL.md`
8. `./backbone/modules.py` and the `./backbone/{lstm,gru,alstm}_example.py`
   selected by each submit's `z_model.model_name`
9. `./agents/RAPA/portfolio_example.py` (reference only; modify or replace it
   when implementing the production portfolio pipeline)
10. `./agents/CLEM/clem.md`
11. `./agents/CLEM/schema.json`
12. the current-contract structural examples:
   - `./examples/README.md`
   - `./examples/EXP_000/config.json`
   - `./examples/EXP_000/result.json`
   - `./examples/EXP_000/logs/training_metrics.jsonl`
   - `./examples/EXP_001/config.json`
   - `./examples/EXP_001/decision.json`
   - `./examples/EXP_001/result.json`
   - `./examples/EXP_001/validation_gate.json`
   - `./examples/memory/experiments.jsonl`
   - `./schemas/result.schema.json`

Every path above is relative to the repository root
`C:\Users\Andy\Desktop\DigAgent_WWW`. Do not resolve it relative to `docs/`.
The files under `examples/` are synthetic structural references, not real
experiment evidence and never formal Memory Bank input. The historical files
under `legacy_unsafe_examples/` are explicitly excluded from implementation,
formal execution, Agent context, tests, and retrieval.

Before execution, inspect (do not rewrite) these actual data locations:

```text
./data/cn_data/
./data/portfolio/c_2_c_1D.csv
./data/portfolio/mask_limit_up_1D.csv
./data/portfolio/mask_limit_down_1D.csv
./data/portfolio/o_2_o_1D.csv
./data/portfolio/mask_limit_up_open_1D.csv
./data/portfolio/mask_limit_down_open_1D.csv
```

`./agents/FAMA/` still needs its production prompt/schema/runtime integration.
RASS now has an Agent-exploration prompt and skill, but its machine output
schema, evidence builder, and runtime integration remain to be implemented.

## First task: repository mapping

Before writing implementation code in this repository:

1. Locate and parse every current `submit/*.json`; design the runtime to require
   one explicit `--submit submit/<file>.json` selection and never guess a
   default.
2. List the actual fields under `z_alpha`, `z_model`, and `z_portfolio`.
3. Initialize Qlib with repository-relative `./data/cn_data` and inspect the
   existing training, prediction, portfolio, and evaluation entry points.
4. Use the return panel and two directional masks selected by
   `task.prediction_mode`; verify their date and asset coverage before Round 0.
5. Identify the train/validation/test split definitions and every path that can
   expose test outputs.
6. Map Qlib long-form predictions to the canonical wide alpha CSV contract in
   `AGENTS.md`, and map outputs to the exact DiagAgent artifacts/result fields.
7. Report inconsistencies between the repository and frozen contract. Do not
   silently redesign either one or silently shorten a split.
8. Treat the matching `backbone/*_example.py` and `backbone/modules.py` as
   architecture/logic examples. They may be rewritten or replaced; the
   resulting runnable design must keep the model
   architecture separate from training/Qlib/metric/checkpoint responsibilities
   as assigned by the specification.

After mapping and before `EXP_000`, expand the explicitly selected submit JSON
into one complete runtime configuration at
`runs/<instruments>_<agent-name-or-provider>_<prediction-mode>/<task.run_id>/<model-name>/<MMDDHHMM>/configs/initial.json`.
Validate every path component, record the selected submit path and SHA256 in
provenance, and keep all mutable state, experiments, and memory under that
timestamped run root. Store only deterministic RASS evidence at
`runs/<instruments>_<agent-name-or-provider>_<prediction-mode>/<task.run_id>/rass_train_evidence.json`.
For year ID `Y`, build evidence from complete calendar years `Y-3`, `Y-2`, and
`Y-1`, purging each year's last two signal dates. Matching backbones under the
same year reuse only this evidence; different years do not. Every trajectory
still runs EXP_000 with five anchors and performs a fresh RASS Agent selection
for EXP_001. No factor selection is cached or replayed across trajectories. If
rejected, RASS may select a different three-factor set. Validate reuse with the
model-independent `rass_evidence_context_hash`, not the full model configuration
hash. Resume an
existing run only if its recorded submit hash matches. The Dataset/Handler,
processors, data keys, loader behavior, and `d_feat` derivation are fixed by the
seed's `pipeline_contract=qlib_ts_lstm_v1` and the authoritative Markdown
contract; batch size, worker count, and label normalization remain explicit in
the seed. Do not choose replacements for either source. Supply only unavoidable
absent runtime fields. Make every effective value explicit and persist its provenance,
dependency/data hashes, and final configuration SHA256. Validate and then
freeze this file; do not overwrite it or fall back to implicit defaults after
Round 0 starts.

## Implementation sequence

Implement in independently testable phases:

```text
Phase 1  config/schema/frozen-contract/state-store
Phase 2  Qlib adapters + deterministic Round 0
Phase 3  evidence builder + exact result writer
Phase 4  Contract Validator
Phase 5  Validation Gate + atomic promotion/rollback
Phase 6  shared Memory Bank + retrieval views
Phase 7  CLEM and specialist integrations
Phase 8  end-to-end orchestrator + restartability
Phase 9  persistent physically isolated researcher test runner (one test Dataset load per experiment)
```

After each phase, run its focused tests. Do not proceed past Phase 2 until the same Round 0 input reproduces the same validation artifacts and metrics twice.

## Non-negotiable implementation behavior

- Inspect existing project runners for useful logic, but freely replace them
  when a simpler or more reliable Qlib pipeline is needed; compatibility with
  the example code is not a requirement.
- Resolve all configured data paths from the repository root and never modify
  source files under `data/`.
- Write validation alpha in the exact date-by-stock wide format and stock-column
  order of the return panel selected by `task.prediction_mode`; preserve missing
  predictions as NaN.
- Generate alpha using exactly the universe named by `task.instruments`. Treat
  stocks in that configured universe with finite alpha on the signal date as
  the complete ranking universe; no second membership filter is applied. Read
  the benchmark instrument from `task.benchmark` through the configured Qlib
  provider and compute the matching C2C or O2O T+1-to-T+2 return; never substitute an
  equal-weight stock benchmark. Use Top100 new-entry candidates union
  all existing holdings as the Mean-Variance optimizer universe. Keep
  `candidate_count=100`; it is not a cap on holdings. Run the Top100/Drop10
  comparison from the same finite-alpha universe, execution dates, masks, and
  costs. Return/mask panels do not define membership; ignore extra mask columns
  after reindexing and handle missing masks conservatively.
- Treat `full_position: true` as a 100% invested-weight target, not permission to
  violate frozen constraints. If directional bounds, `max_weight`, or the
  buyable universe make 100% mathematically infeasible, invest the maximum
  feasible weight and retain the remainder as zero-return cash. Do not fail the
  experiment solely for this condition. Write target/feasible/actual invested
  weights, shortfall, feasibility flag, and a deterministic reason code to
  `artifacts/portfolio_daily_diagnostics.csv`; shortfall remains a Validation
  Gate trade-off guardrail.
- Implement `configs/agent_models.json` as the provider/API-key/model catalog.
  Read the active Agent base model and reasoning effort from `agent` in the
  selected `submit/*.json`, validate both against that catalog, and freeze them
  into the trajectory's initial configuration. The selection applies to
  LLM-backed CLEM/RASS/FAMA/RAPA. RASS explores factors using frozen annual
  development evidence for 2021, 2022, 2023, and 2024; because 2024 is visible
  during factor selection, it must not later be claimed as unseen validation
  evidence; deterministic code validates its boundaries but does not choose its
  factors. Resolve API
  keys from environment references, never log secrets, never silently use a
  fallback model, and treat a model/effort change as a new trajectory.
- Select the model-specific FAMA space and architecture example from frozen
  `z_model.model_name`; reject unsupported names without merging spaces or
  falling back. Keep every backbone architecture-only. ALSTM's
  `attention_hidden_size` must change the actual attention-layer width.
  Implement training separately and
  support validation checkpoint selection by `ic`, `icir`, `rank_ic`, or
  `rank_icir`, defaulting to `ic`.
- Use the raw label bound by `task.prediction_mode`; train MSE on its daily cross-sectional
  standardized form while retaining the raw label separately for all four IC
  metrics. Treat predictions as relative scores and never inverse-transform
  them with future label statistics.
- Before portfolio optimization, normalize each date's prediction cross-section
  exactly as specified in `AGENTS.md`: exclude non-finite values, winsorize at
  1%/99%, z-score with `ddof=0`, handle a constant cross-section as zero, and
  then multiply by `alpha_scale`. Use the unstandardized portfolio return panel
  for realized P&L.
- Use `task.seed` to make training reproducible under the frozen environment.
  Treat `task.GPU` as a requested CUDA index: if CUDA or that index is
  unavailable at startup, warn and use CPU; persist and freeze the resolved
  device. Do not silently switch after a CUDA OOM/runtime failure. Also,
  validation/test evaluation must never drop the last batch.
- Keep contract validation, dependency resolution, metric computation, validation verdicts, state transitions, and test isolation deterministic.
- Never add test-derived data to prompts, memory, result files, routing, acceptance, rollback, or stopping.
- Keep per-epoch `train`/`train_valid` metrics in normal logs. Immediately after each
  epoch checkpoint, one persistent researcher-side evaluator appends that epoch's
  `train`/`train_valid`/`test` IC, ICIR, RankIC, and RankICIR to
  `test/epoch_metrics.jsonl` without returning test values to the trainer. After
  checkpoint and adaptive-output sealing, it stores the complete 27-field test
  result as `test/result.json`; it may also store
  `test/test_alpha.csv` and `test/portfolio_daily_diagnostics.csv`. Adaptive
  components must have no read path to that directory. A researcher-test job
  failure is reported separately and cannot change the sealed adaptive state.
- Never change the exact `result.json` keys.
- Produce all 27 frozen result fields, including strategy and excess ARR, Vol,
  MDD, Sharpe/IR, Calmar, Sortino, win rates, and turnover diagnostics, using
  the formulas in `AGENTS.md` and `schemas/result.schema.json`.
- Never treat example values as tuned defaults beyond the frozen values explicitly named in `AGENTS.md`.
- Implement the original composite Sharpe/mechanism/trade-off Gate exactly as
  written in `configs/frozen_contract.json`, using fixed
  `epsilon_sharpe = 0.002` instead of a Bootstrap-derived Sharpe epsilon.
  `SUPPORTED` and `PARTIALLY_SUPPORTED` promote. Apply the same Gate to RASS:
  reject and retry from the five-factor parent with a non-identical three-factor
  set, then freeze the original five factors after three complete failures. Also
  implement the FAMA-only validation-IC override: when full-precision mean
  daily ensemble validation IC improves by more than `0.001` versus the accepted
  parent, the ordinary verdict is `UNCERTAIN`, at least two of three seed deltas
  are positive, and their median exceeds `0.001`, promote the FAMA candidate.
  Never override `FALSIFIED`, worse-Sharpe, opposite-mechanism, or material-
  trade-off outcomes. Preserve
  that ordinary verdict for audit and record promotion reason
  `fama_validation_ic_override`. Never apply this override to RASS or RAPA.

## Completion definition

The system is complete only when:

1. Round 0 runs reproducibly from the real initial JSON.
2. A legal round can progress through decision, validation, execution, gate, state update/rollback, and memory.
3. Every illegal cross-layer/cross-group/frozen-field proposal is rejected before execution.
4. Dependency-aware reruns are correct for RASS, FAMA, and RAPA.
5. The adaptive process cannot read test artifacts even if they exist.
6. Budget exhaustion is the only normal adaptive stopping rule and terminates cleanly.
7. The final accepted config is frozen and independently testable.
