# DiagAgent implementation decisions

This log records conservative implementation choices where the frozen contract
defines the required behavior but not a low-level mechanism. It is not an
experimental-contract amendment.

## ID-001: Resolve zero-byte Qlib placeholders as unavailable observations

- Decision: Resolve `task.instruments` through Qlib's dated market-membership
  spans, then remove only instruments whose required OHLCV/VWAP feature binary
  is absent or zero bytes before asking Qlib to evaluate expressions.
- Rationale: The supplied provider contains zero-byte placeholder feature files.
  Qlib 0.9.7 raises `IndexError` when they are passed to `D.features`; such an
  instrument cannot produce a finite alpha observation on any date.
- Alternatives: Patch installed Qlib, treat the loader crash as fatal, or pass
  an undated union list. The first mutates a dependency, the second prevents the
  supplied data from running, and the third loses constituent-date semantics.
- Impact: No observable stock is removed and no second portfolio membership
  filter is introduced. Dated CSI500 membership remains authoritative.

## ID-002: Standard non-circular moving-block bootstrap

- Decision: Draw each block start uniformly from `0..n-block_length`,
  concatenate `ceil(n/block_length)` blocks, and truncate to `n` rows.
- Rationale: The frozen contract specifies a moving-block bootstrap but does
  not select circular wrapping. This is the conventional conservative form.
- Alternatives: Circular blocks or stationary bootstrap. Both change the
  resampling distribution beyond the stated algorithm.
- Impact: Only deterministic epsilon estimation is affected; `B=2000`, block
  length 20, seed 0, pairing, `ddof=1`, and multiplier 1.96 remain frozen.

## ID-003: Completed-config hash stored in a companion provenance record

- Decision: Store all provenance inputs inside `initial.json` and store the
  SHA256 of its completed bytes in `initial.provenance.json`.
- Rationale: A file cannot contain its own ordinary SHA256 without a circular
  fixed-point problem. A companion record permits exact verification without
  rewriting the frozen initial configuration.
- Alternatives: Hash an object with the hash field omitted, or overwrite the
  initial file after materialization. The former is ambiguous and the latter
  violates write-once materialization.
- Impact: None on experiment semantics; resume checks verify both files.

## ID-004: Environment-file secret resolution

- Decision: If the selected provider key is not already in the process
  environment, load only the named key from repository-root `.env`.
- Rationale: The target credential is supplied there. Runtime resolution is
  required, while resolved secrets are forbidden from persisted artifacts.
- Alternatives: Require callers to export the variable manually.
- Impact: Connection setup only. Neither the key nor a fingerprint of it is
  written to configs, logs, decisions, prompts, artifacts, or memory.

## ID-005: Idempotent stage markers

- Decision: Each deterministic stage writes a completion record last, using an
  atomic replace. On restart, the stage is reused only after every recorded
  output hash verifies; otherwise execution fails instead of guessing.
- Rationale: The contract requires restartability and stale-output denial but
  does not prescribe a journal format.
- Alternatives: Re-run whole experiments or infer completion from file
  presence. Both risk duplicate LLM calls or partial-artifact reuse.
- Impact: No metric or routing change; restart behavior becomes auditable.

## ID-006: Sequence warmup before the training segment

- Decision: Load exactly `sequence_window` prior trading dates before
  `train_start_time`, while fitting `ZScoreNorm` only on the frozen train
  interval and retaining train samples only inside that interval.
- Rationale: `TSDatasetH` needs prior feature rows to construct a complete first
  training sequence. This is feature history, not a widened label or fit split.
- Alternatives: Backfill all first-window gaps from the first training row or
  discard the first 20 train dates.
- Impact: Early train sequences use historically available inputs; no
  validation or test observation enters preprocessing or learning.

## ID-007: Missing realized stock return means unchanged price

- Decision: For an already selected holding whose aligned return-panel cell is
  missing, use zero realized return for that holding on that period.
- Rationale: The supplied A-share return panel uses missing cells for
  unavailable or suspended observations, and the reference execution logic
  treats those as no price movement. Directional mask missingness remains
  separately conservative in both directions.
- Alternatives: Fail the whole period or liquidate the holding. Both invent an
  executable price or trade and conflict with the no-silent-shortening rule.
- Impact: Suspended holdings carry at unchanged value until a finite return is
  available; candidate membership still requires a finite alpha.

## ID-008: Canonicalize unambiguous specialist config paths

- Decision: The Contract Validator accepts a FAMA path written either in the
  canonical dotted form (`z_model.train_params.dropout`) or as an unescaped
  JSON Pointer (`/z_model/train_params/dropout`). It immediately canonicalizes
  the latter before validating, recording, and applying the diff. Empty or
  escaped pointer components remain invalid.
- Rationale: The frozen contract determines the exact field and value space but
  does not assign experimental meaning to the path serialization. The selected
  Agent returned the correct single legal field using standard JSON Pointer
  notation. Treating this as a representation only avoids another LLM call and
  preserves the frozen prompt and schema files after trajectory materialization.
- Alternatives: Modify the already-frozen FAMA prompt/schema, or reject every
  semantically valid JSON Pointer spelling. The first would mutate formal
  trajectory inputs and the second would make formatting affect the experiment.
- Impact: None on the candidate configuration or intervention semantics. The
  persisted validated diff always uses the canonical dotted spelling.

## ID-009: Make feasibility saturation explicit in Agent evidence

- Decision: Validation evidence reports mean feasible minimum, feasible
  maximum, actual invested weight, and deterministic feasibility reason counts,
  together with the identity that objective coefficients cannot exceed frozen
  bounds when actual weight already equals the feasible maximum.
- Rationale: These values are required daily portfolio diagnostics and are
  necessary to distinguish optimizer preference from mathematical infeasibility.
  The earlier summary exposed only the shortfall and count of infeasible dates,
  which permitted an incorrect causal interpretation.
- Alternatives: Let the Agent infer feasibility from shortfall alone, or add an
  adaptive portfolio constraint. The former omits decisive validation evidence;
  the latter would violate the frozen intervention space.
- Impact: Routing receives a faithful summary of already-produced validation
  evidence. No result, gate rule, parameter, threshold, or portfolio behavior
  changes.

## ID-010: Researcher temporary workspace remains inside its test boundary

- Decision: Each researcher subprocess points Python and dependency temporary
  files to `EXP_xxx/test/_tmp` and removes that directory after the job.
- Rationale: A dependency probes filesystem behavior using a temporary file at
  import time. The researcher write guard correctly rejected the default system
  temp path, causing isolated evaluation to exit before reading test data.
- Alternatives: Permit arbitrary system-temp writes or weaken the researcher
  write guard. Both enlarge the write boundary unnecessarily.
- Impact: Researcher-side dependency scratch writes and artifacts remain
  physically confined to the experiment's `test/` tree; no test value is
  returned to the adaptive process.

## ID-011: Researcher-only epoch-log recovery

- Decision: A researcher subprocess may deterministically reconstruct a failed
  `test/epoch_metrics.jsonl` by replaying every recorded checkpoint in epoch
  order against the frozen test split, using the already-sealed adaptive
  train/validation epoch metrics. It then performs the normal full test job.
- Rationale: Early formal checkpoints were created correctly, but their
  researcher subprocesses exited at dependency initialization before test data
  was read. Re-training would change the timing record and is unnecessary.
- Alternatives: Leave the required audit artifact incomplete or rerun full
  model training. Neither is needed when all checkpoints and sample-order
  artifacts are hash-stable and available.
- Impact: Only missing researcher-only artifacts are completed. Recovery runs
  under the same test write boundary and emits no test values to its caller;
  adaptive routing, acceptance, rollback, stopping, and final configuration
  remain sealed and unchanged.

## ID-012: Auxiliary runner state preconditions

- Decision: The Round-0 helper reuses a verified completed Round-0 execution
  whenever any accepted state already exists. The single-round helper returns
  the sealed final configuration after completion, and the orchestrator rejects
  any direct adaptive-round call whose number/status does not exactly match the
  active state or exceeds `task.trials`.
- Rationale: The formal runner already enforced the budget, but auxiliary entry
  points must be equally restart-safe and must never create `EXP_011`.
- Alternatives: Treat helper scripts as non-production debugging utilities.
  That would leave a state-mutating path around the frozen stop rule.
- Impact: No active trajectory decision changes. Repeated or out-of-order CLI
  invocations are deterministic and non-mutating.
