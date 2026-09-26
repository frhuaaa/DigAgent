---
name: rass-agent-factor-exploration
description: "Guide RASS in a bounded, gate-controlled search over at most three distinct eight-factor candidates."
---

# RASS Agent factor exploration

## Purpose

RASS is an LLM-backed specialist, not a deterministic feature selector. It
operates only during the bounded initial alpha search. It explores
the frozen Qlib `z_alpha.feature_pool`, preserves the ordered anchors, and
proposes exactly enough additions (three for the current target) to reach eight.

Deterministic code prepares evidence and validates boundaries. The Agent owns
the factor choice and explanation. Do not replace the Agent with fixed IC
thresholds, a greedy minimum-correlation algorithm, a feature-importance rank,
or any other rule that directly chooses the three factors.

## Hard boundaries

- Use only the frozen RASS development evidence for the three complete calendar
  years immediately preceding `task.run_id` (`Y-3`, `Y-2`, and `Y-1`). Test
  metrics, predictions, backtests, and test-derived summaries or memories are
  forbidden. On retries, only the validation-safe summaries explicitly supplied
  by the runtime for previously executed RASS candidates may be used.
- Each period purges its final two trading signal dates so the T+2 realized
  label remains inside that period. Exact trading dates are resolved from the
  frozen Qlib calendar rather than hard-coded in the prompt.
- Freeze and hash the factor catalog, ordered anchor expressions, target label,
  training period, data version, and evidence-generation code before RASS runs.
- Every proposed addition must resolve to one unique entry in the frozen
  catalog and must not duplicate an anchor or another addition.
- Preserve every anchor expression and its order. The only legal diff appends
  exactly enough catalog factors to make `z_alpha.selected_features` reach eight.
- Do not modify factor definitions, label, split, model, or portfolio settings.
- Use the ordinary agent-valid portfolio Validation Gate. An accepted candidate
  freezes eight factors. After three complete rejected candidates, freeze the
  original five-factor baseline and permanently disable RASS.
- A retry may reuse one or two previously tried factors, but its complete
  unordered three-factor addition set must be new.

## Evidence supplied to the Agent

For every catalog candidate, deterministic tools should provide a compact,
provenance-tagged RASS-development evidence record where computable:

- factor identifier, expression, and category;
- coverage and missing rate;
- daily cross-sectional IC and RankIC summaries against the raw label selected
  by `task.prediction_mode` (`c2c` or `o2o`);
- ICIR, RankICIR, sign consistency, and time/regime stability summaries;
- separate summaries for the three complete calendar-year periods;
- maximum absolute train correlation with the anchors and pairwise redundancy
  information among promising candidates;
- optional development-period nonlinear importance or mutual-information diagnostics;
- data, catalog, label, configuration, and evidence-code hashes.

These values are evidence, not pass/fail rules. Do not hide weak or conflicting
evidence, and do not label a deterministic ranking as the Agent's decision.

## Exploration procedure

1. Confirm that all development inputs belong to the three frozen calendar-year
   periods derived from `task.run_id`, contain no test data, and hashes match
   the frozen run. Retry feedback must be limited to the supplied validation-safe
   summaries of previously executed RASS candidates.
2. Diagnose what information the current five anchors do and do not represent.
3. In the first Agent stage, explore the entire eligible frozen catalog and
   select exactly twelve candidates for detailed analysis.
4. Prioritize cross-period predictive stability rather than an aggregate
   headline metric. Use the deterministic shortlist evidence to compare daily
   cross-sectional redundancy and marginal information after removing linear
   anchor exposure, including its three rolling-period views.
5. Choose three complementary factors. On retries, incorporate the supplied
   validation-safe failure outcome and choose a non-identical joint set.
6. State why the selected set is preferable to the strongest alternatives.
7. Return the exact bounded feature diff, expected validation signature, and a
   falsification condition. Validation feedback from earlier RASS candidates is
   allowed only through the supplied validation-safe memory; test is forbidden.

RASS must not use a fixed weighted score or mechanically take the metric Top-K.
The rolling-period metrics organize evidence for Agent judgment; they are not a
new optimizer. A candidate with modest standalone
IC may be selected for complementarity, stability, or coverage, but the reason
must be explicit and grounded in the supplied development evidence. Economic
stories without quantitative evidence are insufficient.

## Required output

Return a machine-readable proposal containing at least:

```yaml
method: agent_factor_exploration
data_split: rass_development
selected_group: "initial_feature_bootstrap"
target_subset_size: 8
initial_features: []
removed_features: []
added_features: []
selected_features: []
hypothesis: ""
local_diagnosis: ""
reasoning: ""
alternatives_considered: []
expected_signature: {}
falsification_condition: ""
confidence: 0.0
evidence_frozen_after_success: true
evidence_refs: []
provenance:
  data_hash: ""
  catalog_hash: ""
  label_hash: ""
  evidence_context_hash: ""
  evidence_code_hash: ""
  shortlist_evidence_hash: ""
```

The Contract Validator—not the Agent—verifies exact counts, bootstrap order or
one-feature replacement position,
catalog membership, duplicate absence, evidence provenance, allowed paths,
prior-set non-repetition, and test isolation. Every candidate runs the full
dependency path, and its normal Validation Gate verdict controls promotion or
rollback.

## Failure conditions

Fail the structural round rather than fabricate a proposal when:

- any test-derived value enters RASS context, or validation-derived content is
  supplied outside the bounded retry summaries produced by the runtime;
- the catalog or evidence provenance is missing or changed after exploration;
- an addition is outside the catalog or duplicates another feature;
- an anchor is removed, replaced, reordered, or edited;
- bootstrap does not append exactly enough additions to reach eight;
- the Agent cannot provide one coherent hypothesis and train-evidence-based
  rationale for every addition;
- any development evidence falls outside the frozen three-year set, or retry
  feedback falls outside the bounded validation-safe summaries supplied by the runtime.
