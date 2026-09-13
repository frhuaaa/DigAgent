---
name: rass-agent-factor-exploration
description: "Guide the RASS specialist in exploring a frozen Qlib factor catalog with train-only evidence and proposing an auditable append-only feature intervention."
---

# RASS Agent factor exploration

## Purpose

RASS is an LLM-backed specialist, not a deterministic feature selector. It runs
once in `EXP_001`, after the six-feature `EXP_000` baseline. It explores the
frozen Qlib `z_alpha.feature_pool`, preserves the six ordered anchors, and
proposes exactly four additions so the selected feature list reaches ten.

Deterministic code prepares evidence and validates boundaries. The Agent owns
the factor choice and explanation. Do not replace the Agent with fixed IC
thresholds, a greedy minimum-correlation algorithm, a feature-importance rank,
or any other rule that directly chooses the four factors.

## Hard boundaries

- Use train-derived data only. Validation and test metrics, predictions,
  backtests, portfolio results, summaries, and memories are forbidden.
- Freeze and hash the factor catalog, six anchor expressions, target label,
  training period, data version, and evidence-generation code before RASS runs.
- Every proposed addition must resolve to one unique entry in the frozen
  catalog and must not duplicate an anchor or another addition.
- Preserve every anchor expression and its order. The only legal diff appends
  exactly four catalog factors to `z_alpha.selected_features`.
- Do not modify factor definitions, label, split, model, or portfolio settings.
- After successful execution, emit `alpha_frozen: true`; RASS cannot run again
  in the same trajectory.

## Evidence supplied to the Agent

For every catalog candidate, deterministic tools should provide a compact,
provenance-tagged train-only evidence record where computable:

- factor identifier, expression, and category;
- coverage and missing rate;
- daily cross-sectional IC and RankIC summaries against the raw configured
  close-to-close label;
- ICIR, RankICIR, sign consistency, and time/regime stability summaries;
- maximum absolute train correlation with the anchors and pairwise redundancy
  information among promising candidates;
- optional train-only nonlinear importance or mutual-information diagnostics;
- data, catalog, label, configuration, and evidence-code hashes.

These values are evidence, not pass/fail rules. Do not hide weak or conflicting
evidence, and do not label a deterministic ranking as the Agent's decision.

## Exploration procedure

1. Confirm that all inputs are train-only and hashes match the frozen run.
2. Diagnose what information the six anchors do and do not represent.
3. In the first Agent stage, explore the entire eligible frozen catalog and
   select exactly twelve candidates for detailed analysis.
4. Use the deterministic shortlist evidence to compare daily cross-sectional
   redundancy and marginal information after removing linear anchor exposure.
5. Choose four complementary factors from the twelve-factor shortlist under one coherent hypothesis about why
   broader representation should improve downstream learning.
6. State why the selected set is preferable to the strongest alternatives.
7. Return the exact append-only diff, expected validation signature, and a
   falsification condition. Do not inspect validation while choosing factors.

RASS may use quantitative evidence flexibly. A candidate with modest standalone
IC may be selected for complementarity, stability, or coverage, but the reason
must be explicit and grounded in the supplied train evidence. Economic stories
without train evidence are insufficient.

## Required output

Return a machine-readable proposal containing at least:

```yaml
method: agent_factor_exploration
data_split: train
selected_group: initial_feature_bootstrap
target_subset_size: 10
initial_features: []
added_features: []
selected_features: []
hypothesis: ""
local_diagnosis: ""
reasoning: ""
alternatives_considered: []
expected_signature: {}
falsification_condition: ""
confidence: 0.0
alpha_frozen_after_success: true
evidence_refs: []
provenance:
  data_hash: ""
  catalog_hash: ""
  label_hash: ""
  config_hash: ""
  evidence_code_hash: ""
  shortlist_evidence_hash: ""
```

The Contract Validator—not the Agent—verifies exact counts, append-only order,
catalog membership, duplicate absence, evidence provenance, allowed paths, and
test isolation. A legal successful `EXP_001` is promoted as the frozen
structural initialization; its normal Validation Gate verdict remains audit
evidence and does not retrospectively change the Agent's selection.

## Failure conditions

Fail the structural round rather than fabricate a proposal when:

- any validation/test-derived value enters RASS context;
- the catalog or evidence provenance is missing or changed after exploration;
- an addition is outside the catalog or duplicates another feature;
- an anchor is removed, replaced, reordered, or edited;
- the proposal does not contain exactly six anchors followed by four additions;
- the Agent cannot provide one coherent hypothesis and train-evidence-based
  rationale for all four additions;
- selected features change after `alpha_frozen: true`.
