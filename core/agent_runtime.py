from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from pathlib import Path

import jsonschema
import pandas as pd
import yaml

from adapters.llm_client import AgentClient
from core.errors import ContractError
from core.io_utils import load_json, sha256_file
from core.isolation import assert_validation_safe_value
from core.rass_policy import (
    RASS_MAX_FAILED_ATTEMPTS,
    rass_attempt_summaries,
    rass_failed_attempt_count,
)
from memory.memory_store import MemoryStore


MODEL_METRICS = ("ic", "icir", "rank_ic", "rank_icir")
MODEL_METRIC_DECIMALS = 4
EXPERIMENT_ID_PATTERN = re.compile(r"EXP_[0-9]{3}")
LAYER_TO_AGENT = {"alpha": "RASS", "model": "FAMA", "portfolio": "RAPA"}
POST_BOOTSTRAP_AGENTS = frozenset({"FAMA", "RAPA"})
RAPA_SEARCH_PARAMETERS = ("risk_aversion", "turnover_penalty")
RAPA_MAX_ATTEMPTS_PER_PARAMETER = 4


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric_output(value: object) -> float | None:
    number = _finite_float(value)
    return None if number is None else round(number, MODEL_METRIC_DECIMALS)


def canonicalize_rass_selected_features(proposal: dict, current_features: list[str]) -> dict:
    """Derive RASS's executable feature list from frozen parents and expressions."""
    normalized = deepcopy(proposal)
    additions = normalized.get("added_features")
    if not isinstance(additions, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("expression"), str)
        for item in additions
    ):
        return normalized
    expressions = [item["expression"] for item in additions]
    group = normalized.get("selected_group")
    if group == "initial_feature_bootstrap":
        normalized["selected_features"] = list(current_features) + expressions
    elif group == "feature_refinement":
        removed = normalized.get("removed_features")
        if len(expressions) == 1 and isinstance(removed, list) and len(removed) == 1 and removed[0] in current_features:
            selected = list(current_features)
            selected[selected.index(removed[0])] = expressions[0]
            normalized["selected_features"] = selected
    return normalized


def canonicalize_rass_contract_metadata(proposal: dict, evidence: dict, joint_evidence: dict) -> dict:
    """Fill non-decision RASS provenance from trusted runtime evidence.

    Hashes and evidence paths are execution metadata, not specialist choices.  Deriving
    them here prevents an otherwise valid round from being lost to an LLM copy typo.
    """
    normalized = deepcopy(proposal)
    normalized["provenance"] = {
        "data_hash": evidence["input_data_hash"],
        "catalog_hash": evidence["catalog_hash"],
        "label_hash": evidence["label_hash"],
        "evidence_context_hash": evidence["rass_evidence_context_hash"],
        "evidence_code_hash": evidence["evidence_code_hash"],
        "shortlist_evidence_hash": joint_evidence["evidence_hash"],
    }
    normalized["evidence_refs"] = [
        "configs/rass_train_evidence.json",
        "logs/rass_shortlist_evidence.json",
        f"evidence_hash: {joint_evidence['evidence_hash']}",
    ]
    return normalized


def consecutive_unaccepted_layer(records: list[dict], threshold: int = 3) -> str | None:
    """Return the blocked Agent after a trailing same-layer rejection streak."""
    adaptive = [item for item in records if item.get("layer") in LAYER_TO_AGENT]
    if not adaptive or adaptive[-1].get("accepted") is not False:
        return None
    layer = adaptive[-1]["layer"]
    count = 0
    for item in reversed(adaptive):
        if item.get("layer") != layer or item.get("accepted") is not False:
            break
        count += 1
    return LAYER_TO_AGENT[layer] if count >= threshold else None


def build_rapa_search_state(records: list[dict], config: dict) -> dict:
    """Summarize validation-only directional search state for RAPA."""

    parameters = {}
    for parameter in RAPA_SEARCH_PARAMETERS:
        path = f"z_portfolio.{parameter}"
        history = []
        for item in records:
            if item.get("layer") != "portfolio" or item.get("verdict") == "CONTRACT_REJECTED":
                continue
            for change in item.get("intervention") or []:
                if isinstance(change, dict) and change.get("path") == path:
                    if "old_value" not in change or "new_value" not in change:
                        continue
                    old_value = float(change["old_value"])
                    new_value = float(change["new_value"])
                    history.append({
                        "experiment_id": item["experiment_id"],
                        "old_value": old_value,
                        "new_value": new_value,
                        "direction": "increase" if new_value > old_value else "decrease",
                        "step": abs(new_value - old_value),
                        "accepted": bool(item.get("accepted", False)),
                        "verdict": item.get("verdict"),
                    })
        attempts = len(history)
        if not history:
            phase = "PROBE"
        elif history[-1]["accepted"]:
            phase = "EXPAND"
        else:
            phase = "REVERSE_PROBE"
        parameters[parameter] = {
            "current_value": float(config["z_portfolio"][parameter]),
            "attempts": attempts,
            "remaining_attempts": max(0, RAPA_MAX_ATTEMPTS_PER_PARAMETER - attempts),
            "phase": phase,
            "history": history,
        }
    return {
        "maximum_attempts_per_parameter": RAPA_MAX_ATTEMPTS_PER_PARAMETER,
        "parameters": parameters,
        "available_parameters": [
            parameter
            for parameter, state in parameters.items()
            if state["remaining_attempts"] > 0
        ],
        "selection_rule": "retain_best_via_validation_gate_promotion_and_rollback",
        "contains_test_derived_data": False,
    }


def resolve_model_training_source(experiment_dir: Path) -> Path:
    """Resolve a reused checkpoint chain to the experiment that trained it."""
    current = experiment_dir.resolve()
    visited: set[str] = set()
    while True:
        if current.name in visited:
            raise ContractError(f"MODEL_TRAINING_SOURCE_CYCLE: {current.name}")
        visited.add(current.name)
        if (current / "logs" / "training_metrics.jsonl").is_file():
            return current
        metadata_path = current / "artifacts" / "selected_checkpoint.json"
        if not metadata_path.is_file():
            raise ContractError(f"MODEL_TRAINING_LOG_MISSING: {current.name}")
        try:
            metadata = load_json(metadata_path)
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            raise ContractError(f"MODEL_TRAINING_SOURCE_METADATA_INVALID: {current.name}") from exc
        reused_from = metadata.get("reused_from") if isinstance(metadata, dict) else None
        if not isinstance(reused_from, str) or EXPERIMENT_ID_PATTERN.fullmatch(reused_from) is None:
            raise ContractError(f"MODEL_TRAINING_SOURCE_ID_INVALID: {current.name}")
        candidate = (current.parent / reused_from).resolve()
        if candidate.parent != current.parent.resolve() or not candidate.is_dir():
            raise ContractError(f"MODEL_TRAINING_SOURCE_MISSING: {reused_from}")
        current = candidate


def _model_training_records(experiment_dir: Path) -> tuple[list[dict], dict, Path]:
    training_log = experiment_dir / "logs" / "training_metrics.jsonl"
    if not training_log.is_file():
        raise ContractError(f"MODEL_TRAINING_LOG_MISSING: {experiment_dir.name}")
    try:
        records = [
            json.loads(line)
            for line in training_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (json.JSONDecodeError, OSError) as exc:
        raise ContractError(f"MODEL_TRAINING_LOG_INVALID: {experiment_dir.name}") from exc
    epochs = [record for record in records if "epoch" in record and record.get("event") is None]
    selections = [record for record in records if record.get("event") == "checkpoint_selected"]
    if not epochs:
        raise ContractError(f"MODEL_TRAINING_EPOCHS_MISSING: {experiment_dir.name}")
    epoch_numbers = [int(record["epoch"]) for record in epochs]
    if epoch_numbers != sorted(epoch_numbers) or len(epoch_numbers) != len(set(epoch_numbers)):
        raise ContractError(f"MODEL_TRAINING_EPOCH_ORDER_INVALID: {experiment_dir.name}")
    if len(selections) != 1:
        raise ContractError(f"MODEL_CHECKPOINT_SELECTION_COUNT_INVALID: {experiment_dir.name}")
    return epochs, selections[0], training_log


def _split_metric(record: dict, split: str, metric: str) -> float | None:
    values = record.get(split)
    return _finite_float(values.get(metric)) if isinstance(values, dict) else None


def build_model_training_summary(experiment_dir: Path) -> dict:
    """Build compact train/train-valid model evidence supplied to CLEM."""
    requested_experiment_dir = experiment_dir.resolve()
    source_experiment_dir = resolve_model_training_source(requested_experiment_dir)
    epochs, selected, _ = _model_training_records(source_experiment_dir)
    metric_label = str(selected.get("metric", ""))
    metric = metric_label.removeprefix("train_valid.")
    if metric not in MODEL_METRICS:
        raise ContractError(f"MODEL_CHECKPOINT_METRIC_INVALID: {metric_label}")
    selected_epoch = int(selected["epoch"])
    selected_records = [record for record in epochs if int(record["epoch"]) == selected_epoch]
    if len(selected_records) != 1:
        raise ContractError(f"MODEL_SELECTED_EPOCH_MISSING: {source_experiment_dir.name}")
    best_record = selected_records[0]
    train_valid_scores = [
        score
        for record in epochs
        if (score := _split_metric(record, "train_valid", metric)) is not None
    ]
    if not train_valid_scores:
        raise ContractError(f"MODEL_TRAIN_VALID_METRIC_EMPTY: {source_experiment_dir.name}")
    selected_score = _finite_float(selected.get("score"))
    selected_record_score = _split_metric(best_record, "train_valid", metric)
    rounding_tolerance = 10 ** (-MODEL_METRIC_DECIMALS)
    if (
        selected_score is None
        or selected_record_score is None
        or not math.isclose(selected_score, selected_record_score, rel_tol=0.0, abs_tol=rounding_tolerance)
        or selected_score < max(train_valid_scores) - rounding_tolerance
    ):
        raise ContractError(f"MODEL_CHECKPOINT_SELECTION_MISMATCH: {source_experiment_dir.name}")

    first_record = epochs[0]
    last_record = epochs[-1]
    first_train = _split_metric(first_record, "train", metric)
    first_train_valid = _split_metric(first_record, "train_valid", metric)
    train_at_best = _split_metric(best_record, "train", metric)
    last_train = _split_metric(last_record, "train", metric)
    last_train_valid = _split_metric(last_record, "train_valid", metric)

    def difference(left: float | None, right: float | None) -> float | None:
        return None if left is None or right is None else _metric_output(left - right)

    config = load_json(source_experiment_dir / "config.json")
    configured_epochs = int(config["z_model"]["base_params"]["n_epochs"])
    summary = {
        "model_source_experiment_id": source_experiment_dir.name,
        "model_reused": source_experiment_dir != requested_experiment_dir,
        "epochs_completed": len(epochs),
        "configured_epochs": configured_epochs,
        "first_epoch": int(first_record["epoch"]),
        "last_epoch": int(last_record["epoch"]),
        "selected_metric": metric,
        "selected_epoch": selected_epoch,
        "best_train_valid_score": _metric_output(selected_score),
        "first_train_score": _metric_output(first_train),
        "first_train_valid_score": _metric_output(first_train_valid),
        "train_score_at_best": _metric_output(train_at_best),
        "last_train_score": _metric_output(last_train),
        "last_train_valid_score": _metric_output(last_train_valid),
        "generalization_gap_at_best": difference(train_at_best, selected_score),
        "generalization_gap_last": difference(last_train, last_train_valid),
        "post_best_train_valid_drop": difference(selected_score, last_train_valid),
        "post_best_train_change": difference(last_train, train_at_best),
        "epochs_after_best": int(last_record["epoch"]) - selected_epoch,
        "early_stopped": len(epochs) < configured_epochs,
        "checkpoint_selection": {**selected, "score": _metric_output(selected_score)},
        "contains_test_derived_data": False,
    }
    assert_validation_safe_value(summary)
    return summary


def build_fama_training_history(experiment_dir: Path) -> dict:
    """Build the complete train/train-valid epoch curve supplied only to FAMA."""
    requested_experiment_dir = experiment_dir.resolve()
    source_experiment_dir = resolve_model_training_source(requested_experiment_dir)
    epochs, selected, training_log = _model_training_records(source_experiment_dir)
    columns = ["epoch", "training_loss"]
    for metric in MODEL_METRICS:
        columns.extend((f"train_{metric}", f"train_valid_{metric}"))
    rows = []
    for record in epochs:
        row: list[int | float | None] = [
            int(record["epoch"]),
            _metric_output(record.get("training_loss")),
        ]
        for metric in MODEL_METRICS:
            row.extend((
                _metric_output(_split_metric(record, "train", metric)),
                _metric_output(_split_metric(record, "train_valid", metric)),
            ))
        rows.append(row)
    history = {
        "model_source_experiment_id": source_experiment_dir.name,
        "model_reused": source_experiment_dir != requested_experiment_dir,
        "columns": columns,
        "rows": rows,
        "checkpoint_selection": {**selected, "score": _metric_output(selected.get("score"))},
        "derived_summary": build_model_training_summary(requested_experiment_dir),
        "training_log_sha256": sha256_file(training_log),
        "contains_test_derived_data": False,
    }
    config = load_json(source_experiment_dir / "config.json")
    ensemble = config.get("ensemble", {})
    if ensemble.get("enabled") is True:
        members = []
        for seed in ensemble.get("seeds", []):
            member_log = source_experiment_dir / "logs" / f"training_metrics_seed_{seed}.jsonl"
            if not member_log.is_file():
                raise ContractError(f"MODEL_ENSEMBLE_TRAINING_LOG_MISSING: seed={seed}")
            records = [
                json.loads(line)
                for line in member_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            member_epochs = [record for record in records if "epoch" in record and record.get("event") is None]
            member_selections = [record for record in records if record.get("event") == "checkpoint_selected"]
            if not member_epochs or len(member_selections) != 1:
                raise ContractError(f"MODEL_ENSEMBLE_TRAINING_LOG_INVALID: seed={seed}")
            member_rows = []
            for record in member_epochs:
                row: list[int | float | None] = [
                    int(record["epoch"]),
                    _metric_output(record.get("training_loss")),
                ]
                for metric in MODEL_METRICS:
                    row.extend((
                        _metric_output(_split_metric(record, "train", metric)),
                        _metric_output(_split_metric(record, "train_valid", metric)),
                    ))
                member_rows.append(row)
            members.append({
                "seed": int(seed),
                "rows": member_rows,
                "checkpoint_selection": member_selections[0],
                "training_log_sha256": sha256_file(member_log),
            })
        history["ensemble"] = {
            "method": ensemble["method"],
            "checkpoint_selection": ensemble["checkpoint_selection"],
            "columns": columns,
            "members": members,
        }
    assert_validation_safe_value(history)
    return history


def build_validation_evidence(experiment_dir: Path) -> dict:
    result = load_json(experiment_dir / "result.json")
    diagnostics = pd.read_csv(experiment_dir / "artifacts" / "portfolio_daily_diagnostics.csv")
    signal_daily = pd.read_csv(experiment_dir / "artifacts" / "signal_daily_metrics.csv")
    def _mean(column: str) -> float | None:
        if column not in diagnostics:
            return None
        value = float(pd.to_numeric(diagnostics[column], errors="coerce").mean())
        return value if math.isfinite(value) else None

    def _sharpe(column: str) -> float | None:
        if column not in diagnostics:
            return None
        values = pd.to_numeric(diagnostics[column], errors="coerce").dropna()
        if len(values) < 2:
            return None
        std = float(values.std(ddof=1))
        value = float(values.mean() / std * math.sqrt(252.0)) if std > 0 else float("nan")
        return value if math.isfinite(value) else None

    turnover = _mean("one_way_turn")
    gross_mean = _mean("gross_return")
    evidence = {
        "experiment_id": experiment_dir.name,
        "result": result,
        "signal_daily_agent_valid_date_count": int(len(signal_daily)),
        "portfolio": {
            "missing_signal_dates": int(diagnostics["missing_signal"].astype(bool).sum()),
            "full_position_infeasible_dates": int((~diagnostics["full_position_feasible"].astype(bool)).sum()),
            "feasible_min_invested_weight_mean": float(diagnostics["feasible_min_invested_weight"].mean()),
            "feasible_max_invested_weight_mean": float(diagnostics["feasible_max_invested_weight"].mean()),
            "invested_weight_mean": float(diagnostics["invested_weight"].mean()),
            "transaction_cost_drag_mean": float(diagnostics["transaction_cost_drag"].mean()),
            "gross_return_mean": gross_mean,
            "net_return_mean": _mean("net_return"),
            "gross_sharpe_ratio": _sharpe("gross_return"),
            "net_sharpe_ratio_recomputed": _sharpe("net_return"),
            "one_way_turnover_mean": turnover,
            "gross_return_per_unit_turnover": (
                gross_mean / turnover
                if gross_mean is not None and turnover is not None and turnover > 1e-12
                else None
            ),
            "signal_weight_rank_corr_mean": _mean("signal_weight_rank_corr"),
            "signal_trade_rank_corr_mean": _mean("signal_trade_rank_corr"),
            "signal_rank_autocorrelation_mean": _mean("signal_rank_autocorrelation"),
            "weight_persistence_mean": _mean("weight_persistence"),
            "portfolio_concentration_mean": float(diagnostics["portfolio_concentration"].mean()),
            "invested_weight_shortfall_mean": float(diagnostics["invested_weight_shortfall"].mean()),
            "alpha_term_mean": float(diagnostics["alpha_term"].mean()),
            "risk_term_mean": float(diagnostics["risk_term"].mean()),
            "turnover_term_mean": float(diagnostics["turnover_term"].mean()),
            "feasibility_reason_counts": {
                str(reason): int(count)
                for reason, count in diagnostics["feasibility_reason_code"].value_counts(dropna=False).sort_index().items()
            },
            "constraint_interpretation": (
                "When invested_weight equals feasible_max_invested_weight, cash is caused by frozen "
                "directional/per-stock/universe bounds; RAPA objective coefficients cannot increase "
                "the feasible maximum."
            ),
        },
        "contains_test_derived_data": False,
    }
    topk_path = experiment_dir / "artifacts" / "topk_result.json"
    if topk_path.is_file():
        evidence["topk_comparison"] = load_json(topk_path)
    training_log = experiment_dir / "logs" / "training_metrics.jsonl"
    if training_log.is_file():
        evidence["model"] = build_model_training_summary(experiment_dir)
    else:
        evidence["model"] = {"reused": True}
    ensemble_path = experiment_dir / "artifacts" / "ensemble_validation_diagnostics.json"
    if ensemble_path.is_file():
        evidence["ensemble"] = load_json(ensemble_path)
    assert_validation_safe_value(evidence)
    return evidence


def validate_rass_shortlist_payload(repo_root: Path, run_root: Path, config: dict, proposal: dict) -> None:
    schema = load_json(repo_root / "agents" / "RASS" / "shortlist.schema.json")
    jsonschema.Draft202012Validator(schema).validate(proposal)
    evidence = load_json(run_root / "configs" / "rass_train_evidence.json")
    if proposal["evidence_hash"] != evidence["evidence_hash"]:
        raise ContractError("RASS_SHORTLIST_EVIDENCE_HASH_MISMATCH")
    eligible_pairs = {
        (record["id"], record["expression"])
        for record in evidence["records"]
        if record.get("eligible", False) and record["expression"] not in config["z_alpha"]["selected_features"]
    }
    pairs = [(item["id"], item["expression"]) for item in proposal["shortlist"]]
    if len(set(pairs)) != 12:
        raise ContractError("RASS_SHORTLIST_DUPLICATE")
    if any(pair not in eligible_pairs for pair in pairs):
        raise ContractError("RASS_SHORTLIST_INELIGIBLE_OR_UNKNOWN")
    assert_validation_safe_value(proposal)


class AgentRuntime:
    def __init__(self, repo_root: Path, run_root: Path, config: dict):
        self.repo_root = repo_root.resolve()
        self.run_root = run_root.resolve()
        self.config = config
        self.client = AgentClient(repo_root, config["agent"])
        self.memory = MemoryStore(run_root)

    def _prompt(self, agent: str) -> str:
        if agent == "CLEM":
            return (self.repo_root / "agents" / "CLEM" / "clem.md").read_text(encoding="utf-8") + "\n" + (self.repo_root / "agents" / "CLEM" / "SYSTEM.md").read_text(encoding="utf-8")
        if agent == "RASS_SHORTLIST":
            return (self.repo_root / "agents" / "RASS" / "SYSTEM.md").read_text(encoding="utf-8") + "\n" + (self.repo_root / "agents" / "RASS" / "SHORTLIST.md").read_text(encoding="utf-8")
        base = (self.repo_root / "agents" / agent / "SYSTEM.md").read_text(encoding="utf-8")
        if agent == "RASS":
            base += "\n" + (self.repo_root / "agents" / "RASS" / "SKILL.md").read_text(encoding="utf-8")
        return base

    def _schema(self, agent: str) -> dict:
        if agent == "RASS_SHORTLIST":
            return load_json(self.repo_root / "agents" / "RASS" / "shortlist.schema.json")
        path = self.repo_root / "agents" / ("CLEM" if agent == "CLEM" else agent) / ("schema.json" if agent == "CLEM" else "intervention.schema.json")
        return load_json(path)

    def _clem_route_schema(
        self,
        *,
        alpha_frozen: bool,
        forced_agent: str | None,
    ) -> tuple[dict, list[str]]:
        """Return a CLEM schema whose routing enum matches the frozen state."""

        schema = deepcopy(self._schema("CLEM"))
        eligible_agents = ["FAMA", "RAPA"] if alpha_frozen else ["RASS"]
        if forced_agent is not None:
            eligible_agents = [forced_agent]
        schema["properties"]["selected_agent"]["enum"] = eligible_agents
        return schema, eligible_agents

    @staticmethod
    def _adaptive_config(config: dict) -> dict:
        task = config["task"]
        return {
            "task": {
                "name": task["name"],
                "instruments": task["instruments"],
                "benchmark": task["benchmark"],
                "train_period": [task["train_start_time"], task["train_end_time"]],
                "train_valid_period": [
                    task["train_valid_start_time"],
                    task["train_valid_end_time"],
                ],
                "agent_valid_period": [
                    task["agent_valid_start_time"],
                    task["agent_valid_end_time"],
                ],
                "seed": task["seed"],
            },
            "z_alpha": config["z_alpha"],
            "z_model": config["z_model"],
            "z_portfolio": config["z_portfolio"],
        }

    def clem(self, round_number: int, accepted_experiment: str, evidence: dict) -> dict:
        complete_memory = self.memory.projection("cross_layer")
        blocked_agent = consecutive_unaccepted_layer(complete_memory)
        alpha_frozen = bool(self.config["z_alpha"].get("alpha_frozen", False))
        rass_failures = rass_failed_attempt_count(complete_memory)
        if not alpha_frozen and rass_failures >= RASS_MAX_FAILED_ATTEMPTS:
            raise ContractError("RASS_FAILURE_LIMIT_REACHED_WITHOUT_ALPHA_FALLBACK")
        rapa_search_state = build_rapa_search_state(complete_memory, self.config)
        forced_agent = None
        if alpha_frozen and blocked_agent in POST_BOOTSTRAP_AGENTS:
            forced_agent = next(iter(POST_BOOTSTRAP_AGENTS - {blocked_agent}))
        if alpha_frozen and not rapa_search_state["available_parameters"]:
            forced_agent = "FAMA"
        clem_schema, eligible_agents = self._clem_route_schema(
            alpha_frozen=alpha_frozen,
            forced_agent=forced_agent,
        )
        context = {
            "round": round_number,
            "accepted_experiment_id": accepted_experiment,
            "accepted_adaptive_config": self._adaptive_config(self.config),
            "current_validation_evidence": evidence,
            "complete_memory_bank": complete_memory,
            "frozen_constraints": {
                "minimum_confidence": 0.5,
                "primary_adaptive_objective": "validation_sharpe_ratio_on_net_daily_returns",
                "excess_and_benchmark_relative_metrics": "diagnostic_only_cannot_trigger_intervention",
                "rapa_requires_validation_supported_portfolio_opportunity": True,
                "positive_validation_sharpe_does_not_block_rapa": True,
                "turnover_target_or_reference": None,
                "rass_bootstrap_active": not alpha_frozen,
                "rass_failed_attempts": rass_failures,
                "rass_max_failed_attempts": RASS_MAX_FAILED_ATTEMPTS,
                "rass_failure_action": "freeze_five_factor_alpha_and_continue_with_fama_or_rapa",
                "eligible_agents": eligible_agents,
                "alpha_frozen_permanently_disables_rass": alpha_frozen,
                "one_layer_only": True,
                "three_consecutive_unaccepted_same_layer_cooldown": {
                    "blocked_agent_this_round": blocked_agent,
                    "forced_agent_this_round": forced_agent,
                    "required_action": "select_and_rediagnose_the_other_post_bootstrap_layer" if forced_agent else None,
                    "cooldown_rounds": 1,
                },
                "rapa_available_parameters": rapa_search_state["available_parameters"],
            },
        }
        hard_route_instruction = (
            "\n\nHARD ROUTING CONSTRAINT: selected_agent must be exactly one of "
            f"{eligible_agents}. Do not select or discuss an ineligible route as the decision."
        )
        decision = self.client.complete_json(
            self._prompt("CLEM") + hard_route_instruction,
            context,
            clem_schema,
            "clem_decision",
        )
        self.validate_clem(decision, round_number, evidence)
        return decision

    def validate_clem(self, decision: dict, round_number: int, evidence: dict | None = None) -> None:
        jsonschema.Draft202012Validator(self._schema("CLEM")).validate(decision)
        rass_active = len(self.config["z_alpha"]["selected_features"]) == 5 and not self.config["z_alpha"]["alpha_frozen"]
        rass_failures = rass_failed_attempt_count(self.memory.projection("cross_layer"))
        if rass_active and rass_failures >= RASS_MAX_FAILED_ATTEMPTS:
            raise ContractError("RASS_FAILURE_LIMIT_REACHED_WITHOUT_ALPHA_FALLBACK")
        if rass_active and decision["selected_agent"] != "RASS":
            raise ContractError("CLEM_MANDATORY_RASS_ROUTE_VIOLATION")
        if not rass_active and decision["selected_agent"] == "RASS":
            raise ContractError("CLEM_RASS_ONLY_LEGAL_DURING_BOOTSTRAP_SEARCH")
        blocked_agent = consecutive_unaccepted_layer(self.memory.projection("cross_layer"))
        if self.config["z_alpha"].get("alpha_frozen", False) and blocked_agent is not None and decision["selected_agent"] == blocked_agent:
            raise ContractError("CLEM_THREE_CONSECUTIVE_UNACCEPTED_LAYER_COOLDOWN")
        if decision["confidence"] < 0.5:
            raise ContractError("CLEM_CONFIDENCE_THRESHOLD_VIOLATION")
        if decision["selected_agent"] == "RAPA":
            assessment = decision["primary_objective_assessment"]
            forced_rapa = bool(self.config["z_alpha"].get("alpha_frozen", False)) and blocked_agent == "FAMA"
            if not assessment["failure_supported"] and not forced_rapa:
                raise ContractError("CLEM_RAPA_PORTFOLIO_OPPORTUNITY_NOT_SUPPORTED")
            if evidence is None:
                raise ContractError("CLEM_RAPA_PRIMARY_OBJECTIVE_EVIDENCE_MISSING")
            observed = _finite_float(assessment["observed_value"])
            actual = _finite_float(evidence.get("result", {}).get("sharpe_ratio"))
            if observed is None or actual is None or not math.isclose(observed, actual, rel_tol=0.0, abs_tol=5e-4):
                raise ContractError("CLEM_RAPA_VALIDATION_SHARPE_MISMATCH")
        assert_validation_safe_value(decision)

    def validate_rass_shortlist(self, proposal: dict) -> None:
        if self.config["z_alpha"].get("alpha_frozen", False):
            raise ContractError("RASS_DISABLED_AFTER_ALPHA_RESOLUTION")
        validate_rass_shortlist_payload(self.repo_root, self.run_root, self.config, proposal)

    def rass_shortlist(self, clem: dict, contract_feedback: dict | None = None) -> dict:
        if self.config["z_alpha"].get("alpha_frozen", False):
            raise ContractError("RASS_DISABLED_AFTER_ALPHA_RESOLUTION")
        evidence = load_json(self.run_root / "configs" / "rass_train_evidence.json")
        prior_attempts = rass_attempt_summaries(self.memory.projection("cross_layer"))
        compact_fields = (
            "id", "expression", "category", "eligible", "eligibility_reasons",
            "coverage", "daily_coverage_p05", "nonconstant_cross_section_ratio",
            "ic", "icir", "rank_ic", "rank_icir", "period_ic",
            "period_rank_ic", "positive_ic_period_ratio", "worst_period_ic",
            "rolling_period_evidence",
            "max_abs_anchor_correlation", "strongest_candidate_redundancy",
        )
        context = {
            "clem_goal": {
                "failure_summary": clem["failure_summary"],
                "intervention_goal": clem["intervention_goal"],
            },
            "initial_features": self.config["z_alpha"]["selected_features"],
            "candidate_count": evidence["candidate_count"],
            "eligible_candidate_count": evidence["eligible_candidate_count"],
            "candidate_evidence": [
                {key: record.get(key) for key in compact_fields}
                for record in evidence["records"]
            ],
            "evidence_hash": evidence["evidence_hash"],
            "data_split": "rass_development",
            "prior_agent_valid_rass_attempts": prior_attempts,
            "retry_contract": {
                "maximum_failed_attempts": RASS_MAX_FAILED_ATTEMPTS,
                "next_three_factor_set_must_not_equal_any_prior_set": True,
                "partial_overlap_with_prior_sets_allowed": True,
                "test_evidence_forbidden": True,
            },
        }
        if contract_feedback is not None:
            context["contract_feedback"] = contract_feedback
        assert_validation_safe_value(context)
        proposal = self.client.complete_json(
            self._prompt("RASS_SHORTLIST"),
            context,
            self._schema("RASS_SHORTLIST"),
            "rass_shortlist",
        )
        self.validate_rass_shortlist(proposal)
        return proposal

    def rass_final(self, clem: dict, shortlist: dict, joint_evidence: dict, contract_feedback: dict | None = None) -> dict:
        evidence = load_json(self.run_root / "configs" / "rass_train_evidence.json")
        current_features = list(self.config["z_alpha"]["selected_features"])
        bootstrap = len(current_features) == 5 and not self.config["z_alpha"].get("alpha_frozen", False)
        prior_attempts = rass_attempt_summaries(self.memory.projection("cross_layer"))
        context = {
            "clem_goal": {
                "failure_summary": clem["failure_summary"],
                "intervention_goal": clem["intervention_goal"],
            },
            "initial_features": current_features,
            "operation_contract": {
                "selected_group": "initial_feature_bootstrap" if bootstrap else "feature_refinement",
                "target_subset_size": 8,
                "required_additions": 8 - len(current_features) if bootstrap else 1,
                "required_removals": 0 if bootstrap else 1,
                "immutable_anchor_prefix": current_features[:5],
                "refinement_rule": None if bootstrap else "replace one feature at index 5..7 and preserve its position",
                "selected_features_source": "deterministically derived from current features and added_features[].expression",
                "provenance_source": "deterministically derived from trusted runtime evidence; Agent output is overwritten",
                "evidence_refs_source": "deterministically derived from trusted runtime evidence; Agent output is overwritten",
                "evidence_frozen_after_success": True,
                "maximum_failed_attempts": RASS_MAX_FAILED_ATTEMPTS,
                "next_three_factor_set_must_not_equal_any_prior_set": True,
                "partial_overlap_with_prior_sets_allowed": True,
            },
            "prior_agent_valid_rass_attempts": prior_attempts,
            "shortlist_decision": shortlist,
            "joint_rass_development_evidence": joint_evidence,
            "base_evidence_provenance": {
                key: evidence[key]
                for key in (
                    "input_data_hash",
                    "catalog_hash",
                    "label_hash",
                    "rass_evidence_context_hash",
                    "evidence_code_hash",
                    "evidence_hash",
                )
            },
            "required_evidence_refs": [
                "configs/rass_train_evidence.json",
                "logs/rass_shortlist_evidence.json",
                f"evidence_hash: {joint_evidence['evidence_hash']}",
            ],
        }
        if contract_feedback is not None:
            context["contract_feedback"] = contract_feedback
        assert_validation_safe_value(context)
        proposal = self.client.complete_json(self._prompt("RASS"), context, self._schema("RASS"), "rass_proposal")
        proposal = canonicalize_rass_selected_features(proposal, current_features)
        return canonicalize_rass_contract_metadata(proposal, evidence, joint_evidence)

    def specialist(
        self,
        agent: str,
        clem: dict,
        evidence: dict,
        contract_feedback: dict | None = None,
        model_training_history: dict | None = None,
    ) -> dict:
        context = {
            "clem_diagnosis": clem,
            "accepted_adaptive_config": self._adaptive_config(self.config),
        }
        if agent == "RASS":
            context = {
                "clem_goal": {
                    "failure_summary": clem["failure_summary"],
                    "intervention_goal": clem["intervention_goal"],
                },
                "initial_features": self.config["z_alpha"]["selected_features"],
                "frozen_catalog": load_json(self.run_root / "configs" / "frozen_feature_catalog.json"),
                "rass_development_evidence": load_json(self.run_root / "configs" / "rass_train_evidence.json"),
                "required_evidence_ref": "configs/rass_train_evidence.json",
            }
        elif agent == "FAMA":
            if model_training_history is None:
                raise ContractError("FAMA_MODEL_TRAINING_HISTORY_REQUIRED")
            assert_validation_safe_value(model_training_history)
            router = yaml.safe_load((self.repo_root / "agents" / "FAMA" / "intervention_space.yaml").read_text(encoding="utf-8"))
            model_name = self.config["z_model"]["model_name"]
            relative = router["model_spaces"].get(model_name)
            if relative is None:
                raise ContractError("FAMA_MODEL_UNSUPPORTED")
            context.update({
                "current_validation_evidence": evidence,
                "model_training_history": model_training_history,
                "complete_model_memory": self.memory.projection("model"),
                "intervention_router": router,
                "resolved_model_space": yaml.safe_load((self.repo_root / "agents" / "FAMA" / relative).read_text(encoding="utf-8")),
                "diff_path_contract": {
                    "train_params": "z_model.train_params.<parameter>",
                    "model_params": "z_model.model_params.<parameter>",
                    "rule": "Every diff path must use the full dotted path for the selected_group.",
                },
            })
        elif agent == "RAPA":
            portfolio_memory = self.memory.projection("portfolio")
            context.update({
                "current_validation_evidence": evidence,
                "complete_portfolio_memory": portfolio_memory,
                "intervention_space": yaml.safe_load((self.repo_root / "agents" / "RAPA" / "intervention_space.yaml").read_text(encoding="utf-8")),
                "directional_search_state": build_rapa_search_state(portfolio_memory, self.config),
            })
        else:
            raise ContractError(f"UNKNOWN_SPECIALIST: {agent}")
        if contract_feedback is not None:
            context["contract_feedback"] = contract_feedback
        assert_validation_safe_value(context)
        return self.client.complete_json(self._prompt(agent), context, self._schema(agent), f"{agent.lower()}_proposal")
