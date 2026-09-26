from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from core.configuration import refresh_effective_config_hash
from core.errors import ContractError
from core.io_utils import canonical_json_bytes, load_json, sha256_bytes
from core.isolation import assert_validation_safe_value
from core.rass_policy import (
    RASS_MAX_FAILED_ATTEMPTS,
    load_memory_records,
    proposed_addition_signature,
    rass_attempted_addition_signatures,
    rass_failed_attempt_count,
)


@dataclass(frozen=True)
class ValidationOutcome:
    status: str
    reason_codes: tuple[str, ...]
    verified: tuple[str, ...]
    candidate_diff: tuple[dict, ...]

    @property
    def passed(self) -> bool:
        return self.status == "PASSED"

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "verified": list(self.verified),
        }


AGENT_LAYER = {"RASS": "alpha", "FAMA": "model", "RAPA": "portfolio"}
RAPA_MAX_ATTEMPTS_PER_PARAMETER = 4
RAPA_INITIAL_PROBE_MAX_ABS_STEP = 0.1


def _rapa_parameter_history(run_root: Path, parameter: str) -> list[dict]:
    memory_path = run_root / "memory" / "experiments.jsonl"
    if not memory_path.is_file():
        return []
    path = f"z_portfolio.{parameter}"
    history = []
    with memory_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("layer") != "portfolio" or item.get("verdict") == "CONTRACT_REJECTED":
                continue
            for change in item.get("intervention") or []:
                if isinstance(change, dict) and _canonical_config_path(str(change.get("path", ""))) == path:
                    if "old_value" not in change or "new_value" not in change:
                        continue
                    history.append({
                        "old_value": float(change["old_value"]),
                        "new_value": float(change["new_value"]),
                        "accepted": bool(item.get("accepted", False)),
                    })
    return history


def _previously_rejected_paths(run_root: Path, parent_id: str, layer: str, group: str) -> set[str]:
    memory_path = run_root / "memory" / "experiments.jsonl"
    if not memory_path.is_file():
        return set()
    rejected: set[str] = set()
    with memory_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if (
                item.get("parent") != parent_id
                or item.get("layer") != layer
                or item.get("group") != group
                or item.get("accepted") is not False
                or item.get("verdict") not in {"FALSIFIED", "UNCERTAIN"}
            ):
                continue
            for change in item.get("intervention") or []:
                if isinstance(change, dict) and isinstance(change.get("path"), str):
                    rejected.add(_canonical_config_path(change["path"]))
    return rejected


def _previously_rejected_rass_refinements(run_root: Path, parent_id: str) -> set[tuple[str, str]]:
    memory_path = run_root / "memory" / "experiments.jsonl"
    if not memory_path.is_file():
        return set()
    rejected: set[tuple[str, str]] = set()
    with memory_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if (
                item.get("parent") != parent_id
                or item.get("layer") != "alpha"
                or item.get("group") != "feature_refinement"
                or item.get("accepted") is not False
                or item.get("verdict") not in {"FALSIFIED", "UNCERTAIN"}
            ):
                continue
            intervention = item.get("intervention") or {}
            removed = intervention.get("removed_features") or []
            added = intervention.get("added_features") or []
            if len(removed) == 1 and len(added) == 1 and isinstance(added[0], dict):
                rejected.add((str(removed[0]), str(added[0].get("expression"))))
    return rejected


def _canonical_config_path(path: str) -> str:
    """Normalize an unambiguous JSON Pointer spelling to the dotted contract spelling."""
    if path.startswith("/") and "~" not in path:
        components = path[1:].split("/")
        if components and all(components):
            return ".".join(components)
    return path


def _canonical_fama_path(path: str, group: str) -> str:
    """Resolve an unambiguous group-local FAMA path to its full config path."""
    canonical = _canonical_config_path(path)
    if canonical and "." not in canonical and "/" not in canonical:
        return f"z_model.{group}.{canonical}"
    return canonical


def _get_path(config: dict, path: str) -> Any:
    current: Any = config
    for component in path.split("."):
        if not isinstance(current, dict) or component not in current:
            raise KeyError(path)
        current = current[component]
    return current


def _set_path(config: dict, path: str, value: Any) -> None:
    components = path.split(".")
    current = config
    for component in components[:-1]:
        current = current[component]
    current[components[-1]] = value


def _schema_errors(schema_path: Path, proposal: dict) -> list[str]:
    schema = load_json(schema_path)
    validator = jsonschema.Draft202012Validator(schema)
    return ["SCHEMA_INVALID"] if list(validator.iter_errors(proposal)) else []


def _value_legal(specification: dict, value: Any) -> bool:
    kind = specification["type"]
    if kind == "boolean":
        return type(value) is bool and ("allowed" not in specification or value in specification["allowed"])
    if kind == "integer":
        if type(value) is not int:
            return False
        if "allowed" in specification:
            return value in specification["allowed"]
        minimum, maximum = specification["minimum"], specification["maximum"]
        if not minimum <= value <= maximum:
            return False
        return (value - minimum) % specification.get("step", 1) == 0
    if kind == "float":
        if type(value) not in {int, float} or type(value) is bool:
            return False
        numeric = float(value)
        if specification.get("allowed_zero") and numeric == 0:
            return True
        minimum = specification.get("minimum_nonzero", specification.get("minimum"))
        maximum = specification["maximum"]
        if not minimum <= numeric <= maximum:
            return False
        step = specification.get("step")
        if step is not None:
            quotient = (numeric - float(minimum)) / float(step)
            return abs(quotient - round(quotient)) <= 1e-9
        return True
    return False


def validate_proposal(
    repo_root: Path,
    run_root: Path,
    accepted_config: dict,
    experiment_id: str,
    clem: dict,
    proposal: dict,
) -> ValidationOutcome:
    agent = clem.get("selected_agent")
    reasons: list[str] = []
    verified: list[str] = []
    candidate_diff: list[dict] = []
    if agent not in AGENT_LAYER:
        return ValidationOutcome("REJECTED", ("CLEM_ROUTE_NOT_INTERVENTION",), (), ())
    if agent == "RASS" and accepted_config["z_alpha"].get("alpha_frozen", False):
        return ValidationOutcome(
            "REJECTED",
            ("RASS_DISABLED_AFTER_ALPHA_RESOLUTION",),
            ("hard_alpha_freeze",),
            (),
        )
    schema_path = repo_root / "agents" / agent / "intervention.schema.json"
    reasons.extend(_schema_errors(schema_path, proposal))
    try:
        assert_validation_safe_value({"clem": clem, "specialist": proposal})
        verified.append("no_test_access")
    except Exception:
        reasons.append("TEST_REFERENCE_FORBIDDEN")
    if clem.get("selected_layer") != AGENT_LAYER[agent] or proposal.get("agent") != agent:
        reasons.append("ROUTE_LAYER_MISMATCH")
    if float(clem.get("confidence", 0.0)) < 0.5:
        reasons.append("CLEM_CONFIDENCE_BELOW_THRESHOLD")
    if reasons:
        return ValidationOutcome("REJECTED", tuple(dict.fromkeys(reasons)), tuple(verified), ())

    if agent == "FAMA":
        router = yaml.safe_load((repo_root / "agents" / "FAMA" / "intervention_space.yaml").read_text(encoding="utf-8"))
        model_name = accepted_config["z_model"]["model_name"]
        relative_space = router["model_spaces"].get(model_name)
        if relative_space is None:
            reasons.append("FAMA_MODEL_UNSUPPORTED")
        else:
            space = yaml.safe_load((repo_root / "agents" / "FAMA" / relative_space).read_text(encoding="utf-8"))
            group = proposal["selected_group"]
            diff = proposal["diff"]
            if len(diff) > int(router["max_changed_parameters_per_round"]):
                reasons.append("FAMA_PARAMETER_CAP_EXCEEDED")
            seen_paths = set()
            for change in diff:
                path = _canonical_fama_path(change["path"], group)
                prefix = f"z_model.{group}."
                if not path.startswith(prefix) or path.count(".") != 2:
                    reasons.append("FAMA_CROSS_GROUP_PATH")
                    continue
                parameter = path.split(".")[-1]
                if parameter not in space.get(group, {}):
                    reasons.append("FAMA_PARAMETER_NOT_IN_MODEL_SPACE")
                    continue
                if path in seen_paths:
                    reasons.append("DUPLICATE_CHANGED_PATH")
                seen_paths.add(path)
                try:
                    old = _get_path(accepted_config, path)
                except KeyError:
                    reasons.append("OLD_PATH_MISSING")
                    continue
                if change["old_value"] != old:
                    reasons.append("OLD_VALUE_PARENT_MISMATCH")
                if change["new_value"] == old:
                    reasons.append("NO_OP_CHANGE")
                if not _value_legal(space[group][parameter], change["new_value"]):
                    reasons.append("FAMA_VALUE_OUTSIDE_SPACE")
                canonical_change = copy.deepcopy(change)
                canonical_change["path"] = path
                candidate_diff.append(canonical_change)
            verified.extend(["model_specific_space", "one_parameter_group", "frozen_fields"])

    elif agent == "RAPA":
        space = yaml.safe_load((repo_root / "agents" / "RAPA" / "intervention_space.yaml").read_text(encoding="utf-8"))
        diff = proposal["diff"]
        if len(diff) != 1:
            reasons.append("RAPA_EXACTLY_ONE_PARAMETER_REQUIRED")
        for change in diff:
            parameter = change["path"].split(".")[-1]
            specification = space["parameters"].get(parameter)
            if specification is None or change["path"] != f"z_portfolio.{parameter}":
                reasons.append("RAPA_PATH_OUTSIDE_SPACE")
                continue
            old = _get_path(accepted_config, change["path"])
            if float(change["old_value"]) != float(old):
                reasons.append("OLD_VALUE_PARENT_MISMATCH")
            if float(change["new_value"]) == float(old):
                reasons.append("NO_OP_CHANGE")
            if not _value_legal(specification, change["new_value"]):
                reasons.append("RAPA_VALUE_OUTSIDE_SPACE")
            history = _rapa_parameter_history(run_root, parameter)
            old_value = float(change["old_value"])
            new_value = float(change["new_value"])
            step = abs(new_value - old_value)
            direction = 1 if new_value > old_value else -1
            if len(history) >= RAPA_MAX_ATTEMPTS_PER_PARAMETER:
                reasons.append("RAPA_PARAMETER_ATTEMPT_CAP_EXCEEDED")
            elif not history:
                if step > RAPA_INITIAL_PROBE_MAX_ABS_STEP + 1e-12:
                    reasons.append("RAPA_INITIAL_PROBE_TOO_LARGE")
            else:
                previous = history[-1]
                previous_direction = 1 if previous["new_value"] > previous["old_value"] else -1
                previous_step = abs(previous["new_value"] - previous["old_value"])
                if previous["accepted"]:
                    if direction != previous_direction:
                        reasons.append("RAPA_ACCEPTED_DIRECTION_MUST_CONTINUE")
                    if step + 1e-12 < previous_step:
                        reasons.append("RAPA_EXPANSION_STEP_MUST_NOT_SHRINK")
                else:
                    if direction == previous_direction:
                        reasons.append("RAPA_REJECTED_DIRECTION_MUST_REVERSE")
                    if step > RAPA_INITIAL_PROBE_MAX_ABS_STEP + 1e-12:
                        reasons.append("RAPA_REVERSE_PROBE_TOO_LARGE")
            if parameter == "turnover_penalty":
                expected_turnover_direction = (
                    "decrease"
                    if float(change["new_value"]) > float(change["old_value"])
                    else "increase"
                )
                declared = {
                    item.get("direction")
                    for item in proposal.get("expected_signature", [])
                    if item.get("metric") == "one_way_turnover_mean"
                }
                if declared != {expected_turnover_direction}:
                    reasons.append("RAPA_TURNOVER_PENALTY_MECHANISM_DIRECTION_MISMATCH")
            candidate_diff.append(copy.deepcopy(change))
        verified.extend(["one_parameter_group", "one_parameter_change", "frozen_fields"])

    else:
        catalog = load_json(run_root / "configs" / "frozen_feature_catalog.json")
        anchors = accepted_config["z_alpha"]["selected_features"]
        additions = proposal["added_features"]
        selected = proposal["selected_features"]
        removed = proposal["removed_features"]
        group = proposal["selected_group"]
        bootstrap = group == "initial_feature_bootstrap"
        if proposal["initial_features"] != anchors:
            reasons.append("RASS_PARENT_FEATURES_MISMATCH")
        memory_records = load_memory_records(run_root)
        prior_rass_failures = rass_failed_attempt_count(memory_records)
        if bootstrap and (
            len(anchors) != 5
            or accepted_config["z_alpha"].get("alpha_frozen")
            or prior_rass_failures >= RASS_MAX_FAILED_ATTEMPTS
        ):
            reasons.append("RASS_BOOTSTRAP_NOT_LEGAL_IN_THIS_ROUND")
        if not bootstrap and (len(anchors) != 8 or not accepted_config["z_alpha"].get("alpha_frozen")):
            reasons.append("RASS_REFINEMENT_REQUIRES_FROZEN_EIGHT_FEATURE_STATE")
        required_additions = 8 - len(anchors) if bootstrap else 1
        expressions = [item["expression"] for item in additions]
        identifiers = [item["id"] for item in additions]
        if len(expressions) != required_additions or len(set(expressions)) != required_additions or len(set(identifiers)) != required_additions:
            reasons.append("RASS_UNIQUE_ADDITIONS_TO_EIGHT_REQUIRED")
        if bootstrap:
            if removed or selected != anchors + expressions or len(selected) != 8:
                reasons.append("RASS_APPEND_ONLY_TO_EIGHT_REQUIRED")
            signature = proposed_addition_signature(expressions)
            if signature is not None and signature in rass_attempted_addition_signatures(memory_records):
                reasons.append("RASS_REPEATED_THREE_FACTOR_SET")
        else:
            if len(removed) != 1 or removed[0] not in anchors[5:]:
                reasons.append("RASS_REFINEMENT_MUST_REMOVE_ONE_NON_ANCHOR")
            else:
                expected_selected = list(anchors)
                expected_selected[anchors.index(removed[0])] = expressions[0]
                if selected != expected_selected or selected[:5] != anchors[:5] or len(set(selected)) != 8:
                    reasons.append("RASS_REFINEMENT_MUST_REPLACE_IN_PLACE")
        catalog_pairs = {(item["id"], item["expression"]) for item in catalog["entries"]}
        if any((item["id"], item["expression"]) not in catalog_pairs for item in additions):
            reasons.append("RASS_CATALOG_MEMBERSHIP_FAILED")
        if any(expression in anchors for expression in expressions):
            reasons.append("RASS_DUPLICATES_ANCHOR")
        evidence = load_json(run_root / "configs" / "rass_train_evidence.json")
        shortlist_path = run_root / "experiments" / experiment_id / "logs" / "rass_shortlist_evidence.json"
        shortlist_evidence = load_json(shortlist_path) if shortlist_path.is_file() else None
        evidence_ids = {item["id"] for item in evidence["records"]}
        if any(identifier not in evidence_ids for identifier in identifiers):
            reasons.append("RASS_DEVELOPMENT_EVIDENCE_MISSING")
        eligible_ids = {item["id"] for item in evidence["records"] if item.get("eligible", False)}
        if any(identifier not in eligible_ids for identifier in identifiers):
            reasons.append("RASS_SELECTED_INELIGIBLE_FACTOR")
        shortlist_hash = None
        if shortlist_evidence is None:
            reasons.append("RASS_SHORTLIST_EVIDENCE_MISSING")
        else:
            stored_hash = shortlist_evidence.get("evidence_hash")
            unhashed = copy.deepcopy(shortlist_evidence)
            unhashed.pop("evidence_hash", None)
            shortlist_hash = sha256_bytes(canonical_json_bytes(unhashed))
            if stored_hash != shortlist_hash:
                reasons.append("RASS_SHORTLIST_EVIDENCE_HASH_MISMATCH")
            shortlist_pairs = {(item["id"], item["expression"]) for item in shortlist_evidence.get("shortlist", [])}
            if any((item["id"], item["expression"]) not in shortlist_pairs for item in additions):
                reasons.append("RASS_SELECTION_OUTSIDE_SHORTLIST")
        expected_provenance = {
            "data_hash": evidence["input_data_hash"],
            "catalog_hash": evidence["catalog_hash"],
            "label_hash": evidence["label_hash"],
            "evidence_context_hash": evidence["rass_evidence_context_hash"],
            "evidence_code_hash": evidence["evidence_code_hash"],
            "shortlist_evidence_hash": shortlist_hash,
        }
        if proposal["provenance"] != expected_provenance:
            reasons.append("RASS_PROVENANCE_MISMATCH")
        allowed_refs = {
            "configs/rass_train_evidence.json",
            "logs/rass_shortlist_evidence.json",
            f"evidence_hash: {evidence['evidence_hash']}",
            f"evidence_hash: {shortlist_hash}",
        }
        required_refs = {"configs/rass_train_evidence.json", "logs/rass_shortlist_evidence.json", f"evidence_hash: {shortlist_hash}"}
        if not required_refs.issubset(set(proposal["evidence_refs"])) or any(reference not in allowed_refs for reference in proposal["evidence_refs"]):
            reasons.append("RASS_EVIDENCE_REFERENCE_INVALID")
        if bootstrap:
            candidate_diff.append({"op": "append", "path": "z_alpha.selected_features", "values": expressions})
            verified.extend([
                "append_only",
                "initial_plus_additions_equals_eight",
                "three_factor_set_not_previously_executed",
                "rass_failed_attempt_limit",
            ])
        elif len(removed) == 1 and len(expressions) == 1 and removed[0] in anchors:
            candidate_diff.append({
                "op": "replace_feature",
                "path": "z_alpha.selected_features",
                "index": anchors.index(removed[0]),
                "old_value": removed[0],
                "new_value": expressions[0],
            })
            verified.extend(["one_feature_refinement", "anchor_prefix_preserved"])
        verified.extend(["shared_year_scoped_rass_evidence", "catalog_membership", "frozen_fields"])

    unique_reasons = tuple(dict.fromkeys(reasons))
    if not unique_reasons and agent == "FAMA":
        state_path = run_root / "configs" / "state.json"
        if state_path.is_file():
            parent_id = load_json(state_path).get("accepted_experiment_id")
            if parent_id:
                rejected_paths = _previously_rejected_paths(
                    run_root,
                    parent_id,
                    AGENT_LAYER[agent],
                    proposal["selected_group"],
                )
                current_paths = {
                    _canonical_config_path(change["path"])
                    for change in candidate_diff
                    if change.get("op") == "replace"
                }
                if rejected_paths & current_paths:
                    unique_reasons = ("REPEATED_REJECTED_PARAMETER_SAME_PARENT",)
    if not unique_reasons and agent == "RAPA":
        change = candidate_diff[0]
        parameter = change["path"].split(".")[-1]
        tested_values = {
            float(item["new_value"])
            for item in _rapa_parameter_history(run_root, parameter)
        }
        if float(change["new_value"]) in tested_values:
            unique_reasons = ("RAPA_REPEATED_TESTED_VALUE",)
    if not unique_reasons and agent == "RASS" and proposal.get("selected_group") == "feature_refinement":
        state_path = run_root / "configs" / "state.json"
        if state_path.is_file():
            parent_id = load_json(state_path).get("accepted_experiment_id")
            removed = proposal.get("removed_features") or []
            added = proposal.get("added_features") or []
            signature = (
                str(removed[0]),
                str(added[0].get("expression")),
            ) if len(removed) == 1 and len(added) == 1 else None
            if parent_id and signature in _previously_rejected_rass_refinements(run_root, parent_id):
                unique_reasons = ("REPEATED_REJECTED_RASS_REFINEMENT_SAME_PARENT",)
    status = "REJECTED" if unique_reasons else "PASSED"
    return ValidationOutcome(status, unique_reasons, tuple(dict.fromkeys(verified)), tuple(candidate_diff if not unique_reasons else ()))


def apply_validated_proposal(accepted_config: dict, outcome: ValidationOutcome, agent: str) -> dict:
    if not outcome.passed:
        raise ContractError("CANNOT_APPLY_REJECTED_PROPOSAL")
    candidate = copy.deepcopy(accepted_config)
    for change in outcome.candidate_diff:
        if change["op"] == "replace":
            _set_path(candidate, change["path"], change["new_value"])
        elif change["op"] == "append" and change["path"] == "z_alpha.selected_features":
            candidate["z_alpha"]["selected_features"].extend(change["values"])
        elif change["op"] == "replace_feature" and change["path"] == "z_alpha.selected_features":
            index = int(change["index"])
            if candidate["z_alpha"]["selected_features"][index] != change["old_value"]:
                raise ContractError("RASS_REFINEMENT_PARENT_MISMATCH")
            candidate["z_alpha"]["selected_features"][index] = change["new_value"]
        else:
            raise ContractError("UNKNOWN_VALIDATED_DIFF_OPERATION")
    candidate["z_model"]["derived"]["d_feat"] = len(candidate["z_alpha"]["selected_features"])
    refresh_effective_config_hash(candidate)
    return candidate
