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


def _canonical_config_path(path: str) -> str:
    """Normalize an unambiguous JSON Pointer spelling to the dotted contract spelling."""
    if path.startswith("/") and "~" not in path:
        components = path[1:].split("/")
        if components and all(components):
            return ".".join(components)
    return path


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
                path = _canonical_config_path(change["path"])
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
            candidate_diff.append(copy.deepcopy(change))
        verified.extend(["one_parameter_group", "one_parameter_change", "frozen_fields"])

    else:
        evidence = load_json(run_root / "configs" / "rass_train_evidence.json")
        catalog = load_json(run_root / "configs" / "frozen_feature_catalog.json")
        shortlist_path = run_root / "experiments" / experiment_id / "logs" / "rass_shortlist_evidence.json"
        shortlist_evidence = load_json(shortlist_path) if shortlist_path.is_file() else None
        anchors = accepted_config["z_alpha"]["selected_features"]
        additions = proposal["added_features"]
        selected = proposal["selected_features"]
        if experiment_id != "EXP_001" or accepted_config["z_alpha"].get("alpha_frozen"):
            reasons.append("RASS_NOT_LEGAL_IN_THIS_ROUND")
        if len(anchors) != 6 or proposal["initial_features"] != anchors:
            reasons.append("RASS_ANCHORS_CHANGED")
        expressions = [item["expression"] for item in additions]
        identifiers = [item["id"] for item in additions]
        if len(expressions) != 4 or len(set(expressions)) != 4 or len(set(identifiers)) != 4:
            reasons.append("RASS_FOUR_UNIQUE_ADDITIONS_REQUIRED")
        if selected != anchors + expressions or len(selected) != 10:
            reasons.append("RASS_APPEND_ONLY_SIX_PLUS_FOUR_REQUIRED")
        catalog_pairs = {(item["id"], item["expression"]) for item in catalog["entries"]}
        if any((item["id"], item["expression"]) not in catalog_pairs for item in additions):
            reasons.append("RASS_CATALOG_MEMBERSHIP_FAILED")
        if any(expression in anchors for expression in expressions):
            reasons.append("RASS_DUPLICATES_ANCHOR")
        evidence_ids = {item["id"] for item in evidence["records"]}
        if any(identifier not in evidence_ids for identifier in identifiers):
            reasons.append("RASS_TRAIN_EVIDENCE_MISSING")
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
            "config_hash": evidence["effective_config_hash"],
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
        candidate_diff.append({"op": "append", "path": "z_alpha.selected_features", "values": expressions})
        verified.extend(["train_only_evidence", "append_only", "six_plus_four_equals_ten", "catalog_membership", "frozen_fields"])

    unique_reasons = tuple(dict.fromkeys(reasons))
    if not unique_reasons and agent in {"FAMA", "RAPA"}:
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
        else:
            raise ContractError("UNKNOWN_VALIDATED_DIFF_OPERATION")
    candidate["z_model"]["derived"]["d_feat"] = len(candidate["z_alpha"]["selected_features"])
    refresh_effective_config_hash(candidate)
    return candidate
