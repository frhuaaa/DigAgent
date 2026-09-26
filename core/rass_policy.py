from __future__ import annotations

import json
from pathlib import Path


RASS_MAX_FAILED_ATTEMPTS = 3
RASS_EXECUTED_VERDICTS = {
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "UNCERTAIN",
    "FALSIFIED",
}
RASS_FAILED_VERDICTS = {"UNCERTAIN", "FALSIFIED"}


def load_memory_records(run_root: Path) -> list[dict]:
    path = run_root / "memory" / "experiments.jsonl"
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return sorted(records, key=lambda item: (int(item["round"]), item["experiment_id"]))


def _is_executed_rass_bootstrap(record: dict) -> bool:
    return (
        record.get("layer") == "alpha"
        and record.get("group") == "initial_feature_bootstrap"
        and record.get("verdict") in RASS_EXECUTED_VERDICTS
    )


def addition_signature_from_intervention(intervention: object) -> tuple[str, ...] | None:
    if not isinstance(intervention, dict):
        return None
    additions = intervention.get("added_features")
    if not isinstance(additions, list):
        return None
    expressions = [
        str(item.get("expression"))
        for item in additions
        if isinstance(item, dict) and isinstance(item.get("expression"), str)
    ]
    if len(expressions) != 3 or len(set(expressions)) != 3:
        return None
    return tuple(sorted(expressions))


def proposed_addition_signature(expressions: list[str]) -> tuple[str, ...] | None:
    if len(expressions) != 3 or len(set(expressions)) != 3:
        return None
    return tuple(sorted(expressions))


def rass_executed_attempts(records: list[dict]) -> list[dict]:
    return [record for record in records if _is_executed_rass_bootstrap(record)]


def rass_failed_attempt_count(records: list[dict]) -> int:
    return sum(
        1
        for record in rass_executed_attempts(records)
        if record.get("accepted") is False and record.get("verdict") in RASS_FAILED_VERDICTS
    )


def rass_attempted_addition_signatures(records: list[dict]) -> set[tuple[str, ...]]:
    signatures = {
        addition_signature_from_intervention(record.get("intervention"))
        for record in rass_executed_attempts(records)
    }
    signatures.discard(None)
    return signatures


def rass_attempt_summaries(records: list[dict]) -> list[dict]:
    summaries = []
    for attempt, record in enumerate(rass_executed_attempts(records), start=1):
        intervention = record.get("intervention")
        additions = intervention.get("added_features", []) if isinstance(intervention, dict) else []
        observed = record.get("observed_signature") or {}
        summaries.append({
            "attempt": attempt,
            "experiment_id": record.get("experiment_id"),
            "added_features": [
                item.get("expression")
                for item in additions
                if isinstance(item, dict) and isinstance(item.get("expression"), str)
            ],
            "validation_ic": observed.get("validation_ic"),
            "validation_sharpe": observed.get("validation_sharpe"),
            "sharpe_state": observed.get("sharpe_state"),
            "mechanism_state": observed.get("mechanism_state"),
            "tradeoff_state": observed.get("tradeoff_state"),
            "verdict": record.get("verdict"),
            "accepted": bool(record.get("accepted", False)),
        })
    return summaries
