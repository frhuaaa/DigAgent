from __future__ import annotations

import json
from pathlib import Path

from core.errors import ResumeError
from core.io_utils import atomic_write_json
from core.isolation import assert_validation_safe_value


ROLES = ("cross_layer", "alpha", "model", "portfolio")


class MemoryStore:
    def __init__(self, run_root: Path):
        self.root = run_root / "memory"
        self.path = self.root / "experiments.jsonl"
        self.summary_dir = self.root / "summaries"
        self.root.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    assert_validation_safe_value(item)
                    records.append(item)
        return sorted(records, key=lambda item: (int(item["round"]), item["experiment_id"]))

    def append(self, record: dict) -> None:
        assert_validation_safe_value(record)
        records = self.records()
        existing = [item for item in records if item["experiment_id"] == record["experiment_id"]]
        if existing:
            if existing[0] != record:
                raise ResumeError(f"MEMORY_IDEMPOTENCY_MISMATCH: {record['experiment_id']}")
            return
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
        self.materialize_summaries()

    def projection(self, role: str) -> list[dict]:
        if role not in ROLES:
            raise ValueError(role)
        records = self.records()
        if role == "cross_layer":
            return records
        layer = {"alpha": "alpha", "model": "model", "portfolio": "portfolio"}[role]
        return [
            {
                "experiment_id": item["experiment_id"],
                "round": item["round"],
                "layer_relevant": item.get("layer") == layer,
                "state": item.get("state"),
                "diagnosis": item.get("clem_diagnosis"),
                "layer": item.get("layer"),
                "group": item.get("group"),
                "intervention": item.get("intervention"),
                "expected_signature": item.get("expected_signature"),
                "observed_signature": item.get("observed_signature"),
                "verdict": item.get("verdict"),
                "accepted": item.get("accepted"),
                "acceptance_reason": item.get("acceptance_reason"),
            }
            for item in records
        ]

    def materialize_summaries(self) -> None:
        for role in ROLES:
            atomic_write_json(self.summary_dir / f"{role}.json", {
                "role": role,
                "retrieval": "entire_validation_safe_memory",
                "records": self.projection(role),
                "contains_test_derived_data": False,
            })

