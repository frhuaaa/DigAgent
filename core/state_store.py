from __future__ import annotations

import copy
from pathlib import Path

from core.configuration import refresh_effective_config_hash
from core.errors import ResumeError
from core.io_utils import atomic_write_bytes, atomic_write_json, canonical_json_bytes, load_json, sha256_file


class StateStore:
    def __init__(self, run_root: Path):
        self.run_root = run_root.resolve()
        self.config_dir = self.run_root / "configs"
        self.current_path = self.config_dir / "current.json"
        self.state_path = self.config_dir / "state.json"
        self.transaction_path = self.config_dir / ".promotion_transaction.json"
        self.recover_transaction()

    def current(self) -> dict:
        return load_json(self.current_path)

    def state(self) -> dict:
        return load_json(self.state_path)

    def recover_transaction(self) -> None:
        if not self.transaction_path.exists():
            return
        transaction = load_json(self.transaction_path)
        if transaction.get("phase") == "COMMITTED":
            self.transaction_path.unlink()
            return
        atomic_write_bytes(self.current_path, transaction["parent_config_json"].encode("utf-8"))
        atomic_write_bytes(self.state_path, transaction["parent_state_json"].encode("utf-8"))
        self.transaction_path.unlink()

    def accept_round0(self, config: dict) -> None:
        state = self.state()
        if state["accepted_experiment_id"] == "EXP_000":
            return
        if state["accepted_experiment_id"] is not None or state["next_round"] != 0:
            raise ResumeError("ROUND0_STATE_INVALID")
        state.update({"status": "ADAPTING", "accepted_experiment_id": "EXP_000", "next_round": 1})
        atomic_write_json(self.state_path, state)

    def promote(self, candidate: dict, experiment_id: str, next_round: int, structural_rass: bool = False) -> dict:
        parent_config_bytes = self.current_path.read_bytes()
        parent_state_bytes = self.state_path.read_bytes()
        promoted = copy.deepcopy(candidate)
        if structural_rass:
            promoted["z_alpha"]["alpha_frozen"] = True
            refresh_effective_config_hash(promoted)
        state = self.state()
        state.update({"status": "ADAPTING", "accepted_experiment_id": experiment_id, "next_round": next_round})
        transaction = {
            "phase": "PREPARED",
            "experiment_id": experiment_id,
            "parent_config_json": parent_config_bytes.decode("utf-8"),
            "parent_state_json": parent_state_bytes.decode("utf-8"),
        }
        atomic_write_json(self.transaction_path, transaction)
        try:
            atomic_write_json(self.current_path, promoted)
            atomic_write_json(self.state_path, state)
            transaction["phase"] = "COMMITTED"
            atomic_write_json(self.transaction_path, transaction)
            self.transaction_path.unlink()
        except Exception:
            atomic_write_bytes(self.current_path, parent_config_bytes)
            atomic_write_bytes(self.state_path, parent_state_bytes)
            if self.transaction_path.exists():
                self.transaction_path.unlink()
            raise
        return promoted

    def rollback(self, experiment_id: str, next_round: int) -> None:
        state = self.state()
        state.update({"status": "ADAPTING", "last_rejected_experiment_id": experiment_id, "next_round": next_round})
        atomic_write_json(self.state_path, state)

    def fail(self, reason: str, experiment_id: str | None = None) -> None:
        state = self.state()
        state.update({"status": "FAILED", "failure_reason": reason})
        if experiment_id is not None:
            state["failed_experiment_id"] = experiment_id
        atomic_write_json(self.state_path, state)

    def finalize(self, reason: str) -> Path:
        final_path = self.config_dir / "final_frozen.json"
        current_bytes = self.current_path.read_bytes()
        if final_path.exists():
            if final_path.read_bytes() != current_bytes:
                raise ResumeError("FINAL_FROZEN_CONFIG_MISMATCH")
        else:
            atomic_write_bytes(final_path, current_bytes)
        state = self.state()
        state.update({
            "status": "FINISHED",
            "stop_reason": reason,
            "final_config_sha256": sha256_file(final_path),
        })
        atomic_write_json(self.state_path, state)
        return final_path
