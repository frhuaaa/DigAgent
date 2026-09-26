from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from core.agent_runtime import (
    AgentRuntime,
    build_fama_training_history,
    build_validation_evidence,
    validate_rass_shortlist_payload,
)
from core.configuration import materialize_initial_config
from core.contract_validator import apply_validated_proposal, validate_proposal
from core.errors import ContractError, DiagAgentError, IsolationError, OfflineEvidenceRequired, ResumeError
from core.evidence_builder import build_rass_shortlist_evidence
from core.executor import ExecutionOutcome, execute_validation
from core.io_utils import atomic_write_json, load_json
from core.isolation import assert_validation_safe_value, install_adaptive_test_path_guard
from core.researcher_boundary import dispatch_researcher_outputs_audit
from core.rass_policy import (
    RASS_MAX_FAILED_ATTEMPTS,
    rass_executed_attempts,
    rass_failed_attempt_count,
)
from core.state_store import StateStore
from core.validation_gate import evaluate_gate
from evaluation.result_writer import print_portfolio_report
from memory.memory_store import MemoryStore


ENVELOPE_KEYS = {"experiment_id", "round", "accepted_parent_experiment_id", "candidate_diff"}


def _experiment_config(config: dict, experiment_id: str, round_number: int, parent: str | None, candidate_diff: list[dict]) -> dict:
    envelope = copy.deepcopy(config)
    envelope.update({
        "experiment_id": experiment_id,
        "round": round_number,
        "accepted_parent_experiment_id": parent,
        "candidate_diff": candidate_diff,
    })
    return envelope


def _core_config(envelope: dict) -> dict:
    return {key: copy.deepcopy(value) for key, value in envelope.items() if key not in ENVELOPE_KEYS}


def _load_outcome(repo_root: Path, experiment_dir: Path) -> ExecutionOutcome:
    marker = load_json(experiment_dir / "logs" / "execution_complete.json")
    return ExecutionOutcome(
        result=load_json(experiment_dir / "result.json"),
        diagnostics=pd.read_csv(experiment_dir / "artifacts" / "portfolio_daily_diagnostics.csv", parse_dates=["signal_date", "trade_date", "return_date"]),
        signal_daily=pd.read_csv(experiment_dir / "artifacts" / "signal_daily_metrics.csv", index_col=0, parse_dates=True),
        checkpoint=repo_root / marker["selected_checkpoint"],
        manifest=marker,
    )


def _memory_baseline(result: dict) -> dict:
    return {
        "experiment_id": "EXP_000",
        "round": 0,
        "parent": None,
        "state": "initial_feature_baseline",
        "clem_diagnosis": None,
        "layer": None,
        "group": None,
        "intervention": None,
        "expected_signature": None,
        "observed_signature": {"validation_ic": result["ic"], "validation_sharpe": result["sharpe_ratio"]},
        "verdict": "BASELINE",
        "accepted": True,
        "test_derived": False,
    }


def _memory_candidate(
    experiment_id: str,
    round_number: int,
    parent: str,
    clem: dict,
    proposal: dict,
    gate: dict,
    result: dict,
    candidate_config: dict,
) -> dict:
    accepted = bool(gate["promotion"])
    return {
        "experiment_id": experiment_id,
        "round": round_number,
        "parent": parent,
        "state": f"accepted_parent={parent}",
        "clem_diagnosis": clem["failure_summary"],
        "layer": clem["selected_layer"],
        "group": proposal["selected_group"],
        "intervention": proposal.get("diff", {
            "removed_features": proposal.get("removed_features", []),
            "added_features": proposal.get("added_features", []),
        }),
        "expected_signature": proposal["expected_signature"],
        "observed_signature": {
            "validation_ic": result["ic"],
            "validation_sharpe": result["sharpe_ratio"],
            "sharpe_state": gate["sharpe"]["state"],
            "mechanism_state": gate["mechanism_state"],
            "tradeoff_state": gate["tradeoff_state"],
            "partial_promotion_stability": gate.get("partial_promotion_stability"),
            "fama_ic_override_applied": bool(gate.get("fama_ic_override", {}).get("applied", False)),
        },
        "verdict": gate["gate_verdict"],
        "accepted": accepted,
        "acceptance_reason": gate["promotion_reason"],
        "alpha_frozen": bool(
            gate.get("alpha_frozen_after_promotion", False)
            or candidate_config["z_alpha"].get("alpha_frozen", False)
        ),
        "test_derived": False,
    }


def _memory_contract_rejection(
    experiment_id: str,
    round_number: int,
    parent: str,
    clem: dict,
    proposal: dict,
    reason_codes: list[str],
    accepted_config: dict,
) -> dict:
    intervention = proposal.get("diff", {
        "removed_features": proposal.get("removed_features", []),
        "added_features": proposal.get("added_features", []),
    })
    expected_signature = proposal.get("expected_signature")
    try:
        assert_validation_safe_value({
            "intervention": intervention,
            "expected_signature": expected_signature,
        })
    except IsolationError:
        intervention = {"redacted": True, "reason": "validation_isolation"}
        expected_signature = None
    return {
        "experiment_id": experiment_id,
        "round": round_number,
        "parent": parent,
        "state": f"accepted_parent={parent}",
        "clem_diagnosis": clem["failure_summary"],
        "layer": clem["selected_layer"],
        "group": proposal.get("selected_group"),
        "intervention": intervention,
        "expected_signature": expected_signature,
        "observed_signature": {"contract_reason_codes": list(reason_codes)},
        "verdict": "CONTRACT_REJECTED",
        "accepted": False,
        "acceptance_reason": "contract_rejected_after_one_correction_attempt",
        "alpha_frozen": bool(accepted_config["z_alpha"].get("alpha_frozen", False)),
        "test_derived": False,
    }


class Orchestrator:
    def __init__(self, repo_root: Path, submit_path: Path):
        self.repo_root = repo_root.resolve()
        self.initial_config, self.run_root = materialize_initial_config(self.repo_root, submit_path)
        install_adaptive_test_path_guard(self.run_root)
        self.store = StateStore(self.run_root)
        self.memory = MemoryStore(self.run_root)

    @property
    def experiments_dir(self) -> Path:
        return self.run_root / "experiments"

    def _finish(self, reason: str) -> Path:
        final_path = self.store.finalize(reason)
        dispatch_researcher_outputs_audit(self.repo_root, self.run_root)
        return final_path

    def prepare_round0(self) -> str:
        """Materialize the immutable Round-0 input without executing Qlib."""
        experiment_dir = self.experiments_dir / "EXP_000"
        state = self.store.state()
        if state.get("accepted_experiment_id") is not None:
            if not (experiment_dir / "logs" / "execution_complete.json").is_file():
                raise ResumeError("ROUND0_ACCEPTED_BUT_ARTIFACTS_INCOMPLETE")
            return "EXP_000"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config_path = experiment_dir / "config.json"
        if not config_path.exists():
            atomic_write_json(config_path, _experiment_config(self.initial_config, "EXP_000", 0, None, []))
        elif _core_config(load_json(config_path)) != self.initial_config:
            raise ResumeError("ROUND0_CONFIG_MISMATCH")
        if (experiment_dir / "decision.json").exists():
            raise ContractError("ROUND0_MUST_NOT_HAVE_DECISION")
        return "EXP_000"

    def finalize_round0(self) -> ExecutionOutcome:
        """Commit an already executed Round 0 to accepted state."""
        experiment_dir = self.experiments_dir / "EXP_000"
        if not (experiment_dir / "logs" / "execution_complete.json").is_file():
            raise ResumeError("ROUND0_EXECUTION_NOT_COMPLETE")
        outcome = _load_outcome(self.repo_root, experiment_dir)
        self.memory.append(_memory_baseline(outcome.result))
        self.store.accept_round0(self.initial_config)
        return outcome

    def run_round0(self) -> ExecutionOutcome:
        self.prepare_round0()
        outcome = self.execute_prepared_round("EXP_000")
        self.finalize_round0()
        return outcome

    def _decision(self, runtime: AgentRuntime, round_number: int, parent_id: str, evidence: dict, experiment_dir: Path) -> dict:
        path = experiment_dir / "logs" / "clem_response.json"
        if path.exists():
            decision = load_json(path)
            runtime.validate_clem(decision, round_number, evidence)
            return decision
        decision = runtime.clem(round_number, parent_id, evidence)
        atomic_write_json(path, decision)
        return decision

    def _proposal(
        self,
        runtime: AgentRuntime,
        agent: str,
        clem: dict,
        evidence: dict,
        experiment_dir: Path,
        attempt: int = 1,
        contract_feedback: dict | None = None,
        defer_rass_evidence: bool = False,
    ) -> dict:
        filename = "specialist_response.json" if attempt == 1 else f"specialist_response_retry_{attempt}.json"
        path = experiment_dir / "logs" / filename
        if path.exists():
            return load_json(path)
        if agent == "RASS":
            shortlist_path = experiment_dir / "logs" / "rass_shortlist_response.json"
            if shortlist_path.exists():
                shortlist = load_json(shortlist_path)
                runtime.validate_rass_shortlist(shortlist)
            else:
                shortlist = runtime.rass_shortlist(clem)
                atomic_write_json(shortlist_path, shortlist)
            joint_path = experiment_dir / "logs" / "rass_shortlist_evidence.json"
            if joint_path.exists():
                joint_evidence = load_json(joint_path)
                if joint_evidence.get("base_evidence_hash") != shortlist["evidence_hash"] or joint_evidence.get("shortlist") != shortlist["shortlist"]:
                    raise ResumeError("RASS_SHORTLIST_EVIDENCE_RESUME_MISMATCH")
            else:
                if defer_rass_evidence:
                    raise OfflineEvidenceRequired("RASS_SHORTLIST_EVIDENCE_REQUIRED")
                joint_evidence = build_rass_shortlist_evidence(
                    self.repo_root,
                    runtime.config,
                    shortlist["shortlist"],
                    shortlist["evidence_hash"],
                )
                atomic_write_json(joint_path, joint_evidence)
            proposal = runtime.rass_final(clem, shortlist, joint_evidence, contract_feedback=contract_feedback)
        else:
            model_training_history = None
            if agent == "FAMA":
                model_training_history = build_fama_training_history(
                    self.experiments_dir / evidence["experiment_id"]
                )
            proposal = runtime.specialist(
                agent,
                clem,
                evidence,
                contract_feedback=contract_feedback,
                model_training_history=model_training_history,
            )
        atomic_write_json(path, proposal)
        return proposal

    def prepare_adaptive_round(self, round_number: int, *, defer_rass_evidence: bool = False) -> str | None:
        """Run validation-only agent reasoning and persist a legal candidate.

        Returns the experiment id when GPU execution is required or ``None``
        after a contract rejection. Adaptive routing continues until budget exhaustion.
        """
        state = self.store.state()
        if state.get("status") != "ADAPTING":
            raise ResumeError(f"ADAPTIVE_ROUND_STATE_INVALID: {state.get('status')}")
        if round_number != int(state.get("next_round", -1)):
            raise ResumeError("ADAPTIVE_ROUND_NUMBER_DOES_NOT_MATCH_STATE")
        if not 1 <= round_number <= int(self.initial_config["task"]["trials"]):
            raise ContractError("ADAPTIVE_ROUND_OUTSIDE_BUDGET")
        parent_id = state["accepted_experiment_id"]
        if not parent_id:
            raise ContractError("ADAPTIVE_ROUND_WITHOUT_ACCEPTED_PARENT")
        accepted = self.store.current()
        experiment_id = f"EXP_{round_number:03d}"
        experiment_dir = self.experiments_dir / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        (experiment_dir / "logs").mkdir(parents=True, exist_ok=True)
        parent_dir = self.experiments_dir / parent_id
        evidence = build_validation_evidence(parent_dir)
        runtime = AgentRuntime(self.repo_root, self.run_root, accepted)
        clem = self._decision(runtime, round_number, parent_id, evidence, experiment_dir)
        proposal = self._proposal(
            runtime,
            clem["selected_agent"],
            clem,
            evidence,
            experiment_dir,
            defer_rass_evidence=defer_rass_evidence,
        )
        agent = clem["selected_agent"]
        contract = validate_proposal(self.repo_root, self.run_root, accepted, experiment_id, clem, proposal)
        contract_attempts = [{"attempt": 1, "proposal": proposal, "contract": contract.as_dict()}]
        if not contract.passed:
            feedback = {
                "previous_reason_codes": list(contract.reason_codes),
                "previous_proposal": proposal,
                "instruction": "Correct only the contract violations while preserving the same CLEM diagnosis and one coherent hypothesis.",
                "attempt": 2,
                "maximum_attempts": 2,
            }
            proposal = self._proposal(
                runtime,
                agent,
                clem,
                evidence,
                experiment_dir,
                attempt=2,
                contract_feedback=feedback,
                defer_rass_evidence=defer_rass_evidence,
            )
            contract = validate_proposal(self.repo_root, self.run_root, accepted, experiment_id, clem, proposal)
            contract_attempts.append({"attempt": 2, "proposal": proposal, "contract": contract.as_dict()})
        decision = {
            "experiment_id": experiment_id,
            "round": round_number,
            "clem": clem,
            "specialist": proposal,
            "contract": contract.as_dict(),
            "contract_attempts": contract_attempts,
        }
        assert_validation_safe_value(decision)
        atomic_write_json(experiment_dir / "decision.json", decision)
        if not contract.passed:
            atomic_write_json(experiment_dir / "logs" / "invalid_proposal.json", {"reason_codes": list(contract.reason_codes), "contains_test_derived_data": False})
            self.memory.append(_memory_contract_rejection(
                experiment_id,
                round_number,
                parent_id,
                clem,
                proposal,
                list(contract.reason_codes),
                accepted,
            ))
            self.store.rollback(experiment_id, round_number + 1)
            return None

        candidate = apply_validated_proposal(accepted, contract, agent)
        config_path = experiment_dir / "config.json"
        if config_path.exists():
            persisted = _core_config(load_json(config_path))
            if persisted != candidate:
                raise ContractError("RESUME_CANDIDATE_CONFIG_MISMATCH")
        else:
            atomic_write_json(config_path, _experiment_config(candidate, experiment_id, round_number, parent_id, list(contract.candidate_diff)))
        return experiment_id

    def execute_rass_evidence_stage(self, experiment_id: str) -> Path:
        """Build RASS joint shortlist evidence without constructing an Agent client."""
        experiment_dir = self.experiments_dir / experiment_id
        logs_dir = experiment_dir / "logs"
        output_path = logs_dir / "rass_shortlist_evidence.json"
        if output_path.is_file():
            return output_path
        state = self.store.state()
        round_number = int(experiment_id.removeprefix("EXP_"))
        if state.get("status") != "ADAPTING" or int(state.get("next_round", -1)) != round_number:
            raise ResumeError("RASS_EVIDENCE_STATE_MISMATCH")
        clem = load_json(logs_dir / "clem_response.json")
        if clem.get("selected_agent") != "RASS":
            raise ContractError("RASS_EVIDENCE_WITHOUT_RASS_SELECTION")
        shortlist = load_json(logs_dir / "rass_shortlist_response.json")
        accepted = self.store.current()
        validate_rass_shortlist_payload(self.repo_root, self.run_root, accepted, shortlist)
        joint_evidence = build_rass_shortlist_evidence(
            self.repo_root,
            accepted,
            shortlist["shortlist"],
            shortlist["evidence_hash"],
        )
        atomic_write_json(output_path, joint_evidence)
        return output_path

    def execute_prepared_round(self, experiment_id: str) -> ExecutionOutcome:
        """Execute one persisted experiment without constructing an LLM client.

        This is the only orchestration entry point used on a Slurm compute node.
        It deterministically revalidates an adaptive decision before execution.
        """
        experiment_dir = self.experiments_dir / experiment_id
        config_path = experiment_dir / "config.json"
        if not config_path.is_file():
            raise ResumeError(f"PREPARED_CONFIG_MISSING: {experiment_id}")
        envelope = load_json(config_path)
        if envelope.get("experiment_id") != experiment_id:
            raise ResumeError("PREPARED_EXPERIMENT_ID_MISMATCH")
        if (experiment_dir / "logs" / "execution_complete.json").is_file():
            return _load_outcome(self.repo_root, experiment_dir)
        candidate = _core_config(envelope)
        round_number = int(envelope.get("round", -1))
        parent_id = envelope.get("accepted_parent_experiment_id")
        parent_dir = None
        agent = None
        if experiment_id == "EXP_000":
            if round_number != 0 or parent_id is not None or envelope.get("candidate_diff") != []:
                raise ContractError("ROUND0_ENVELOPE_INVALID")
            if candidate != self.initial_config:
                raise ContractError("ROUND0_CONFIG_MISMATCH")
            if (experiment_dir / "decision.json").exists():
                raise ContractError("ROUND0_MUST_NOT_HAVE_DECISION")
        else:
            state = self.store.state()
            if state.get("status") != "ADAPTING" or int(state.get("next_round", -1)) != round_number:
                raise ResumeError("PREPARED_ROUND_STATE_MISMATCH")
            if state.get("accepted_experiment_id") != parent_id:
                raise ResumeError("PREPARED_PARENT_STATE_MISMATCH")
            decision_path = experiment_dir / "decision.json"
            if not decision_path.is_file():
                raise ResumeError("PREPARED_DECISION_MISSING")
            decision = load_json(decision_path)
            clem = decision["clem"]
            proposal = decision["specialist"]
            agent = clem["selected_agent"]
            contract = validate_proposal(
                self.repo_root,
                self.run_root,
                self.store.current(),
                experiment_id,
                clem,
                proposal,
            )
            if not contract.passed:
                raise ContractError(f"PREPARED_CONTRACT_REVALIDATION_FAILED: {contract.reason_codes}")
            expected_candidate = apply_validated_proposal(self.store.current(), contract, agent)
            if expected_candidate != candidate or list(contract.candidate_diff) != envelope.get("candidate_diff"):
                raise ResumeError("PREPARED_CANDIDATE_CONFIG_MISMATCH")
            parent_dir = self.experiments_dir / parent_id
        outcome = execute_validation(self.repo_root, candidate, experiment_dir, agent, parent_dir)
        print_portfolio_report(experiment_id, outcome.result)
        return outcome

    def finalize_adaptive_round(self, round_number: int) -> None:
        """Evaluate validation output and atomically promote or roll back."""
        experiment_id = f"EXP_{round_number:03d}"
        state = self.store.state()
        if int(state.get("next_round", -1)) > round_number:
            return
        if state.get("status") != "ADAPTING" or int(state.get("next_round", -1)) != round_number:
            raise ResumeError("FINALIZE_ROUND_STATE_MISMATCH")
        experiment_dir = self.experiments_dir / experiment_id
        if not (experiment_dir / "logs" / "execution_complete.json").is_file():
            raise ResumeError("ADAPTIVE_EXECUTION_NOT_COMPLETE")
        envelope = load_json(experiment_dir / "config.json")
        candidate = _core_config(envelope)
        parent_id = envelope.get("accepted_parent_experiment_id")
        if state.get("accepted_experiment_id") != parent_id:
            raise ResumeError("FINALIZE_PARENT_STATE_MISMATCH")
        parent_dir = self.experiments_dir / parent_id
        decision = load_json(experiment_dir / "decision.json")
        clem = decision["clem"]
        proposal = decision["specialist"]
        agent = clem["selected_agent"]
        outcome = _load_outcome(self.repo_root, experiment_dir)
        parent_outcome = _load_outcome(self.repo_root, parent_dir)
        gate_path = experiment_dir / "validation_gate.json"
        is_rass_bootstrap = agent == "RASS" and proposal["selected_group"] == "initial_feature_bootstrap"
        if gate_path.exists():
            gate = load_json(gate_path)
        else:
            gate = evaluate_gate(
                parent_outcome.diagnostics,
                outcome.diagnostics,
                parent_outcome.signal_daily,
                outcome.signal_daily,
                proposal["expected_signature"],
                freeze_alpha_on_promotion=is_rass_bootstrap,
                fama_ic_override=agent == "FAMA",
                parent_ensemble_diagnostics=load_json(parent_dir / "artifacts" / "ensemble_validation_diagnostics.json"),
                candidate_ensemble_diagnostics=load_json(experiment_dir / "artifacts" / "ensemble_validation_diagnostics.json"),
            )
            gate.update({"experiment_id": experiment_id, "accepted_parent_experiment_id": parent_id})
            atomic_write_json(gate_path, gate)
        prior_records = [
            record
            for record in self.memory.records()
            if record.get("experiment_id") != experiment_id
        ]
        memory_record = _memory_candidate(
            experiment_id,
            round_number,
            parent_id,
            clem,
            proposal,
            gate,
            outcome.result,
            candidate,
        )
        fallback_to_five = False
        if is_rass_bootstrap:
            attempt_number = len(rass_executed_attempts(prior_records)) + 1
            failed_attempts_after_round = rass_failed_attempt_count(prior_records) + (0 if gate["promotion"] else 1)
            fallback_to_five = bool(
                not gate["promotion"]
                and failed_attempts_after_round >= RASS_MAX_FAILED_ATTEMPTS
            )
            memory_record.update({
                "rass_attempt_number": attempt_number,
                "rass_failed_attempts_after_round": failed_attempts_after_round,
                "rass_fallback_to_five_features": fallback_to_five,
                "alpha_frozen": bool(gate["promotion"] or fallback_to_five),
            })
        self.memory.append(memory_record)
        state = self.store.state()
        if state.get("accepted_experiment_id") != experiment_id and state.get("next_round", 0) <= round_number:
            if gate["promotion"]:
                self.store.promote(
                    candidate,
                    experiment_id,
                    round_number + 1,
                    freeze_alpha_on_promotion=is_rass_bootstrap,
                )
            elif fallback_to_five:
                self.store.freeze_five_factor_alpha_after_rass_failures(
                    experiment_id,
                    round_number + 1,
                    RASS_MAX_FAILED_ATTEMPTS,
                )
            else:
                self.store.rollback(experiment_id, round_number + 1)

    def run_adaptive_round(self, round_number: int) -> str | None:
        prepared = self.prepare_adaptive_round(round_number)
        if prepared is None:
            return prepared
        self.execute_prepared_round(prepared)
        self.finalize_adaptive_round(round_number)
        return None

    def run(self) -> Path:
        state = self.store.state()
        if state["status"] == "FINISHED":
            dispatch_researcher_outputs_audit(self.repo_root, self.run_root)
            return self.run_root / "configs" / "final_frozen.json"
        if state["status"] == "FAILED":
            raise ContractError(f"TRAJECTORY_ALREADY_FAILED: {state.get('failure_reason')}")
        if state["accepted_experiment_id"] is None:
            self.run_round0()
        while True:
            state = self.store.state()
            round_number = int(state["next_round"])
            if round_number > int(self.initial_config["task"]["trials"]):
                return self._finish("ADAPTIVE_ROUND_BUDGET_EXHAUSTED")
            self.run_adaptive_round(round_number)


def run_submission(repo_root: Path, submit_path: Path) -> Path:
    return Orchestrator(repo_root, submit_path).run()
