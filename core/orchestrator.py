from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from core.agent_runtime import AgentRuntime, build_fama_training_history, build_validation_evidence
from core.configuration import materialize_initial_config
from core.contract_validator import apply_validated_proposal, validate_proposal
from core.errors import ContractError, DiagAgentError, IsolationError, ResumeError
from core.evidence_builder import build_rass_shortlist_evidence
from core.executor import ExecutionOutcome, execute_validation
from core.io_utils import atomic_write_json, load_json
from core.isolation import assert_validation_safe_value, install_adaptive_test_path_guard
from core.researcher_boundary import ResearcherBoundary, dispatch_researcher_outputs_audit
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
        "state": "six_feature_baseline",
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
        "intervention": proposal.get("diff", proposal.get("added_features")),
        "expected_signature": proposal["expected_signature"],
        "observed_signature": {
            "validation_ic": result["ic"],
            "validation_sharpe": result["sharpe_ratio"],
            "sharpe_state": gate["sharpe"]["state"],
            "mechanism_state": gate["mechanism_state"],
            "tradeoff_state": gate["tradeoff_state"],
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
    intervention = proposal.get("diff", proposal.get("added_features"))
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

    def run_round0(self) -> ExecutionOutcome:
        experiment_dir = self.experiments_dir / "EXP_000"
        state = self.store.state()
        if state.get("accepted_experiment_id") is not None:
            if not (experiment_dir / "logs" / "execution_complete.json").is_file():
                raise ResumeError("ROUND0_ACCEPTED_BUT_ARTIFACTS_INCOMPLETE")
            return _load_outcome(self.repo_root, experiment_dir)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config_path = experiment_dir / "config.json"
        if not config_path.exists():
            atomic_write_json(config_path, _experiment_config(self.initial_config, "EXP_000", 0, None, []))
        if (experiment_dir / "decision.json").exists():
            raise ContractError("ROUND0_MUST_NOT_HAVE_DECISION")
        outcome = execute_validation(self.repo_root, self.initial_config, experiment_dir, None, None)
        print_portfolio_report("EXP_000", outcome.result)
        self.memory.append(_memory_baseline(outcome.result))
        self.store.accept_round0(self.initial_config)
        boundary = ResearcherBoundary(self.repo_root, config_path, experiment_dir / "logs" / "researcher_dispatch.jsonl")
        boundary.full(outcome.checkpoint)
        return outcome

    def _decision(self, runtime: AgentRuntime, round_number: int, parent_id: str, evidence: dict, experiment_dir: Path) -> dict:
        path = experiment_dir / "logs" / "clem_response.json"
        if path.exists():
            decision = load_json(path)
            runtime.validate_clem(decision, round_number)
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

    def run_adaptive_round(self, round_number: int) -> str | None:
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
        if clem["selected_agent"] == "NO_INTERVENTION":
            decision = {"experiment_id": experiment_id, "round": round_number, "clem": clem, "specialist": None, "contract": {"status": "NOT_APPLICABLE", "reason_codes": [], "verified": ["no_intervention"]}}
            atomic_write_json(experiment_dir / "decision.json", decision)
            self.store.finalize("NO_INTERVENTION")
            return "NO_INTERVENTION"

        agent = clem["selected_agent"]
        proposal = self._proposal(runtime, agent, clem, evidence, experiment_dir)
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
            if round_number == 1 and agent == "RASS":
                self.store.fail("MANDATORY_RASS_PROPOSAL_INVALID", experiment_id)
                raise ContractError(f"MANDATORY_RASS_PROPOSAL_INVALID: {contract.reason_codes}")
            return None

        candidate = apply_validated_proposal(accepted, contract, agent)
        config_path = experiment_dir / "config.json"
        if config_path.exists():
            persisted = _core_config(load_json(config_path))
            if persisted != candidate:
                raise ContractError("RESUME_CANDIDATE_CONFIG_MISMATCH")
        else:
            atomic_write_json(config_path, _experiment_config(candidate, experiment_id, round_number, parent_id, list(contract.candidate_diff)))
        outcome = execute_validation(self.repo_root, candidate, experiment_dir, agent, parent_dir)
        print_portfolio_report(experiment_id, outcome.result)
        parent_outcome = _load_outcome(self.repo_root, parent_dir)
        gate_path = experiment_dir / "validation_gate.json"
        structural = round_number == 1 and agent == "RASS" and proposal["selected_group"] == "initial_feature_bootstrap"
        if gate_path.exists():
            gate = load_json(gate_path)
        else:
            gate = evaluate_gate(
                parent_outcome.diagnostics,
                outcome.diagnostics,
                parent_outcome.signal_daily,
                outcome.signal_daily,
                proposal["expected_signature"],
                structural_rass_promotion=structural,
            )
            gate.update({"experiment_id": experiment_id, "accepted_parent_experiment_id": parent_id})
            atomic_write_json(gate_path, gate)
        self.memory.append(_memory_candidate(
            experiment_id,
            round_number,
            parent_id,
            clem,
            proposal,
            gate,
            outcome.result,
            candidate,
        ))
        state = self.store.state()
        if state.get("accepted_experiment_id") != experiment_id and state.get("next_round", 0) <= round_number:
            if gate["promotion"]:
                self.store.promote(candidate, experiment_id, round_number + 1, structural_rass=structural)
            else:
                self.store.rollback(experiment_id, round_number + 1)
        boundary = ResearcherBoundary(self.repo_root, config_path, experiment_dir / "logs" / "researcher_dispatch.jsonl")
        boundary.full(outcome.checkpoint, source_experiment_dir=parent_dir if agent == "RAPA" else None)
        return None

    def run(self) -> Path:
        state = self.store.state()
        if state["status"] == "FINISHED":
            dispatch_researcher_outputs_audit(self.repo_root, self.run_root)
            return self.run_root / "configs" / "final_frozen.json"
        if state["status"] == "FAILED":
            if state.get("failure_reason") == "MANDATORY_RASS_PROPOSAL_INVALID" and state.get("failed_experiment_id") == "EXP_001":
                failed_dir = self.experiments_dir / "EXP_001"
                decision = load_json(failed_dir / "decision.json")
                revalidated = validate_proposal(
                    self.repo_root,
                    self.run_root,
                    self.store.current(),
                    "EXP_001",
                    decision["clem"],
                    decision["specialist"],
                )
                if revalidated.passed:
                    for key in ("failure_reason", "failed_experiment_id", "last_rejected_experiment_id"):
                        state.pop(key, None)
                    state.update({"status": "ADAPTING", "next_round": 1})
                    atomic_write_json(self.run_root / "configs" / "state.json", state)
                else:
                    raise ContractError(f"TRAJECTORY_ALREADY_FAILED: {state.get('failure_reason')}")
            else:
                raise ContractError(f"TRAJECTORY_ALREADY_FAILED: {state.get('failure_reason')}")
        if state["accepted_experiment_id"] is None:
            self.run_round0()
        while True:
            state = self.store.state()
            round_number = int(state["next_round"])
            if round_number > int(self.initial_config["task"]["trials"]):
                return self._finish("ADAPTIVE_ROUND_BUDGET_EXHAUSTED")
            stopped = self.run_adaptive_round(round_number)
            if stopped == "NO_INTERVENTION":
                dispatch_researcher_outputs_audit(self.repo_root, self.run_root)
                return self.run_root / "configs" / "final_frozen.json"


def run_submission(repo_root: Path, submit_path: Path) -> Path:
    return Orchestrator(repo_root, submit_path).run()
