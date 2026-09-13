from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pandas as pd
import yaml

from adapters.llm_client import AgentClient
from core.errors import ContractError
from core.io_utils import load_json
from core.isolation import assert_validation_safe_value
from memory.memory_store import MemoryStore


def build_validation_evidence(experiment_dir: Path) -> dict:
    result = load_json(experiment_dir / "result.json")
    diagnostics = pd.read_csv(experiment_dir / "artifacts" / "portfolio_daily_diagnostics.csv")
    signal_daily = pd.read_csv(experiment_dir / "artifacts" / "signal_daily_metrics.csv")
    evidence = {
        "experiment_id": experiment_dir.name,
        "result": result,
        "signal_daily_valid_date_count": int(len(signal_daily)),
        "portfolio": {
            "missing_signal_dates": int(diagnostics["missing_signal"].astype(bool).sum()),
            "full_position_infeasible_dates": int((~diagnostics["full_position_feasible"].astype(bool)).sum()),
            "feasible_min_invested_weight_mean": float(diagnostics["feasible_min_invested_weight"].mean()),
            "feasible_max_invested_weight_mean": float(diagnostics["feasible_max_invested_weight"].mean()),
            "invested_weight_mean": float(diagnostics["invested_weight"].mean()),
            "transaction_cost_drag_mean": float(diagnostics["transaction_cost_drag"].mean()),
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
        records = [json.loads(line) for line in training_log.read_text(encoding="utf-8").splitlines() if line.strip()]
        epochs = [record for record in records if "epoch" in record and record.get("event") is None]
        selected = next((record for record in records if record.get("event") == "checkpoint_selected"), None)
        evidence["model"] = {
            "epochs_completed": len(epochs),
            "first_epoch": epochs[0] if epochs else None,
            "last_epoch": epochs[-1] if epochs else None,
            "checkpoint_selection": selected,
        }
    else:
        evidence["model"] = {"reused": True}
    assert_validation_safe_value(evidence)
    return evidence


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

    @staticmethod
    def _adaptive_config(config: dict) -> dict:
        task = config["task"]
        return {
            "task": {
                "name": task["name"],
                "instruments": task["instruments"],
                "benchmark": task["benchmark"],
                "train_period": [task["train_start_time"], task["train_end_time"]],
                "valid_period": [task["valid_start_time"], task["valid_end_time"]],
                "seed": task["seed"],
            },
            "z_alpha": config["z_alpha"],
            "z_model": config["z_model"],
            "z_portfolio": config["z_portfolio"],
        }

    def clem(self, round_number: int, accepted_experiment: str, evidence: dict) -> dict:
        context = {
            "round": round_number,
            "accepted_experiment_id": accepted_experiment,
            "accepted_adaptive_config": self._adaptive_config(self.config),
            "current_validation_evidence": evidence,
            "complete_memory_bank": self.memory.projection("cross_layer"),
            "frozen_constraints": {
                "minimum_confidence": 0.5,
                "mandatory_rass_exp001": round_number == 1 and len(self.config["z_alpha"]["selected_features"]) == 6 and not self.config["z_alpha"]["alpha_frozen"],
                "rass_forbidden_after_alpha_frozen": self.config["z_alpha"]["alpha_frozen"],
                "one_layer_only": True,
            },
        }
        decision = self.client.complete_json(self._prompt("CLEM"), context, self._schema("CLEM"), "clem_decision")
        self.validate_clem(decision, round_number)
        return decision

    def validate_clem(self, decision: dict, round_number: int) -> None:
        jsonschema.Draft202012Validator(self._schema("CLEM")).validate(decision)
        mandatory = round_number == 1 and len(self.config["z_alpha"]["selected_features"]) == 6 and not self.config["z_alpha"]["alpha_frozen"]
        if mandatory and decision["selected_agent"] != "RASS":
            raise ContractError("CLEM_MANDATORY_RASS_ROUTE_VIOLATION")
        if self.config["z_alpha"]["alpha_frozen"] and decision["selected_agent"] == "RASS":
            raise ContractError("CLEM_RASS_AFTER_ALPHA_FREEZE")
        if decision["selected_agent"] != "NO_INTERVENTION" and decision["confidence"] < 0.5:
            raise ContractError("CLEM_CONFIDENCE_THRESHOLD_VIOLATION")
        if decision["selected_agent"] == "NO_INTERVENTION" and (decision["selected_layer"] != "none" or decision["intervention_goal"] is not None):
            raise ContractError("CLEM_NO_INTERVENTION_SHAPE_INVALID")
        assert_validation_safe_value(decision)

    def validate_rass_shortlist(self, proposal: dict) -> None:
        jsonschema.Draft202012Validator(self._schema("RASS_SHORTLIST")).validate(proposal)
        evidence = load_json(self.run_root / "configs" / "rass_train_evidence.json")
        if proposal["evidence_hash"] != evidence["evidence_hash"]:
            raise ContractError("RASS_SHORTLIST_EVIDENCE_HASH_MISMATCH")
        eligible_pairs = {
            (record["id"], record["expression"])
            for record in evidence["records"]
            if record.get("eligible", False) and record["expression"] not in self.config["z_alpha"]["selected_features"]
        }
        pairs = [(item["id"], item["expression"]) for item in proposal["shortlist"]]
        if len(set(pairs)) != 12:
            raise ContractError("RASS_SHORTLIST_DUPLICATE")
        if any(pair not in eligible_pairs for pair in pairs):
            raise ContractError("RASS_SHORTLIST_INELIGIBLE_OR_UNKNOWN")
        assert_validation_safe_value(proposal)

    def rass_shortlist(self, clem: dict, contract_feedback: dict | None = None) -> dict:
        evidence = load_json(self.run_root / "configs" / "rass_train_evidence.json")
        compact_fields = (
            "id", "expression", "category", "eligible", "eligibility_reasons",
            "coverage", "daily_coverage_p05", "nonconstant_cross_section_ratio",
            "ic", "icir", "rank_ic", "rank_icir", "yearly_ic",
            "yearly_rank_ic", "positive_ic_year_ratio", "worst_year_ic",
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
            "data_split": "train",
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
        context = {
            "clem_goal": {
                "failure_summary": clem["failure_summary"],
                "intervention_goal": clem["intervention_goal"],
            },
            "initial_features": self.config["z_alpha"]["selected_features"],
            "shortlist_decision": shortlist,
            "joint_train_only_evidence": joint_evidence,
            "base_evidence_provenance": {
                key: evidence[key]
                for key in ("input_data_hash", "catalog_hash", "label_hash", "effective_config_hash", "evidence_code_hash", "evidence_hash")
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
        return self.client.complete_json(self._prompt("RASS"), context, self._schema("RASS"), "rass_proposal")

    def specialist(self, agent: str, clem: dict, evidence: dict, contract_feedback: dict | None = None) -> dict:
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
                "train_only_evidence": load_json(self.run_root / "configs" / "rass_train_evidence.json"),
                "required_evidence_ref": "configs/rass_train_evidence.json",
            }
        elif agent == "FAMA":
            router = yaml.safe_load((self.repo_root / "agents" / "FAMA" / "intervention_space.yaml").read_text(encoding="utf-8"))
            model_name = self.config["z_model"]["model_name"]
            relative = router["model_spaces"].get(model_name)
            if relative is None:
                raise ContractError("FAMA_MODEL_UNSUPPORTED")
            context.update({
                "current_validation_evidence": evidence,
                "complete_model_memory": self.memory.projection("model"),
                "intervention_router": router,
                "resolved_model_space": yaml.safe_load((self.repo_root / "agents" / "FAMA" / relative).read_text(encoding="utf-8")),
            })
        elif agent == "RAPA":
            context.update({
                "current_validation_evidence": evidence,
                "complete_portfolio_memory": self.memory.projection("portfolio"),
                "intervention_space": yaml.safe_load((self.repo_root / "agents" / "RAPA" / "intervention_space.yaml").read_text(encoding="utf-8")),
            })
        else:
            raise ContractError(f"UNKNOWN_SPECIALIST: {agent}")
        if contract_feedback is not None:
            context["contract_feedback"] = contract_feedback
        assert_validation_safe_value(context)
        return self.client.complete_json(self._prompt(agent), context, self._schema(agent), f"{agent.lower()}_proposal")
