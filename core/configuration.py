from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import platform
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from jsonschema import Draft202012Validator
from qlib.data import D

from adapters.portfolio_runner import read_panel
from adapters.qlib_runner import (
    initialize_qlib,
    instrument_to_panel_code,
    resolve_feature_catalog,
    resolve_market_instruments,
)
from core.errors import ContractError, DataCoverageError, ResumeError
from core.evidence_builder import build_rass_train_evidence, rass_evidence_context
from core.io_utils import (
    atomic_write_json,
    canonical_json_bytes,
    load_json,
    sha256_bytes,
    sha256_file,
    sha256_paths,
    write_json_exclusive,
)
from core.prediction_mode import apply_prediction_mode, prediction_mode, prediction_mode_preset
from core.split_guard import aligned_trade_return_dates, purged_signal_dates, split_window, validate_panel_coverage


TASK_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
ANCHOR_FEATURES = [
    "$open/$close",
    "$high/$close",
    "$low/$close",
    "$vwap/$close",
    "Ref($close, 1)/$close",
]


def _flatten_paths(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        result = []
        for key in sorted(value):
            result.extend(_flatten_paths(value[key], f"{prefix}.{key}" if prefix else key))
        return result
    if isinstance(value, list):
        return [prefix]
    return [prefix]


def _semantic_config(config: dict) -> dict:
    value = copy.deepcopy(config)
    value.pop("provenance", None)
    return value


def refresh_effective_config_hash(config: dict) -> str:
    digest = sha256_bytes(canonical_json_bytes(_semantic_config(config)))
    config.setdefault("provenance", {})["effective_config_hash"] = digest
    return digest


def _load_agent_catalog(repo_root: Path, config: dict) -> None:
    catalog = load_json(repo_root / "configs" / "agent_models.json")
    selection = config.get("agent", {})
    model = selection.get("model")
    effort = selection.get("reasoning_effort")
    if model not in catalog.get("models", {}):
        raise ContractError(f"AGENT_MODEL_NOT_CATALOGED: {model}")
    if effort not in catalog["models"][model].get("reasoning_efforts", []):
        raise ContractError(f"AGENT_REASONING_EFFORT_NOT_ALLOWED: {model}/{effort}")


def _resolve_device(config: dict) -> dict:
    requested_index = config["task"]["GPU"]
    if type(requested_index) is not int or requested_index < 0:
        raise ContractError("GPU_INDEX_MUST_BE_NONNEGATIVE_INTEGER")
    requested = f"cuda:{requested_index}"
    if torch.cuda.is_available() and requested_index < torch.cuda.device_count():
        resolved = requested
        reason = None
    elif not torch.cuda.is_available():
        resolved = "cpu"
        reason = "cuda_unavailable"
    else:
        resolved = "cpu"
        reason = "cuda_index_out_of_range"
    if reason:
        warnings.warn(f"Requested {requested}; using cpu ({reason}).", RuntimeWarning, stacklevel=2)
    return {
        "requested_device": requested,
        "resolved_device": resolved,
        "fallback_reason": reason,
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
    }


def _validate_seed(config: dict) -> None:
    apply_prediction_mode(config)
    task = config.get("task", {})
    preset = prediction_mode_preset(config)
    name = task.get("name", "")
    if not TASK_NAME_PATTERN.fullmatch(name):
        raise ContractError("TASK_NAME_INVALID")
    if not isinstance(task.get("run_id"), str) or not re.fullmatch(r"\d{4}", task["run_id"]):
        raise ContractError("TASK_RUN_ID_MUST_BE_FOUR_DIGIT_YEAR")
    if config.get("pipeline_contract") != "qlib_ts_lstm_v1":
        raise ContractError("PIPELINE_CONTRACT_INVALID")
    if task.get("provider_uri") != "./data/cn_data":
        raise ContractError("QLIB_PROVIDER_PATH_FROZEN")
    if task.get("target") != preset["target"]:
        raise ContractError("PREDICTION_MODE_TARGET_MISMATCH")
    if task.get("primary_metric") != "sharpe":
        raise ContractError("PRIMARY_METRIC_FROZEN")
    if type(task.get("trials")) is not int or not 0 <= task["trials"] <= 10:
        raise ContractError("ADAPTIVE_ROUND_BUDGET_INVALID")
    ensemble = config.get("ensemble", {})
    if ensemble.get("enabled") is not True:
        raise ContractError("ENSEMBLE_MUST_BE_ENABLED")
    if ensemble.get("seeds") != [0, 1, 2]:
        raise ContractError("ENSEMBLE_SEEDS_MUST_BE_0_1_2")
    if ensemble.get("method") != "cross_sectional_zscore_mean":
        raise ContractError("ENSEMBLE_METHOD_INVALID")
    if ensemble.get("checkpoint_selection") != "independent_train_valid_ic":
        raise ContractError("ENSEMBLE_CHECKPOINT_SELECTION_INVALID")
    windows = {
        split: (
            pd.Timestamp(task[f"{split}_start_time"]),
            pd.Timestamp(task[f"{split}_end_time"]),
        )
        for split in ("train", "train_valid", "agent_valid", "test")
    }
    if any(start > end for start, end in windows.values()):
        raise ContractError("DATA_SPLIT_START_AFTER_END")
    train_start, train_end = windows["train"]
    train_valid_start, train_valid_end = windows["train_valid"]
    agent_valid_start, agent_valid_end = windows["agent_valid"]
    test_start, test_end = windows["test"]
    # The two validation windows may overlap or be identical.  This keeps the
    # current setting reproducible while making their responsibilities explicit.
    # Neither window may overlap training or the isolated researcher test.
    if not (
        train_start <= train_end < train_valid_start
        and train_end < agent_valid_start
        and train_valid_end < test_start <= test_end
        and agent_valid_end < test_start
    ):
        raise ContractError("DATA_SPLITS_OVERLAP_OR_UNORDERED")
    alpha = config.get("z_alpha", {})
    if alpha.get("feature_pool") != "Alpha158" or alpha.get("selected_features") != ANCHOR_FEATURES:
        raise ContractError("ROUND0_ANCHOR_FEATURES_INVALID")
    model = config.get("z_model", {})
    if model.get("model_name") not in {"lstm", "gru", "alstm"}:
        raise ContractError("MODEL_NAME_UNSUPPORTED")
    if model.get("sequence_mode") != "sampler" or model.get("sequence_window") != 20:
        raise ContractError("SEQUENCE_CONTRACT_INVALID")
    base = model.get("base_params", {})
    expected_base = {"loss": "mse", "optimizer": "adam", "batch_size": 2048, "n_jobs": 8, "label_norm": True}
    for key, expected in expected_base.items():
        if base.get(key) != expected:
            raise ContractError(f"MODEL_BASE_FIELD_FROZEN: {key}")
    if base.get("metric", "ic") != "ic":
        raise ContractError("CHECKPOINT_METRIC_MUST_BE_TRAIN_VALID_IC")
    portfolio = config.get("z_portfolio", {})
    expected_portfolio = {
        "ret_path": preset["ret_path"],
        "limit_up_mask_path": preset["limit_up_mask_path"],
        "limit_down_mask_path": preset["limit_down_mask_path"],
        "execution_price": preset["execution_price"],
        "return_definition": preset["return_definition"],
        "buy_commission": 0.0001,
        "sell_commission": 0.0001,
        "sell_tax": 0.0005,
        "buy_slippage": preset["buy_slippage"],
        "sell_slippage": preset["sell_slippage"],
        "full_position": True,
        "long_only": True,
        "trade_delay_days": 1,
        "return_delay_days": 2,
        "max_weight": 0.02,
        "min_weight": 0.0,
        "candidate_count": 100,
        "risk_model_mode": "diagonal",
        "risk_aversion": 1,
        "turnover_penalty": 1,
        "alpha_scale": 0.001,
    }
    for key, expected in expected_portfolio.items():
        if portfolio.get(key) != expected:
            raise ContractError(f"PORTFOLIO_FIELD_FROZEN_OR_BASELINE_INVALID: {key}")
    buy_cost = float(portfolio["buy_commission"]) + float(portfolio["buy_slippage"])
    sell_cost = float(portfolio["sell_commission"]) + float(portfolio["sell_tax"]) + float(portfolio["sell_slippage"])
    expected_buy_cost = 0.0001 + float(preset["buy_slippage"])
    expected_sell_cost = 0.0001 + 0.0005 + float(preset["sell_slippage"])
    if abs(buy_cost - expected_buy_cost) > 1e-15 or abs(sell_cost - expected_sell_cost) > 1e-15:
        raise ContractError("PORTFOLIO_TRANSACTION_COSTS_FROZEN")


def _handler_start(calendar: list[pd.Timestamp], train_start: pd.Timestamp, sequence_window: int) -> str:
    before = [date for date in calendar if date < train_start]
    if len(before) < sequence_window:
        raise DataCoverageError("INSUFFICIENT_SEQUENCE_WARMUP_CALENDAR")
    return before[-sequence_window].strftime("%Y-%m-%d")


def _relevant_qlib_files(repo_root: Path, config: dict, instruments: dict[str, Any]) -> list[Path]:
    provider = (repo_root / config["task"]["provider_uri"]).resolve()
    task = config["task"]
    paths = [provider / "calendars" / "day.txt", provider / "instruments" / f"{task['instruments']}.txt"]
    for instrument in sorted(instruments):
        for field in ("open", "high", "low", "close", "vwap", "volume"):
            paths.append(provider / "features" / instrument.lower() / f"{field}.day.bin")
    price_field = prediction_mode_preset(config)["price_field"]
    paths.append(provider / "features" / task["benchmark"].lower() / f"{price_field}.day.bin")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise DataCoverageError(f"QLIB_RELEVANT_INPUT_MISSING: {missing[:3]}")
    return paths


def _validate_data_coverage(repo_root: Path, config: dict, calendar: list[pd.Timestamp], instruments: dict[str, Any]) -> dict[str, str]:
    portfolio = config["z_portfolio"]
    returns_path = (repo_root / portfolio["ret_path"]).resolve()
    up_path = (repo_root / portfolio["limit_up_mask_path"]).resolve()
    down_path = (repo_root / portfolio["limit_down_mask_path"]).resolve()
    returns = read_panel(returns_path)
    up = read_panel(up_path)
    down = read_panel(down_path)
    mapped = {instrument_to_panel_code(instrument) for instrument in instruments}
    # Historical index membership may contain securities that were subsequently
    # delisted or renamed and therefore do not exist in the portfolio panels.
    # Portfolio execution is defined on the daily intersection of agent-valid alpha
    # and portfolio data, so partial membership mismatch is legal.  Still catch
    # code-format mistakes or a completely unrelated panel before execution.
    if not mapped.intersection(returns.columns):
        raise DataCoverageError("QLIB_RETURN_PANEL_NO_ASSET_OVERLAP")
    for split in ("agent_valid", "test"):
        window = split_window(config, split, researcher=True)
        signal_dates = purged_signal_dates(calendar, window)
        alignment = aligned_trade_return_dates(calendar, signal_dates)
        validate_panel_coverage(signal_dates, alignment, returns.index, up.index, down.index)
    qlib_files = _relevant_qlib_files(repo_root, config, instruments)
    return {
        "return_panel": sha256_file(returns_path),
        "limit_up_panel": sha256_file(up_path),
        "limit_down_panel": sha256_file(down_path),
        "qlib_relevant_manifest": sha256_paths(repo_root, qlib_files),
    }


def _dependency_versions() -> dict[str, str | None]:
    versions = {"python": platform.python_version(), "python_executable": sys.executable}
    for package in ("pyqlib", "torch", "numpy", "pandas", "scipy", "jsonschema", "pyyaml"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    versions["platform"] = platform.platform()
    return versions


def _contract_files(repo_root: Path) -> list[Path]:
    paths = [
        repo_root / "AGENTS.md",
        repo_root / "docs" / "DiagAgent_Implementation_Spec.md",
        repo_root / "configs" / "frozen_contract.json",
        repo_root / "configs" / "agent_models.json",
        repo_root / "schemas" / "result.schema.json",
        repo_root / "agents" / "CLEM" / "clem.md",
        repo_root / "agents" / "CLEM" / "schema.json",
        repo_root / "agents" / "RASS" / "SYSTEM.md",
        repo_root / "agents" / "RASS" / "SKILL.md",
        repo_root / "agents" / "RASS" / "SHORTLIST.md",
        repo_root / "agents" / "RASS" / "shortlist.schema.json",
        repo_root / "agents" / "RASS" / "intervention_space.yaml",
        repo_root / "agents" / "FAMA" / "intervention_space.yaml",
        repo_root / "agents" / "RAPA" / "intervention_space.yaml",
    ]
    paths.extend(sorted((repo_root / "agents" / "FAMA" / "intervention_spaces").glob("*.yaml")))
    paths.extend(path for path in [
        repo_root / "agents" / "FAMA" / "SYSTEM.md",
        repo_root / "agents" / "FAMA" / "intervention.schema.json",
        repo_root / "agents" / "RAPA" / "SYSTEM.md",
        repo_root / "agents" / "RAPA" / "intervention.schema.json",
        repo_root / "agents" / "RASS" / "intervention.schema.json",
    ] if path.exists())
    return paths


def load_submit(repo_root: Path, submit_path: Path) -> tuple[dict, str, str]:
    resolved = submit_path.resolve()
    submit_root = (repo_root / "submit").resolve()
    if submit_root not in resolved.parents or resolved.suffix.lower() != ".json":
        raise ContractError("SUBMIT_PATH_MUST_BE_EXPLICIT_REPOSITORY_SUBMIT_JSON")
    if not resolved.is_file():
        raise ContractError(f"SUBMIT_FILE_MISSING: {resolved}")
    relative = resolved.relative_to(repo_root.resolve()).as_posix()
    return load_json(resolved), relative, sha256_file(resolved)


RUN_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _run_component(value: Any, field: str) -> str:
    if not isinstance(value, str) or not RUN_COMPONENT_PATTERN.fullmatch(value):
        raise ContractError(f"RUN_PATH_COMPONENT_INVALID: {field}")
    return value


def _agent_name(repo_root: Path, config: dict) -> str:
    explicit = config["agent"].get("name")
    if explicit:
        return _run_component(explicit, "agent.name")
    model = config["agent"]["model"]
    catalog_path = repo_root.resolve() / "configs" / "agent_models.json"
    if catalog_path.is_file():
        provider = load_json(catalog_path).get("models", {}).get(model, {}).get("provider")
        if provider:
            return _run_component(provider, "agent.provider")
    return _run_component(model, "agent.model")


def resolve_run_layout(repo_root: Path, seed: dict, submit_path: Path) -> tuple[Path, Path, dict[str, str]]:
    """Return the shared RASS namespace and immutable per-submission run root.

    The login controller freezes ``DIAGAGENT_RUN_TIMESTAMP`` once and Slurm
    workers inherit it. ``task.run_id``, when present, is a stable study/year
    directory above that timestamp, never a replacement for it. Direct
    non-controller use falls back to the submit-file modification minute. A
    same-minute collision is detected by the existing submit hash/provenance
    checks instead of silently mixing trajectories.
    """
    instruments = _run_component(seed["task"]["instruments"], "task.instruments")
    agent_name = _agent_name(repo_root, seed)
    mode = prediction_mode(seed)
    model_name = _run_component(seed["z_model"]["model_name"], "z_model.model_name")
    explicit_run_id = seed["task"].get("run_id")
    study_id = _run_component(explicit_run_id, "task.run_id") if explicit_run_id is not None else None
    run_timestamp = os.environ.get("DIAGAGENT_RUN_TIMESTAMP")
    if run_timestamp is None:
        run_timestamp = datetime.fromtimestamp(submit_path.resolve().stat().st_mtime).astimezone().strftime("%m%d%H%M")
    if not re.fullmatch(r"\d{8}", run_timestamp):
        raise ContractError("RUN_TIMESTAMP_INVALID")
    namespace = repo_root.resolve() / "runs" / f"{instruments}_{agent_name}_{mode}"
    run_root = namespace
    if study_id is not None:
        run_root = run_root / study_id
    run_root = run_root / model_name / run_timestamp
    return namespace, run_root, {
        "instruments": instruments,
        "agent_name": agent_name,
        "prediction_mode": mode,
        "model_name": model_name,
        "run_id": study_id or run_timestamp,
        "run_id_source": "task.run_id" if study_id is not None else "controller_or_submit_timestamp",
        "study_id": study_id,
        "run_timestamp": run_timestamp,
    }


def rass_evidence_path(repo_root: Path, config: dict) -> Path:
    instruments = _run_component(config["task"]["instruments"], "task.instruments")
    agent_name = _agent_name(repo_root, config)
    mode = prediction_mode(config)
    run_id = _run_component(config["task"].get("run_id"), "task.run_id")
    return repo_root.resolve() / "runs" / f"{instruments}_{agent_name}_{mode}" / run_id / "rass_train_evidence.json"


def _validated_cached_rass_evidence(
    repo_root: Path,
    config: dict,
    evidence: dict,
    catalog_hash: str,
) -> dict:
    if (
        evidence.get("method") != "deterministic_rass_development_factor_evidence_v5"
        or evidence.get("data_split") != "rass_development"
        or evidence.get("contains_test_derived_data") is not False
    ):
        raise ContractError("RASS_EVIDENCE_CACHE_CONTRACT_INVALID")
    expected_context = rass_evidence_context(config, catalog_hash)
    context = evidence.get("evidence_context")
    if context != expected_context:
        raise ContractError("RASS_EVIDENCE_CACHE_CONTEXT_MISMATCH")
    expected_context_hash = sha256_bytes(canonical_json_bytes(expected_context))
    if evidence.get("rass_evidence_context_hash") != expected_context_hash:
        raise ContractError("RASS_EVIDENCE_CACHE_CONTEXT_HASH_MISMATCH")
    periods = expected_context["evidence_periods"]
    expected = {
        "prediction_mode": expected_context["prediction_mode"],
        "target_label": expected_context["target_label"],
        "label_hash": expected_context["label_hash"],
        "initial_features": ANCHOR_FEATURES,
        "catalog_hash": catalog_hash,
        "input_data_hash": expected_context["input_data_hash"],
        "evidence_code_hash": expected_context["evidence_code_hash"],
        "configured_training_period": expected_context["configured_training_period"],
        "evidence_period": [periods[0]["start_time"], periods[-1]["signal_end_time"]],
    }
    for key, value in expected.items():
        if evidence.get(key) != value:
            raise ContractError(f"RASS_EVIDENCE_CACHE_IDENTITY_MISMATCH: {key}")
    rolling = evidence.get("rolling_evidence_protocol")
    if not isinstance(rolling, dict) or rolling.get("periods") != periods or rolling.get("test_used") is not False:
        raise ContractError("RASS_EVIDENCE_CACHE_PERIOD_MISMATCH")
    records = evidence.get("records")
    if not isinstance(records, list) or evidence.get("candidate_count") != len(records):
        raise ContractError("RASS_EVIDENCE_CACHE_RECORDS_INVALID")
    stored_hash = evidence.get("evidence_hash")
    unhashed = copy.deepcopy(evidence)
    unhashed.pop("evidence_hash", None)
    if not isinstance(stored_hash, str) or stored_hash != sha256_bytes(canonical_json_bytes(unhashed)):
        raise ContractError("RASS_EVIDENCE_CACHE_HASH_MISMATCH")
    return evidence


def _set_round0_alpha_baseline(config: dict) -> None:
    """Enforce the five submitted anchors independently of shared RASS cache state."""
    if config["z_alpha"].get("selected_features") != ANCHOR_FEATURES:
        raise ContractError("ROUND0_ANCHOR_FEATURES_INVALID")
    config["z_alpha"]["alpha_frozen"] = False
    config["z_model"]["derived"] = {"d_feat": len(ANCHOR_FEATURES)}


def materialize_initial_config(repo_root: Path, submit_path: Path) -> tuple[dict, Path]:
    repo_root = repo_root.resolve()
    seed, relative_submit, submit_hash = load_submit(repo_root, submit_path)
    _validate_seed(seed)
    _load_agent_catalog(repo_root, seed)
    namespace, run_root, run_layout = resolve_run_layout(repo_root, seed, submit_path)
    initial_path = run_root / "configs" / "initial.json"
    provenance_path = run_root / "configs" / "initial.provenance.json"
    if initial_path.exists():
        initial = load_json(initial_path)
        source = initial.get("source_submit", {})
        if source != {"path": relative_submit, "sha256": submit_hash}:
            raise ResumeError("EXISTING_RUN_SUBMIT_MISMATCH")
        if not provenance_path.is_file() or load_json(provenance_path).get("initial_config_sha256") != sha256_file(initial_path):
            raise ResumeError("INITIAL_CONFIG_PROVENANCE_MISMATCH")
        return initial, run_root

    config = copy.deepcopy(seed)
    config["source_submit"] = {"path": relative_submit, "sha256": submit_hash}
    config["runtime"] = _resolve_device(config)
    config["runtime"]["run_layout"] = run_layout
    _set_round0_alpha_baseline(config)
    config["pipeline_runtime"] = {
        "dataset": "TSDatasetH",
        "handler": "DataHandlerLP",
        "process_type": "PTYPE_A",
        "data_loader": "QlibDataLoader",
        "feature_processors": ["ProcessInf", "ZScoreNorm(train-fit)", "Fillna(0)"],
        "label_processors": ["DropnaLabel", "CSZScoreNorm"],
        "sampler_fillna_type": "ffill+bfill",
        "train_loader": {"shuffle": True, "drop_last": True},
        "evaluation_loaders": {"shuffle": False, "drop_last": False},
        "train_valid_learning_key": "DK_L",
        "raw_label_key": "DK_R",
    }
    initialize_qlib(repo_root, config)
    complete_calendar = [pd.Timestamp(item).normalize() for item in D.calendar(start_time="2000-01-01", end_time=config["task"]["test_end_time"], freq="day")]
    config["task"]["runtime_handler_start_time"] = _handler_start(
        complete_calendar,
        pd.Timestamp(config["task"]["train_start_time"]),
        int(config["z_model"]["sequence_window"]),
    )
    instruments, unavailable = resolve_market_instruments(repo_root, config)
    calendar = [date for date in complete_calendar if date >= pd.Timestamp(config["task"]["runtime_handler_start_time"])]
    input_hashes = _validate_data_coverage(repo_root, config, calendar, instruments)
    config["provenance"] = {
        "source_submit": {"path": relative_submit, "sha256": submit_hash},
        "dependency_versions": _dependency_versions(),
        "input_hashes": input_hashes,
        "contract_inputs_hash": sha256_paths(repo_root, _contract_files(repo_root)),
        "resolved_market": {
            "name": config["task"]["instruments"],
            "available_instrument_count": len(instruments),
            "unavailable_placeholder_count": len(unavailable),
        },
        "field_sources": {
            "seed_fields": _flatten_paths(seed),
            "seed_source": relative_submit,
            "derived_fields": {
                "runtime": "frozen startup CUDA resolution",
                "z_alpha.alpha_frozen": "false for the mandatory five-factor Round-0 baseline",
                "z_model.derived.d_feat": "len(z_alpha.selected_features)",
                "pipeline_runtime": "qlib_ts_lstm_v1 frozen contract",
                "ensemble": "three independent seeds; per-seed train-valid IC checkpoints; cross-sectional z-score mean",
                "task.prediction_mode_and_bound_execution_fields": "c2c/o2o mode preset",
                "task.runtime_handler_start_time": "20-trading-day sequence warmup",
            },
        },
        "contains_test_derived_data": False,
    }
    refresh_effective_config_hash(config)
    catalog = resolve_feature_catalog(config["z_alpha"]["feature_pool"])
    catalog_payload = {"pool": "Alpha158", "entries": catalog, "contains_test_derived_data": False}
    catalog_hash = sha256_bytes(canonical_json_bytes(catalog_payload))
    catalog_payload["catalog_hash"] = catalog_hash
    shared_evidence_path = rass_evidence_path(repo_root, config)
    evidence_cache_status = "created"
    if shared_evidence_path.is_file():
        try:
            evidence = _validated_cached_rass_evidence(
                repo_root,
                config,
                load_json(shared_evidence_path),
                catalog_hash,
            )
            evidence_cache_status = "reused"
        except (ContractError, json.JSONDecodeError):
            evidence = build_rass_train_evidence(repo_root, config, catalog, catalog_hash)
            atomic_write_json(shared_evidence_path, evidence)
            evidence_cache_status = "regenerated"
    else:
        evidence = build_rass_train_evidence(repo_root, config, catalog, catalog_hash)
        atomic_write_json(shared_evidence_path, evidence)
    config["provenance"]["shared_rass_evidence"] = {
        "path": shared_evidence_path.relative_to(repo_root).as_posix(),
        "sha256": sha256_file(shared_evidence_path),
        "status": evidence_cache_status,
        "selection_cached": False,
    }
    config["provenance"]["feature_catalog"] = {"path": "configs/frozen_feature_catalog.json", "sha256": sha256_bytes(canonical_json_bytes(catalog_payload))}
    if evidence is not None:
        config["provenance"]["rass_train_evidence"] = {"path": "configs/rass_train_evidence.json", "sha256": sha256_bytes(canonical_json_bytes(evidence))}
    configs_dir = run_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(configs_dir / "frozen_feature_catalog.json", catalog_payload)
    if evidence is not None:
        atomic_write_json(configs_dir / "rass_train_evidence.json", evidence)
    write_json_exclusive(initial_path, config)
    initial_hash = sha256_file(initial_path)
    write_json_exclusive(provenance_path, {
        "initial_config_sha256": initial_hash,
        "source_submit": {"path": relative_submit, "sha256": submit_hash},
        "contains_test_derived_data": False,
    })
    write_json_exclusive(configs_dir / "current.json", config)
    atomic_write_json(configs_dir / "state.json", {
        "status": "MATERIALIZED",
        "accepted_experiment_id": None,
        "next_round": 0,
        "source_submit": {"path": relative_submit, "sha256": submit_hash},
        "contains_test_derived_data": False,
    })
    return config, run_root


def validate_result_schema(repo_root: Path, result: dict) -> None:
    schema = load_json(repo_root / "schemas" / "result.schema.json")
    Draft202012Validator(schema).validate(result)
