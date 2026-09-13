from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from qlib.data import D

from adapters.portfolio_runner import read_panel, run_mean_variance, run_topk
from adapters.qlib_runner import (
    benchmark_returns,
    build_bundle,
    initialize_qlib,
    validate_alpha_panel,
    write_alpha_panel,
)
from core.errors import ContractError
from core.io_utils import atomic_write_json, load_json, sha256_file, verify_hashes
from core.researcher_boundary import ResearcherBoundary
from core.split_guard import aligned_trade_return_dates, purged_signal_dates, split_window, validate_panel_coverage
from evaluation.result_writer import build_result, write_result
from training.model_trainer import evaluate_model, load_model_from_checkpoint, train_model


@dataclass
class ExecutionOutcome:
    result: dict
    diagnostics: pd.DataFrame
    signal_daily: pd.DataFrame
    checkpoint: Path
    manifest: dict


def dependency_plan(agent: str | None) -> tuple[list[str], list[str]]:
    if agent is None:
        return ["factors", "model", "predictions", "portfolio", "validation"], ["source_data"]
    if agent == "RASS":
        return ["factors", "model", "predictions", "portfolio", "validation"], ["source_data"]
    if agent == "FAMA":
        return ["model", "predictions", "portfolio", "validation"], ["selected_factors", "source_data"]
    if agent == "RAPA":
        return ["portfolio", "validation"], ["factors", "model", "predictions"]
    raise ContractError(f"UNKNOWN_DEPENDENCY_AGENT: {agent}")


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _stage_outputs(repo_root: Path, experiment_dir: Path, marker: dict) -> ExecutionOutcome:
    verify_hashes(repo_root, marker["output_hashes"])
    result = load_json(experiment_dir / "result.json")
    diagnostics = pd.read_csv(experiment_dir / "artifacts" / "portfolio_daily_diagnostics.csv", parse_dates=["signal_date", "trade_date", "return_date"])
    signal_daily = pd.read_csv(experiment_dir / "artifacts" / "signal_daily_metrics.csv", index_col=0, parse_dates=True)
    checkpoint = repo_root / marker["selected_checkpoint"]
    return ExecutionOutcome(result, diagnostics, signal_daily, checkpoint, marker)


def _signal_summary(daily: pd.DataFrame) -> dict:
    result = {}
    for column, ratio_name in (("ic", "icir"), ("rank_ic", "rank_icir")):
        values = pd.to_numeric(daily[column], errors="coerce").dropna()
        result[column] = float(values.mean()) if len(values) else None
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        result[ratio_name] = float(values.mean() / std) if np.isfinite(std) and std > 0 else None
    return result


def _expected_checkpoint_config_hash(
    config: dict,
    agent: str | None,
    parent_experiment_dir: Path | None,
    selected_checkpoint_metadata: dict,
) -> str:
    if agent != "RAPA":
        return config["provenance"]["effective_config_hash"]
    if parent_experiment_dir is None:
        raise ContractError("RAPA_PARENT_REQUIRED")
    if selected_checkpoint_metadata.get("reused_from") != parent_experiment_dir.name:
        raise ContractError("RECOVERY_RAPA_PARENT_PROVENANCE_MISMATCH")
    parent_config = load_json(parent_experiment_dir / "config.json")
    return parent_config["provenance"]["effective_config_hash"]


def _recover_completed_validation_artifacts(
    repo_root: Path,
    config: dict,
    experiment_dir: Path,
    reference_columns: list[str],
    rerun: list[str],
    reused: list[str],
    agent: str | None,
    parent_experiment_dir: Path | None,
) -> ExecutionOutcome | None:
    artifact_dir = experiment_dir / "artifacts"
    required = [
        artifact_dir / "selected_checkpoint.json",
        artifact_dir / "valid_alpha.csv",
        artifact_dir / "signal_daily_metrics.csv",
        artifact_dir / "portfolio_daily_diagnostics.csv",
        artifact_dir / "topk_daily_diagnostics.csv",
    ]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return None
    selected = load_json(required[0])
    checkpoint = repo_root / selected["repository_path"] if "repository_path" in selected else experiment_dir / selected["path"]
    if not checkpoint.is_file():
        raise ContractError("RECOVERY_SELECTED_CHECKPOINT_MISSING")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_checkpoint_config_hash = _expected_checkpoint_config_hash(
        config,
        agent,
        parent_experiment_dir,
        selected,
    )
    if payload.get("config_hash") != expected_checkpoint_config_hash:
        raise ContractError("RECOVERY_CHECKPOINT_CONFIG_HASH_MISMATCH")
    diagnostics = pd.read_csv(required[3], parse_dates=["signal_date", "trade_date", "return_date"])
    topk = pd.read_csv(required[4], parse_dates=["signal_date", "trade_date", "return_date"])
    signal_daily = pd.read_csv(required[2], index_col=0, parse_dates=True)
    signal_dates = [pd.Timestamp(item).normalize() for item in diagnostics["signal_date"]]
    if signal_dates != [pd.Timestamp(item).normalize() for item in topk["signal_date"]]:
        raise ContractError("RECOVERY_TOPK_DATES_MISMATCH")
    validate_alpha_panel(required[1], signal_dates, reference_columns)
    result = build_result(_signal_summary(signal_daily), diagnostics)
    topk_result = build_result(_signal_summary(signal_daily), topk)
    write_result(result, experiment_dir / "result.json", repo_root / "schemas" / "result.schema.json")
    write_result(topk_result, artifact_dir / "topk_result.json", repo_root / "schemas" / "result.schema.json")
    paths = [experiment_dir / "result.json", *required, artifact_dir / "topk_result.json"]
    manifest = {
        "status": "COMPLETE",
        "recovered_from_completed_artifacts": True,
        "rerun_stages": rerun,
        "reused_stages": reused,
        "selected_checkpoint": checkpoint.resolve().relative_to(repo_root.resolve()).as_posix(),
        "selected_checkpoint_sha256": sha256_file(checkpoint),
        "output_hashes": {path.resolve().relative_to(repo_root.resolve()).as_posix(): sha256_file(path) for path in paths},
        "contains_test_derived_data": False,
    }
    atomic_write_json(experiment_dir / "logs" / "execution_complete.json", manifest)
    return ExecutionOutcome(dict(result), diagnostics, signal_daily, checkpoint, manifest)


def execute_validation(
    repo_root: Path,
    config: dict,
    experiment_dir: Path,
    agent: str | None,
    parent_experiment_dir: Path | None,
) -> ExecutionOutcome:
    marker_path = experiment_dir / "logs" / "execution_complete.json"
    if marker_path.exists():
        return _stage_outputs(repo_root, experiment_dir, load_json(marker_path))
    artifact_dir = experiment_dir / "artifacts"
    log_dir = experiment_dir / "logs"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    boundary = ResearcherBoundary(repo_root, experiment_dir / "config.json", log_dir / "researcher_dispatch.jsonl")
    reference_returns = read_panel((repo_root / config["z_portfolio"]["ret_path"]).resolve())
    limit_up = read_panel((repo_root / config["z_portfolio"]["limit_up_mask_path"]).resolve())
    limit_down = read_panel((repo_root / config["z_portfolio"]["limit_down_mask_path"]).resolve())

    rerun, reused = dependency_plan(agent)
    recovered = _recover_completed_validation_artifacts(
        repo_root,
        config,
        experiment_dir,
        list(reference_returns.columns),
        rerun,
        reused,
        agent,
        parent_experiment_dir,
    )
    if recovered is not None:
        return recovered
    if agent == "RAPA":
        if parent_experiment_dir is None:
            raise ContractError("RAPA_PARENT_REQUIRED")
        parent_alpha = parent_experiment_dir / "artifacts" / "valid_alpha.csv"
        parent_signal = parent_experiment_dir / "artifacts" / "signal_daily_metrics.csv"
        parent_selected = load_json(parent_experiment_dir / "artifacts" / "selected_checkpoint.json")
        checkpoint = (
            repo_root / parent_selected["repository_path"]
            if "repository_path" in parent_selected
            else parent_experiment_dir / parent_selected["path"]
        )
        for source, target in [(parent_alpha, artifact_dir / "valid_alpha.csv"), (parent_signal, artifact_dir / "signal_daily_metrics.csv")]:
            shutil.copyfile(source, target)
            if sha256_file(source) != sha256_file(target):
                raise ContractError("REUSED_ARTIFACT_HASH_MISMATCH")
        atomic_write_json(artifact_dir / "selected_checkpoint.json", {
            "reused_from": parent_experiment_dir.name,
            "repository_path": checkpoint.resolve().relative_to(repo_root.resolve()).as_posix(),
            "sha256": sha256_file(checkpoint),
        })
        alpha = read_panel(artifact_dir / "valid_alpha.csv")
        signal_daily = pd.read_csv(artifact_dir / "signal_daily_metrics.csv", index_col=0, parse_dates=True)
        signal_summary = {key: load_json(parent_experiment_dir / "result.json")[key] for key in ("ic", "icir", "rank_ic", "rank_icir")}
        initialize_qlib(repo_root, config)
        calendar = [pd.Timestamp(item).normalize() for item in D.calendar(start_time=config["task"]["runtime_handler_start_time"], end_time=config["task"]["test_end_time"], freq="day")]
    else:
        bundle = build_bundle(repo_root, config)
        train_sampler, train_raw = bundle.prepare("train", researcher=False)
        valid_sampler, valid_raw = bundle.prepare("valid", researcher=False)
        outcome = train_model(
            config,
            train_sampler,
            train_raw,
            valid_sampler,
            valid_raw,
            experiment_dir,
            boundary.epoch,
        )
        checkpoint = outcome.selected_checkpoint
        model = load_model_from_checkpoint(config, checkpoint)
        prediction, signal_metrics = evaluate_model(
            model,
            valid_sampler,
            valid_raw,
            len(config["z_alpha"]["selected_features"]),
            config,
            torch.device(config["runtime"]["resolved_device"]),
        )
        signal_summary = signal_metrics.summary
        signal_daily = signal_metrics.daily
        signal_daily.to_csv(artifact_dir / "signal_daily_metrics.csv", encoding="utf-8", lineterminator="\n")
        calendar = bundle.calendar
        valid_window = split_window(config, "valid", researcher=False)
        signal_dates = purged_signal_dates(calendar, valid_window)
        alpha = write_alpha_panel(prediction, signal_dates, list(reference_returns.columns), artifact_dir / "valid_alpha.csv")

    valid_window = split_window(config, "valid", researcher=False)
    signal_dates = purged_signal_dates(calendar, valid_window)
    alignment = aligned_trade_return_dates(calendar, signal_dates)
    validate_panel_coverage(signal_dates, alignment, reference_returns.index, limit_up.index, limit_down.index)
    benchmark = benchmark_returns(config, calendar, signal_dates)
    if not alpha.index.equals(pd.DatetimeIndex(signal_dates, name="date")):
        alpha = alpha.reindex(pd.DatetimeIndex(signal_dates))
    portfolio_run = run_mean_variance(config, alpha, reference_returns, limit_up, limit_down, alignment, benchmark)
    topk_run = run_topk(config, alpha, reference_returns, limit_up, limit_down, alignment, benchmark, drop_count=10)
    diagnostics_path = artifact_dir / "portfolio_daily_diagnostics.csv"
    _write_frame(portfolio_run.diagnostics, diagnostics_path)
    _write_frame(portfolio_run.orders, artifact_dir / "order_details.csv")
    _write_frame(portfolio_run.objective, artifact_dir / "objective_diagnostics.csv")
    _write_frame(topk_run.diagnostics, artifact_dir / "topk_daily_diagnostics.csv")
    result = build_result(signal_summary, portfolio_run.diagnostics)
    write_result(result, experiment_dir / "result.json", repo_root / "schemas" / "result.schema.json")
    topk_result = build_result(signal_summary, topk_run.diagnostics)
    write_result(topk_result, artifact_dir / "topk_result.json", repo_root / "schemas" / "result.schema.json")
    selected_meta = artifact_dir / "selected_checkpoint.json"
    if not selected_meta.exists():
        raise ContractError("SELECTED_CHECKPOINT_METADATA_MISSING")
    paths = [
        experiment_dir / "result.json",
        artifact_dir / "valid_alpha.csv",
        diagnostics_path,
        artifact_dir / "signal_daily_metrics.csv",
        selected_meta,
        artifact_dir / "topk_daily_diagnostics.csv",
        artifact_dir / "topk_result.json",
    ]
    manifest = {
        "status": "COMPLETE",
        "rerun_stages": rerun,
        "reused_stages": reused,
        "selected_checkpoint": checkpoint.resolve().relative_to(repo_root.resolve()).as_posix(),
        "selected_checkpoint_sha256": sha256_file(checkpoint),
        "output_hashes": {path.resolve().relative_to(repo_root.resolve()).as_posix(): sha256_file(path) for path in paths},
        "contains_test_derived_data": False,
    }
    atomic_write_json(marker_path, manifest)
    return ExecutionOutcome(dict(result), portfolio_run.diagnostics, signal_daily, checkpoint, manifest)
