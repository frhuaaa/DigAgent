from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from qlib.data import D

from adapters.portfolio_runner import align_portfolio_asset_panels, read_panel, run_mean_variance, run_topk
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
from evaluation.prediction_metrics import compute_signal_metrics
from training.ensemble import (
    ensemble_predictions,
    ensemble_seeds,
    normalize_prediction_cross_sections,
    prediction_from_frame,
)


@dataclass
class ExecutionOutcome:
    result: dict
    diagnostics: pd.DataFrame
    signal_daily: pd.DataFrame
    checkpoint: Path
    manifest: dict


def _marker_checkpoint_paths(repo_root: Path, marker: dict) -> list[Path]:
    records = marker.get("selected_checkpoints")
    if records is None:
        return [(repo_root / marker["selected_checkpoint"]).resolve()]
    paths = [(repo_root / record["repository_path"]).resolve() for record in records]
    if not paths:
        raise ContractError("ENSEMBLE_CHECKPOINTS_EMPTY")
    return paths


def _selected_checkpoint_paths(repo_root: Path, experiment_dir: Path, selected: dict) -> list[Path]:
    records = selected.get("checkpoints")
    if records is None:
        path = repo_root / selected["repository_path"] if "repository_path" in selected else experiment_dir / selected["path"]
        return [path.resolve()]
    paths = [(repo_root / record["repository_path"]).resolve() for record in records]
    if not paths:
        raise ContractError("ENSEMBLE_CHECKPOINTS_EMPTY")
    return paths


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
    checkpoints = _marker_checkpoint_paths(repo_root, marker)
    records = marker.get("selected_checkpoints", [])
    if records and len(records) != len(checkpoints):
        raise ContractError("EXECUTION_MARKER_CHECKPOINT_COUNT_MISMATCH")
    for checkpoint, record in zip(checkpoints, records):
        if not checkpoint.is_file() or sha256_file(checkpoint) != record.get("sha256"):
            raise ContractError("EXECUTION_MARKER_CHECKPOINT_HASH_MISMATCH")
    checkpoint = checkpoints[0]
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


def _visible_cuda_devices() -> list[str]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured:
        return [item.strip() for item in configured.split(",") if item.strip()]
    return [str(index) for index in range(torch.cuda.device_count())]


def _run_ensemble_seed_workers(repo_root: Path, experiment_dir: Path, config: dict) -> list[dict]:
    seeds = list(ensemble_seeds(config))
    devices = _visible_cuda_devices()
    if not torch.cuda.is_available() or len(devices) < len(seeds):
        raise ContractError(
            f"ENSEMBLE_REQUIRES_{len(seeds)}_VISIBLE_GPUS: found={len(devices)}"
        )
    processes: list[tuple[int, subprocess.Popen[bytes], object, object]] = []
    for seed, device in zip(seeds, devices):
        stdout_path = experiment_dir / "logs" / f"seed_{seed}_worker.out"
        stderr_path = experiment_dir / "logs" / f"seed_{seed}_worker.err"
        stdout_handle = stdout_path.open("wb")
        stderr_handle = stderr_path.open("wb")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = device
        environment["PYTHONPATH"] = str(repo_root) + os.pathsep + environment.get("PYTHONPATH", "")
        environment["OMP_NUM_THREADS"] = "8"
        environment["MKL_NUM_THREADS"] = "8"
        command = [
            sys.executable,
            str(repo_root / "scripts" / "run_seed_training.py"),
            "--config",
            str(experiment_dir / "config.json"),
            "--seed",
            str(seed),
        ]
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        processes.append((seed, process, stdout_handle, stderr_handle))

    failures: list[str] = []
    for seed, process, stdout_handle, stderr_handle in processes:
        return_code = process.wait()
        stdout_handle.close()
        stderr_handle.close()
        if return_code != 0:
            failures.append(f"seed={seed}:exit={return_code}")
    if failures:
        raise ContractError("ENSEMBLE_SEED_WORKER_FAILED: " + ",".join(failures))

    markers = [load_json(experiment_dir / "logs" / f"seed_{seed}_complete.json") for seed in seeds]
    if [marker.get("seed") for marker in markers] != seeds:
        raise ContractError("ENSEMBLE_SEED_MARKER_ORDER_MISMATCH")
    return markers


def _aggregate_ensemble_validation(
    repo_root: Path,
    experiment_dir: Path,
    config: dict,
    reference_columns: list[str],
) -> tuple[Path, list[Path], pd.DataFrame, dict, pd.DataFrame, list[pd.Timestamp]]:
    artifact_dir = experiment_dir / "artifacts"
    seeds = list(ensemble_seeds(config))
    markers = _run_ensemble_seed_workers(repo_root, experiment_dir, config)
    # Preserve the established compact training-summary entry point while the
    # complete per-seed curves remain available in their namespaced logs.
    shutil.copyfile(
        experiment_dir / "logs" / f"training_metrics_seed_{seeds[0]}.jsonl",
        experiment_dir / "logs" / "training_metrics.jsonl",
    )
    bundle = build_bundle(repo_root, config)
    _, agent_valid_raw = bundle.prepare("agent_valid", researcher=False)
    predictions = [
        prediction_from_frame(pd.read_csv(artifact_dir / f"agent_valid_prediction_seed_{seed}.csv"))
        for seed in seeds
    ]
    normalized_members = [normalize_prediction_cross_sections(prediction) for prediction in predictions]
    member_signal_metrics = [
        compute_signal_metrics(prediction, agent_valid_raw)
        for prediction in normalized_members
    ]
    ensemble_prediction = ensemble_predictions(predictions)
    ensemble_metrics = compute_signal_metrics(ensemble_prediction, agent_valid_raw)
    ensemble_metrics.daily.to_csv(
        artifact_dir / "signal_daily_metrics.csv",
        encoding="utf-8",
        lineterminator="\n",
    )
    agent_valid_window = split_window(config, "agent_valid", researcher=False)
    signal_dates = purged_signal_dates(bundle.calendar, agent_valid_window)
    alpha = write_alpha_panel(
        ensemble_prediction,
        signal_dates,
        reference_columns,
        artifact_dir / "agent_valid_alpha.csv",
    )
    for seed, prediction in zip(seeds, normalized_members):
        write_alpha_panel(
            prediction,
            signal_dates,
            reference_columns,
            artifact_dir / f"agent_valid_alpha_seed_{seed}.csv",
        )

    checkpoints = [(repo_root / marker["selected_checkpoint"]).resolve() for marker in markers]
    records = []
    for seed, checkpoint, marker in zip(seeds, checkpoints, markers):
        if not checkpoint.is_file() or sha256_file(checkpoint) != marker["selected_checkpoint_sha256"]:
            raise ContractError(f"ENSEMBLE_CHECKPOINT_HASH_MISMATCH: seed={seed}")
        records.append({
            "seed": seed,
            "repository_path": checkpoint.relative_to(repo_root).as_posix(),
            "sha256": marker["selected_checkpoint_sha256"],
            "epoch": marker["selected_epoch"],
            "score": marker["selected_score"],
            "config_hash": config["provenance"]["effective_config_hash"],
        })
    selected = {
        "ensemble": True,
        "method": config["ensemble"]["method"],
        "checkpoint_selection": config["ensemble"]["checkpoint_selection"],
        "checkpoints": records,
        "contains_test_derived_data": False,
    }
    atomic_write_json(artifact_dir / "selected_checkpoint.json", selected)
    diagnostics = {
        "enabled": True,
        "method": config["ensemble"]["method"],
        "seeds": seeds,
        "seed_signal_metrics": [
            {"seed": seed, **metrics.summary}
            for seed, metrics in zip(seeds, member_signal_metrics)
        ],
        "ensemble_signal_metrics": ensemble_metrics.summary,
        "selected_checkpoints": records,
        "contains_test_derived_data": False,
    }
    atomic_write_json(artifact_dir / "ensemble_validation_diagnostics.json", diagnostics)
    return checkpoints[0], checkpoints, alpha, ensemble_metrics.summary, ensemble_metrics.daily, bundle.calendar


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
        artifact_dir / "ensemble_validation_diagnostics.json",
        artifact_dir / "agent_valid_alpha.csv",
        artifact_dir / "signal_daily_metrics.csv",
        artifact_dir / "portfolio_daily_diagnostics.csv",
        artifact_dir / "topk_daily_diagnostics.csv",
    ]
    required.extend(
        artifact_dir / f"agent_valid_alpha_seed_{seed}.csv"
        for seed in ensemble_seeds(config)
    )
    if agent != "RAPA":
        required.append(experiment_dir / "logs" / "training_metrics.jsonl")
        for seed in ensemble_seeds(config):
            required.extend([
                experiment_dir / "logs" / f"seed_{seed}_complete.json",
                experiment_dir / "logs" / f"training_metrics_seed_{seed}.jsonl",
                artifact_dir / f"split_indices_seed_{seed}.json",
                artifact_dir / f"selected_checkpoint_seed_{seed}.json",
            ])
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return None
    selected = load_json(required[0])
    checkpoints = _selected_checkpoint_paths(repo_root, experiment_dir, selected)
    expected_seeds = list(ensemble_seeds(config))
    selected_records = selected.get("checkpoints", [])
    if [record.get("seed") for record in selected_records] != expected_seeds:
        raise ContractError("RECOVERY_ENSEMBLE_SEEDS_MISMATCH")
    expected_checkpoint_config_hash = _expected_checkpoint_config_hash(
        config,
        agent,
        parent_experiment_dir,
        selected,
    )
    for seed, checkpoint, record in zip(expected_seeds, checkpoints, selected_records):
        if not checkpoint.is_file():
            raise ContractError("RECOVERY_SELECTED_CHECKPOINT_MISSING")
        if record.get("sha256") != sha256_file(checkpoint):
            raise ContractError("RECOVERY_SELECTED_CHECKPOINT_HASH_MISMATCH")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        record_config_hash = record.get("config_hash")
        expected_record_hash = (
            record_config_hash
            if agent == "RAPA" and isinstance(record_config_hash, str)
            else expected_checkpoint_config_hash
        )
        if payload.get("config_hash") != expected_record_hash or payload.get("seed") != seed:
            raise ContractError("RECOVERY_CHECKPOINT_CONFIG_OR_SEED_MISMATCH")
    diagnostics = pd.read_csv(required[4], parse_dates=["signal_date", "trade_date", "return_date"])
    topk = pd.read_csv(required[5], parse_dates=["signal_date", "trade_date", "return_date"])
    signal_daily = pd.read_csv(required[3], index_col=0, parse_dates=True)
    signal_dates = [pd.Timestamp(item).normalize() for item in diagnostics["signal_date"]]
    if signal_dates != [pd.Timestamp(item).normalize() for item in topk["signal_date"]]:
        raise ContractError("RECOVERY_TOPK_DATES_MISMATCH")
    validate_alpha_panel(required[2], signal_dates, reference_columns)
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
        "selected_checkpoint": checkpoints[0].resolve().relative_to(repo_root.resolve()).as_posix(),
        "selected_checkpoint_sha256": sha256_file(checkpoints[0]),
        "selected_checkpoints": selected_records,
        "output_hashes": {path.resolve().relative_to(repo_root.resolve()).as_posix(): sha256_file(path) for path in paths},
        "contains_test_derived_data": False,
    }
    atomic_write_json(experiment_dir / "logs" / "execution_complete.json", manifest)
    return ExecutionOutcome(dict(result), diagnostics, signal_daily, checkpoints[0], manifest)


def execute_validation(
    repo_root: Path,
    config: dict,
    experiment_dir: Path,
    agent: str | None,
    parent_experiment_dir: Path | None,
) -> ExecutionOutcome:
    marker_path = experiment_dir / "logs" / "execution_complete.json"
    artifact_dir = experiment_dir / "artifacts"
    log_dir = experiment_dir / "logs"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    boundary = ResearcherBoundary(repo_root, experiment_dir / "config.json", log_dir / "researcher_dispatch.jsonl")
    if marker_path.exists():
        marker = load_json(marker_path)
        completed = _stage_outputs(repo_root, experiment_dir, marker)
        boundary.full(_marker_checkpoint_paths(repo_root, marker), source_experiment_dir=parent_experiment_dir if agent == "RAPA" else None)
        boundary.close_epoch_worker()
        return completed
    reference_returns = read_panel((repo_root / config["z_portfolio"]["ret_path"]).resolve())
    limit_up = read_panel((repo_root / config["z_portfolio"]["limit_up_mask_path"]).resolve())
    limit_down = read_panel((repo_root / config["z_portfolio"]["limit_down_mask_path"]).resolve())
    reference_returns, limit_up, limit_down, asset_coverage = align_portfolio_asset_panels(
        reference_returns, limit_up, limit_down
    )
    atomic_write_json(log_dir / "portfolio_asset_coverage.json", asset_coverage)

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
        boundary.full(_marker_checkpoint_paths(repo_root, recovered.manifest), source_experiment_dir=parent_experiment_dir if agent == "RAPA" else None)
        boundary.close_epoch_worker()
        return recovered
    if agent == "RAPA":
        if parent_experiment_dir is None:
            raise ContractError("RAPA_PARENT_REQUIRED")
        parent_alpha = parent_experiment_dir / "artifacts" / "agent_valid_alpha.csv"
        parent_signal = parent_experiment_dir / "artifacts" / "signal_daily_metrics.csv"
        parent_selected = load_json(parent_experiment_dir / "artifacts" / "selected_checkpoint.json")
        checkpoints = _selected_checkpoint_paths(repo_root, parent_experiment_dir, parent_selected)
        checkpoint = checkpoints[0]
        for source, target in [
            (parent_alpha, artifact_dir / "agent_valid_alpha.csv"),
            (parent_signal, artifact_dir / "signal_daily_metrics.csv"),
            (parent_experiment_dir / "artifacts" / "ensemble_validation_diagnostics.json", artifact_dir / "ensemble_validation_diagnostics.json"),
        ]:
            shutil.copyfile(source, target)
            if sha256_file(source) != sha256_file(target):
                raise ContractError("REUSED_ARTIFACT_HASH_MISMATCH")
        for seed in ensemble_seeds(config):
            source = parent_experiment_dir / "artifacts" / f"agent_valid_alpha_seed_{seed}.csv"
            target = artifact_dir / source.name
            shutil.copyfile(source, target)
            if sha256_file(source) != sha256_file(target):
                raise ContractError("REUSED_ARTIFACT_HASH_MISMATCH")
        atomic_write_json(
            artifact_dir / "selected_checkpoint.json",
            {**parent_selected, "reused_from": parent_experiment_dir.name},
        )
        alpha = read_panel(artifact_dir / "agent_valid_alpha.csv")
        signal_daily = pd.read_csv(artifact_dir / "signal_daily_metrics.csv", index_col=0, parse_dates=True)
        signal_summary = {key: load_json(parent_experiment_dir / "result.json")[key] for key in ("ic", "icir", "rank_ic", "rank_icir")}
        initialize_qlib(repo_root, config)
        calendar = [pd.Timestamp(item).normalize() for item in D.calendar(start_time=config["task"]["runtime_handler_start_time"], end_time=config["task"]["test_end_time"], freq="day")]
    else:
        checkpoint, checkpoints, alpha, signal_summary, signal_daily, calendar = _aggregate_ensemble_validation(
            repo_root,
            experiment_dir,
            config,
            list(reference_returns.columns),
        )

    dropped_prediction_assets = list(alpha.attrs.get("dropped_prediction_assets", []))
    asset_coverage["dropped_prediction_asset_count"] = len(dropped_prediction_assets)
    asset_coverage["dropped_prediction_assets"] = dropped_prediction_assets
    atomic_write_json(log_dir / "portfolio_asset_coverage.json", asset_coverage)

    agent_valid_window = split_window(config, "agent_valid", researcher=False)
    signal_dates = purged_signal_dates(calendar, agent_valid_window)
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
        artifact_dir / "agent_valid_alpha.csv",
        diagnostics_path,
        artifact_dir / "signal_daily_metrics.csv",
        selected_meta,
        artifact_dir / "ensemble_validation_diagnostics.json",
        artifact_dir / "topk_daily_diagnostics.csv",
        artifact_dir / "topk_result.json",
    ]
    paths.extend(artifact_dir / f"agent_valid_alpha_seed_{seed}.csv" for seed in ensemble_seeds(config))
    if agent != "RAPA":
        paths.append(log_dir / "training_metrics.jsonl")
        for seed in ensemble_seeds(config):
            paths.extend([
                log_dir / f"seed_{seed}_complete.json",
                log_dir / f"training_metrics_seed_{seed}.jsonl",
                artifact_dir / f"split_indices_seed_{seed}.json",
                artifact_dir / f"selected_checkpoint_seed_{seed}.json",
            ])
    selected_checkpoint_metadata = load_json(selected_meta)
    checkpoints = _selected_checkpoint_paths(repo_root, experiment_dir, selected_checkpoint_metadata)
    manifest = {
        "status": "COMPLETE",
        "rerun_stages": rerun,
        "reused_stages": reused,
        "selected_checkpoint": checkpoint.resolve().relative_to(repo_root.resolve()).as_posix(),
        "selected_checkpoint_sha256": sha256_file(checkpoint),
        "selected_checkpoints": selected_checkpoint_metadata["checkpoints"],
        "output_hashes": {path.resolve().relative_to(repo_root.resolve()).as_posix(): sha256_file(path) for path in paths},
        "contains_test_derived_data": False,
    }
    atomic_write_json(marker_path, manifest)
    completed = ExecutionOutcome(dict(result), portfolio_run.diagnostics, signal_daily, checkpoint, manifest)
    boundary.full(checkpoints, source_experiment_dir=parent_experiment_dir if agent == "RAPA" else None)
    boundary.close_epoch_worker()
    return completed
