from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import pandas as pd
import torch
from qlib.data import D

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.portfolio_runner import align_portfolio_asset_panels, read_panel, run_mean_variance
from adapters.qlib_runner import benchmark_returns, build_bundle, initialize_qlib, write_alpha_panel
from core.io_utils import atomic_write_json, load_json, sha256_file
from core.isolation import install_researcher_write_guard
from core.split_guard import aligned_trade_return_dates, purged_signal_dates, split_window, validate_panel_coverage
from evaluation.result_writer import build_result, write_result
from evaluation.prediction_metrics import compute_signal_metrics
from training.ensemble import ensemble_predictions, normalize_prediction_cross_sections
from training.model_trainer import evaluate_model, load_model_from_checkpoint, persisted_metric


RESEARCHER_EVALUATION_N_JOBS = 0


def zip_equal_lengths(*sequences):
    """Zip sequences strictly while remaining compatible with Python 3.9."""
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) > 1:
        raise RuntimeError("RESEARCHER_ENSEMBLE_MEMBER_COUNT_MISMATCH")
    return zip(*sequences)


def seed_output_dir(experiment_dir: Path, seed: int | None) -> Path:
    output = experiment_dir / "test"
    return output if seed is None else output / "seeds" / f"seed_{seed}"


def researcher_runtime_config(config: dict) -> dict:
    """Use a single-process loader inside the nested isolated researcher process."""
    runtime = copy.deepcopy(config)
    runtime["z_model"]["base_params"]["n_jobs"] = RESEARCHER_EVALUATION_N_JOBS
    return runtime


def configure_researcher_temp(output_dir: Path) -> Path:
    """Keep Python and joblib scratch files inside the isolated test boundary."""
    temporary_dir = output_dir / "_tmp"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    for variable in ("TMP", "TEMP", "TMPDIR", "JOBLIB_TEMP_FOLDER"):
        os.environ[variable] = str(temporary_dir)
    tempfile.tempdir = str(temporary_dir)
    return temporary_dir


def persist_researcher_error(output_dir: Path, mode: str, epoch: int | None) -> None:
    """Persist diagnostics only inside test/ so adaptive code cannot observe them."""
    header = f"mode={mode} epoch={epoch}\n"
    (output_dir / "error.log").write_text(header + traceback.format_exc(), encoding="utf-8")


def ensure_recovered_epoch_record(
    experiment_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    test_metrics: dict,
) -> None:
    """Recover only the selected epoch when historical checkpoints no longer exist."""
    path = output_dir / "epoch_metrics.jsonl"
    if path.is_file() and path.stat().st_size > 0:
        return
    training_path = experiment_dir / "logs" / "training_metrics.jsonl"
    if not training_path.is_file():
        return
    records = [json.loads(line) for line in training_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selections = [record for record in records if record.get("event") == "checkpoint_selected"]
    if not selections:
        return
    selection = selections[-1]
    epoch = int(selection["epoch"])
    epoch_records = [record for record in records if record.get("epoch") == epoch and record.get("event") is None]
    if len(epoch_records) != 1:
        return
    record = epoch_records[0]
    recovered = {
        "epoch": epoch,
        "checkpoint_id": checkpoint.name,
        "recovery_scope": "selected_checkpoint_only",
        **{f"train_{key}": persisted_metric(record.get("train", {}).get(key)) for key in ("ic", "icir", "rank_ic", "rank_icir")},
        **{
            f"train_valid_{key}": persisted_metric(record.get("train_valid", {}).get(key))
            for key in ("ic", "icir", "rank_ic", "rank_icir")
        },
        **{f"test_{key}": persisted_metric(test_metrics.get(key)) for key in ("ic", "icir", "rank_ic", "rank_icir")},
    }
    path.write_text(json.dumps(recovered, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def prepare_test_context(config: dict):
    bundle = build_bundle(REPO_ROOT, config)
    sampler, raw_label = bundle.prepare("test", researcher=True)
    return bundle, sampler, raw_label


def run_epoch_prepared(
    config: dict,
    experiment_dir: Path,
    checkpoint: Path,
    epoch: int,
    train_metrics: dict,
    train_valid_metrics: dict,
    sampler,
    raw_label: pd.Series,
    seed: int | None = None,
) -> None:
    model = load_model_from_checkpoint(config, checkpoint)
    _, metrics = evaluate_model(
        model, sampler, raw_label, len(config["z_alpha"]["selected_features"]),
        config, torch.device(config["runtime"]["resolved_device"]),
    )
    output_dir = seed_output_dir(experiment_dir, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "epoch_metrics.jsonl"
    record = {
        "epoch": epoch,
        "checkpoint_id": checkpoint.name,
        "seed": seed,
        **{f"train_{key}": persisted_metric(train_metrics.get(key)) for key in ("ic", "icir", "rank_ic", "rank_icir")},
        **{
            f"train_valid_{key}": persisted_metric(train_valid_metrics.get(key))
            for key in ("ic", "icir", "rank_ic", "rank_icir")
        },
        **{f"test_{key}": persisted_metric(metrics.summary.get(key)) for key in ("ic", "icir", "rank_ic", "rank_icir")},
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def run_epoch(
    config: dict,
    experiment_dir: Path,
    checkpoint: Path,
    epoch: int,
    train_metrics: dict,
    train_valid_metrics: dict,
    seed: int | None = None,
) -> None:
    _, sampler, raw_label = prepare_test_context(config)
    run_epoch_prepared(
        config,
        experiment_dir,
        checkpoint,
        epoch,
        train_metrics,
        train_valid_metrics,
        sampler,
        raw_label,
        seed,
    )


def repair_epoch_log(config: dict, experiment_dir: Path) -> None:
    """Researcher-only recovery for legacy runs that retained every epoch checkpoint."""
    adaptive_log = experiment_dir / "logs" / "training_metrics.jsonl"
    records = [
        json.loads(line)
        for line in adaptive_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    epochs = [record for record in records if "epoch" in record and record.get("event") is None]
    if not epochs:
        raise RuntimeError("NO_ADAPTIVE_EPOCH_RECORDS_FOR_RESEARCHER_REPAIR")
    bundle = build_bundle(REPO_ROOT, config)
    sampler, raw_label = bundle.prepare("test", researcher=True)
    output_dir = experiment_dir / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_dir / "epoch_metrics.repair.jsonl"
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in epochs:
            epoch = int(record["epoch"])
            checkpoint = experiment_dir / "artifacts" / "checkpoints" / f"epoch_{epoch:03d}.pt"
            if not checkpoint.is_file():
                raise RuntimeError(
                    "EPOCH_METRIC_REPAIR_UNAVAILABLE: non-selected epoch checkpoints "
                    "are intentionally not retained"
                )
            model = load_model_from_checkpoint(config, checkpoint)
            _, metrics = evaluate_model(
                model, sampler, raw_label, len(config["z_alpha"]["selected_features"]),
                config, torch.device(config["runtime"]["resolved_device"]),
            )
            output = {
                "epoch": epoch,
                "checkpoint_id": checkpoint.name,
                **{f"train_{key}": persisted_metric(record["train"].get(key)) for key in ("ic", "icir", "rank_ic", "rank_icir")},
                **{
                    f"train_valid_{key}": persisted_metric(record["train_valid"].get(key))
                    for key in ("ic", "icir", "rank_ic", "rank_icir")
                },
                **{f"test_{key}": persisted_metric(metrics.summary.get(key)) for key in ("ic", "icir", "rank_ic", "rank_icir")},
            }
            handle.write(json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    temporary_path.replace(output_dir / "epoch_metrics.jsonl")


def _copy_reused_test_artifact(source: Path, target: Path) -> None:
    """Copy one researcher-only artifact atomically and verify its content hash."""
    temporary = target.with_name(target.name + ".reuse.tmp")
    shutil.copyfile(source, temporary)
    if sha256_file(source) != sha256_file(temporary):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("RESEARCHER_REUSED_TEST_ARTIFACT_HASH_MISMATCH")
    temporary.replace(target)


def reuse_parent_test_alpha(config: dict, experiment_dir: Path, source_experiment_dir: Path) -> bool:
    """Rerun only the test portfolio for RAPA using its accepted parent's alpha.

    This function runs inside the physically isolated researcher process.  It never
    exposes the copied alpha or any resulting test metric to adaptive components.
    """
    source_output = source_experiment_dir / "test"
    output_dir = experiment_dir / "test"
    required_sources = (
        "epoch_metrics.jsonl",
        "test_alpha.csv",
        "ensemble_validation_diagnostics.json",
        "portfolio_asset_coverage.json",
        "result.json",
    )
    if not all((source_output / name).is_file() and (source_output / name).stat().st_size > 0 for name in required_sources):
        return False

    for name in required_sources[:-1]:
        _copy_reused_test_artifact(source_output / name, output_dir / name)
    for source in sorted(source_output.glob("test_alpha_seed_*.csv")):
        _copy_reused_test_artifact(source, output_dir / source.name)

    alpha = read_panel(output_dir / "test_alpha.csv")
    returns = read_panel((REPO_ROOT / config["z_portfolio"]["ret_path"]).resolve())
    up = read_panel((REPO_ROOT / config["z_portfolio"]["limit_up_mask_path"]).resolve())
    down = read_panel((REPO_ROOT / config["z_portfolio"]["limit_down_mask_path"]).resolve())
    returns, up, down, _ = align_portfolio_asset_panels(returns, up, down)

    initialize_qlib(REPO_ROOT, config)
    calendar = [
        pd.Timestamp(item).normalize()
        for item in D.calendar(
            start_time=config["task"]["runtime_handler_start_time"],
            end_time=config["task"]["test_end_time"],
            freq="day",
        )
    ]
    window = split_window(config, "test", researcher=True)
    signal_dates = purged_signal_dates(calendar, window)
    expected_index = pd.DatetimeIndex(signal_dates)
    if not alpha.index.equals(expected_index):
        raise RuntimeError("RAPA_REUSED_TEST_ALPHA_INDEX_MISMATCH")
    alignment = aligned_trade_return_dates(calendar, signal_dates)
    validate_panel_coverage(signal_dates, alignment, returns.index, up.index, down.index)
    benchmark = benchmark_returns(config, calendar, signal_dates)
    portfolio = run_mean_variance(config, alpha, returns, up, down, alignment, benchmark)
    portfolio.diagnostics.to_csv(
        output_dir / "portfolio_daily_diagnostics.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    source_result = load_json(source_output / "result.json")
    ensemble_diagnostics = load_json(source_output / "ensemble_validation_diagnostics.json")
    full_precision_summary = ensemble_diagnostics.get("ensemble_signal_metrics")
    signal_summary = (
        {key: full_precision_summary[key] for key in ("ic", "icir", "rank_ic", "rank_icir")}
        if isinstance(full_precision_summary, dict)
        else {key: source_result[key] for key in ("ic", "icir", "rank_ic", "rank_icir")}
    )
    result = build_result(signal_summary, portfolio.diagnostics)
    write_result(result, output_dir / "result.json", REPO_ROOT / "schemas" / "result.schema.json")
    atomic_write_json(
        output_dir / "researcher_complete.json",
        {
            "status": "COMPLETE",
            "alpha_reused_from": source_experiment_dir.name,
            "contains_test_derived_data": True,
        },
    )
    return True


def run_full(
    config: dict,
    experiment_dir: Path,
    checkpoints: list[Path],
    source_experiment_dir: Path | None,
    prepared_test_context=None,
) -> None:
    output_dir = experiment_dir / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_epoch_log = source_experiment_dir / "test" / "epoch_metrics.jsonl" if source_experiment_dir is not None else None
    if source_epoch_log is not None and source_epoch_log.is_file() and not (output_dir / "epoch_metrics.jsonl").exists():
        shutil.copyfile(source_epoch_log, output_dir / "epoch_metrics.jsonl")
    required_outputs = (
        "epoch_metrics.jsonl",
        "test_alpha.csv",
        "ensemble_validation_diagnostics.json",
        "portfolio_daily_diagnostics.csv",
        "result.json",
        "researcher_complete.json",
    )
    if all((output_dir / name).is_file() and (output_dir / name).stat().st_size > 0 for name in required_outputs):
        return
    if source_experiment_dir is not None and reuse_parent_test_alpha(config, experiment_dir, source_experiment_dir):
        return
    if prepared_test_context is None:
        bundle, sampler, raw_label = prepare_test_context(config)
    else:
        bundle, sampler, raw_label = prepared_test_context
    if not checkpoints:
        raise RuntimeError("RESEARCHER_CHECKPOINTS_EMPTY")
    predictions = []
    seed_metrics = []
    checkpoint_seeds = []
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_seed = int(payload.get("seed", config["task"]["seed"]))
        checkpoint_seeds.append(checkpoint_seed)
        model = load_model_from_checkpoint(config, checkpoint)
        prediction, metrics = evaluate_model(
            model, sampler, raw_label, len(config["z_alpha"]["selected_features"]),
            config, torch.device(config["runtime"]["resolved_device"]),
        )
        predictions.append(prediction)
        seed_metrics.append(metrics)
    if len(predictions) == 1:
        prediction = predictions[0]
        metrics = seed_metrics[0]
        ensure_recovered_epoch_record(experiment_dir, output_dir, checkpoints[0], metrics.summary)
    else:
        prediction = ensemble_predictions(predictions)
        metrics = compute_signal_metrics(prediction, raw_label.reindex(prediction.index))
    returns = read_panel((REPO_ROOT / config["z_portfolio"]["ret_path"]).resolve())
    up = read_panel((REPO_ROOT / config["z_portfolio"]["limit_up_mask_path"]).resolve())
    down = read_panel((REPO_ROOT / config["z_portfolio"]["limit_down_mask_path"]).resolve())
    returns, up, down, asset_coverage = align_portfolio_asset_panels(returns, up, down)
    window = split_window(config, "test", researcher=True)
    signal_dates = purged_signal_dates(bundle.calendar, window)
    alignment = aligned_trade_return_dates(bundle.calendar, signal_dates)
    validate_panel_coverage(signal_dates, alignment, returns.index, up.index, down.index)
    if len(predictions) > 1:
        for seed, member in zip_equal_lengths(checkpoint_seeds, predictions):
            write_alpha_panel(
                normalize_prediction_cross_sections(member),
                signal_dates,
                list(returns.columns),
                output_dir / f"test_alpha_seed_{seed}.csv",
            )
        selected_records = [
            {
                "seed": seed,
                "checkpoint_id": checkpoint.name,
                **{
                    f"test_{key}": persisted_metric(member_metrics.summary.get(key))
                    for key in ("ic", "icir", "rank_ic", "rank_icir")
                },
            }
            for seed, checkpoint, member_metrics in zip_equal_lengths(
                checkpoint_seeds, checkpoints, seed_metrics
            )
        ]
        (output_dir / "epoch_metrics.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for record in selected_records),
            encoding="utf-8",
        )
        atomic_write_json(
            output_dir / "ensemble_validation_diagnostics.json",
            {
                "seeds": checkpoint_seeds,
                "method": config["ensemble"]["method"],
                "seed_signal_metrics": [item.summary for item in seed_metrics],
                "ensemble_signal_metrics": metrics.summary,
                "contains_test_derived_data": True,
            },
        )
    alpha = write_alpha_panel(prediction, signal_dates, list(returns.columns), output_dir / "test_alpha.csv")
    asset_coverage["dropped_prediction_asset_count"] = len(alpha.attrs.get("dropped_prediction_assets", []))
    asset_coverage["dropped_prediction_assets"] = list(alpha.attrs.get("dropped_prediction_assets", []))
    atomic_write_json(output_dir / "portfolio_asset_coverage.json", asset_coverage)
    benchmark = benchmark_returns(config, bundle.calendar, signal_dates)
    portfolio = run_mean_variance(config, alpha, returns, up, down, alignment, benchmark)
    portfolio.diagnostics.to_csv(output_dir / "portfolio_daily_diagnostics.csv", index=False, encoding="utf-8", lineterminator="\n")
    result = build_result(metrics.summary, portfolio.diagnostics)
    write_result(result, output_dir / "result.json", REPO_ROOT / "schemas" / "result.schema.json")
    atomic_write_json(output_dir / "researcher_complete.json", {"status": "COMPLETE"})


def _ack(event: str, ok: bool, epoch: int | None = None) -> None:
    payload = {"event": event, "ok": ok}
    if epoch is not None:
        payload["epoch"] = epoch
    print("DIAGAGENT_ACK " + json.dumps(payload, separators=(",", ":")), flush=True)


def run_epoch_stream(config: dict, experiment_dir: Path, seed: int | None = None) -> None:
    """Own the test Dataset for one experiment and accept write-only requests."""
    context = prepare_test_context(config)
    _ack("worker_ready", True)
    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        command = message.get("command")
        if command == "close":
            _ack("closed", True)
            return
        epoch = int(message["epoch"]) if command == "epoch" else None
        try:
            if command == "epoch":
                checkpoint = Path(message["checkpoint"]).resolve()
                run_epoch_prepared(
                    config,
                    experiment_dir,
                    checkpoint,
                    epoch,
                    message["train_metrics"],
                    message["train_valid_metrics"],
                    context[1],
                    context[2],
                    seed,
                )
                _ack("epoch_complete", True, epoch)
            elif command == "full":
                source = message.get("source_experiment_dir")
                run_full(
                    config,
                    experiment_dir,
                    [Path(item).resolve() for item in message.get("checkpoints", [message.get("checkpoint")]) if item],
                    Path(source).resolve() if source else None,
                    prepared_test_context=context,
                )
                _ack("full_complete", True)
            else:
                raise RuntimeError("RESEARCHER_STREAM_COMMAND_INVALID")
        except Exception:
            persist_researcher_error(experiment_dir / "test", "epoch-stream", epoch)
            _ack("epoch_complete" if command == "epoch" else "full_complete", False, epoch)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["epoch", "epoch-stream", "full", "repair"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoints-json")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--train-metrics-json")
    parser.add_argument("--train-valid-metrics-json")
    parser.add_argument("--source-experiment-dir")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = researcher_runtime_config(load_json(config_path))
    experiment_dir = config_path.parent
    output_dir = experiment_dir / "test"
    temporary_dir = configure_researcher_temp(output_dir)
    install_researcher_write_guard(output_dir)
    error_path = output_dir / "error.log"
    error_path.unlink(missing_ok=True)
    checkpoints = (
        [Path(item).resolve() for item in json.loads(args.checkpoints_json)]
        if args.checkpoints_json
        else ([Path(args.checkpoint).resolve()] if args.checkpoint else [])
    )
    try:
        try:
            if args.mode == "epoch-stream":
                run_epoch_stream(config, experiment_dir, args.seed)
            elif args.mode == "epoch":
                if not checkpoints:
                    raise RuntimeError("RESEARCHER_CHECKPOINT_REQUIRED")
                run_epoch(
                    config,
                    experiment_dir,
                    checkpoints[0],
                    args.epoch,
                    json.loads(args.train_metrics_json),
                    json.loads(args.train_valid_metrics_json),
                    args.seed,
                )
            elif args.mode == "repair":
                if not checkpoints:
                    raise RuntimeError("RESEARCHER_CHECKPOINT_REQUIRED")
                repair_epoch_log(config, experiment_dir)
                run_full(config, experiment_dir, checkpoints, None)
            else:
                if not checkpoints:
                    raise RuntimeError("RESEARCHER_CHECKPOINT_REQUIRED")
                run_full(config, experiment_dir, checkpoints, Path(args.source_experiment_dir).resolve() if args.source_experiment_dir else None)
        except Exception:
            persist_researcher_error(output_dir, args.mode, args.epoch)
            raise
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
