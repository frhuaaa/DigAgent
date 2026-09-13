from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.portfolio_runner import read_panel, run_mean_variance
from adapters.qlib_runner import benchmark_returns, build_bundle, write_alpha_panel
from core.io_utils import atomic_write_json, load_json
from core.isolation import install_researcher_write_guard
from core.split_guard import aligned_trade_return_dates, purged_signal_dates, split_window, validate_panel_coverage
from evaluation.result_writer import build_result, write_result
from training.model_trainer import evaluate_model, load_model_from_checkpoint, persisted_metric


def run_epoch(config: dict, experiment_dir: Path, checkpoint: Path, epoch: int, train_metrics: dict, valid_metrics: dict) -> None:
    bundle = build_bundle(REPO_ROOT, config)
    sampler, raw_label = bundle.prepare("test", researcher=True)
    model = load_model_from_checkpoint(config, checkpoint)
    _, metrics = evaluate_model(
        model, sampler, raw_label, len(config["z_alpha"]["selected_features"]),
        config, torch.device(config["runtime"]["resolved_device"]),
    )
    output_dir = experiment_dir / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "epoch_metrics.jsonl"
    record = {
        "epoch": epoch,
        "checkpoint_id": checkpoint.name,
        **{f"train_{key}": persisted_metric(train_metrics.get(key)) for key in ("ic", "icir", "rank_ic", "rank_icir")},
        **{f"valid_{key}": persisted_metric(valid_metrics.get(key)) for key in ("ic", "icir", "rank_ic", "rank_icir")},
        **{f"test_{key}": persisted_metric(metrics.summary.get(key)) for key in ("ic", "icir", "rank_ic", "rank_icir")},
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


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
                **{f"valid_{key}": persisted_metric(record["valid"].get(key)) for key in ("ic", "icir", "rank_ic", "rank_icir")},
                **{f"test_{key}": persisted_metric(metrics.summary.get(key)) for key in ("ic", "icir", "rank_ic", "rank_icir")},
            }
            handle.write(json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    temporary_path.replace(output_dir / "epoch_metrics.jsonl")


def run_full(config: dict, experiment_dir: Path, checkpoint: Path, source_experiment_dir: Path | None) -> None:
    output_dir = experiment_dir / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_epoch_log = source_experiment_dir / "test" / "epoch_metrics.jsonl" if source_experiment_dir is not None else None
    if source_epoch_log is not None and source_epoch_log.is_file() and not (output_dir / "epoch_metrics.jsonl").exists():
        shutil.copyfile(source_epoch_log, output_dir / "epoch_metrics.jsonl")
    bundle = build_bundle(REPO_ROOT, config)
    sampler, raw_label = bundle.prepare("test", researcher=True)
    model = load_model_from_checkpoint(config, checkpoint)
    prediction, metrics = evaluate_model(
        model, sampler, raw_label, len(config["z_alpha"]["selected_features"]),
        config, torch.device(config["runtime"]["resolved_device"]),
    )
    returns = read_panel((REPO_ROOT / config["z_portfolio"]["ret_path"]).resolve())
    up = read_panel((REPO_ROOT / config["z_portfolio"]["limit_up_mask_path"]).resolve())
    down = read_panel((REPO_ROOT / config["z_portfolio"]["limit_down_mask_path"]).resolve())
    window = split_window(config, "test", researcher=True)
    signal_dates = purged_signal_dates(bundle.calendar, window)
    alignment = aligned_trade_return_dates(bundle.calendar, signal_dates)
    validate_panel_coverage(signal_dates, alignment, returns.index, up.index, down.index)
    alpha = write_alpha_panel(prediction, signal_dates, list(returns.columns), output_dir / "test_alpha.csv")
    benchmark = benchmark_returns(config, bundle.calendar, signal_dates)
    portfolio = run_mean_variance(config, alpha, returns, up, down, alignment, benchmark)
    portfolio.diagnostics.to_csv(output_dir / "portfolio_daily_diagnostics.csv", index=False, encoding="utf-8", lineterminator="\n")
    result = build_result(metrics.summary, portfolio.diagnostics)
    write_result(result, output_dir / "result.json", REPO_ROOT / "schemas" / "result.schema.json")
    atomic_write_json(output_dir / "researcher_complete.json", {"status": "COMPLETE"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["epoch", "full", "repair"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--train-metrics-json")
    parser.add_argument("--valid-metrics-json")
    parser.add_argument("--source-experiment-dir")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    experiment_dir = config_path.parent
    output_dir = experiment_dir / "test"
    temporary_dir = output_dir / "_tmp"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    for variable in ("TMP", "TEMP", "TMPDIR"):
        os.environ[variable] = str(temporary_dir)
    tempfile.tempdir = str(temporary_dir)
    install_researcher_write_guard(output_dir)
    checkpoint = Path(args.checkpoint).resolve()
    try:
        if args.mode == "epoch":
            run_epoch(config, experiment_dir, checkpoint, args.epoch, json.loads(args.train_metrics_json), json.loads(args.valid_metrics_json))
        elif args.mode == "repair":
            repair_epoch_log(config, experiment_dir)
            run_full(config, experiment_dir, checkpoint, None)
        else:
            run_full(config, experiment_dir, checkpoint, Path(args.source_experiment_dir).resolve() if args.source_experiment_dir else None)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
