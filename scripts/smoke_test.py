from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
from qlib.data import D

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.portfolio_runner import align_portfolio_asset_panels, read_panel, run_mean_variance
from adapters.qlib_runner import benchmark_returns, build_bundle, initialize_qlib, write_alpha_panel
from core.configuration import refresh_effective_config_hash
from core.io_utils import atomic_write_json, load_json
from core.split_guard import aligned_trade_return_dates, purged_signal_dates, split_window, validate_panel_coverage
from evaluation.result_writer import build_result, validate_result
from training.model_trainer import evaluate_model, load_model_from_checkpoint, train_model


def main() -> int:
    config = copy.deepcopy(load_json(REPO_ROOT / "submit" / "lstm_csi500_deepseek_2024.json"))
    config["task"].update({
        "train_start_time": "2023-06-01", "train_end_time": "2023-12-29",
        "train_valid_start_time": "2024-01-02", "train_valid_end_time": "2024-01-31",
        "agent_valid_start_time": "2024-01-02", "agent_valid_end_time": "2024-01-31",
        "test_start_time": "2024-02-01", "test_end_time": "2024-02-29",
        "runtime_handler_start_time": "2023-05-04",
    })
    config["runtime"] = {"requested_device": "cuda:0", "resolved_device": "cpu", "fallback_reason": "smoke_cpu"}
    config["z_alpha"]["alpha_frozen"] = False
    config["z_model"]["derived"] = {"d_feat": 5}
    config["z_model"]["base_params"]["n_epochs"] = 1
    config["z_model"]["base_params"]["early_stop"] = 1
    config["provenance"] = {"effective_config_hash": "pending", "contains_test_derived_data": False}
    refresh_effective_config_hash(config)
    with tempfile.TemporaryDirectory(prefix="diagagent_smoke_") as tmp:
        experiment = Path(tmp) / "EXP_SMOKE"
        experiment.mkdir(parents=True)
        atomic_write_json(experiment / "config.json", config)
        bundle = build_bundle(REPO_ROOT, config)
        train_sampler, train_raw = bundle.prepare("train", researcher=False)
        train_valid_sampler, train_valid_raw = bundle.prepare("train_valid", researcher=False)
        outcome = train_model(
            config,
            train_sampler,
            train_raw,
            train_valid_sampler,
            train_valid_raw,
            experiment,
            lambda *_: None,
        )
        model = load_model_from_checkpoint(config, outcome.selected_checkpoint)
        agent_valid_sampler, agent_valid_raw = bundle.prepare("agent_valid", researcher=False)
        prediction, signal = evaluate_model(
            model,
            agent_valid_sampler,
            agent_valid_raw,
            5,
            config,
            torch.device("cpu"),
        )
        returns = read_panel(REPO_ROOT / config["z_portfolio"]["ret_path"])
        up = read_panel(REPO_ROOT / config["z_portfolio"]["limit_up_mask_path"])
        down = read_panel(REPO_ROOT / config["z_portfolio"]["limit_down_mask_path"])
        returns, up, down, _ = align_portfolio_asset_panels(returns, up, down)
        window = split_window(config, "agent_valid", researcher=False)
        signal_dates = purged_signal_dates(bundle.calendar, window)
        alignment = aligned_trade_return_dates(bundle.calendar, signal_dates)
        validate_panel_coverage(signal_dates, alignment, returns.index, up.index, down.index)
        alpha = write_alpha_panel(
            prediction,
            signal_dates,
            list(returns.columns),
            experiment / "artifacts" / "agent_valid_alpha.csv",
        )
        benchmark = benchmark_returns(config, bundle.calendar, signal_dates)
        portfolio = run_mean_variance(config, alpha, returns, up, down, alignment, benchmark)
        result = build_result(signal.summary, portfolio.diagnostics)
        validate_result(result, REPO_ROOT / "schemas" / "result.schema.json")
        print({
            "train_samples": len(train_sampler),
            "train_valid_samples": len(train_valid_sampler),
            "agent_valid_samples": len(agent_valid_sampler),
            "periods": result["n_periods"],
            "ic": result["ic"],
            "sharpe": result["sharpe_ratio"],
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
