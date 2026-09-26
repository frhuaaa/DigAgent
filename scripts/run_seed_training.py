from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.qlib_runner import build_bundle
from core.errors import ContractError
from core.io_utils import atomic_write_json, load_json, sha256_file, verify_hashes
from core.process_lock import advisory_lock
from core.researcher_boundary import ResearcherBoundary
from training.ensemble import prediction_to_frame
from training.model_trainer import evaluate_model, load_model_from_checkpoint, train_model


def run_seed(config_path: Path, seed: int) -> Path:
    experiment_dir = config_path.resolve().parent
    artifact_dir = experiment_dir / "artifacts"
    log_dir = experiment_dir / "logs"
    marker_path = log_dir / f"seed_{seed}_complete.json"
    lock_path = log_dir / f"seed_{seed}_training.lock"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    with advisory_lock(lock_path):
        if marker_path.is_file():
            marker = load_json(marker_path)
            verify_hashes(REPO_ROOT, marker["output_hashes"])
            return marker_path

        config = load_json(config_path)
        configured = config.get("ensemble", {}).get("seeds", [])
        if seed not in configured:
            raise ContractError("SEED_NOT_IN_FROZEN_ENSEMBLE")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ContractError("ENSEMBLE_SEED_WORKER_REQUIRES_ONE_VISIBLE_GPU")
        config["runtime"]["resolved_device"] = "cuda:0"
        namespace = f"seed_{seed}"
        boundary = ResearcherBoundary(
            REPO_ROOT,
            config_path,
            log_dir / f"researcher_dispatch_{namespace}.jsonl",
            seed=seed,
        )
        bundle = build_bundle(REPO_ROOT, config)
        train_sampler, train_raw = bundle.prepare("train", researcher=False)
        train_valid_sampler, train_valid_raw = bundle.prepare("train_valid", researcher=False)
        try:
            outcome = train_model(
                config,
                train_sampler,
                train_raw,
                train_valid_sampler,
                train_valid_raw,
                experiment_dir,
                boundary.epoch,
                seed=seed,
                artifact_namespace=namespace,
            )
        except Exception:
            boundary.close_epoch_worker(force=True)
            raise
        finally:
            boundary.close_epoch_worker()

        model = load_model_from_checkpoint(config, outcome.selected_checkpoint)
        agent_valid_sampler, agent_valid_raw = bundle.prepare("agent_valid", researcher=False)
        prediction, signal_metrics = evaluate_model(
            model,
            agent_valid_sampler,
            agent_valid_raw,
            len(config["z_alpha"]["selected_features"]),
            config,
            torch.device("cuda:0"),
        )
        prediction_path = artifact_dir / f"agent_valid_prediction_{namespace}.csv"
        prediction_to_frame(prediction).to_csv(
            prediction_path,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        signal_path = artifact_dir / f"signal_daily_metrics_{namespace}.csv"
        signal_metrics.daily.to_csv(signal_path, encoding="utf-8", lineterminator="\n")
        selected_path = artifact_dir / f"selected_checkpoint_{namespace}.json"
        outputs = [
            outcome.selected_checkpoint,
            prediction_path,
            signal_path,
            selected_path,
            log_dir / f"training_metrics_{namespace}.jsonl",
            artifact_dir / f"split_indices_{namespace}.json",
        ]
        marker = {
            "status": "COMPLETE",
            "seed": seed,
            "selected_epoch": outcome.selected_epoch,
            "selected_score": outcome.selected_score,
            "selected_checkpoint": outcome.selected_checkpoint.relative_to(REPO_ROOT).as_posix(),
            "selected_checkpoint_sha256": sha256_file(outcome.selected_checkpoint),
            "signal_summary": signal_metrics.summary,
            "output_hashes": {
                path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
                for path in outputs
            },
            "contains_test_derived_data": False,
        }
        atomic_write_json(marker_path, marker)
        return marker_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train one frozen DiagAgent ensemble seed.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    print(run_seed(Path(args.config), args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
