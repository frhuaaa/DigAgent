from __future__ import annotations

import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from backbone import ALSTMModel, GRUModel, LSTMModel
from core.errors import ContractError
from core.io_utils import atomic_write_json
from evaluation.prediction_metrics import SignalMetricResult, compute_signal_metrics


ResearcherDispatch = Callable[[int, Path, dict, dict], None]
TRAINING_METRIC_DECIMALS = 4


def persisted_metric(value: object) -> float | None:
    """Round a finite training metric for artifacts without changing internal math."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, TRAINING_METRIC_DECIMALS) if np.isfinite(number) else None


def persisted_metric_summary(values: dict) -> dict:
    return {key: persisted_metric(value) for key, value in values.items()}


def persisted_epoch_record(record: dict) -> dict:
    return {
        **record,
        "training_loss": persisted_metric(record.get("training_loss")),
        "train": persisted_metric_summary(record.get("train", {})),
        "valid": persisted_metric_summary(record.get("valid", {})),
    }


class SamplerDataset(Dataset):
    def __init__(self, sampler, feature_count: int):
        self.sampler = sampler
        self.feature_count = feature_count

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, index: int):
        sample = np.asarray(self.sampler[index], dtype=np.float32)
        features = sample[:, : self.feature_count]
        target = sample[-1, self.feature_count]
        return torch.from_numpy(features), torch.tensor(target, dtype=torch.float32), index


class FixedOrderSampler(Sampler[int]):
    def __init__(self, order: np.ndarray):
        self.order = [int(item) for item in order]

    def __iter__(self) -> Iterator[int]:
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)


@dataclass(frozen=True)
class TrainingOutcome:
    selected_checkpoint: Path
    selected_epoch: int
    selected_score: float | None
    history: list[dict]


def select_checkpoint_epoch(history: list[dict], metric: str | None = None) -> tuple[int, float | None]:
    metric = metric or "ic"
    if metric not in {"ic", "icir", "rank_ic", "rank_icir"}:
        raise ContractError("CHECKPOINT_METRIC_INVALID")
    if not history:
        raise ContractError("NO_CHECKPOINT_HISTORY")
    best_epoch = int(history[0]["epoch"])
    best_score = -math.inf
    for record in history:
        value = record["valid"].get(metric)
        score = float(value) if value is not None and np.isfinite(value) else -math.inf
        if score > best_score:
            best_epoch = int(record["epoch"])
            best_score = score
    return best_epoch, None if best_score == -math.inf else best_score


def checkpoint_score(value: object) -> float:
    """Normalize a configured validation metric for strict best-checkpoint selection."""
    if value is None:
        return -math.inf
    try:
        score = float(value)
    except (TypeError, ValueError):
        return -math.inf
    return score if np.isfinite(score) else -math.inf


def is_checkpoint_improvement(value: object, best_score: float) -> bool:
    """Return True only for a finite validation score strictly above the incumbent."""
    return checkpoint_score(value) > best_score


def _checkpoint_payload(model: nn.Module, config: dict, epoch: int, feature_count: int) -> dict:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "model_name": config["z_model"]["model_name"],
        "d_feat": feature_count,
        "config_hash": config["provenance"]["effective_config_hash"],
    }


def _atomic_torch_save(payload: dict, destination: Path) -> None:
    """Replace the one persistent best checkpoint without exposing a partial file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _transient_checkpoint(payload: dict, epoch: int) -> Path:
    """Create an ephemeral hand-off for the synchronous isolated epoch evaluator."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"diagagent_epoch_{epoch:03d}_",
        suffix=".pt",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(config: dict) -> nn.Module:
    model_name = config["z_model"]["model_name"]
    train = config["z_model"]["train_params"]
    params = config["z_model"]["model_params"]
    common = {
        "d_feat": len(config["z_alpha"]["selected_features"]),
        "hidden_size": int(train["hidden_size"]),
        "num_layers": int(params["model_layer"]),
        "dropout": float(train["dropout"]),
        "use_pe": bool(params.get("use_pe", False)),
        "use_bn": bool(params.get("use_bn", False)),
    }
    if model_name == "lstm":
        return LSTMModel(**common, use_ln=bool(params.get("use_ln", False)))
    if model_name == "gru":
        return GRUModel(**common, use_ln=bool(params.get("use_ln", False)))
    if model_name == "alstm":
        return ALSTMModel(**common, attention_hidden_size=int(params["attention_hidden_size"]))
    raise ContractError(f"UNSUPPORTED_MODEL_NAME: {model_name}")


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_eval_loader(dataset: Dataset, config: dict) -> DataLoader:
    base = config["z_model"]["base_params"]
    return DataLoader(
        dataset,
        batch_size=int(base["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(base["n_jobs"]),
        worker_init_fn=_worker_seed,
    )


def _predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    indexes: list[np.ndarray] = []
    with torch.no_grad():
        for features, _, batch_indexes in loader:
            output = model(features.to(device)).reshape(-1)
            predictions.append(output.detach().cpu().numpy())
            indexes.append(batch_indexes.detach().cpu().numpy())
    if not predictions:
        return np.array([], dtype=float), np.array([], dtype=int)
    return np.concatenate(predictions), np.concatenate(indexes).astype(int)


def evaluate_model(
    model: nn.Module,
    sampler,
    raw_label: pd.Series,
    feature_count: int,
    config: dict,
    device: torch.device,
) -> tuple[pd.Series, SignalMetricResult]:
    dataset = SamplerDataset(sampler, feature_count)
    values, positions = _predict_loader(model, make_eval_loader(dataset, config), device)
    index = sampler.get_index()[positions]
    prediction = pd.Series(values, index=index, name="prediction")
    metrics = compute_signal_metrics(prediction, raw_label.reindex(index))
    return prediction, metrics


def train_model(
    config: dict,
    train_sampler,
    train_raw_label: pd.Series,
    valid_sampler,
    valid_raw_label: pd.Series,
    experiment_dir: Path,
    researcher_dispatch: ResearcherDispatch,
) -> TrainingOutcome:
    seed = int(config["task"]["seed"])
    seed_everything(seed)
    device = torch.device(config["runtime"]["resolved_device"])
    model = build_model(config).to(device)
    base = config["z_model"]["base_params"]
    train_params = config["z_model"]["train_params"]
    if base["loss"] != "mse" or base["optimizer"] != "adam":
        raise ContractError("TRAINING_LOSS_OR_OPTIMIZER_FROZEN")
    metric_name = base.get("metric", "ic")
    if metric_name not in {"ic", "icir", "rank_ic", "rank_icir"}:
        raise ContractError("CHECKPOINT_METRIC_INVALID")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_params["lr"]),
        weight_decay=float(train_params["weight_decay"]),
    )
    loss_function = nn.MSELoss()
    feature_count = len(config["z_alpha"]["selected_features"])
    train_dataset = SamplerDataset(train_sampler, feature_count)
    valid_dataset = SamplerDataset(valid_sampler, feature_count)
    artifact_dir = experiment_dir / "artifacts"
    checkpoint_dir = artifact_dir / "checkpoints"
    log_dir = experiment_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    for stale_checkpoint in checkpoint_dir.glob("*.pt"):
        stale_checkpoint.unlink()
    index_payload = {
        "train": [[pd.Timestamp(date).strftime("%Y-%m-%d"), str(instrument)] for date, instrument in train_sampler.get_index()],
        "valid": [[pd.Timestamp(date).strftime("%Y-%m-%d"), str(instrument)] for date, instrument in valid_sampler.get_index()],
        "loader_contract": {
            "training": {"shuffle": True, "drop_last": True, "batch_size": int(base["batch_size"]), "n_jobs": int(base["n_jobs"])},
            "evaluation": {"shuffle": False, "drop_last": False, "batch_size": int(base["batch_size"]), "n_jobs": int(base["n_jobs"])},
        },
    }
    atomic_write_json(artifact_dir / "split_indices.json", index_payload)

    history: list[dict] = []
    best_score = -math.inf
    best_epoch = -1
    best_checkpoint = checkpoint_dir / "best.pt"
    has_best_checkpoint = False
    no_improvement = 0
    metrics_log = log_dir / "training_metrics.jsonl"
    with metrics_log.open("w", encoding="utf-8", newline="\n") as log_handle:
        for epoch in range(int(base["n_epochs"])):
            order_generator = torch.Generator()
            order_generator.manual_seed(seed + epoch)
            order = torch.randperm(len(train_dataset), generator=order_generator).numpy()
            np.save(log_dir / f"train_sample_order_epoch_{epoch:03d}.npy", order, allow_pickle=False)
            train_loader = DataLoader(
                train_dataset,
                batch_size=int(base["batch_size"]),
                sampler=FixedOrderSampler(order),
                drop_last=True,
                num_workers=int(base["n_jobs"]),
                worker_init_fn=_worker_seed,
            )
            model.train()
            losses: list[float] = []
            for features, target, _ in train_loader:
                features = features.to(device)
                target = target.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(features).reshape(-1)
                loss = loss_function(prediction, target)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            if not losses:
                raise ContractError("TRAINING_DROP_LAST_REMOVED_ALL_BATCHES")
            _, train_metrics = evaluate_model(model, train_sampler, train_raw_label, feature_count, config, device)
            _, valid_metrics = evaluate_model(model, valid_sampler, valid_raw_label, feature_count, config, device)
            record = {
                "epoch": epoch,
                "training_loss": float(np.mean(losses)),
                "train": train_metrics.summary,
                "valid": valid_metrics.summary,
                "excluded_dates": {"train": train_metrics.excluded, "valid": valid_metrics.excluded},
            }
            history.append(record)
            log_handle.write(json.dumps(persisted_epoch_record(record), ensure_ascii=False, separators=(",", ":")) + "\n")
            log_handle.flush()
            score_value = valid_metrics.summary.get(metric_name)
            score = checkpoint_score(score_value)
            improved = is_checkpoint_improvement(score_value, best_score)
            payload = _checkpoint_payload(model, config, epoch, feature_count)
            if improved:
                _atomic_torch_save(payload, best_checkpoint)
                best_score = score
                best_epoch = epoch
                has_best_checkpoint = True
                no_improvement = 0
            else:
                no_improvement += 1
            dispatch_checkpoint = best_checkpoint if improved else _transient_checkpoint(payload, epoch)
            try:
                researcher_dispatch(epoch, dispatch_checkpoint, train_metrics.summary, valid_metrics.summary)
            finally:
                if not improved:
                    dispatch_checkpoint.unlink(missing_ok=True)
            if no_improvement >= int(base["early_stop"]):
                break
        selected_epoch, selected_score = select_checkpoint_epoch(history, metric_name)
        if not has_best_checkpoint or selected_score is None or not best_checkpoint.is_file():
            raise ContractError("NO_FINITE_VALIDATION_METRIC_FOR_CHECKPOINT")
        if selected_epoch != best_epoch or not math.isclose(selected_score, best_score, rel_tol=0.0, abs_tol=0.0):
            raise ContractError("INCREMENTAL_CHECKPOINT_SELECTION_MISMATCH")
        selection = {
            "event": "checkpoint_selected",
            "metric": f"valid.{metric_name}",
            "mode": "max",
            "epoch": best_epoch,
            "score": None if best_score == -math.inf else best_score,
        }
        persisted_selection = {**selection, "score": persisted_metric(selection["score"])}
        log_handle.write(json.dumps(persisted_selection, ensure_ascii=False, separators=(",", ":")) + "\n")
    atomic_write_json(artifact_dir / "selected_checkpoint.json", {**persisted_selection, "path": best_checkpoint.relative_to(experiment_dir).as_posix()})
    return TrainingOutcome(best_checkpoint, best_epoch, None if best_score == -math.inf else best_score, history)


def load_model_from_checkpoint(config: dict, checkpoint: Path) -> nn.Module:
    device = torch.device(config["runtime"]["resolved_device"])
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model
