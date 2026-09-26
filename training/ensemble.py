from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from core.errors import ContractError


ENSEMBLE_METHOD = "cross_sectional_zscore_mean"


def ensemble_seeds(config: dict) -> tuple[int, ...]:
    settings = config.get("ensemble", {})
    if settings.get("enabled") is not True:
        return (int(config["task"]["seed"]),)
    seeds = settings.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 2:
        raise ContractError("ENSEMBLE_SEEDS_INVALID")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ContractError("ENSEMBLE_SEEDS_INVALID")
    if len(set(seeds)) != len(seeds):
        raise ContractError("ENSEMBLE_SEEDS_DUPLICATED")
    if settings.get("method") != ENSEMBLE_METHOD:
        raise ContractError("ENSEMBLE_METHOD_INVALID")
    return tuple(seeds)


def normalize_prediction_cross_sections(prediction: pd.Series) -> pd.Series:
    """Winsorize and z-score every dated cross-section without changing its index."""
    if not isinstance(prediction.index, pd.MultiIndex):
        raise ContractError("PREDICTION_INDEX_NOT_MULTIINDEX")
    if prediction.index.has_duplicates:
        raise ContractError("PREDICTION_INDEX_DUPLICATED")
    values = pd.to_numeric(prediction, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if values.isna().any():
        raise ContractError("ENSEMBLE_PREDICTION_NONFINITE")

    def normalize(group: pd.Series) -> pd.Series:
        lower, upper = group.quantile([0.01, 0.99])
        winsorized = group.clip(lower=float(lower), upper=float(upper))
        std = float(winsorized.std(ddof=0))
        if not np.isfinite(std) or std <= 1e-12:
            return pd.Series(0.0, index=group.index, dtype=float)
        return (winsorized - float(winsorized.mean())) / std

    normalized = values.groupby(level="datetime", group_keys=False, sort=False).apply(normalize)
    normalized = normalized.reindex(prediction.index)
    normalized.name = "prediction"
    return normalized


def ensemble_predictions(predictions: Iterable[pd.Series]) -> pd.Series:
    members = list(predictions)
    if len(members) < 2:
        raise ContractError("ENSEMBLE_REQUIRES_MULTIPLE_MEMBERS")
    reference_index = members[0].index
    if any(not member.index.equals(reference_index) for member in members[1:]):
        raise ContractError("ENSEMBLE_PREDICTION_INDEX_MISMATCH")
    normalized = [normalize_prediction_cross_sections(member) for member in members]
    frame = pd.concat(normalized, axis=1, keys=range(len(normalized)))
    if frame.isna().any().any():
        raise ContractError("ENSEMBLE_PREDICTION_ALIGNMENT_MISSING")
    averaged = frame.mean(axis=1)
    averaged.name = "prediction"
    return normalize_prediction_cross_sections(averaged)


def prediction_to_frame(prediction: pd.Series) -> pd.DataFrame:
    if not isinstance(prediction.index, pd.MultiIndex):
        raise ContractError("PREDICTION_INDEX_NOT_MULTIINDEX")
    frame = prediction.rename("prediction").reset_index()
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.strftime("%Y-%m-%d")
    return frame


def prediction_from_frame(frame: pd.DataFrame) -> pd.Series:
    required = ["datetime", "instrument", "prediction"]
    if list(frame.columns) != required:
        raise ContractError("ENSEMBLE_PREDICTION_COLUMNS_INVALID")
    frame = frame.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    prediction = frame.set_index(["datetime", "instrument"])["prediction"]
    prediction.name = "prediction"
    if prediction.index.has_duplicates:
        raise ContractError("PREDICTION_INDEX_DUPLICATED")
    return prediction
