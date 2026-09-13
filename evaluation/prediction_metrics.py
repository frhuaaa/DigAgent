from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


METRIC_NAMES = ("ic", "icir", "rank_ic", "rank_icir")


@dataclass(frozen=True)
class SignalMetricResult:
    summary: dict[str, float | None]
    daily: pd.DataFrame
    excluded: dict[str, int]


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _corr_for_date(frame: pd.DataFrame) -> tuple[float, float] | None:
    pair = frame[["prediction", "label"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 2:
        return None
    if pair["prediction"].nunique(dropna=True) < 2 or pair["label"].nunique(dropna=True) < 2:
        return None
    ic = pair["prediction"].corr(pair["label"], method="pearson")
    rank_ic = pair["prediction"].corr(pair["label"], method="spearman")
    if not np.isfinite(ic) or not np.isfinite(rank_ic):
        return None
    return float(ic), float(rank_ic)


def compute_signal_metrics(prediction: pd.Series, raw_label: pd.Series) -> SignalMetricResult:
    """Compute daily cross-sectional IC metrics against unstandardized labels."""
    prediction = prediction.rename("prediction")
    raw_label = raw_label.rename("label")
    joined = pd.concat([prediction, raw_label], axis=1, join="inner")
    if not isinstance(joined.index, pd.MultiIndex) or "datetime" not in joined.index.names:
        raise ValueError("Signal inputs require a MultiIndex containing 'datetime'.")

    rows: list[dict[str, object]] = []
    excluded_too_few = 0
    excluded_constant = 0
    for date, frame in joined.groupby(level="datetime", sort=True):
        pair = frame[["prediction", "label"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 2:
            excluded_too_few += 1
            continue
        if pair["prediction"].nunique() < 2 or pair["label"].nunique() < 2:
            excluded_constant += 1
            continue
        values = _corr_for_date(pair)
        if values is None:
            excluded_constant += 1
            continue
        rows.append({"datetime": pd.Timestamp(date), "ic": values[0], "rank_ic": values[1], "n_pairs": len(pair)})

    daily = pd.DataFrame(rows)
    if not daily.empty:
        daily = daily.set_index("datetime").sort_index()
    else:
        daily = pd.DataFrame(columns=["ic", "rank_ic", "n_pairs"], index=pd.DatetimeIndex([], name="datetime"))

    def aggregate(column: str) -> tuple[float | None, float | None]:
        values = pd.to_numeric(daily[column], errors="coerce").dropna()
        if values.empty:
            return None, None
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        ratio = mean / std if np.isfinite(std) and std > 0 else float("nan")
        return _finite_or_none(mean), _finite_or_none(ratio)

    ic, icir = aggregate("ic")
    rank_ic, rank_icir = aggregate("rank_ic")
    return SignalMetricResult(
        summary={"ic": ic, "icir": icir, "rank_ic": rank_ic, "rank_icir": rank_icir},
        daily=daily,
        excluded={"too_few_pairs": excluded_too_few, "constant_vector": excluded_constant},
    )

