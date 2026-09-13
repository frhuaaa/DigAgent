from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from core.errors import ContractError, DataCoverageError, IsolationError


@dataclass(frozen=True)
class SplitWindow:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


def split_window(config: dict, split: str, researcher: bool = False) -> SplitWindow:
    if split not in {"train", "valid", "test"}:
        raise ContractError(f"UNKNOWN_SPLIT: {split}")
    if split == "test" and not researcher:
        raise IsolationError("ADAPTIVE_TEST_SPLIT_DENIED")
    task = config["task"]
    return SplitWindow(split, pd.Timestamp(task[f"{split}_start_time"]), pd.Timestamp(task[f"{split}_end_time"]))


def purged_signal_dates(calendar: Iterable, window: SplitWindow) -> list[pd.Timestamp]:
    dates = [pd.Timestamp(item).normalize() for item in calendar]
    positions = {date: index for index, date in enumerate(dates)}
    selected: list[pd.Timestamp] = []
    for date in dates:
        if date < window.start or date > window.end:
            continue
        index = positions[date]
        if index + 2 >= len(dates):
            continue
        if dates[index + 1] > window.end or dates[index + 2] > window.end:
            continue
        selected.append(date)
    return selected


def aligned_trade_return_dates(calendar: Iterable, signal_dates: Iterable[pd.Timestamp]) -> dict[pd.Timestamp, tuple[pd.Timestamp, pd.Timestamp]]:
    dates = [pd.Timestamp(item).normalize() for item in calendar]
    positions = {date: index for index, date in enumerate(dates)}
    result = {}
    for signal in signal_dates:
        signal = pd.Timestamp(signal).normalize()
        index = positions.get(signal)
        if index is None or index + 2 >= len(dates):
            raise DataCoverageError(f"ALIGNMENT_DATE_MISSING: {signal.date()}")
        result[signal] = (dates[index + 1], dates[index + 2])
    return result


def validate_panel_coverage(
    signal_dates: Iterable[pd.Timestamp],
    alignment: dict[pd.Timestamp, tuple[pd.Timestamp, pd.Timestamp]],
    return_dates: Iterable,
    limit_up_dates: Iterable,
    limit_down_dates: Iterable,
) -> None:
    returns = {pd.Timestamp(item).normalize() for item in return_dates}
    ups = {pd.Timestamp(item).normalize() for item in limit_up_dates}
    downs = {pd.Timestamp(item).normalize() for item in limit_down_dates}
    for signal in signal_dates:
        trade, realized = alignment[pd.Timestamp(signal).normalize()]
        if trade not in returns or realized not in returns:
            raise DataCoverageError(f"RETURN_PANEL_COVERAGE_MISSING: signal={signal.date()} trade={trade.date()} return={realized.date()}")
        if trade not in ups:
            raise DataCoverageError(f"LIMIT_UP_PANEL_COVERAGE_MISSING: {trade.date()}")
        if trade not in downs:
            raise DataCoverageError(f"LIMIT_DOWN_PANEL_COVERAGE_MISSING: {trade.date()}")

