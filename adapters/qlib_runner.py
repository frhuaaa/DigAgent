from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import qlib
from qlib.contrib.data.handler import Alpha158DL
from qlib.data import D
from qlib.data.dataset import DatasetH, TSDataSampler, TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import QlibDataLoader
from qlib.data.dataset.processor import CSZScoreNorm, DropnaLabel, Fillna, ProcessInf, ZScoreNorm

from core.errors import ContractError, DataCoverageError, IsolationError
from core.split_guard import aligned_trade_return_dates, purged_signal_dates, split_window


RAW_LABEL_EXPRESSION = "Ref($close, -2) / Ref($close, -1) - 1"


class ContractTSDatasetH(TSDatasetH):
    """Qlib TSDatasetH with the frozen ffill+bfill sampler behavior."""

    def _prepare_seg(self, slc: slice, **kwargs) -> TSDataSampler:
        dtype = kwargs.pop("dtype", None)
        if not isinstance(slc, slice):
            slc = slice(*slc)
        flt_col = kwargs.pop("flt_col", None)
        if flt_col is None:
            flt_col = self.flt_col
        extended = self._extend_slice(slc, self.cal, self.step_len)
        data = DatasetH._prepare_seg(self, extended, **kwargs)
        filter_kwargs = deepcopy(kwargs)
        if flt_col is not None:
            filter_kwargs["col_set"] = flt_col
            filter_data = DatasetH._prepare_seg(self, extended, **filter_kwargs)
            if len(filter_data.columns) != 1:
                raise ContractError("TSDATASET_FILTER_COLUMN_COUNT")
        else:
            filter_data = None
        return TSDataSampler(
            data=data,
            start=slc.start,
            end=slc.stop,
            step_len=self.step_len,
            fillna_type="ffill+bfill",
            dtype=dtype,
            flt_data=filter_data,
        )


@dataclass
class QlibBundle:
    dataset: ContractTSDatasetH
    handler: DataHandlerLP
    calendar: list[pd.Timestamp]
    instruments: dict[str, Any]
    config: dict

    def prepare(self, split: str, researcher: bool = False) -> tuple[TSDataSampler, pd.Series]:
        if split == "test" and not researcher:
            raise IsolationError("ADAPTIVE_TEST_SPLIT_DENIED")
        window = split_window(self.config, split, researcher=researcher)
        purged = purged_signal_dates(self.calendar, window)
        if not purged:
            raise DataCoverageError(f"NO_PURGED_SIGNAL_DATES: {split}")
        sampler = self.dataset.prepare(
            split,
            col_set="__all",
            data_key=DataHandlerLP.DK_L,
            dtype=np.float32,
        )
        allowed = set(purged)
        indexes = sampler.get_index()
        keep = [index for index, pair in enumerate(indexes) if pd.Timestamp(pair[0]).normalize() in allowed]
        if len(keep) != len(indexes):
            sampler.idx_map = sampler.idx_map[np.asarray(keep, dtype=int)]
            sampler.data_index = sampler.data_index[np.asarray(keep, dtype=int)]
        raw = self.handler.fetch(
            selector=slice(window.start, window.end),
            col_set=["raw_label"],
            data_key=DataHandlerLP.DK_R,
        )
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0]
        raw = raw.rename("raw_label").reindex(sampler.get_index())
        return sampler, raw


def initialize_qlib(repo_root: Path, config: dict) -> None:
    provider = (repo_root / config["task"]["provider_uri"]).resolve()
    if not provider.is_dir():
        raise DataCoverageError(f"QLIB_PROVIDER_MISSING: {provider}")
    qlib.init(provider_uri=str(provider), region="cn", kernels=1)


def _feature_file(repo_root: Path, config: dict, instrument: str, field: str = "close") -> Path:
    provider = (repo_root / config["task"]["provider_uri"]).resolve()
    return provider / "features" / instrument.lower() / f"{field}.day.bin"


def resolve_market_instruments(repo_root: Path, config: dict) -> tuple[dict[str, Any], list[str]]:
    task = config["task"]
    market = D.instruments(market=task["instruments"])
    spans = D.list_instruments(
        market,
        start_time=task["train_start_time"],
        end_time=task["test_end_time"],
        as_list=False,
    )
    available: dict[str, Any] = {}
    unavailable: list[str] = []
    required_fields = ("open", "high", "low", "close", "vwap", "volume")
    for instrument, intervals in sorted(spans.items()):
        paths = [_feature_file(repo_root, config, instrument, field) for field in required_fields]
        if all(path.is_file() and path.stat().st_size > 0 for path in paths):
            available[instrument] = intervals
        else:
            unavailable.append(instrument)
    if not available:
        raise DataCoverageError(f"QLIB_MARKET_HAS_NO_AVAILABLE_INSTRUMENTS: {task['instruments']}")
    return available, unavailable


def resolve_feature_catalog(pool_name: str) -> list[dict[str, str]]:
    if pool_name != "Alpha158":
        raise ContractError(f"UNSUPPORTED_FEATURE_POOL: {pool_name}")
    expressions, names = Alpha158DL.get_feature_config()
    if len(expressions) != len(names) or len(set(names)) != len(names) or len(set(expressions)) != len(expressions):
        raise ContractError("FEATURE_CATALOG_NOT_UNIQUE")
    catalog = []
    for name, expression in zip(names, expressions):
        category = "kbar" if name.startswith("K") else "price" if name in {"OPEN0", "HIGH0", "LOW0", "VWAP0"} else "rolling"
        catalog.append({"id": name, "expression": expression, "category": category})
    return catalog


def build_bundle(repo_root: Path, config: dict) -> QlibBundle:
    if config.get("pipeline_contract") != "qlib_ts_lstm_v1":
        raise ContractError("UNSUPPORTED_PIPELINE_CONTRACT")
    if config["task"]["target"] != RAW_LABEL_EXPRESSION:
        raise ContractError("RAW_LABEL_EXPRESSION_FROZEN")
    initialize_qlib(repo_root, config)
    instruments, _ = resolve_market_instruments(repo_root, config)
    task = config["task"]
    features = list(config["z_alpha"]["selected_features"])
    feature_names = [f"FEATURE_{index:03d}" for index in range(len(features))]
    loader = QlibDataLoader(
        config={
            "feature": (features, feature_names),
            "label": ([RAW_LABEL_EXPRESSION], ["LABEL0"]),
            "raw_label": ([RAW_LABEL_EXPRESSION], ["RAW_LABEL0"]),
        },
        freq="day",
    )
    handler = DataHandlerLP(
        instruments=instruments,
        start_time=task["runtime_handler_start_time"],
        end_time=task["test_end_time"],
        data_loader=loader,
        infer_processors=[
            ProcessInf(),
            ZScoreNorm(
                fit_start_time=task["train_start_time"],
                fit_end_time=task["train_end_time"],
                fields_group="feature",
            ),
            Fillna(fields_group="feature", fill_value=0),
        ],
        learn_processors=[DropnaLabel(fields_group="label"), CSZScoreNorm(fields_group="label")],
        process_type=DataHandlerLP.PTYPE_A,
        drop_raw=False,
    )
    segments = {
        name: (task[f"{name}_start_time"], task[f"{name}_end_time"])
        for name in ("train", "valid", "test")
    }
    dataset = ContractTSDatasetH(handler=handler, segments=segments, step_len=int(config["z_model"]["sequence_window"]))
    calendar = [pd.Timestamp(item).normalize() for item in D.calendar(start_time=task["runtime_handler_start_time"], end_time=task["test_end_time"], freq="day")]
    return QlibBundle(dataset, handler, calendar, instruments, config)


def instrument_to_panel_code(instrument: str) -> str:
    if instrument.startswith("SZ") and len(instrument) == 8:
        return f"{instrument[2:]}_XSHE"
    if instrument.startswith("SH") and len(instrument) == 8:
        return f"{instrument[2:]}_XSHG"
    raise ContractError(f"UNKNOWN_QLIB_INSTRUMENT_PREFIX: {instrument}")


def write_alpha_panel(
    prediction: pd.Series,
    signal_dates: list[pd.Timestamp],
    reference_columns: list[str],
    output_path: Path,
) -> pd.DataFrame:
    if not isinstance(prediction.index, pd.MultiIndex):
        raise ContractError("PREDICTION_INDEX_NOT_MULTIINDEX")
    long = prediction.rename("score").reset_index()
    long["panel_code"] = long["instrument"].map(instrument_to_panel_code)
    if long.duplicated(["datetime", "panel_code"]).any():
        raise ContractError("PREDICTION_PANEL_COLLISION")
    unknown = sorted(set(long["panel_code"]) - set(reference_columns))
    finite_unknown = long[long["panel_code"].isin(unknown)]["score"].replace([np.inf, -np.inf], np.nan).notna()
    if finite_unknown.any():
        raise DataCoverageError(f"PREDICTION_ASSET_NOT_IN_RETURN_HEADER: {unknown[:5]}")
    wide = long.pivot(index="datetime", columns="panel_code", values="score")
    wide.index = pd.to_datetime(wide.index).normalize()
    wide = wide.reindex(index=pd.DatetimeIndex(signal_dates), columns=reference_columns)
    wide.index = pd.Index(wide.index.strftime("%Y%m%d"), name=None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(output_path, encoding="utf-8", na_rep="", index=True)
    validate_alpha_panel(output_path, signal_dates, reference_columns)
    wide.index = pd.to_datetime(wide.index, format="%Y%m%d")
    wide.index.name = "date"
    return wide


def validate_alpha_panel(path: Path, signal_dates: list[pd.Timestamp], reference_columns: list[str]) -> None:
    first_line = path.open("r", encoding="utf-8").readline().rstrip("\r\n")
    if not first_line.startswith(","):
        raise ContractError("ALPHA_INDEX_HEADER_NOT_UNNAMED")
    frame = pd.read_csv(path, index_col=0)
    if list(frame.columns) != reference_columns:
        raise ContractError("ALPHA_CANONICAL_HEADER_MISMATCH")
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ContractError("ALPHA_PANEL_DUPLICATES")
    expected = [date.strftime("%Y%m%d") for date in signal_dates]
    actual = [str(item) for item in frame.index]
    if actual != expected:
        raise ContractError("ALPHA_SIGNAL_DATES_MISMATCH")


def benchmark_returns(config: dict, calendar: list[pd.Timestamp], signal_dates: list[pd.Timestamp]) -> pd.Series:
    alignment = aligned_trade_return_dates(calendar, signal_dates)
    task = config["task"]
    closes = D.features(
        [task["benchmark"]],
        ["$close"],
        start_time=min(signal_dates),
        end_time=max(value[1] for value in alignment.values()),
        freq="day",
    )
    if closes.empty:
        raise DataCoverageError(f"BENCHMARK_DATA_MISSING: {task['benchmark']}")
    close_series = closes.iloc[:, 0]
    close_series.index = close_series.index.get_level_values("datetime")
    rows = {}
    for signal, (trade, realized) in alignment.items():
        if trade not in close_series.index or realized not in close_series.index:
            raise DataCoverageError(f"BENCHMARK_ALIGNMENT_MISSING: {signal.date()}")
        denominator = float(close_series.loc[trade])
        numerator = float(close_series.loc[realized])
        if not np.isfinite(denominator) or denominator == 0 or not np.isfinite(numerator):
            raise DataCoverageError(f"BENCHMARK_CLOSE_INVALID: {signal.date()}")
        rows[signal] = numerator / denominator - 1.0
    return pd.Series(rows, name="benchmark_return").sort_index()
