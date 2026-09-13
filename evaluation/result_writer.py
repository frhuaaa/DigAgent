from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import jsonschema
import numpy as np
import pandas as pd

from core.errors import ContractError
from core.io_utils import atomic_write_bytes


RESULT_KEYS = (
    "ic", "icir", "rank_ic", "rank_icir", "final_nav", "total_return",
    "annual_return", "annual_volatility", "max_drawdown", "sharpe_ratio",
    "calmar_ratio", "sortino_ratio", "win_rate", "excess_annual_return",
    "excess_annual_volatility", "excess_max_drawdown", "information_ratio",
    "excess_calmar_ratio", "excess_sortino_ratio", "excess_win_rate",
    "sell_turn_mean", "buy_turn_mean", "one_way_turnover_mean",
    "holding_count_mean", "optimizer_universe_count_mean", "invested_weight_mean",
    "n_periods",
)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= 1e-20:
        return None
    value = numerator / denominator
    return float(value) if np.isfinite(value) else None


def _series_metrics(values: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    n = len(values)
    if n == 0:
        return {name: None for name in ("final_nav", "total_return", "annual_return", "annual_volatility", "max_drawdown", "sharpe", "calmar", "sortino", "win_rate")}
    nav = (1.0 + values).cumprod()
    final_nav = float(nav.iloc[-1])
    total_return = final_nav - 1.0
    annual_return = float(final_nav ** (252.0 / n) - 1.0) if final_nav >= 0 else None
    std = float(values.std(ddof=1)) if n > 1 else float("nan")
    annual_volatility = float(std * math.sqrt(252.0)) if np.isfinite(std) else None
    sharpe = _safe_ratio(float(values.mean()) * math.sqrt(252.0), std)
    drawdown = nav / nav.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = _safe_ratio(float(annual_return) if annual_return is not None else float("nan"), abs(max_drawdown))
    downside = np.minimum(values.to_numpy(), 0.0)
    downside_dev = float(np.sqrt(np.mean(downside ** 2)))
    sortino = _safe_ratio(float(values.mean()) * math.sqrt(252.0), downside_dev)
    return {
        "final_nav": final_nav,
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "calmar": calmar,
        "sortino": sortino,
        "win_rate": float((values > 0).sum() / n),
    }


def build_result(signal_metrics: Mapping[str, float | None], diagnostics: pd.DataFrame) -> OrderedDict:
    required = {
        "net_return", "benchmark_return", "sell_turn", "buy_turn", "one_way_turn",
        "holding_count", "optimizer_universe_count", "invested_weight",
    }
    missing = sorted(required - set(diagnostics.columns))
    if missing:
        raise ContractError(f"RESULT_DIAGNOSTIC_FIELDS_MISSING: {missing}")
    finite = diagnostics.replace([np.inf, -np.inf], np.nan).dropna(subset=["net_return", "benchmark_return"])
    strategy = _series_metrics(finite["net_return"])
    excess = _series_metrics(finite["net_return"] - finite["benchmark_return"])
    raw = {
        **{name: signal_metrics.get(name) for name in ("ic", "icir", "rank_ic", "rank_icir")},
        "final_nav": strategy["final_nav"],
        "total_return": strategy["total_return"],
        "annual_return": strategy["annual_return"],
        "annual_volatility": strategy["annual_volatility"],
        "max_drawdown": strategy["max_drawdown"],
        "sharpe_ratio": strategy["sharpe"],
        "calmar_ratio": strategy["calmar"],
        "sortino_ratio": strategy["sortino"],
        "win_rate": strategy["win_rate"],
        "excess_annual_return": excess["annual_return"],
        "excess_annual_volatility": excess["annual_volatility"],
        "excess_max_drawdown": excess["max_drawdown"],
        "information_ratio": excess["sharpe"],
        "excess_calmar_ratio": excess["calmar"],
        "excess_sortino_ratio": excess["sortino"],
        "excess_win_rate": excess["win_rate"],
        "sell_turn_mean": finite["sell_turn"].mean(),
        "buy_turn_mean": finite["buy_turn"].mean(),
        "one_way_turnover_mean": finite["one_way_turn"].mean(),
        "holding_count_mean": finite["holding_count"].mean(),
        "optimizer_universe_count_mean": finite["optimizer_universe_count"].mean(),
        "invested_weight_mean": finite["invested_weight"].mean(),
        "n_periods": int(len(finite)),
    }
    result = OrderedDict()
    for key in RESULT_KEYS:
        value = raw[key]
        if key == "n_periods":
            result[key] = int(value) if value is not None else None
        elif value is None or not np.isfinite(float(value)):
            result[key] = None
        else:
            result[key] = round(float(value), 3)
    return result


def validate_result(result: Mapping, schema_path: Path) -> None:
    if tuple(result.keys()) != RESULT_KEYS:
        raise ContractError("RESULT_KEYS_MISMATCH")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(dict(result))


def serialize_result(result: Mapping) -> bytes:
    """Emit JSON numbers with exactly three fractional digits."""
    if tuple(result.keys()) != RESULT_KEYS:
        raise ContractError("RESULT_KEYS_MISMATCH")
    lines = ["{"]
    for index, key in enumerate(RESULT_KEYS):
        value = result[key]
        if value is None:
            token = "null"
        elif key == "n_periods":
            if isinstance(value, bool) or int(value) != value:
                raise ContractError("RESULT_N_PERIODS_NOT_INTEGER")
            token = str(int(value))
        else:
            number = float(value)
            if not math.isfinite(number):
                raise ContractError(f"RESULT_NONFINITE: {key}")
            token = f"{number:.3f}"
        suffix = "," if index < len(RESULT_KEYS) - 1 else ""
        lines.append(f"  {json.dumps(key)}: {token}{suffix}")
    lines.append("}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_result(result: Mapping, path: Path, schema_path: Path) -> None:
    validate_result(result, schema_path)
    atomic_write_bytes(path, serialize_result(result))


def _report_value(value: object, *, percent: bool = False) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "N/A"
    return f"{number * 100.0:.3f}%" if percent else f"{number:.3f}"


def format_portfolio_report(experiment_id: str, result: Mapping) -> str:
    """Build an additive human-readable validation report without changing result.json."""
    net = (
        ("ARR", "annual_return", True),
        ("Vol", "annual_volatility", True),
        ("MDD", "max_drawdown", True),
        ("Sharpe", "sharpe_ratio", False),
        ("Calmar", "calmar_ratio", False),
        ("Sortino", "sortino_ratio", False),
        ("Win rate", "win_rate", True),
    )
    excess = (
        ("ARR", "excess_annual_return", True),
        ("Vol", "excess_annual_volatility", True),
        ("MDD", "excess_max_drawdown", True),
        ("Sharpe (IR)", "information_ratio", False),
        ("Calmar", "excess_calmar_ratio", False),
        ("Sortino", "excess_sortino_ratio", False),
        ("Win rate", "excess_win_rate", True),
    )

    def row(spec: tuple[tuple[str, str, bool], ...]) -> list[str]:
        return [
            " | ".join(label for label, _, _ in spec),
            " | ".join(_report_value(result.get(key), percent=percent) for _, key, percent in spec),
        ]

    lines = [f"=== {experiment_id} validation portfolio report ===", "Net return"]
    lines.extend(row(net))
    lines.extend(["Excess return", *row(excess)])
    lines.extend([
        "Turnover",
        "Sell | Buy | One-way",
        " | ".join(_report_value(result.get(key), percent=True) for key in (
            "sell_turn_mean", "buy_turn_mean", "one_way_turnover_mean",
        )),
        "Other metrics",
        "IC | ICIR | Rank IC | Rank ICIR | Final NAV | Total return | Avg holdings | Avg universe | Invested | Periods",
        " | ".join((
            _report_value(result.get("ic")),
            _report_value(result.get("icir")),
            _report_value(result.get("rank_ic")),
            _report_value(result.get("rank_icir")),
            _report_value(result.get("final_nav")),
            _report_value(result.get("total_return"), percent=True),
            _report_value(result.get("holding_count_mean")),
            _report_value(result.get("optimizer_universe_count_mean")),
            _report_value(result.get("invested_weight_mean"), percent=True),
            str(result.get("n_periods")) if result.get("n_periods") is not None else "N/A",
        )),
    ])
    return "\n".join(lines)


def print_portfolio_report(experiment_id: str, result: Mapping) -> None:
    print(format_portfolio_report(experiment_id, result), flush=True)
