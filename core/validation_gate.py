from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from core.errors import ContractError


Statistic = Callable[[pd.DataFrame], float]


# Bootstrap dispersion can collapse to machine precision when parent and
# candidate are numerically identical.  Keep a deterministic numerical floor
# so floating-point round-off is never interpreted as economic evidence.
NUMERICAL_ABS_TOL = 1e-12
NUMERICAL_REL_TOL = 1e-10


def _numerical_tolerance(parent_value: float, candidate_value: float) -> float:
    scale = max(abs(parent_value), abs(candidate_value))
    return max(NUMERICAL_ABS_TOL, NUMERICAL_REL_TOL * scale)


def _sharpe(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < 2:
        return float("nan")
    std = float(values.std(ddof=1))
    return float(values.mean() / std * math.sqrt(252.0)) if np.isfinite(std) and std > 0 else float("nan")


def _annual_volatility(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < 2:
        return float("nan")
    return float(values.std(ddof=1) * math.sqrt(252.0))


def _drawdown_magnitude(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    nav = (1.0 + values).cumprod()
    return abs(float((nav / nav.cummax() - 1.0).min()))


def _ratio(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < 2:
        return float("nan")
    std = float(values.std(ddof=1))
    return float(values.mean() / std) if np.isfinite(std) and std > 0 else float("nan")


def _aligned(parent: pd.DataFrame, candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "signal_date" in parent.columns:
        parent = parent.set_index(pd.to_datetime(parent["signal_date"])).drop(columns=["signal_date"])
    if "signal_date" in candidate.columns:
        candidate = candidate.set_index(pd.to_datetime(candidate["signal_date"])).drop(columns=["signal_date"])
    common = parent.index.intersection(candidate.index).sort_values()
    return parent.reindex(common), candidate.reindex(common)


def paired_moving_block_epsilon(
    parent: pd.DataFrame,
    candidate: pd.DataFrame,
    statistic: Statistic,
    samples: int = 2000,
    block_length: int = 20,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    parent, candidate = _aligned(parent, candidate)
    n = len(parent)
    observed_parent = statistic(parent)
    observed_candidate = statistic(candidate)
    observed_delta = observed_candidate - observed_parent
    if n < block_length or not np.isfinite(observed_delta):
        return None, None
    numerical_tolerance = _numerical_tolerance(observed_parent, observed_candidate)
    if abs(observed_delta) <= numerical_tolerance:
        observed_delta = 0.0
    rng = np.random.default_rng(seed)
    number_blocks = math.ceil(n / block_length)
    deltas = np.empty(samples, dtype=float)
    max_start = n - block_length
    for sample in range(samples):
        starts = rng.integers(0, max_start + 1, size=number_blocks)
        positions = np.concatenate([np.arange(start, start + block_length) for start in starts])[:n]
        delta = statistic(candidate.iloc[positions]) - statistic(parent.iloc[positions])
        if not np.isfinite(delta):
            return float(observed_delta), None
        deltas[sample] = delta
    bootstrap_epsilon = 1.96 * float(np.std(deltas, ddof=1))
    if not np.isfinite(bootstrap_epsilon):
        return float(observed_delta), None
    effective_epsilon = max(bootstrap_epsilon, numerical_tolerance)
    return float(observed_delta), float(effective_epsilon)


def _mean_column(column: str) -> Statistic:
    return lambda frame: float(pd.to_numeric(frame[column], errors="coerce").mean()) if column in frame else float("nan")


def _metric_frames(
    metric: str,
    parent_diagnostics: pd.DataFrame,
    candidate_diagnostics: pd.DataFrame,
    parent_signal: pd.DataFrame,
    candidate_signal: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Statistic] | None:
    if metric == "sharpe_ratio":
        return parent_diagnostics, candidate_diagnostics, lambda frame: _sharpe(frame["net_return"])
    if metric == "annual_volatility":
        return parent_diagnostics, candidate_diagnostics, lambda frame: _annual_volatility(frame["net_return"])
    if metric == "ic":
        return parent_signal, candidate_signal, _mean_column("ic")
    if metric == "rank_ic":
        return parent_signal, candidate_signal, _mean_column("rank_ic")
    if metric == "icir":
        return parent_signal, candidate_signal, lambda frame: _ratio(frame["ic"])
    if metric == "rank_icir":
        return parent_signal, candidate_signal, lambda frame: _ratio(frame["rank_ic"])
    mapping = {
        "one_way_turnover_mean": "one_way_turn",
        "transaction_cost_drag_mean": "transaction_cost_drag",
        "portfolio_concentration_mean": "portfolio_concentration",
        "invested_weight_shortfall_mean": "invested_weight_shortfall",
        "alpha_term_mean": "alpha_term",
        "risk_term_mean": "risk_term",
        "turnover_term_mean": "turnover_term",
    }
    if metric in mapping:
        return parent_diagnostics, candidate_diagnostics, _mean_column(mapping[metric])
    return None


def evaluate_gate(
    parent_diagnostics: pd.DataFrame,
    candidate_diagnostics: pd.DataFrame,
    parent_signal_daily: pd.DataFrame,
    candidate_signal_daily: pd.DataFrame,
    expected_signature: list[dict],
    structural_rass_promotion: bool = False,
) -> dict:
    sharpe_delta, sharpe_epsilon = paired_moving_block_epsilon(
        parent_diagnostics,
        candidate_diagnostics,
        lambda frame: _sharpe(frame["net_return"]),
    )
    required_unavailable: list[str] = []
    if sharpe_delta is None or sharpe_epsilon is None:
        sharpe_state = "UNAVAILABLE"
        required_unavailable.append("sharpe")
    elif sharpe_delta > sharpe_epsilon:
        sharpe_state = "IMPROVED"
    elif sharpe_delta < -sharpe_epsilon:
        sharpe_state = "WORSE"
    else:
        sharpe_state = "UNCHANGED"

    mechanism_records = []
    states = []
    for declaration in expected_signature:
        metric = declaration["metric"]
        resolved = _metric_frames(metric, parent_diagnostics, candidate_diagnostics, parent_signal_daily, candidate_signal_daily)
        if resolved is None:
            mechanism_records.append({"metric": metric, "state": "UNAVAILABLE", "reason": "unsupported_metric"})
            states.append("UNAVAILABLE")
            required_unavailable.append(f"mechanism:{metric}")
            continue
        parent_frame, candidate_frame, statistic = resolved
        delta, epsilon = paired_moving_block_epsilon(parent_frame, candidate_frame, statistic)
        if delta is None or epsilon is None:
            state = "UNAVAILABLE"
            required_unavailable.append(f"mechanism:{metric}")
            oriented = None
        else:
            oriented = delta if declaration["direction"] in {"increase", "stable_or_improve"} else -delta
            if oriented > epsilon:
                state = "MATCH"
            elif oriented < -epsilon:
                state = "OPPOSITE"
            else:
                state = "INCONCLUSIVE"
        states.append(state)
        mechanism_records.append({
            "metric": metric,
            "expected_direction": declaration["direction"],
            "observed_delta": delta,
            "oriented_delta": oriented,
            "epsilon": epsilon,
            "state": state,
        })
    if "OPPOSITE" in states:
        mechanism_state = "OPPOSITE"
    elif "UNAVAILABLE" in states or not states:
        mechanism_state = "UNAVAILABLE"
        if not states:
            required_unavailable.append("mechanism:none_declared")
    elif all(state == "MATCH" for state in states):
        mechanism_state = "MATCH"
    elif any(state == "MATCH" for state in states):
        mechanism_state = "PARTIAL"
    else:
        mechanism_state = "INCONCLUSIVE"

    guardrails = [
        ("max_drawdown_magnitude", lambda frame: _drawdown_magnitude(frame["net_return"])),
        ("annual_volatility", lambda frame: _annual_volatility(frame["net_return"])),
        ("one_way_turnover_mean", _mean_column("one_way_turn")),
        ("transaction_cost_drag_mean", _mean_column("transaction_cost_drag")),
        ("portfolio_concentration_mean", _mean_column("portfolio_concentration")),
        ("invested_weight_shortfall_mean", _mean_column("invested_weight_shortfall")),
    ]
    guardrail_records = []
    degraded = False
    for metric, statistic in guardrails:
        delta, epsilon = paired_moving_block_epsilon(parent_diagnostics, candidate_diagnostics, statistic)
        if delta is None or epsilon is None:
            state = "UNAVAILABLE"
            required_unavailable.append(f"guardrail:{metric}")
        elif delta > epsilon:
            state = "MATERIAL_DEGRADATION"
            degraded = True
        else:
            state = "NO_MATERIAL_DEGRADATION"
        guardrail_records.append({"metric": metric, "oriented_delta": delta, "epsilon": epsilon, "state": state})
    tradeoff_state = "MATERIAL_DEGRADATION" if degraded else "UNAVAILABLE" if any(item["state"] == "UNAVAILABLE" for item in guardrail_records) else "NO_MATERIAL_DEGRADATION"

    if sharpe_state == "WORSE":
        verdict = "FALSIFIED"
    elif mechanism_state == "OPPOSITE":
        verdict = "FALSIFIED"
    elif degraded:
        verdict = "FALSIFIED"
    elif required_unavailable:
        verdict = "UNCERTAIN"
    elif sharpe_state == "IMPROVED" and mechanism_state == "MATCH":
        verdict = "SUPPORTED"
    elif sharpe_state == "IMPROVED":
        verdict = "PARTIALLY_SUPPORTED"
    elif sharpe_state == "UNCHANGED" and mechanism_state in {"MATCH", "PARTIAL"}:
        verdict = "PARTIALLY_SUPPORTED"
    else:
        verdict = "UNCERTAIN"

    ordinary_promotion = verdict in {"SUPPORTED", "PARTIALLY_SUPPORTED"}
    promotion = bool(structural_rass_promotion or ordinary_promotion)
    return {
        "gate_verdict": verdict,
        "sharpe": {"delta": sharpe_delta, "epsilon": sharpe_epsilon, "state": sharpe_state},
        "mechanism_state": mechanism_state,
        "mechanism_metrics": mechanism_records,
        "tradeoff_state": tradeoff_state,
        "guardrails": guardrail_records,
        "required_evidence_unavailable": sorted(set(required_unavailable)),
        "promotion": promotion,
        "promotion_reason": "mandatory_rass_structural_initialization" if structural_rass_promotion else "validation_gate" if ordinary_promotion else "rollback",
        "gate_verdict_usage": "audit_only_for_exp_001" if structural_rass_promotion else "controls_promotion",
        "alpha_frozen_after_promotion": bool(structural_rass_promotion),
        "contains_test_derived_data": False,
    }
