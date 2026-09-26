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
SHARPE_ACCEPTANCE_DELTA = 0.002
FAMA_IC_ACCEPTANCE_DELTA = 0.001


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
    freeze_alpha_on_promotion: bool = False,
    fama_ic_override: bool = False,
    parent_ensemble_diagnostics: dict | None = None,
    candidate_ensemble_diagnostics: dict | None = None,
) -> dict:
    aligned_parent, aligned_candidate = _aligned(parent_diagnostics, candidate_diagnostics)
    observed_parent_sharpe = _sharpe(aligned_parent["net_return"])
    observed_candidate_sharpe = _sharpe(aligned_candidate["net_return"])
    sharpe_delta = observed_candidate_sharpe - observed_parent_sharpe
    if not np.isfinite(sharpe_delta):
        sharpe_delta = None
    elif abs(sharpe_delta) <= _numerical_tolerance(observed_parent_sharpe, observed_candidate_sharpe):
        sharpe_delta = 0.0
    sharpe_epsilon = SHARPE_ACCEPTANCE_DELTA if sharpe_delta is not None else None
    required_unavailable: list[str] = []
    if sharpe_delta is None:
        sharpe_state = "UNAVAILABLE"
        required_unavailable.append("sharpe")
    elif sharpe_delta > SHARPE_ACCEPTANCE_DELTA:
        sharpe_state = "IMPROVED"
    elif sharpe_delta < -SHARPE_ACCEPTANCE_DELTA:
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

    aligned_parent_signal, aligned_candidate_signal = _aligned(parent_signal_daily, candidate_signal_daily)
    parent_ic = _mean_column("ic")(aligned_parent_signal)
    candidate_ic = _mean_column("ic")(aligned_candidate_signal)
    ic_delta = candidate_ic - parent_ic
    ic_numerical_tolerance = _numerical_tolerance(parent_ic, candidate_ic)
    if not np.isfinite(ic_delta):
        ic_delta = None
        fama_ic_state = "UNAVAILABLE"
    elif abs(ic_delta) <= ic_numerical_tolerance:
        ic_delta = 0.0
        fama_ic_state = "UNCHANGED"
    elif ic_delta > FAMA_IC_ACCEPTANCE_DELTA + ic_numerical_tolerance:
        fama_ic_state = "IMPROVED"
    elif ic_delta < -FAMA_IC_ACCEPTANCE_DELTA - ic_numerical_tolerance:
        fama_ic_state = "WORSE"
    else:
        fama_ic_state = "UNCHANGED"

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

    seed_ic_deltas: list[dict] = []
    if parent_ensemble_diagnostics is not None and candidate_ensemble_diagnostics is not None:
        parent_by_seed = {
            int(item["seed"]): item
            for item in parent_ensemble_diagnostics.get("seed_signal_metrics", [])
        }
        candidate_by_seed = {
            int(item["seed"]): item
            for item in candidate_ensemble_diagnostics.get("seed_signal_metrics", [])
        }
        if set(parent_by_seed) == set(candidate_by_seed) == {0, 1, 2}:
            for seed in sorted(parent_by_seed):
                parent_value = parent_by_seed[seed].get("ic")
                candidate_value = candidate_by_seed[seed].get("ic")
                delta = (
                    float(candidate_value) - float(parent_value)
                    if parent_value is not None and candidate_value is not None
                    else None
                )
                seed_ic_deltas.append({"seed": seed, "delta": delta})
    finite_seed_deltas = [item["delta"] for item in seed_ic_deltas if item["delta"] is not None and np.isfinite(item["delta"])]
    positive_seed_count = sum(delta > 0.0 for delta in finite_seed_deltas)
    median_seed_delta = float(np.median(finite_seed_deltas)) if len(finite_seed_deltas) == 3 else None
    seed_consistent = bool(
        len(finite_seed_deltas) == 3
        and positive_seed_count >= 2
        and median_seed_delta is not None
        and median_seed_delta > FAMA_IC_ACCEPTANCE_DELTA
    )

    ordinary_promotion = verdict in {"SUPPORTED", "PARTIALLY_SUPPORTED"}
    # FAMA's IC rule is a narrow rescue for an otherwise UNCERTAIN candidate.
    # It cannot overturn Sharpe deterioration, an opposite mechanism, or a
    # material trade-off, and it requires improvement in the ensemble plus a
    # majority-consistent improvement across the three frozen seeds.
    fama_ic_override_applied = bool(
        fama_ic_override
        and verdict == "UNCERTAIN"
        and fama_ic_state == "IMPROVED"
        and sharpe_state != "WORSE"
        and mechanism_state != "OPPOSITE"
        and not degraded
        and seed_consistent
    )
    # RASS bootstrap candidates use the same portfolio-based Gate as every
    # other candidate.  The flag only freezes alpha after an ordinary promotion;
    # it never overrides an UNCERTAIN or FALSIFIED verdict.
    promotion = bool(ordinary_promotion or fama_ic_override_applied)
    if fama_ic_override_applied and not ordinary_promotion:
        promotion_reason = "fama_validation_ic_override"
        gate_verdict_usage = "overridden_by_fama_validation_ic"
    else:
        promotion_reason = "validation_gate" if ordinary_promotion else "rollback"
        gate_verdict_usage = "controls_promotion"
    return {
        "gate_verdict": verdict,
        "sharpe": {
            "delta": sharpe_delta,
            "epsilon": sharpe_epsilon,
            "acceptance_delta": SHARPE_ACCEPTANCE_DELTA,
            "comparison": "fixed_symmetric_epsilon",
            "state": sharpe_state,
        },
        "mechanism_state": mechanism_state,
        "mechanism_metrics": mechanism_records,
        "tradeoff_state": tradeoff_state,
        "guardrails": guardrail_records,
        "required_evidence_unavailable": sorted(set(required_unavailable)),
        "fama_ic_override": {
            "enabled": bool(fama_ic_override),
            "parent_ic": float(parent_ic) if np.isfinite(parent_ic) else None,
            "candidate_ic": float(candidate_ic) if np.isfinite(candidate_ic) else None,
            "delta": ic_delta,
            "acceptance_delta": FAMA_IC_ACCEPTANCE_DELTA,
            "state": fama_ic_state,
            "seed_ic_deltas": seed_ic_deltas,
            "positive_seed_count": positive_seed_count,
            "median_seed_delta": median_seed_delta,
            "seed_consistent": seed_consistent,
            "applied": fama_ic_override_applied,
        },
        "promotion": promotion,
        "promotion_reason": promotion_reason,
        "gate_verdict_usage": gate_verdict_usage,
        "alpha_frozen_after_promotion": bool(freeze_alpha_on_promotion and promotion),
        "contains_test_derived_data": False,
    }
