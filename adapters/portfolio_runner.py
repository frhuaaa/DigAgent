from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core.errors import ContractError, DataCoverageError


@dataclass(frozen=True)
class PortfolioRun:
    diagnostics: pd.DataFrame
    orders: pd.DataFrame
    objective: pd.DataFrame


def read_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    try:
        frame.index = pd.to_datetime(frame.index.astype(str), format="%Y%m%d")
    except ValueError as exc:
        raise DataCoverageError(f"PANEL_DATE_FORMAT_INVALID: {path}") from exc
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise DataCoverageError(f"PANEL_DUPLICATES: {path}")
    frame.index.name = "date"
    return frame.apply(pd.to_numeric, errors="coerce").sort_index()


def preprocess_prediction_cross_section(values: pd.Series) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return clean
    lower, upper = clean.quantile([0.01, 0.99])
    winsorized = clean.clip(lower=float(lower), upper=float(upper))
    std = float(winsorized.std(ddof=0))
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=winsorized.index, dtype=float)
    return (winsorized - float(winsorized.mean())) / std


def project_box_simplex(vector: np.ndarray, target: float, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if not (len(vector) == len(lower) == len(upper)):
        raise ContractError("BOX_SIMPLEX_LENGTH_MISMATCH")
    if np.any(lower > upper + 1e-12):
        raise ContractError("BOX_SIMPLEX_INCONSISTENT_BOUNDS")
    if target < lower.sum() - 1e-10 or target > upper.sum() + 1e-10:
        raise ContractError("BOX_SIMPLEX_INFEASIBLE_TARGET")
    lo_theta = float(np.min(vector - upper))
    hi_theta = float(np.max(vector - lower))
    for _ in range(120):
        theta = (lo_theta + hi_theta) / 2.0
        weights = np.clip(vector - theta, lower, upper)
        if weights.sum() > target:
            lo_theta = theta
        else:
            hi_theta = theta
    weights = np.clip(vector - (lo_theta + hi_theta) / 2.0, lower, upper)
    difference = target - float(weights.sum())
    if abs(difference) > 1e-10:
        free = np.where((weights > lower + 1e-10) & (weights < upper - 1e-10))[0]
        if len(free):
            weights[free] += difference / len(free)
            weights = np.clip(weights, lower, upper)
    if abs(float(weights.sum()) - target) > 1e-8:
        raise ContractError("BOX_SIMPLEX_NUMERICAL_FAILURE")
    return weights


def estimate_risk(history: pd.DataFrame, codes: list[str], config: dict) -> tuple[np.ndarray, np.ndarray | None]:
    portfolio = config["z_portfolio"]
    hist = history.reindex(columns=codes).tail(int(portfolio["lookback"]))
    hist = hist.replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0)
    if len(hist) < int(portfolio["min_lookback"]):
        return np.full(len(codes), 0.0004, dtype=float), None
    array = hist.to_numpy(dtype=float)
    array -= array.mean(axis=0, keepdims=True)
    if portfolio["risk_model_mode"] == "sample_cov":
        sample = (array.T @ array) / max(len(hist) - 1, 1)
        diagonal = np.diag(np.diag(sample))
        shrink = float(portfolio["cov_shrink_to_diag"])
        covariance = (1.0 - shrink) * sample + shrink * diagonal
        indexes = np.diag_indices_from(covariance)
        covariance[indexes] = np.maximum(covariance[indexes], float(portfolio["variance_floor"]))
        return np.diag(covariance), covariance
    if portfolio["risk_model_mode"] != "diagonal":
        raise ContractError(f"UNSUPPORTED_RISK_MODEL: {portfolio['risk_model_mode']}")
    variance = np.maximum(array.var(axis=0, ddof=1), float(portfolio["variance_floor"]))
    return variance, None


def optimize_mean_variance(
    mu: np.ndarray,
    variance: np.ndarray,
    covariance: np.ndarray | None,
    old_weights: np.ndarray,
    target: float,
    lower: np.ndarray,
    upper: np.ndarray,
    config: dict,
) -> np.ndarray:
    portfolio = config["z_portfolio"]
    risk_aversion = float(portfolio["risk_aversion"])
    turnover_penalty = float(portfolio["turnover_penalty"])
    if covariance is None:
        coefficient = np.maximum(risk_aversion * variance + turnover_penalty, 1e-12)
        linear = mu + 2.0 * turnover_penalty * old_weights
        lo_theta = float(np.min(linear - 2.0 * coefficient * upper))
        hi_theta = float(np.max(linear - 2.0 * coefficient * lower))
        for _ in range(120):
            theta = (lo_theta + hi_theta) / 2.0
            weights = np.clip((linear - theta) / (2.0 * coefficient), lower, upper)
            if weights.sum() > target:
                lo_theta = theta
            else:
                hi_theta = theta
        weights = np.clip((linear - (lo_theta + hi_theta) / 2.0) / (2.0 * coefficient), lower, upper)
        return project_box_simplex(weights, target, lower, upper)

    weights = project_box_simplex(old_weights.copy(), target, lower, upper)
    lipschitz = 2.0 * risk_aversion * float(np.max(np.sum(np.abs(covariance), axis=1))) + 2.0 * turnover_penalty + 1e-12
    step = min(1.0, 1.0 / lipschitz)
    previous = np.inf
    for _ in range(int(portfolio["max_iter"])):
        gradient = 2.0 * risk_aversion * (covariance @ weights) + 2.0 * turnover_penalty * (weights - old_weights) - mu
        candidate = project_box_simplex(weights - step * gradient, target, lower, upper)
        objective = risk_aversion * float(candidate @ covariance @ candidate) + turnover_penalty * float(np.sum((candidate - old_weights) ** 2)) - float(mu @ candidate)
        weights = candidate
        if abs(previous - objective) < float(portfolio["tol"]):
            break
        previous = objective
    return weights


def _drift(weights: pd.Series, realized: pd.Series, gross_return: float, threshold: float) -> dict[str, float]:
    if weights.empty:
        return {}
    denominator = 1.0 + gross_return
    if not np.isfinite(denominator) or denominator <= 1e-12:
        raise DataCoverageError("PORTFOLIO_NAV_NONPOSITIVE")
    values = weights * (1.0 + realized.reindex(weights.index).fillna(0.0)) / denominator
    values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return {str(code): float(weight) for code, weight in values.items() if weight > threshold}


def _objective(mu: np.ndarray, variance: np.ndarray, covariance: np.ndarray | None, new: np.ndarray, old: np.ndarray, config: dict) -> dict[str, float]:
    portfolio = config["z_portfolio"]
    alpha_term = float(mu @ new)
    risk_raw = float(np.sum(variance * new ** 2)) if covariance is None else float(new @ covariance @ new)
    turnover_raw = float(np.sum((new - old) ** 2))
    risk_term = float(portfolio["risk_aversion"]) * risk_raw
    turnover_term = float(portfolio["turnover_penalty"]) * turnover_raw
    denominator = max(abs(alpha_term), 1e-20)
    return {
        "alpha_term": alpha_term,
        "risk_raw": risk_raw,
        "risk_term": risk_term,
        "turnover_raw": turnover_raw,
        "turnover_term": turnover_term,
        "objective_value": alpha_term - risk_term - turnover_term,
        "risk_alpha_ratio": risk_term / denominator,
        "turnover_alpha_ratio": turnover_term / denominator,
    }


def run_mean_variance(
    config: dict,
    alpha: pd.DataFrame,
    returns: pd.DataFrame,
    limit_up: pd.DataFrame,
    limit_down: pd.DataFrame,
    alignment: dict[pd.Timestamp, tuple[pd.Timestamp, pd.Timestamp]],
    benchmark_by_signal: pd.Series,
) -> PortfolioRun:
    portfolio = config["z_portfolio"]
    reference_columns = list(returns.columns)
    if list(alpha.columns) != reference_columns:
        raise ContractError("ALPHA_RETURN_HEADER_MISMATCH")
    threshold = float(portfolio["weight_threshold_for_drop"])
    max_weight = float(portfolio["max_weight"])
    min_weight = float(portfolio["min_weight"])
    alpha_scale = float(portfolio["alpha_scale"])
    candidate_count = int(portfolio["candidate_count"])
    if bool(portfolio["full_position"]) and candidate_count < math.ceil(1.0 / max_weight):
        raise ContractError("CANDIDATE_COUNT_CANNOT_SUPPORT_FULL_POSITION")

    holdings: dict[str, float] = {}
    diagnostic_rows: list[dict] = []
    order_rows: list[dict] = []
    objective_rows: list[dict] = []

    for signal_date in alpha.index:
        signal_date = pd.Timestamp(signal_date).normalize()
        if signal_date not in alignment:
            raise DataCoverageError(f"SIGNAL_ALIGNMENT_MISSING: {signal_date.date()}")
        trade_date, return_date = alignment[signal_date]
        old = pd.Series(holdings, dtype=float)
        old = old[old > threshold]
        raw = alpha.loc[signal_date]
        processed = preprocess_prediction_cross_section(raw)
        missing_signal = processed.empty

        if missing_signal:
            codes = list(old.index)
        else:
            absent = [code for code in processed.index if code not in returns.columns]
            if absent:
                raise DataCoverageError(f"FINITE_ALPHA_ASSET_NOT_IN_RETURN_PANEL: {absent[:5]}")
            ranked = list(processed.sort_values(ascending=False, kind="mergesort").index)
            top_new = ranked[:candidate_count]
            codes = list(dict.fromkeys(top_new + list(old.index)))

        up_row = limit_up.loc[trade_date].reindex(codes) if codes else pd.Series(dtype=float)
        down_row = limit_down.loc[trade_date].reindex(codes) if codes else pd.Series(dtype=float)
        cannot_buy = up_row.isna() | (up_row > 0)
        cannot_sell = down_row.isna() | (down_row > 0)
        old_vector = np.array([old.get(code, 0.0) for code in codes], dtype=float)
        lower = np.full(len(codes), min_weight, dtype=float)
        upper = np.full(len(codes), max_weight, dtype=float)
        for index, code in enumerate(codes):
            if bool(cannot_sell.get(code, True)):
                lower[index] = max(lower[index], old_vector[index])
                upper[index] = max(upper[index], old_vector[index])
            if bool(cannot_buy.get(code, True)):
                upper[index] = min(upper[index], old_vector[index])
            if bool(cannot_buy.get(code, True)) and bool(cannot_sell.get(code, True)):
                lower[index] = old_vector[index]
                upper[index] = old_vector[index]
            if lower[index] > upper[index] + 1e-12:
                raise ContractError(f"DIRECTIONAL_BOUNDS_INCONSISTENT: {code} {trade_date.date()}")

        feasible_min = float(lower.sum())
        feasible_max = float(upper.sum())
        target_requested = 1.0 if bool(portfolio["full_position"]) else min(1.0, feasible_max)
        full_position_feasible = feasible_min <= target_requested + 1e-12 and feasible_max >= target_requested - 1e-12
        target = min(max(target_requested, feasible_min), feasible_max) if codes else 0.0
        if full_position_feasible:
            reason = "FULL_POSITION_FEASIBLE"
        elif not codes:
            reason = "NO_FINITE_SIGNAL_OR_HOLDINGS"
        elif feasible_max < target_requested:
            reason = "MAXIMUM_FEASIBLE_BELOW_TARGET"
        else:
            reason = "MINIMUM_FEASIBLE_ABOVE_TARGET"

        if missing_signal:
            new_vector = old_vector.copy()
            target = float(new_vector.sum())
            reason = "MISSING_SIGNAL_HOLD_EXISTING" if len(codes) else "MISSING_SIGNAL_CASH_ONLY"
            variance = np.zeros(len(codes), dtype=float)
            covariance = None
            mu = np.zeros(len(codes), dtype=float)
        elif codes:
            scores = processed.reindex(codes).fillna(0.0).to_numpy(dtype=float)
            mu = scores * alpha_scale
            history = returns.loc[returns.index <= signal_date]
            variance, covariance = estimate_risk(history, codes, config)
            new_vector = optimize_mean_variance(mu, variance, covariance, old_vector, target, lower, upper, config)
        else:
            mu = variance = new_vector = old_vector
            covariance = None

        target_weights = pd.Series(new_vector, index=codes, dtype=float)
        all_codes = sorted(set(old.index) | set(target_weights.index))
        old_all = old.reindex(all_codes).fillna(0.0)
        new_all = target_weights.reindex(all_codes).fillna(0.0)
        delta = new_all - old_all
        for code, change in delta.items():
            if change > 1e-10 and bool((limit_up.loc[trade_date].reindex([code]).isna() | (limit_up.loc[trade_date].reindex([code]) > 0)).iloc[0]):
                raise ContractError(f"ILLEGAL_LIMIT_UP_BUY: {code} {trade_date.date()}")
            if change < -1e-10 and bool((limit_down.loc[trade_date].reindex([code]).isna() | (limit_down.loc[trade_date].reindex([code]) > 0)).iloc[0]):
                raise ContractError(f"ILLEGAL_LIMIT_DOWN_SELL: {code} {trade_date.date()}")

        buy_turn = float(delta.clip(lower=0.0).sum())
        sell_turn = float((-delta.clip(upper=0.0)).sum())
        buy_cost = float(portfolio["buy_commission"]) + float(portfolio["buy_slippage"])
        sell_cost = float(portfolio["sell_commission"]) + float(portfolio["sell_tax"]) + float(portfolio["sell_slippage"])
        cost_rate = buy_turn * buy_cost + sell_turn * sell_cost
        realized = returns.loc[return_date].reindex(all_codes).fillna(0.0)
        gross_return = float((new_all * realized).sum())
        net_return = gross_return - cost_rate
        benchmark_return = float(benchmark_by_signal.loc[signal_date])
        invested = float(new_all.sum())
        concentration = float(np.sum(new_all.to_numpy(dtype=float) ** 2))
        obj = _objective(mu, variance, covariance, new_vector, old_vector, config) if codes else {
            "alpha_term": 0.0, "risk_raw": 0.0, "risk_term": 0.0, "turnover_raw": 0.0,
            "turnover_term": 0.0, "objective_value": 0.0, "risk_alpha_ratio": 0.0,
            "turnover_alpha_ratio": 0.0,
        }
        objective_rows.append({"signal_date": signal_date, "trade_date": trade_date, "return_date": return_date, **obj})
        diagnostic_rows.append({
            "signal_date": signal_date,
            "trade_date": trade_date,
            "return_date": return_date,
            "gross_return": gross_return,
            "transaction_cost_drag": cost_rate,
            "net_return": net_return,
            "benchmark_return": benchmark_return,
            "excess_return": net_return - benchmark_return,
            "buy_turn": buy_turn,
            "sell_turn": sell_turn,
            "one_way_turn": (buy_turn + sell_turn) / 2.0,
            "holding_count": int((new_all > threshold).sum()),
            "optimizer_universe_count": len(codes),
            "portfolio_concentration": concentration,
            "target_invested_weight": target_requested,
            "feasible_min_invested_weight": feasible_min,
            "feasible_max_invested_weight": feasible_max,
            "full_position_feasible": bool(full_position_feasible),
            "invested_weight": invested,
            "invested_weight_shortfall": 1.0 - invested,
            "feasibility_reason_code": reason,
            "missing_signal": bool(missing_signal),
            **obj,
        })
        for code, change in delta[delta.abs() > threshold].items():
            order_rows.append({
                "signal_date": signal_date, "trade_date": trade_date, "return_date": return_date,
                "instrument": code, "side": "BUY" if change > 0 else "SELL",
                "old_weight": float(old_all[code]), "target_weight": float(new_all[code]),
                "order_weight": float(change),
            })
        holdings = _drift(new_all, realized, gross_return, threshold)

    diagnostics = pd.DataFrame(diagnostic_rows)
    if diagnostics.empty:
        raise DataCoverageError("NO_PORTFOLIO_PERIODS")
    diagnostics = diagnostics.sort_values("signal_date").reset_index(drop=True)
    return PortfolioRun(diagnostics, pd.DataFrame(order_rows), pd.DataFrame(objective_rows))


def run_topk(
    config: dict,
    alpha: pd.DataFrame,
    returns: pd.DataFrame,
    limit_up: pd.DataFrame,
    limit_down: pd.DataFrame,
    alignment: dict[pd.Timestamp, tuple[pd.Timestamp, pd.Timestamp]],
    benchmark_by_signal: pd.Series,
    drop_count: int = 10,
) -> PortfolioRun:
    """Top100/Drop10 comparison under the same dates, masks, costs, and cap."""
    portfolio = config["z_portfolio"]
    threshold = float(portfolio["weight_threshold_for_drop"])
    max_weight = float(portfolio["max_weight"])
    candidate_count = int(portfolio["candidate_count"])
    holdings: dict[str, float] = {}
    rows: list[dict] = []
    orders: list[dict] = []
    for signal_date in alpha.index:
        signal_date = pd.Timestamp(signal_date).normalize()
        trade_date, return_date = alignment[signal_date]
        old = pd.Series(holdings, dtype=float)
        old = old[old > threshold]
        processed = preprocess_prediction_cross_section(alpha.loc[signal_date])
        missing_signal = processed.empty
        if missing_signal:
            selected = list(old.index)
        else:
            ranked = list(processed.sort_values(ascending=False, kind="mergesort").index)
            rank_position = {code: index for index, code in enumerate(ranked)}
            outside = sorted([code for code in old.index if code not in ranked[:candidate_count]], key=lambda code: rank_position.get(code, len(ranked)), reverse=True)
            dropped = set(outside[:drop_count])
            retained = [code for code in old.index if code not in dropped]
            selected = retained + [code for code in ranked if code not in retained][: max(0, candidate_count - len(retained))]
            selected = list(dict.fromkeys(selected))
        codes = list(dict.fromkeys(selected + list(old.index)))
        old_vector = np.array([old.get(code, 0.0) for code in codes], dtype=float)
        lower = np.zeros(len(codes), dtype=float)
        upper = np.full(len(codes), max_weight, dtype=float)
        up_row = limit_up.loc[trade_date].reindex(codes)
        down_row = limit_down.loc[trade_date].reindex(codes)
        cannot_buy = up_row.isna() | (up_row > 0)
        cannot_sell = down_row.isna() | (down_row > 0)
        for index, code in enumerate(codes):
            if code not in selected:
                upper[index] = 0.0
            if bool(cannot_sell.get(code, True)):
                lower[index] = old_vector[index]
                upper[index] = max(upper[index], old_vector[index])
            if bool(cannot_buy.get(code, True)):
                upper[index] = min(upper[index], old_vector[index])
            if bool(cannot_buy.get(code, True)) and bool(cannot_sell.get(code, True)):
                lower[index] = upper[index] = old_vector[index]
        feasible_min = float(lower.sum())
        feasible_max = float(upper.sum())
        requested = 1.0
        target = min(max(requested, feasible_min), feasible_max) if codes else 0.0
        full_feasible = feasible_min <= requested + 1e-12 and feasible_max >= requested - 1e-12
        if missing_signal:
            new_vector = old_vector.copy()
            reason = "MISSING_SIGNAL_HOLD_EXISTING" if codes else "MISSING_SIGNAL_CASH_ONLY"
        elif codes:
            desired = np.array([1.0 / max(len(selected), 1) if code in selected else 0.0 for code in codes], dtype=float)
            new_vector = project_box_simplex(desired, target, lower, upper)
            reason = "FULL_POSITION_FEASIBLE" if full_feasible else "MAXIMUM_FEASIBLE_BELOW_TARGET"
        else:
            new_vector = old_vector
            reason = "NO_FINITE_SIGNAL_OR_HOLDINGS"
        new = pd.Series(new_vector, index=codes, dtype=float)
        all_codes = sorted(set(old.index) | set(new.index))
        old_all = old.reindex(all_codes).fillna(0.0)
        new_all = new.reindex(all_codes).fillna(0.0)
        delta = new_all - old_all
        for code, change in delta.items():
            if change > 1e-10 and bool((limit_up.loc[trade_date].reindex([code]).isna() | (limit_up.loc[trade_date].reindex([code]) > 0)).iloc[0]):
                raise ContractError(f"TOPK_ILLEGAL_LIMIT_UP_BUY: {code}")
            if change < -1e-10 and bool((limit_down.loc[trade_date].reindex([code]).isna() | (limit_down.loc[trade_date].reindex([code]) > 0)).iloc[0]):
                raise ContractError(f"TOPK_ILLEGAL_LIMIT_DOWN_SELL: {code}")
        buy_turn = float(delta.clip(lower=0.0).sum())
        sell_turn = float((-delta.clip(upper=0.0)).sum())
        cost = buy_turn * (float(portfolio["buy_commission"]) + float(portfolio["buy_slippage"])) + sell_turn * (float(portfolio["sell_commission"]) + float(portfolio["sell_tax"]) + float(portfolio["sell_slippage"]))
        realized = returns.loc[return_date].reindex(all_codes).fillna(0.0)
        gross = float((new_all * realized).sum())
        net = gross - cost
        invested = float(new_all.sum())
        rows.append({
            "signal_date": signal_date, "trade_date": trade_date, "return_date": return_date,
            "gross_return": gross, "transaction_cost_drag": cost, "net_return": net,
            "benchmark_return": float(benchmark_by_signal.loc[signal_date]),
            "excess_return": net - float(benchmark_by_signal.loc[signal_date]),
            "buy_turn": buy_turn, "sell_turn": sell_turn, "one_way_turn": (buy_turn + sell_turn) / 2.0,
            "holding_count": int((new_all > threshold).sum()), "optimizer_universe_count": len(codes),
            "portfolio_concentration": float(np.sum(new_all.to_numpy() ** 2)),
            "target_invested_weight": requested, "feasible_min_invested_weight": feasible_min,
            "feasible_max_invested_weight": feasible_max, "full_position_feasible": bool(full_feasible),
            "invested_weight": invested, "invested_weight_shortfall": 1.0 - invested,
            "feasibility_reason_code": reason, "missing_signal": bool(missing_signal),
            "alpha_term": 0.0, "risk_raw": 0.0, "risk_term": 0.0, "turnover_raw": float(np.sum(delta.to_numpy() ** 2)),
            "turnover_term": 0.0, "objective_value": 0.0, "risk_alpha_ratio": 0.0, "turnover_alpha_ratio": 0.0,
        })
        for code, change in delta[delta.abs() > threshold].items():
            orders.append({"signal_date": signal_date, "trade_date": trade_date, "return_date": return_date, "instrument": code, "side": "BUY" if change > 0 else "SELL", "old_weight": float(old_all[code]), "target_weight": float(new_all[code]), "order_weight": float(change)})
        holdings = _drift(new_all, realized, gross, threshold)
    frame = pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)
    if frame.empty:
        raise DataCoverageError("NO_TOPK_PERIODS")
    return PortfolioRun(frame, pd.DataFrame(orders), pd.DataFrame())
