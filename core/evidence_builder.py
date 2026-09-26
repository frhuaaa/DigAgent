from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from qlib.data import D

from adapters.qlib_runner import initialize_qlib, resolve_market_instruments
from core.errors import ContractError
from core.io_utils import canonical_json_bytes, sha256_bytes, sha256_file
from core.prediction_mode import prediction_mode, prediction_mode_preset
from evaluation.prediction_metrics import compute_signal_metrics

def rass_evidence_periods(config: dict) -> tuple[dict[str, str], ...]:
    run_id = str(config.get("task", {}).get("run_id", ""))
    if len(run_id) != 4 or not run_id.isdigit():
        raise ContractError("RASS_RUN_ID_MUST_BE_FOUR_DIGIT_YEAR")
    target_year = int(run_id)
    years = range(target_year - 3, target_year)
    calendar = pd.DatetimeIndex(D.calendar(
        start_time=f"{target_year - 3}-01-01",
        end_time=f"{target_year - 1}-12-31",
        freq="day",
    )).normalize()
    periods = []
    for year in years:
        year_calendar = calendar[calendar.year == year]
        if len(year_calendar) < 3:
            raise ContractError(f"RASS_EVIDENCE_CALENDAR_INCOMPLETE: {year}")
        periods.append({
            "period_id": str(year),
            "start_time": f"{year}-01-01",
            "end_time": f"{year}-12-31",
            "signal_end_time": year_calendar[-3].strftime("%Y-%m-%d"),
        })
    return tuple(periods)


def rass_evidence_context(config: dict, catalog_hash: str) -> dict:
    """Return the model-independent identity of shared RASS evidence."""

    task = config["task"]
    periods = list(rass_evidence_periods(config))
    label_expression = prediction_mode_preset(config)["target"]
    return {
        "version": 1,
        "instruments": task["instruments"],
        "run_id": str(task["run_id"]),
        "prediction_mode": prediction_mode(config),
        "feature_pool": config["z_alpha"]["feature_pool"],
        "initial_features": list(config["z_alpha"]["selected_features"]),
        "configured_training_period": [task["train_start_time"], task["train_end_time"]],
        "evidence_periods": periods,
        "target_label": label_expression,
        "label_hash": sha256_bytes(label_expression.encode("utf-8")),
        "input_data_hash": config["provenance"]["input_hashes"]["qlib_relevant_manifest"],
        "catalog_hash": catalog_hash,
        "evidence_code_hash": sha256_file(Path(__file__)),
    }


def _rass_period_mask(dates: pd.DatetimeIndex, periods: tuple[dict[str, str], ...]) -> np.ndarray:
    allowed = np.zeros(len(dates), dtype=bool)
    for period in periods:
        allowed |= (
            (dates >= pd.Timestamp(period["start_time"]))
            & (dates <= pd.Timestamp(period["signal_end_time"]))
        )
    return allowed


def _correlation(left: pd.Series, right: pd.Series) -> float | None:
    pair = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 2 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return None
    value = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    return float(value) if np.isfinite(value) else None


def _rolling_signal_evidence(
    daily_metrics: pd.DataFrame,
    evaluation_periods: tuple[dict[str, str], ...],
) -> dict:
    def summarize(frame: pd.DataFrame) -> dict:
        ic = frame["ic"].dropna()
        rank_ic = frame["rank_ic"].dropna()
        return {
            "ic": float(ic.mean()) if len(ic) else None,
            "icir": float(ic.mean() / ic.std(ddof=1)) if len(ic) > 1 and ic.std(ddof=1) > 0 else None,
            "rank_ic": float(rank_ic.mean()) if len(rank_ic) else None,
            "rank_icir": float(rank_ic.mean() / rank_ic.std(ddof=1)) if len(rank_ic) > 1 and rank_ic.std(ddof=1) > 0 else None,
            "valid_ic_dates": int(len(frame)),
        }

    dates = pd.DatetimeIndex(daily_metrics.index)
    folds = []
    combined_mask = np.zeros(len(daily_metrics), dtype=bool)
    for period in evaluation_periods:
        start = pd.Timestamp(period["start_time"])
        signal_end = pd.Timestamp(period["signal_end_time"])
        mask = (dates >= start) & (dates <= signal_end)
        combined_mask |= mask
        folds.append({**period, **summarize(daily_metrics.loc[mask])})
    aggregate = summarize(daily_metrics.loc[combined_mask]) if combined_mask.any() else None

    def fold_consistency(metric: str) -> float | None:
        values = [fold[metric] for fold in folds if fold[metric] is not None]
        if not values:
            return None
        center = aggregate[metric] if aggregate is not None else None
        if center is None:
            return None
        if center == 0:
            return float(sum(value == 0 for value in values) / len(values))
        return float(sum(np.sign(value) == np.sign(center) for value in values) / len(values))

    return {
        "protocol": "year_scoped_three_calendar_year_rass_development_evidence",
        "folds": folds,
        "aggregate": aggregate,
        "ic_fold_sign_consistency": fold_consistency("ic"),
        "rank_ic_fold_sign_consistency": fold_consistency("rank_ic"),
        "worst_fold_ic": min((fold["ic"] for fold in folds if fold["ic"] is not None), default=None),
        "worst_fold_rank_ic": min((fold["rank_ic"] for fold in folds if fold["rank_ic"] is not None), default=None),
    }


def build_rass_train_evidence(
    repo_root: Path,
    config: dict,
    catalog: list[dict[str, str]],
    catalog_hash: str,
) -> dict:
    initialize_qlib(repo_root, config)
    evidence_periods = rass_evidence_periods(config)
    evidence_start_time = evidence_periods[0]["start_time"]
    evidence_end_time = evidence_periods[-1]["signal_end_time"]
    instruments, unavailable = resolve_market_instruments(repo_root, config)
    anchors = list(config["z_alpha"]["selected_features"])
    task = config["task"]
    label_expression = prediction_mode_preset(config)["target"]
    evidence_context = rass_evidence_context(config, catalog_hash)
    expressions = list(dict.fromkeys([item["expression"] for item in catalog] + anchors + [label_expression]))
    data = D.features(
        instruments,
        expressions,
        start_time=evidence_start_time,
        end_time=evidence_end_time,
        freq="day",
    )
    if data.empty:
        raise ContractError("RASS_TRAIN_EVIDENCE_EMPTY")
    data = data.replace([np.inf, -np.inf], np.nan)
    signal_dates = pd.DatetimeIndex(data.index.get_level_values("datetime"))
    data = data.loc[_rass_period_mask(signal_dates, evidence_periods)]
    if data.empty:
        raise ContractError("RASS_TRAIN_EVIDENCE_EMPTY_AFTER_PERIOD_ALIGNMENT")
    raw_label = data[label_expression].rename("raw_label")
    correlation_sample_size = min(len(data), 20_000)
    correlation_positions = np.linspace(0, len(data) - 1, correlation_sample_size, dtype=int)
    correlation_data = data.iloc[np.unique(correlation_positions)]
    candidate_frame = correlation_data[[item["expression"] for item in catalog]].copy()
    candidate_frame.columns = [item["id"] for item in catalog]
    flattened_correlation = candidate_frame.corr(method="pearson", min_periods=20)

    records = []
    total_rows = len(data)
    evidence_dates = data.index.get_level_values("datetime")
    evidence_date_count = int(evidence_dates.nunique())
    for item in catalog:
        factor = data[item["expression"]].rename("prediction")
        metrics = compute_signal_metrics(factor, raw_label)
        finite = factor.replace([np.inf, -np.inf], np.nan).notna()
        daily_ic = metrics.daily["ic"] if "ic" in metrics.daily else pd.Series(dtype=float)
        rolling_evidence = _rolling_signal_evidence(metrics.daily, evidence_periods)
        period_ic = {
            fold["period_id"]: fold["ic"] for fold in rolling_evidence["folds"]
        }
        period_rank_ic = {
            fold["period_id"]: fold["rank_ic"] for fold in rolling_evidence["folds"]
        }
        mean_ic = metrics.summary["ic"]
        if mean_ic is None or daily_ic.empty:
            sign_consistency = None
        elif mean_ic == 0:
            sign_consistency = float((daily_ic == 0).mean())
        else:
            sign_consistency = float((np.sign(daily_ic) == np.sign(mean_ic)).mean())
        sampled_factor = correlation_data[item["expression"]].rename("prediction")
        anchor_correlations = {
            f"anchor_{index:02d}": _correlation(sampled_factor, correlation_data[anchor])
            for index, anchor in enumerate(anchors)
        }
        finite_anchor = [abs(value) for value in anchor_correlations.values() if value is not None]
        redundancy = flattened_correlation[item["id"]].drop(labels=[item["id"]], errors="ignore").dropna().abs()
        strongest = [
            {"id": str(identifier), "abs_correlation": float(value)}
            for identifier, value in redundancy.sort_values(ascending=False, kind="mergesort").head(3).items()
        ]
        daily_coverage = finite.groupby(level="datetime").mean()
        daily_dispersion = factor.groupby(level="datetime").std(ddof=1).replace([np.inf, -np.inf], np.nan)
        nonconstant_ratio = float((daily_dispersion > 0).mean()) if len(daily_dispersion) else 0.0
        eligibility_reasons = []
        if float(finite.sum() / total_rows) < 0.90:
            eligibility_reasons.append("COVERAGE_BELOW_90_PERCENT")
        if int(len(metrics.daily)) < max(20, math.ceil(evidence_date_count * 0.80)):
            eligibility_reasons.append("INSUFFICIENT_VALID_IC_DATES")
        if nonconstant_ratio < 0.90:
            eligibility_reasons.append("TOO_MANY_CONSTANT_CROSS_SECTIONS")
        if item["expression"] in anchors:
            eligibility_reasons.append("DUPLICATES_ANCHOR")
        records.append({
            **item,
            "coverage": float(finite.sum() / total_rows),
            "missing_rate": float(1.0 - finite.sum() / total_rows),
            "ic": metrics.summary["ic"],
            "icir": metrics.summary["icir"],
            "rank_ic": metrics.summary["rank_ic"],
            "rank_icir": metrics.summary["rank_icir"],
            "rolling_period_evidence": rolling_evidence,
            "valid_ic_dates": int(len(metrics.daily)),
            "excluded_dates": metrics.excluded,
            "sign_consistency": sign_consistency,
            "period_ic": period_ic,
            "period_rank_ic": period_rank_ic,
            "positive_ic_period_ratio": float(sum(value > 0 for value in period_ic.values() if value is not None) / sum(value is not None for value in period_ic.values())) if any(value is not None for value in period_ic.values()) else None,
            "worst_period_ic": min((value for value in period_ic.values() if value is not None), default=None),
            "daily_coverage_mean": float(daily_coverage.mean()) if len(daily_coverage) else None,
            "daily_coverage_p05": float(daily_coverage.quantile(0.05)) if len(daily_coverage) else None,
            "daily_coverage_min": float(daily_coverage.min()) if len(daily_coverage) else None,
            "nonconstant_cross_section_ratio": nonconstant_ratio,
            "max_abs_anchor_correlation": max(finite_anchor) if finite_anchor else None,
            "anchor_correlations": anchor_correlations,
            "strongest_candidate_redundancy": strongest,
            "eligible": not eligibility_reasons,
            "eligibility_reasons": eligibility_reasons,
        })

    eligible_candidate_count = sum(record["eligible"] for record in records)
    if eligible_candidate_count < 12:
        raise ContractError("RASS_FEWER_THAN_TWELVE_ELIGIBLE_CANDIDATES")
    payload = {
        "method": "deterministic_rass_development_factor_evidence_v5",
        "data_split": "rass_development",
        "evidence_context": evidence_context,
        "rass_evidence_context_hash": sha256_bytes(canonical_json_bytes(evidence_context)),
        "evidence_period": [evidence_start_time, evidence_end_time],
        "configured_training_period": [task["train_start_time"], task["train_end_time"]],
        "prediction_mode": prediction_mode(config),
        "target_label": label_expression,
        "label_hash": sha256_bytes(label_expression.encode("utf-8")),
        "initial_features": anchors,
        "catalog_hash": catalog_hash,
        "input_data_hash": config["provenance"]["input_hashes"]["qlib_relevant_manifest"],
        "evidence_code_hash": sha256_file(Path(__file__)),
        "available_instrument_count": len(instruments),
        "unavailable_placeholder_count": len(unavailable),
        "row_count": int(total_rows),
        "candidate_count": len(records),
        "eligible_candidate_count": eligible_candidate_count,
        "rolling_evidence_protocol": {
            "method": "three_complete_calendar_year_evaluation",
            "periods": list(evidence_periods),
            "signal_end_alignment_note": "Each period removes its final two trading signal dates so every T+1-to-T+2 label is realized within that period.",
            "selection_role": "agent_evidence_not_deterministic_ranking",
            "development_evidence_ends": evidence_periods[-1]["end_time"],
            "last_usable_signal_date": evidence_end_time,
            "test_used": False,
        },
        "redundancy_estimation": {
            "method": "deterministic_evenly_spaced_row_sample",
            "maximum_rows": 20000,
            "actual_rows": int(len(correlation_data)),
            "final_shortlist_uses_full_daily_cross_sectional_evidence": True,
        },
        "records": records,
        "contains_test_derived_data": False,
    }
    payload["evidence_hash"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def build_rass_shortlist_evidence(
    repo_root: Path,
    config: dict,
    shortlist: list[dict[str, str]],
    base_evidence_hash: str,
) -> dict:
    """Build detailed fixed-period development evidence for an Agent shortlist."""
    initialize_qlib(repo_root, config)
    evidence_periods = rass_evidence_periods(config)
    evidence_start_time = evidence_periods[0]["start_time"]
    evidence_end_time = evidence_periods[-1]["signal_end_time"]
    instruments, _ = resolve_market_instruments(repo_root, config)
    anchors = list(config["z_alpha"]["selected_features"])
    ids = [item["id"] for item in shortlist]
    expressions = [item["expression"] for item in shortlist]
    task = config["task"]
    label_expression = prediction_mode_preset(config)["target"]
    requested = list(dict.fromkeys(anchors + expressions + [label_expression]))
    data = D.features(
        instruments,
        requested,
        start_time=evidence_start_time,
        end_time=evidence_end_time,
        freq="day",
    ).replace([np.inf, -np.inf], np.nan)
    signal_dates = pd.DatetimeIndex(data.index.get_level_values("datetime"))
    data = data.loc[_rass_period_mask(signal_dates, evidence_periods)]
    if data.empty:
        raise ContractError("RASS_SHORTLIST_EVIDENCE_EMPTY")

    renamed = data[expressions].copy()
    renamed.columns = ids
    pairwise: dict[tuple[str, str], list[float]] = {
        (left, right): []
        for index, left in enumerate(ids)
        for right in ids[index + 1:]
    }
    marginal: dict[str, list[tuple[pd.Timestamp, float]]] = {identifier: [] for identifier in ids}
    marginal_rank: dict[str, list[tuple[pd.Timestamp, float]]] = {identifier: [] for identifier in ids}
    for date, candidate_day in renamed.groupby(level="datetime", sort=True):
        candidate_day = candidate_day.droplevel("datetime")
        corr = candidate_day.corr(method="pearson", min_periods=20)
        for pair, values in pairwise.items():
            value = corr.loc[pair[0], pair[1]]
            if np.isfinite(value):
                values.append(float(value))
        source_day = data.xs(date, level="datetime")
        for identifier, expression in zip(ids, expressions):
            frame = source_day[anchors + [expression, label_expression]].dropna()
            if len(frame) < 20:
                continue
            x = frame[anchors].to_numpy(dtype=float)
            candidate_values = frame[expression].to_numpy(dtype=float)
            label_values = frame[label_expression].to_numpy(dtype=float)
            x = np.column_stack([np.ones(len(x)), x])
            residual = candidate_values - x @ np.linalg.lstsq(x, candidate_values, rcond=None)[0]
            if np.std(residual) > 0 and np.std(label_values) > 0:
                value = np.corrcoef(residual, label_values)[0, 1]
                if np.isfinite(value):
                    marginal[identifier].append((pd.Timestamp(date), float(value)))
                rank_value = pd.Series(residual).corr(pd.Series(label_values), method="spearman")
                if np.isfinite(rank_value):
                    marginal_rank[identifier].append((pd.Timestamp(date), float(rank_value)))

    pairwise_records = []
    for (left, right), values in pairwise.items():
        array = np.asarray(values, dtype=float)
        pairwise_records.append({
            "left": left,
            "right": right,
            "valid_dates": int(len(array)),
            "mean_correlation": float(array.mean()) if len(array) else None,
            "median_correlation": float(np.median(array)) if len(array) else None,
            "p90_abs_correlation": float(np.quantile(np.abs(array), 0.90)) if len(array) else None,
        })
    marginal_records = []
    for identifier in ids:
        dated_values = marginal[identifier]
        dated_ranks = marginal_rank[identifier]
        values = np.asarray([value for _, value in dated_values], dtype=float)
        ranks = np.asarray([value for _, value in dated_ranks], dtype=float)
        rolling_residual_evidence = []
        for period in evidence_periods:
            start = pd.Timestamp(period["start_time"])
            signal_end = pd.Timestamp(period["signal_end_time"])
            fold_values = np.asarray([value for date, value in dated_values if start <= date <= signal_end], dtype=float)
            fold_ranks = np.asarray([value for date, value in dated_ranks if start <= date <= signal_end], dtype=float)
            rolling_residual_evidence.append({
                **period,
                "valid_dates": int(len(fold_values)),
                "anchor_residual_ic": float(fold_values.mean()) if len(fold_values) else None,
                "anchor_residual_icir": float(fold_values.mean() / fold_values.std(ddof=1)) if len(fold_values) > 1 and fold_values.std(ddof=1) > 0 else None,
                "anchor_residual_rank_ic": float(fold_ranks.mean()) if len(fold_ranks) else None,
                "positive_residual_ic_ratio": float((fold_values > 0).mean()) if len(fold_values) else None,
            })
        marginal_records.append({
            "id": identifier,
            "valid_dates": int(len(values)),
            "anchor_residual_ic": float(values.mean()) if len(values) else None,
            "anchor_residual_icir": float(values.mean() / values.std(ddof=1)) if len(values) > 1 and values.std(ddof=1) > 0 else None,
            "anchor_residual_rank_ic": float(ranks.mean()) if len(ranks) else None,
            "positive_residual_ic_ratio": float((values > 0).mean()) if len(values) else None,
            "rolling_residual_evidence": rolling_residual_evidence,
        })
    payload = {
        "method": "agent_shortlist_joint_rass_development_evidence_v4",
        "data_split": "rass_development",
        "evidence_period": [evidence_start_time, evidence_end_time],
        "configured_training_period": [task["train_start_time"], task["train_end_time"]],
        "prediction_mode": prediction_mode(config),
        "target_label": label_expression,
        "base_evidence_hash": base_evidence_hash,
        "shortlist": shortlist,
        "rolling_evaluation_periods": list(evidence_periods),
        "pairwise_daily_cross_sectional_correlation": pairwise_records,
        "marginal_information_after_anchor_residualization": marginal_records,
        "contains_test_derived_data": False,
    }
    payload["evidence_hash"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def summarize_rass_evidence(evidence: dict) -> dict:
    """Small deterministic audit summary; it does not select factors."""
    records = evidence["records"]
    return {
        "candidate_count": len(records),
        "coverage_range": [
            min(record["coverage"] for record in records),
            max(record["coverage"] for record in records),
        ],
        "ic_computable_count": sum(record["ic"] is not None for record in records),
        "contains_test_derived_data": False,
    }
