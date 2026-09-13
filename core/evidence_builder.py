from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from qlib.data import D

from adapters.qlib_runner import RAW_LABEL_EXPRESSION, initialize_qlib, resolve_market_instruments
from core.errors import ContractError
from core.io_utils import canonical_json_bytes, sha256_bytes, sha256_file
from evaluation.prediction_metrics import compute_signal_metrics


def _correlation(left: pd.Series, right: pd.Series) -> float | None:
    pair = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 2 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return None
    value = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    return float(value) if np.isfinite(value) else None


def build_rass_train_evidence(
    repo_root: Path,
    config: dict,
    catalog: list[dict[str, str]],
    catalog_hash: str,
) -> dict:
    initialize_qlib(repo_root, config)
    instruments, unavailable = resolve_market_instruments(repo_root, config)
    anchors = list(config["z_alpha"]["selected_features"])
    expressions = list(dict.fromkeys([item["expression"] for item in catalog] + anchors + [RAW_LABEL_EXPRESSION]))
    task = config["task"]
    data = D.features(
        instruments,
        expressions,
        start_time=task["train_start_time"],
        end_time=task["train_end_time"],
        freq="day",
    )
    if data.empty:
        raise ContractError("RASS_TRAIN_EVIDENCE_EMPTY")
    data = data.replace([np.inf, -np.inf], np.nan)
    raw_label = data[RAW_LABEL_EXPRESSION].rename("raw_label")
    correlation_sample_size = min(len(data), 20_000)
    correlation_positions = np.linspace(0, len(data) - 1, correlation_sample_size, dtype=int)
    correlation_data = data.iloc[np.unique(correlation_positions)]
    candidate_frame = correlation_data[[item["expression"] for item in catalog]].copy()
    candidate_frame.columns = [item["id"] for item in catalog]
    flattened_correlation = candidate_frame.corr(method="pearson", min_periods=20)

    records = []
    total_rows = len(data)
    train_date_count = int(data.index.get_level_values("datetime").nunique())
    for item in catalog:
        factor = data[item["expression"]].rename("prediction")
        metrics = compute_signal_metrics(factor, raw_label)
        finite = factor.replace([np.inf, -np.inf], np.nan).notna()
        daily_ic = metrics.daily["ic"] if "ic" in metrics.daily else pd.Series(dtype=float)
        yearly = {
            str(year): (None if values.empty else float(values.mean()))
            for year, values in daily_ic.groupby(daily_ic.index.year)
        }
        yearly_rank_ic = {
            str(year): (None if values.empty else float(values.mean()))
            for year, values in metrics.daily["rank_ic"].groupby(metrics.daily.index.year)
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
        if int(len(metrics.daily)) < max(20, math.ceil(train_date_count * 0.80)):
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
            "valid_ic_dates": int(len(metrics.daily)),
            "excluded_dates": metrics.excluded,
            "sign_consistency": sign_consistency,
            "yearly_ic": yearly,
            "yearly_rank_ic": yearly_rank_ic,
            "positive_ic_year_ratio": float(sum(value > 0 for value in yearly.values() if value is not None) / sum(value is not None for value in yearly.values())) if any(value is not None for value in yearly.values()) else None,
            "worst_year_ic": min((value for value in yearly.values() if value is not None), default=None),
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
        "method": "deterministic_train_only_factor_evidence_v1",
        "data_split": "train",
        "training_period": [task["train_start_time"], task["train_end_time"]],
        "target_label": RAW_LABEL_EXPRESSION,
        "label_hash": sha256_bytes(RAW_LABEL_EXPRESSION.encode("utf-8")),
        "initial_features": anchors,
        "catalog_hash": catalog_hash,
        "effective_config_hash": config["provenance"]["effective_config_hash"],
        "input_data_hash": config["provenance"]["input_hashes"]["qlib_relevant_manifest"],
        "evidence_code_hash": sha256_file(Path(__file__)),
        "available_instrument_count": len(instruments),
        "unavailable_placeholder_count": len(unavailable),
        "row_count": int(total_rows),
        "candidate_count": len(records),
        "eligible_candidate_count": eligible_candidate_count,
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
    """Build detailed train-only joint evidence for an Agent-selected shortlist."""
    initialize_qlib(repo_root, config)
    instruments, _ = resolve_market_instruments(repo_root, config)
    anchors = list(config["z_alpha"]["selected_features"])
    ids = [item["id"] for item in shortlist]
    expressions = [item["expression"] for item in shortlist]
    requested = list(dict.fromkeys(anchors + expressions + [RAW_LABEL_EXPRESSION]))
    task = config["task"]
    data = D.features(
        instruments,
        requested,
        start_time=task["train_start_time"],
        end_time=task["train_end_time"],
        freq="day",
    ).replace([np.inf, -np.inf], np.nan)
    if data.empty:
        raise ContractError("RASS_SHORTLIST_EVIDENCE_EMPTY")

    renamed = data[expressions].copy()
    renamed.columns = ids
    pairwise: dict[tuple[str, str], list[float]] = {
        (left, right): []
        for index, left in enumerate(ids)
        for right in ids[index + 1:]
    }
    marginal: dict[str, list[float]] = {identifier: [] for identifier in ids}
    marginal_rank: dict[str, list[float]] = {identifier: [] for identifier in ids}
    for date, candidate_day in renamed.groupby(level="datetime", sort=True):
        candidate_day = candidate_day.droplevel("datetime")
        corr = candidate_day.corr(method="pearson", min_periods=20)
        for pair, values in pairwise.items():
            value = corr.loc[pair[0], pair[1]]
            if np.isfinite(value):
                values.append(float(value))
        source_day = data.xs(date, level="datetime")
        for identifier, expression in zip(ids, expressions):
            frame = source_day[anchors + [expression, RAW_LABEL_EXPRESSION]].dropna()
            if len(frame) < 20:
                continue
            x = frame[anchors].to_numpy(dtype=float)
            candidate_values = frame[expression].to_numpy(dtype=float)
            label_values = frame[RAW_LABEL_EXPRESSION].to_numpy(dtype=float)
            x = np.column_stack([np.ones(len(x)), x])
            residual = candidate_values - x @ np.linalg.lstsq(x, candidate_values, rcond=None)[0]
            if np.std(residual) > 0 and np.std(label_values) > 0:
                value = np.corrcoef(residual, label_values)[0, 1]
                if np.isfinite(value):
                    marginal[identifier].append(float(value))
                rank_value = pd.Series(residual).corr(pd.Series(label_values), method="spearman")
                if np.isfinite(rank_value):
                    marginal_rank[identifier].append(float(rank_value))

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
        values = np.asarray(marginal[identifier], dtype=float)
        ranks = np.asarray(marginal_rank[identifier], dtype=float)
        marginal_records.append({
            "id": identifier,
            "valid_dates": int(len(values)),
            "anchor_residual_ic": float(values.mean()) if len(values) else None,
            "anchor_residual_icir": float(values.mean() / values.std(ddof=1)) if len(values) > 1 and values.std(ddof=1) > 0 else None,
            "anchor_residual_rank_ic": float(ranks.mean()) if len(ranks) else None,
            "positive_residual_ic_ratio": float((values > 0).mean()) if len(values) else None,
        })
    payload = {
        "method": "agent_shortlist_joint_train_evidence_v1",
        "data_split": "train",
        "training_period": [task["train_start_time"], task["train_end_time"]],
        "base_evidence_hash": base_evidence_hash,
        "shortlist": shortlist,
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
