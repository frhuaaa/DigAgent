from __future__ import annotations

"""Reference-only portfolio example.

This file illustrates the existing portfolio/backtest mechanics but is not the
authoritative DiagAgent implementation. Codex may modify or replace it when
building the runnable system. In particular, the production implementation
must obtain the benchmark from submit JSON ``task.benchmark`` and must not use
the temporary equal-weight benchmark contained in this example.
"""

import argparse
import json
import math
import types
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================

def load_config_from_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cfg = types.SimpleNamespace()

    for k, v in data.items():
        setattr(cfg, k, v)

    return cfg


# ============================================================
# IO / Utility
# ============================================================

def normalize_code(code: str) -> str:
    return str(code).replace(".", "_")


def read_panel(
    path: str,
    normalize_cols: bool = True,
) -> pd.DataFrame:

    df = pd.read_csv(
        path,
        index_col=0,
    )

    df.index = pd.Index(
        df.index.astype(str),
        name="date",
    )

    if normalize_cols:
        df.columns = [
            normalize_code(c)
            for c in df.columns
        ]

    return df.apply(
        pd.to_numeric,
        errors="coerce",
    )


def zscore_alpha(
    alpha: pd.Series,
) -> pd.Series:

    x = (
        alpha
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(x) == 0:
        return x

    # Winsorize
    lo, hi = x.quantile([0.01, 0.99])

    x = x.clip(
        lower=lo,
        upper=hi,
    )

    std = x.std(ddof=0)

    if not np.isfinite(std) or std <= 1e-12:
        return x * 0.0

    return (x - x.mean()) / std


def max_drawdown(
    nav: pd.Series,
) -> float:

    cummax = nav.cummax()

    dd = nav / cummax - 1.0

    return float(dd.min())


# ============================================================
# Box-constrained simplex projection
#
# Project onto:
#
#   sum(w) = z
#   lo_i <= w_i <= hi_i
#
# ============================================================

def project_box_simplex(
    v: np.ndarray,
    z: float,
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:

    v = np.asarray(
        v,
        dtype=float,
    )

    lo = np.asarray(
        lo,
        dtype=float,
    )

    hi = np.asarray(
        hi,
        dtype=float,
    )

    if len(v) != len(lo) or len(v) != len(hi):
        raise ValueError(
            "v, lo and hi must have the same length."
        )

    if np.any(lo > hi + 1e-12):
        bad = np.where(lo > hi + 1e-12)[0]

        raise ValueError(
            f"Lower bound exceeds upper bound "
            f"for indices {bad[:10]}"
        )

    min_sum = float(lo.sum())
    max_sum = float(hi.sum())

    if z < min_sum - 1e-10:
        raise ValueError(
            f"Infeasible target sum: "
            f"target={z:.12f}, "
            f"minimum={min_sum:.12f}"
        )

    if z > max_sum + 1e-10:
        raise ValueError(
            f"Infeasible target sum: "
            f"target={z:.12f}, "
            f"maximum={max_sum:.12f}"
        )

    # Bisection:
    #
    # w_i = clip(v_i - theta, lo_i, hi_i)
    #
    lower_theta = float(
        np.min(v - hi)
    )

    upper_theta = float(
        np.max(v - lo)
    )

    for _ in range(120):

        theta = (
            lower_theta
            + upper_theta
        ) / 2.0

        w = np.clip(
            v - theta,
            lo,
            hi,
        )

        if w.sum() > z:
            lower_theta = theta
        else:
            upper_theta = theta

    theta = (
        lower_theta
        + upper_theta
    ) / 2.0

    w = np.clip(
        v - theta,
        lo,
        hi,
    )

    # Tiny numerical correction
    diff = z - w.sum()

    if abs(diff) > 1e-10:

        free = np.where(
            (w > lo + 1e-10)
            &
            (w < hi - 1e-10)
        )[0]

        if len(free) > 0:

            w[free] += (
                diff / len(free)
            )

            w = np.clip(
                w,
                lo,
                hi,
            )

    return w


# ============================================================
# Risk estimation
# ============================================================

def estimate_risk(
    hist_ret: pd.DataFrame,
    codes: List[str],
    cfg,
):

    hist = (
        hist_ret
        .reindex(columns=codes)
        .tail(cfg.lookback)
    )

    hist = hist.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    hist = hist.dropna(
        how="all",
    )

    hist = hist.fillna(0.0)

    # --------------------------------
    # Not enough history
    # --------------------------------
    if len(hist) < cfg.min_lookback:

        # 2% daily vol squared
        var = np.full(
            len(codes),
            0.0004,
            dtype=float,
        )

        return var, None

    x = hist.to_numpy(
        dtype=float,
    )

    x = (
        x
        - x.mean(
            axis=0,
            keepdims=True,
        )
    )

    # --------------------------------
    # Full sample covariance
    # --------------------------------
    if cfg.risk_model_mode == "sample_cov":

        sample = (
            x.T @ x
        ) / max(
            len(hist) - 1,
            1,
        )

        diag = np.diag(
            np.diag(sample)
        )

        rho = float(
            cfg.cov_shrink_to_diag
        )

        cov = (
            (1.0 - rho) * sample
            + rho * diag
        )

        diag_values = np.maximum(
            np.diag(cov),
            cfg.variance_floor,
        )

        cov[
            np.diag_indices_from(cov)
        ] = diag_values

        return (
            np.diag(cov),
            cov,
        )

    # --------------------------------
    # Diagonal risk model
    # --------------------------------
    var = x.var(
        axis=0,
        ddof=1,
    )

    var = np.maximum(
        var,
        cfg.variance_floor,
    )

    return var, None


# ============================================================
# Mean-Variance Optimization
#
# maximize:
#
#   mu' w
#   - lambda * w' Sigma w
#   - gamma * ||w - old_w||^2
#
# subject to:
#
#   sum(w) = target_sum
#   lower_i <= w_i <= upper_i
#
# ============================================================

def optimize_mean_variance(
    mu: np.ndarray,
    var_diag: np.ndarray,
    cov: np.ndarray | None,
    old_w: np.ndarray,
    cfg,
    target_sum: float,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:

    mu = np.asarray(
        mu,
        dtype=float,
    )

    var_diag = np.asarray(
        var_diag,
        dtype=float,
    )

    old_w = np.asarray(
        old_w,
        dtype=float,
    )

    lower_bounds = np.asarray(
        lower_bounds,
        dtype=float,
    )

    upper_bounds = np.asarray(
        upper_bounds,
        dtype=float,
    )

    n = len(mu)

    if n == 0:
        return np.array(
            [],
            dtype=float,
        )

    min_sum = float(
        lower_bounds.sum()
    )

    max_sum = float(
        upper_bounds.sum()
    )

    if target_sum < min_sum - 1e-10:
        raise ValueError(
            f"Target {target_sum:.8f} "
            f"is smaller than lower-bound sum "
            f"{min_sum:.8f}"
        )

    if target_sum > max_sum + 1e-10:
        raise ValueError(
            f"Target {target_sum:.8f} "
            f"is larger than upper-bound sum "
            f"{max_sum:.8f}"
        )

    # ========================================================
    # Diagonal covariance:
    # exact KKT solution
    # ========================================================

    if cov is None:

        # Minimize:
        #
        # lambda var_i w_i^2
        # + gamma (w_i-old_i)^2
        # - mu_i w_i
        #
        # => a_i w_i^2 - b_i w_i

        a = (
            cfg.risk_aversion
            * var_diag
            + cfg.turnover_penalty
        )

        a = np.maximum(
            a,
            1e-12,
        )

        b = (
            mu
            + 2.0
            * cfg.turnover_penalty
            * old_w
        )

        # w_i(theta)
        #
        # = clip(
        #   (b_i-theta)/(2a_i),
        #   lower_i,
        #   upper_i
        # )

        lo_theta = float(
            np.min(
                b
                - 2.0
                * a
                * upper_bounds
            )
        )

        hi_theta = float(
            np.max(
                b
                - 2.0
                * a
                * lower_bounds
            )
        )

        for _ in range(120):

            theta = (
                lo_theta
                + hi_theta
            ) / 2.0

            w = np.clip(
                (b - theta)
                / (2.0 * a),
                lower_bounds,
                upper_bounds,
            )

            if w.sum() > target_sum:
                lo_theta = theta
            else:
                hi_theta = theta

        theta = (
            lo_theta
            + hi_theta
        ) / 2.0

        w = np.clip(
            (b - theta)
            / (2.0 * a),
            lower_bounds,
            upper_bounds,
        )

        # Final projection to guarantee sum
        w = project_box_simplex(
            w,
            target_sum,
            lower_bounds,
            upper_bounds,
        )

        return w

    # ========================================================
    # Full covariance:
    # projected gradient descent
    # ========================================================

    try:

        w = project_box_simplex(
            old_w.copy(),
            target_sum,
            lower_bounds,
            upper_bounds,
        )

    except Exception:

        initial = (
            lower_bounds
            + upper_bounds
        ) / 2.0

        w = project_box_simplex(
            initial,
            target_sum,
            lower_bounds,
            upper_bounds,
        )

    row_sum_bound = float(
        np.max(
            np.sum(
                np.abs(cov),
                axis=1,
            )
        )
    )

    lip = (
        2.0
        * cfg.risk_aversion
        * row_sum_bound
        +
        2.0
        * cfg.turnover_penalty
        +
        1e-12
    )

    step = min(
        1.0,
        1.0 / lip,
    )

    prev_obj = np.inf

    for _ in range(
        cfg.max_iter
    ):

        grad = (
            2.0
            * cfg.risk_aversion
            * (cov @ w)
            +
            2.0
            * cfg.turnover_penalty
            * (w - old_w)
            -
            mu
        )

        tentative = (
            w
            - step * grad
        )

        new_w = project_box_simplex(
            tentative,
            target_sum,
            lower_bounds,
            upper_bounds,
        )

        risk = float(
            new_w
            @ cov
            @ new_w
        )

        turnover_pen = float(
            np.sum(
                (
                    new_w
                    - old_w
                ) ** 2
            )
        )

        alpha_value = float(
            np.dot(
                mu,
                new_w,
            )
        )

        # minimization form
        obj = (
            cfg.risk_aversion
            * risk
            +
            cfg.turnover_penalty
            * turnover_pen
            -
            alpha_value
        )

        if abs(
            prev_obj - obj
        ) < cfg.tol:

            w = new_w
            break

        w = new_w
        prev_obj = obj

    return w


# ============================================================
# Weight drift after realized return
# ============================================================

def drift_weights_after_return(
    post_trade_weights: pd.Series,
    realized_return: pd.Series,
    gross_return: float,
    drop_threshold: float,
) -> Dict[str, float]:

    if post_trade_weights.empty:
        return {}

    realized = (
        realized_return
        .reindex(
            post_trade_weights.index
        )
        .fillna(0.0)
    )

    denominator = (
        1.0
        + gross_return
    )

    if (
        not np.isfinite(denominator)
        or denominator <= 1e-12
    ):

        return {}

    drifted = (
        post_trade_weights
        * (1.0 + realized)
        / denominator
    )

    drifted = (
        drifted
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    return {
        c: float(w)
        for c, w
        in drifted.items()
        if w > drop_threshold
    }


# ============================================================
# Objective diagnostics
# ============================================================

def calculate_objective_components(
    mu: np.ndarray,
    var_diag: np.ndarray,
    cov: np.ndarray | None,
    new_w: np.ndarray,
    old_w: np.ndarray,
    cfg,
):

    alpha_term = float(
        np.dot(
            mu,
            new_w,
        )
    )

    if cov is None:

        risk_raw = float(
            np.sum(
                var_diag
                * new_w ** 2
            )
        )

    else:

        risk_raw = float(
            new_w
            @ cov
            @ new_w
        )

    turnover_raw = float(
        np.sum(
            (
                new_w
                - old_w
            ) ** 2
        )
    )

    risk_term = float(
        cfg.risk_aversion
        * risk_raw
    )

    turnover_term = float(
        cfg.turnover_penalty
        * turnover_raw
    )

    objective_value = (
        alpha_term
        - risk_term
        - turnover_term
    )

    denom = max(
        abs(alpha_term),
        1e-20,
    )

    return {
        "alpha_term": alpha_term,
        "risk_raw": risk_raw,
        "risk_term": risk_term,
        "turnover_raw": turnover_raw,
        "turnover_term": turnover_term,
        "objective_value": objective_value,
        "risk_alpha_ratio": (
            risk_term / denom
        ),
        "turnover_alpha_ratio": (
            turnover_term / denom
        ),
    }


# ============================================================
# Backtest
# ============================================================

def run_backtest(
    cfg,
) -> Dict[str, float]:

    out = Path(
        cfg.output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load data
    # ========================================================

    alpha = read_panel(
        cfg.alpha_path
    )

    ret = read_panel(
        cfg.ret_path
    )

    limit_up_mask = read_panel(
        cfg.limit_up_mask_path
    )

    limit_down_mask = read_panel(
        cfg.limit_down_mask_path
    )

    # ========================================================
    # Common universe
    # ========================================================

    common_codes = sorted(
        set(alpha.columns)
        &
        set(ret.columns)
        &
        set(limit_up_mask.columns)
        &
        set(limit_down_mask.columns)
    )

    if len(common_codes) == 0:
        raise RuntimeError(
            "No common stock codes "
            "across alpha, return and masks."
        )

    alpha = alpha[
        common_codes
    ]

    ret = ret[
        common_codes
    ]

    limit_up_mask = (
        limit_up_mask[
            common_codes
        ]
    )

    limit_down_mask = (
        limit_down_mask[
            common_codes
        ]
    )

    # ========================================================
    # Dates
    # ========================================================

    ret_dates = list(
        ret.index
    )

    ret_pos = {
        d: i
        for i, d
        in enumerate(ret_dates)
    }

    alpha_dates = [
        d
        for d in alpha.index
        if d in ret_pos
    ]

    # ========================================================
    # State
    # ========================================================

    account_rows = []
    order_rows = []
    objective_rows = []

    weights: Dict[
        str,
        float,
    ] = {}

    account_value = float(
        cfg.init_cash
    )

    # ========================================================
    # Basic feasibility
    # ========================================================

    if cfg.full_position:

        min_required = math.ceil(
            1.0
            / cfg.max_weight
        )

        if (
            cfg.candidate_count
            < min_required
        ):

            raise ValueError(
                "candidate_count must be >= "
                "ceil(1/max_weight). "
                f"Current minimum = {min_required}"
            )

    # ========================================================
    # Main loop
    # ========================================================

    for alpha_date in alpha_dates:

        i = ret_pos.get(
            alpha_date
        )

        if i is None:
            continue

        trade_i = (
            i
            + cfg.trade_delay_days
        )

        return_i = (
            i
            + cfg.return_delay_days
        )

        if (
            trade_i >= len(ret_dates)
            or
            return_i >= len(ret_dates)
        ):
            continue

        trade_date = (
            ret_dates[
                trade_i
            ]
        )

        return_date = (
            ret_dates[
                return_i
            ]
        )

        # ====================================================
        # Mask coverage
        # ====================================================

        if (
            trade_date
            not in limit_up_mask.index
            or
            trade_date
            not in limit_down_mask.index
        ):
            continue

        # ====================================================
        # IMPORTANT MASK SEMANTICS
        #
        # Assumption:
        #
        # mask_limit_up == 1:
        #     stock is limit-up
        #
        # mask_limit_down == 1:
        #     stock is limit-down
        #
        # Therefore:
        #
        # limit-up:
        #     cannot BUY / increase
        #
        # limit-down:
        #     cannot SELL / decrease
        #
        # Missing values treated conservatively
        # as not tradable in that direction.
        # ====================================================

        is_limit_up = (
            limit_up_mask
            .loc[trade_date]
            .reindex(common_codes)
            .fillna(1.0)
            > 0
        )

        is_limit_down = (
            limit_down_mask
            .loc[trade_date]
            .reindex(common_codes)
            .fillna(1.0)
            > 0
        )

        buyable = (
            ~is_limit_up
        )

        sellable = (
            ~is_limit_down
        )

        # ====================================================
        # Current drifted holdings
        # ====================================================

        old = pd.Series(
            weights,
            dtype=float,
        )

        old = old[
            old
            >
            cfg.weight_threshold_for_drop
        ]

        # ====================================================
        # Alpha
        # ====================================================

        a_raw = (
            alpha
            .loc[alpha_date]
            .reindex(common_codes)
        )

        a_z = zscore_alpha(
            a_raw
        )

        if len(a_z) == 0:
            continue

        # ====================================================
        # Candidate selection
        #
        # New-entry candidates:
        #   Top alpha among BUYABLE stocks
        #
        # Existing holdings:
        #   Always retained in optimizer universe
        #
        # ====================================================

        ranked_all = list(
            a_z
            .sort_values(
                ascending=False
            )
            .index
        )

        buyable_ranked = [
            c
            for c in ranked_all
            if bool(
                buyable.get(
                    c,
                    False,
                )
            )
        ]

        top_codes = (
            buyable_ranked[
                : cfg.candidate_count
            ]
        )

        held_codes = [
            c
            for c, w
            in old.items()
            if (
                w
                >
                cfg.weight_threshold_for_drop
            )
        ]

        codes = list(
            dict.fromkeys(
                top_codes
                + held_codes
            )
        )

        if not codes:
            continue

        # ====================================================
        # Old weights
        # ====================================================

        old_w = np.array(
            [
                old.get(
                    c,
                    0.0,
                )
                for c in codes
            ],
            dtype=float,
        )

        # ====================================================
        # Directional bounds
        # ====================================================

        lower_bounds = np.full(
            len(codes),
            cfg.min_weight,
            dtype=float,
        )

        upper_bounds = np.full(
            len(codes),
            cfg.max_weight,
            dtype=float,
        )

        for j, c in enumerate(
            codes
        ):

            old_value = old_w[j]

            can_buy = bool(
                buyable.get(
                    c,
                    False,
                )
            )

            can_sell = bool(
                sellable.get(
                    c,
                    False,
                )
            )

            # --------------------------------
            # Cannot sell:
            # new weight >= old weight
            # --------------------------------

            if not can_sell:

                lower_bounds[j] = max(
                    lower_bounds[j],
                    old_value,
                )

                # Existing drifted position
                # may already exceed max_weight.
                # It must still be feasible
                # because we cannot force a sale.
                upper_bounds[j] = max(
                    upper_bounds[j],
                    old_value,
                )

            # --------------------------------
            # Cannot buy:
            # new weight <= old weight
            # --------------------------------

            if not can_buy:

                upper_bounds[j] = min(
                    upper_bounds[j],
                    old_value,
                )

            # --------------------------------
            # Cannot buy or sell:
            # fully frozen
            # --------------------------------

            if (
                not can_buy
                and
                not can_sell
            ):

                lower_bounds[j] = (
                    old_value
                )

                upper_bounds[j] = (
                    old_value
                )

            if (
                lower_bounds[j]
                >
                upper_bounds[j]
                + 1e-12
            ):

                raise RuntimeError(
                    f"Inconsistent bounds "
                    f"for {c} on {trade_date}: "
                    f"old={old_value:.8f}, "
                    f"lower={lower_bounds[j]:.8f}, "
                    f"upper={upper_bounds[j]:.8f}, "
                    f"buyable={can_buy}, "
                    f"sellable={can_sell}"
                )

        # ====================================================
        # Full-position feasibility
        # ====================================================

        if cfg.full_position:

            target_sum = 1.0

        else:

            target_sum = min(
                1.0,
                float(
                    upper_bounds.sum()
                ),
            )

        min_possible = float(
            lower_bounds.sum()
        )

        max_possible = float(
            upper_bounds.sum()
        )

        # If full position becomes infeasible
        # because too many names are restricted,
        # use the maximum feasible invested weight.
        #
        # This avoids inventing impossible trades.
        if target_sum > max_possible:

            target_sum = (
                max_possible
            )

        if target_sum < min_possible:

            target_sum = (
                min_possible
            )

        # ====================================================
        # Alpha vector
        # ====================================================

        alpha_raw_vector = (
            a_z
            .reindex(codes)
            .fillna(0.0)
            .to_numpy(
                dtype=float
            )
        )

        mu = (
            alpha_raw_vector
            * cfg.alpha_scale
        )

        # ====================================================
        # Risk history
        #
        # Uses data up to alpha_date T.
        #
        # It intentionally does NOT include
        # T+1 information.
        # ====================================================

        hist = ret.iloc[
            : i + 1
        ]

        var_diag, cov = (
            estimate_risk(
                hist,
                codes,
                cfg,
            )
        )

        # ====================================================
        # Optimize
        # ====================================================

        new_w = (
            optimize_mean_variance(
                mu=mu,
                var_diag=var_diag,
                cov=cov,
                old_w=old_w,
                cfg=cfg,
                target_sum=target_sum,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
        )

        target = pd.Series(
            new_w,
            index=codes,
            dtype=float,
        )

        target[
            target.abs()
            <
            cfg.weight_threshold_for_drop
        ] = 0.0

        # ====================================================
        # Objective diagnostics
        # ====================================================

        obj_diag = (
            calculate_objective_components(
                mu=mu,
                var_diag=var_diag,
                cov=cov,
                new_w=new_w,
                old_w=old_w,
                cfg=cfg,
            )
        )

        objective_rows.append({
            "alpha_date":
                alpha_date,

            "trade_date":
                trade_date,

            "return_date":
                return_date,

            "alpha_raw_term":
                float(
                    np.dot(
                        alpha_raw_vector,
                        new_w,
                    )
                ),

            "alpha_term":
                obj_diag[
                    "alpha_term"
                ],

            "risk_raw":
                obj_diag[
                    "risk_raw"
                ],

            "risk_term":
                obj_diag[
                    "risk_term"
                ],

            "turnover_raw":
                obj_diag[
                    "turnover_raw"
                ],

            "turnover_term":
                obj_diag[
                    "turnover_term"
                ],

            "objective_value":
                obj_diag[
                    "objective_value"
                ],

            "risk_alpha_ratio":
                obj_diag[
                    "risk_alpha_ratio"
                ],

            "turnover_alpha_ratio":
                obj_diag[
                    "turnover_alpha_ratio"
                ],

            "candidate_count":
                len(codes),

            "holding_count":
                int(
                    (
                        new_w
                        >
                        cfg.weight_threshold_for_drop
                    ).sum()
                ),

            "limit_up_count":
                int(
                    is_limit_up.sum()
                ),

            "limit_down_count":
                int(
                    is_limit_down.sum()
                ),
        })

        # ====================================================
        # Orders
        # ====================================================

        all_order_codes = sorted(
            set(old.index)
            |
            set(target.index)
        )

        old_all = (
            old
            .reindex(
                all_order_codes
            )
            .fillna(0.0)
        )

        new_all = (
            target
            .reindex(
                all_order_codes
            )
            .fillna(0.0)
        )

        delta = (
            new_all
            - old_all
        )

        # Safety assertions:
        #
        # cannot BUY limit-up
        # cannot SELL limit-down

        for c, dw in delta.items():

            if (
                dw > 1e-10
                and
                bool(
                    is_limit_up.get(
                        c,
                        False,
                    )
                )
            ):

                raise RuntimeError(
                    f"Illegal BUY on limit-up stock "
                    f"{c} at {trade_date}"
                )

            if (
                dw < -1e-10
                and
                bool(
                    is_limit_down.get(
                        c,
                        False,
                    )
                )
            ):

                raise RuntimeError(
                    f"Illegal SELL on limit-down stock "
                    f"{c} at {trade_date}"
                )

        buy_turn = float(
            delta
            .clip(
                lower=0.0
            )
            .sum()
        )

        sell_turn = float(
            (
                -delta
                .clip(
                    upper=0.0
                )
            )
            .sum()
        )

        cost_rate = (
            buy_turn
            * (
                cfg.buy_commission
                +
                cfg.buy_slippage
            )
            +
            sell_turn
            * (
                cfg.sell_commission
                +
                cfg.sell_tax
                +
                cfg.sell_slippage
            )
        )

        # ====================================================
        # Realized return
        # ====================================================

        realized = (
            ret
            .loc[return_date]
            .reindex(
                new_all.index
            )
            .fillna(0.0)
        )

        gross_return = float(
            (
                new_all
                * realized
            ).sum()
        )

        net_return = (
            gross_return
            - cost_rate
        )

        start_value = (
            account_value
        )

        account_value *= (
            1.0
            + net_return
        )

        # ====================================================
        # Temporary equal-weight benchmark
        # ====================================================

        bm_codes = [
            c
            for c in common_codes
            if not bool(
                is_limit_up.get(
                    c,
                    False,
                )
            )
            and not bool(
                is_limit_down.get(
                    c,
                    False,
                )
            )
        ]

        if len(bm_codes):

            bm_return = float(
                ret
                .loc[return_date]
                .reindex(
                    bm_codes
                )
                .dropna()
                .mean()
            )

        else:

            bm_return = 0.0

        excess_return = (
            net_return
            - bm_return
        )

        # ====================================================
        # Save order details
        # ====================================================

        active_orders = delta[
            delta.abs()
            >
            cfg.weight_threshold_for_drop
        ]

        for c, dw in active_orders.items():

            side = (
                "BUY"
                if dw > 0
                else "SELL"
            )

            unit_cost = (
                cfg.buy_commission
                +
                cfg.buy_slippage
                if dw > 0
                else
                cfg.sell_commission
                +
                cfg.sell_tax
                +
                cfg.sell_slippage
            )

            order_rows.append({
                "alpha_date":
                    alpha_date,

                "trade_date":
                    trade_date,

                "return_date":
                    return_date,

                "code":
                    c,

                "side":
                    side,

                "is_limit_up":
                    bool(
                        is_limit_up.get(
                            c,
                            False,
                        )
                    ),

                "is_limit_down":
                    bool(
                        is_limit_down.get(
                            c,
                            False,
                        )
                    ),

                "alpha_z":
                    float(
                        a_z.get(
                            c,
                            np.nan,
                        )
                    ),

                "old_weight":
                    float(
                        old_all[c]
                    ),

                "target_weight":
                    float(
                        new_all[c]
                    ),

                "order_weight":
                    float(dw),

                "estimated_trade_value":
                    float(
                        abs(dw)
                        * start_value
                    ),

                "estimated_cost":
                    float(
                        abs(dw)
                        * start_value
                        * unit_cost
                    ),
            })

        # ====================================================
        # Account details
        # ====================================================

        account_rows.append({
            "alpha_date":
                alpha_date,

            "trade_date":
                trade_date,

            "return_date":
                return_date,

            "start_value":
                start_value,

            "end_value":
                account_value,

            "gross_return":
                gross_return,

            "cost_rate":
                cost_rate,

            "net_return":
                net_return,

            "benchmark_return":
                bm_return,

            "excess_return":
                excess_return,

            "buy_turn":
                buy_turn,

            "sell_turn":
                sell_turn,

            "one_way_turn":
                (
                    buy_turn
                    + sell_turn
                ) / 2.0,

            "holding_count":
                int(
                    (
                        new_all
                        >
                        cfg.weight_threshold_for_drop
                    ).sum()
                ),

            "optimizer_universe_count":
                len(codes),

            "limit_up_count":
                int(
                    is_limit_up.sum()
                ),

            "limit_down_count":
                int(
                    is_limit_down.sum()
                ),

            "invested_weight":
                float(
                    new_all.sum()
                ),

            "risk_model_mode":
                cfg.risk_model_mode,

            "risk_aversion":
                cfg.risk_aversion,

            "turnover_penalty":
                cfg.turnover_penalty,

            "alpha_scale":
                cfg.alpha_scale,
        })

        # ====================================================
        # Drift holdings to next period
        # ====================================================

        weights = (
            drift_weights_after_return(
                post_trade_weights=
                    new_all,

                realized_return=
                    realized,

                gross_return=
                    gross_return,

                drop_threshold=
                    cfg.weight_threshold_for_drop,
            )
        )

    # ========================================================
    # DataFrames
    # ========================================================

    account = pd.DataFrame(
        account_rows
    )

    orders = pd.DataFrame(
        order_rows
    )

    objective_df = pd.DataFrame(
        objective_rows
    )

    if account.empty:

        raise RuntimeError(
            "No backtest rows generated. "
            "Check dates, alpha, returns, "
            "and limit masks."
        )

    # ========================================================
    # Output paths
    # ========================================================

    account_path = (
        out
        / "account_details.csv"
    )

    order_path = (
        out
        / "order_details.csv"
    )

    objective_path = (
        out
        / "objective_diagnostics.csv"
    )

    metric_path = (
        out
        / "metric.json"
    )

    config_path = (
        out
        / "mean_variance_config.json"
    )

    # ========================================================
    # Performance metrics
    # ========================================================

    n = len(account)

    ann = 252.0

    net_ret = (
        account[
            "net_return"
        ]
        .fillna(0.0)
    )

    strategy_nav = (
        1.0
        + net_ret
    ).cumprod()

    final_nav = float(
        strategy_nav.iloc[-1]
    )

    total_return = (
        final_nav - 1.0
    )

    annual_return = (
        float(
            final_nav
            ** (
                ann / n
            )
            - 1.0
        )
        if n > 0
        else np.nan
    )

    daily_mean = float(
        net_ret.mean()
    )

    daily_std = (
        float(
            net_ret.std(
                ddof=1
            )
        )
        if n > 1
        else np.nan
    )

    annual_volatility = (
        float(
            daily_std
            * math.sqrt(ann)
        )
        if np.isfinite(
            daily_std
        )
        else np.nan
    )

    # ========================================================
    # STANDARD SHARPE
    #
    # rf = 0
    #
    # mean(daily return)
    # ------------------ * sqrt(252)
    # std(daily return)
    # ========================================================

    sharpe_ratio = (
        float(
            daily_mean
            / daily_std
            * math.sqrt(ann)
        )
        if (
            np.isfinite(
                daily_std
            )
            and
            daily_std > 0
        )
        else np.nan
    )

    strategy_max_drawdown = (
        max_drawdown(
            strategy_nav
        )
    )

    # ========================================================
    # Objective summary
    # ========================================================

    objective_summary = {}

    if not objective_df.empty:

        for col in [
            "alpha_term",
            "risk_term",
            "turnover_term",
            "risk_alpha_ratio",
            "turnover_alpha_ratio",
        ]:

            values = (
                objective_df[col]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .dropna()
            )

            if len(values):

                objective_summary[
                    col
                ] = {
                    "mean":
                        float(
                            values.mean()
                        ),

                    "median":
                        float(
                            values.median()
                        ),

                    "p25":
                        float(
                            values.quantile(
                                0.25
                            )
                        ),

                    "p75":
                        float(
                            values.quantile(
                                0.75
                            )
                        ),

                    "p90":
                        float(
                            values.quantile(
                                0.90
                            )
                        ),
                }

    # ========================================================
    # Metric
    # ========================================================

    metric = {
        "final_nav":
            final_nav,

        "total_return":
            float(
                total_return
            ),

        "annual_return":
            annual_return,

        "annual_volatility":
            annual_volatility,

        "sharpe_ratio":
            sharpe_ratio,

        "max_drawdown":
            strategy_max_drawdown,

        "sell_turn_mean":
            float(
                account[
                    "sell_turn"
                ].mean()
            ),

        "buy_turn_mean":
            float(
                account[
                    "buy_turn"
                ].mean()
            ),

        "one_way_turnover_mean":
            float(
                account[
                    "one_way_turn"
                ].mean()
            ),

        "holding_count_mean":
            float(
                account[
                    "holding_count"
                ].mean()
            ),

        "optimizer_universe_count_mean":
            float(
                account[
                    "optimizer_universe_count"
                ].mean()
            ),

        "invested_weight_mean":
            float(
                account[
                    "invested_weight"
                ].mean()
            ),

        "n_periods":
            int(n),

        "label_alignment":
            (
                "alpha_t -> "
                "trade T+1 close -> "
                "return row T+2 = "
                "close(T+2)/close(T+1)-1"
            ),

        "execution_constraints":
            {
                "limit_up":
                    "cannot buy/increase weight",

                "limit_down":
                    "cannot sell/decrease weight",
            },

        "optimizer":
            (
                "mean_variance_with_directional_"
                "limit_constraints"
            ),

        "objective":
            (
                "maximize "
                "alpha'w "
                "- risk_aversion*w'Cov*w "
                "- turnover_penalty*"
                "||w-w_old||^2"
            ),

        "objective_diagnostics":
            objective_summary,

        "config":
            vars(cfg),
    }

    # ========================================================
    # NAVs
    # ========================================================

    account[
        "strategy_nav"
    ] = strategy_nav

    if (
        "benchmark_return"
        in account.columns
    ):

        account[
            "benchmark_nav"
        ] = (
            1.0
            +
            account[
                "benchmark_return"
            ].fillna(0.0)
        ).cumprod()

    else:

        account[
            "benchmark_nav"
        ] = np.nan

    if (
        "excess_return"
        in account.columns
    ):

        account[
            "excess_nav"
        ] = (
            1.0
            +
            account[
                "excess_return"
            ].fillna(0.0)
        ).cumprod()

    else:

        account[
            "excess_nav"
        ] = np.nan

    # ========================================================
    # Save CSV / JSON
    # ========================================================

    account.to_csv(
        account_path,
        index=False,
    )

    orders.to_csv(
        order_path,
        index=False,
    )

    objective_df.to_csv(
        objective_path,
        index=False,
    )

    with open(
        metric_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metric,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            vars(cfg),
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Plots
    # ========================================================

    x = pd.to_datetime(
        account[
            "return_date"
        ].astype(str),
        errors="coerce",
    )

    # Strategy NAV
    nav_fig_path = (
        out
        / "strategy_nav.png"
    )

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        x,
        account[
            "strategy_nav"
        ],
        label="Strategy NAV",
    )

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
    )

    plt.title(
        "Strategy Net Value Curve"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "NAV"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        nav_fig_path,
        dpi=150,
    )

    plt.close()

    # Excess NAV
    excess_fig_path = (
        out
        / "excess_nav.png"
    )

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        x,
        account[
            "excess_nav"
        ],
        label="Excess NAV",
    )

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
    )

    plt.title(
        "Excess Net Value Curve"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "NAV"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        excess_fig_path,
        dpi=150,
    )

    plt.close()

    # Combined
    combined_fig_path = (
        out
        / "backtest_performance.png"
    )

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        x,
        account[
            "strategy_nav"
        ],
        label="Strategy NAV",
    )

    if (
        account[
            "benchmark_nav"
        ]
        .notna()
        .any()
    ):

        plt.plot(
            x,
            account[
                "benchmark_nav"
            ],
            label="Benchmark NAV",
        )

    if (
        account[
            "excess_nav"
        ]
        .notna()
        .any()
    ):

        plt.plot(
            x,
            account[
                "excess_nav"
            ],
            label="Excess NAV",
        )

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
    )

    plt.title(
        "Mean-Variance Backtest Performance"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Cumulative NAV"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        combined_fig_path,
        dpi=150,
    )

    plt.close()

    # ========================================================
    # Console
    # ========================================================

    print(
        "\n"
        "========== MEAN-VARIANCE BACKTEST =========="
    )

    print(
        "alignment                : "
        "alpha[T] -> T+1 close -> T+2 return"
    )

    print(
        "limit-up                 : "
        "cannot buy / increase"
    )

    print(
        "limit-down               : "
        "cannot sell / decrease"
    )

    print(
        "risk model               : "
        f"{cfg.risk_model_mode}"
    )

    print(
        "============================================\n"
    )

    print(
        "Saved outputs to:",
        out,
    )

    print_keys = [
        "final_nav",
        "total_return",
        "annual_return",
        "annual_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "sell_turn_mean",
        "buy_turn_mean",
        "one_way_turnover_mean",
        "holding_count_mean",
        "optimizer_universe_count_mean",
        "invested_weight_mean",
        "n_periods",
    ]

    print(
        json.dumps(
            {
                k: metric.get(
                    k,
                    None,
                )
                for k
                in print_keys
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    return metric


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Mean-variance portfolio backtest "
            "with directional A-share "
            "limit-up / limit-down constraints."
        )
    )

    parser.add_argument(
        "--config",
        "--c",
        "-c",
        dest="config",
        required=True,
        help="Path to JSON config file",
    )

    args = parser.parse_args()

    cfg = load_config_from_json(
        args.config
    )

    run_backtest(
        cfg
    )
