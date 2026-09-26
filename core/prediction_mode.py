from __future__ import annotations

from copy import deepcopy

from core.errors import ContractError


PREDICTION_MODE_PRESETS = {
    "c2c": {
        "target": "Ref($close, -2) / Ref($close, -1) - 1",
        "price_field": "close",
        "qlib_price_expression": "$close",
        "ret_path": "./data/portfolio/c_2_c_1D.csv",
        "limit_up_mask_path": "./data/portfolio/mask_limit_up_1D.csv",
        "limit_down_mask_path": "./data/portfolio/mask_limit_down_1D.csv",
        "execution_price": "close",
        "return_definition": "close(T+2) / close(T+1) - 1",
        "buy_slippage": 0.0005,
        "sell_slippage": 0.0005,
    },
    "o2o": {
        "target": "Ref($open, -2) / Ref($open, -1) - 1",
        "price_field": "open",
        "qlib_price_expression": "$open",
        "ret_path": "./data/portfolio/o_2_o_1D.csv",
        "limit_up_mask_path": "./data/portfolio/mask_limit_up_open_1D.csv",
        "limit_down_mask_path": "./data/portfolio/mask_limit_down_open_1D.csv",
        "execution_price": "open",
        "return_definition": "open(T+2) / open(T+1) - 1",
        "buy_slippage": 0.001,
        "sell_slippage": 0.001,
    },
}


def prediction_mode(config: dict) -> str:
    raw = config.get("task", {}).get("prediction_mode", "c2c")
    if not isinstance(raw, str):
        raise ContractError("PREDICTION_MODE_INVALID")
    mode = raw.strip().lower()
    if mode not in PREDICTION_MODE_PRESETS:
        raise ContractError("PREDICTION_MODE_INVALID")
    return mode


def prediction_mode_preset(config: dict) -> dict:
    return deepcopy(PREDICTION_MODE_PRESETS[prediction_mode(config)])


def apply_prediction_mode(config: dict) -> dict:
    """Bind label, execution data, masks, timing, and slippage to one switch."""

    mode = prediction_mode(config)
    preset = PREDICTION_MODE_PRESETS[mode]
    task = config.setdefault("task", {})
    portfolio = config.setdefault("z_portfolio", {})
    task["prediction_mode"] = mode
    task["target"] = preset["target"]
    for key in (
        "ret_path",
        "limit_up_mask_path",
        "limit_down_mask_path",
        "execution_price",
        "return_definition",
        "buy_slippage",
        "sell_slippage",
    ):
        portfolio[key] = preset[key]
    portfolio["trade_delay_days"] = 1
    portfolio["return_delay_days"] = 2
    portfolio["comment_delay"] = (
        f"使用 alpha[T]，在 T+1 {preset['execution_price']}成交，"
        f"第 T+2 行收益为 {preset['return_definition']}"
    )
    return config
