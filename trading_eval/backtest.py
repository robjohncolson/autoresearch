"""Backtest engine: maps predictions to trade results with fees and slippage."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_eval.candidate import Prediction, Signal
from trading_eval.config import EvalConfig


@dataclass(frozen=True, slots=True)
class TradeResult:
    """Result of one non-abstain prediction."""

    timestamp: int
    signal: Signal
    confidence: float
    actual_return_bps: float  # 10_000 * (close[t+h] - close[t]) / close[t]
    cost_bps: float           # fee_bps + slippage_bps
    net_return_bps: float     # actual_return_bps - cost_bps (always deducted)
    pnl_bps: float            # signal * net_return_bps


def run_backtest(
    predictions: pd.Series,
    labels: pd.DataFrame,
    config: EvalConfig,
) -> list[TradeResult]:
    """Execute backtest by joining predictions to labels on timestamp.

    Args:
        predictions: Series of Prediction objects, indexed by timestamp.
        labels: DataFrame with label columns including return_bps_{horizon}.
        config: Evaluation config with fee/slippage/horizon settings.

    Returns:
        List of TradeResult for non-abstain predictions with valid labels.

    Raises:
        ValueError: If prediction timestamps don't align with label data.
    """
    if len(predictions) == 0:
        return []

    return_col = config.return_bps_col
    if return_col not in labels.columns:
        raise ValueError(
            f"Label column {return_col!r} not found. "
            f"Available: {list(labels.columns)}"
        )

    # Build a timestamp -> label lookup
    if "timestamp" in labels.columns:
        label_index = pd.Index(labels["timestamp"])
    else:
        label_index = labels.index

    pred_timestamps = predictions.index
    if not pred_timestamps.isin(label_index).all():
        missing = pred_timestamps.difference(label_index)
        raise ValueError(
            f"{len(missing)} prediction timestamps not found in labels. "
            f"First few: {list(missing[:5])}"
        )

    # Map timestamps to label rows
    if "timestamp" in labels.columns:
        ts_to_row = {int(ts): i for i, ts in enumerate(labels["timestamp"])}
    else:
        ts_to_row = {int(ts): i for i, ts in enumerate(label_index)}

    cost = config.cost_bps
    trades: list[TradeResult] = []

    for ts, pred in predictions.items():
        pred: Prediction
        if pred.signal == Signal.ABSTAIN:
            continue

        row_idx = ts_to_row[int(ts)]
        actual_return_bps = labels[return_col].iloc[row_idx]

        # Skip NaN labels (end of dataset, insufficient future data)
        if np.isnan(actual_return_bps):
            continue

        net_return = actual_return_bps - cost
        pnl = int(pred.signal) * net_return

        trades.append(TradeResult(
            timestamp=int(ts),
            signal=pred.signal,
            confidence=pred.confidence,
            actual_return_bps=float(actual_return_bps),
            cost_bps=cost,
            net_return_bps=float(net_return),
            pnl_bps=float(pnl),
        ))

    return trades
