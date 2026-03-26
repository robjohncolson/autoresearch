"""Trading metrics: direction accuracy, Brier, P&L, drawdown, Sharpe, etc."""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_eval.backtest import TradeResult
from trading_eval.candidate import Prediction, Signal


@dataclass(frozen=True, slots=True)
class Metrics:
    direction_accuracy: float
    brier_score: float
    mae_bps: float
    net_pnl_bps: float
    max_drawdown_bps: float
    sharpe_ratio: float
    sortino_ratio: float
    turnover: float
    hit_rate: float
    trade_count: int
    abstain_count: int

    def to_dict(self) -> dict:
        return {
            "direction_accuracy": self.direction_accuracy,
            "brier_score": self.brier_score,
            "mae_bps": self.mae_bps,
            "net_pnl_bps": self.net_pnl_bps,
            "max_drawdown_bps": self.max_drawdown_bps,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "turnover": self.turnover,
            "hit_rate": self.hit_rate,
            "trade_count": self.trade_count,
            "abstain_count": self.abstain_count,
        }


def compute_metrics(
    trades: list[TradeResult],
    all_predictions: pd.Series,
    periods_per_year: float = 365 * 24,
) -> Metrics:
    """Compute evaluation metrics from trade results and full prediction set.

    Args:
        trades: Non-abstain trade results from the backtest.
        all_predictions: Full Series of Predictions (including abstains).
        periods_per_year: Annualization factor (default: hourly candles).
    """
    total_preds = len(all_predictions)
    abstain_count = sum(
        1 for p in all_predictions if p.signal == Signal.ABSTAIN
    )

    if not trades:
        return Metrics(
            direction_accuracy=0.0,
            brier_score=1.0,
            mae_bps=0.0,
            net_pnl_bps=0.0,
            max_drawdown_bps=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            turnover=0.0,
            hit_rate=0.0,
            trade_count=0,
            abstain_count=abstain_count,
        )

    pnls = np.array([t.pnl_bps for t in trades])
    actual_returns = np.array([t.actual_return_bps for t in trades])
    signals = np.array([int(t.signal) for t in trades])

    # Direction accuracy: fraction where sign(signal) == sign(actual_return)
    actual_signs = np.sign(actual_returns)
    signal_signs = np.sign(signals)
    direction_correct = np.sum(signal_signs == actual_signs)
    direction_accuracy = float(direction_correct / len(trades))

    # Brier score: mean((prob_up - I(actual > 0))^2)
    # Build prob_up from the prediction objects via timestamp lookup
    pred_lookup = {int(p.timestamp): p for p in all_predictions}
    brier_terms = []
    for t in trades:
        pred = pred_lookup.get(t.timestamp)
        prob_up = pred.prob_up if pred else 0.5
        actual_up = 1.0 if t.actual_return_bps > 0 else 0.0
        brier_terms.append((prob_up - actual_up) ** 2)
    brier_score = float(np.mean(brier_terms))

    # MAE on actual returns (baseline reference)
    mae_bps = float(np.mean(np.abs(actual_returns)))

    # Net P&L
    net_pnl_bps = float(np.sum(pnls))

    # Max drawdown
    cum_pnl = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = running_max - cum_pnl
    max_drawdown_bps = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Sharpe ratio (annualized)
    mean_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0
    annualization = math.sqrt(periods_per_year)
    sharpe_ratio = (mean_pnl / std_pnl * annualization) if std_pnl > 0 else 0.0

    # Sortino ratio (annualized, downside deviation)
    downside = pnls[pnls < 0]
    if len(downside) > 1:
        downside_std = float(np.std(downside, ddof=1))
        sortino_ratio = (mean_pnl / downside_std * annualization) if downside_std > 0 else 0.0
    else:
        sortino_ratio = 0.0

    # Turnover
    turnover = float(len(trades) / total_preds) if total_preds > 0 else 0.0

    # Hit rate
    positive_trades = np.sum(pnls > 0)
    hit_rate = float(positive_trades / len(trades))

    return Metrics(
        direction_accuracy=round(direction_accuracy, 6),
        brier_score=round(brier_score, 6),
        mae_bps=round(mae_bps, 4),
        net_pnl_bps=round(net_pnl_bps, 4),
        max_drawdown_bps=round(max_drawdown_bps, 4),
        sharpe_ratio=round(sharpe_ratio, 4),
        sortino_ratio=round(sortino_ratio, 4),
        turnover=round(turnover, 6),
        hit_rate=round(hit_rate, 6),
        trade_count=len(trades),
        abstain_count=abstain_count,
    )
