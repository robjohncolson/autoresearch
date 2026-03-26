"""Tests for metrics computation."""

import numpy as np
import pandas as pd
import pytest

from trading_eval.backtest import TradeResult
from trading_eval.candidate import Prediction, Signal, predictions_to_series
from trading_eval.metrics import Metrics, compute_metrics


def _trade(ts, signal, actual, cost=15.0):
    net = actual - cost
    pnl = int(signal) * net
    return TradeResult(
        timestamp=ts,
        signal=signal,
        confidence=0.8,
        actual_return_bps=actual,
        cost_bps=cost,
        net_return_bps=net,
        pnl_bps=pnl,
    )


def _pred(ts, signal, prob_up=0.5):
    return Prediction(timestamp=ts, signal=signal, confidence=0.8, prob_up=prob_up)


class TestComputeMetrics:
    def test_empty_trades(self):
        preds = predictions_to_series([
            _pred(100, Signal.ABSTAIN),
        ])
        m = compute_metrics([], preds)
        assert m.trade_count == 0
        assert m.abstain_count == 1
        assert m.net_pnl_bps == 0.0

    def test_direction_accuracy_all_correct(self):
        trades = [
            _trade(100, Signal.LONG, 50.0),   # LONG on positive -> correct
            _trade(200, Signal.SHORT, -30.0),  # SHORT on negative -> correct
        ]
        preds = predictions_to_series([
            _pred(100, Signal.LONG), _pred(200, Signal.SHORT),
        ])
        m = compute_metrics(trades, preds)
        assert m.direction_accuracy == 1.0

    def test_direction_accuracy_all_wrong(self):
        trades = [
            _trade(100, Signal.LONG, -50.0),   # LONG on negative -> wrong
            _trade(200, Signal.SHORT, 30.0),    # SHORT on positive -> wrong
        ]
        preds = predictions_to_series([
            _pred(100, Signal.LONG), _pred(200, Signal.SHORT),
        ])
        m = compute_metrics(trades, preds)
        assert m.direction_accuracy == 0.0

    def test_hit_rate(self):
        trades = [
            _trade(100, Signal.LONG, 50.0),    # pnl = 35 (positive)
            _trade(200, Signal.LONG, -50.0),   # pnl = -65 (negative)
            _trade(300, Signal.SHORT, -80.0),  # pnl = 65 (positive)
        ]
        preds = predictions_to_series([
            _pred(100, Signal.LONG),
            _pred(200, Signal.LONG),
            _pred(300, Signal.SHORT),
        ])
        m = compute_metrics(trades, preds)
        # 2 of 3 trades are positive
        assert m.hit_rate == pytest.approx(2 / 3, abs=0.001)

    def test_net_pnl(self):
        trades = [
            _trade(100, Signal.LONG, 50.0, cost=0),   # pnl = 50
            _trade(200, Signal.LONG, -30.0, cost=0),   # pnl = -30
        ]
        preds = predictions_to_series([
            _pred(100, Signal.LONG), _pred(200, Signal.LONG),
        ])
        m = compute_metrics(trades, preds)
        assert m.net_pnl_bps == pytest.approx(20.0)

    def test_max_drawdown(self):
        # Cumulative P&L: 50, 50-30=20. Peak=50, trough=20, drawdown=30
        trades = [
            _trade(100, Signal.LONG, 50.0, cost=0),
            _trade(200, Signal.LONG, -30.0, cost=0),
        ]
        preds = predictions_to_series([
            _pred(100, Signal.LONG), _pred(200, Signal.LONG),
        ])
        m = compute_metrics(trades, preds)
        assert m.max_drawdown_bps == pytest.approx(30.0)

    def test_turnover(self):
        trades = [_trade(100, Signal.LONG, 50.0)]
        preds = predictions_to_series([
            _pred(100, Signal.LONG),
            _pred(200, Signal.ABSTAIN),
            _pred(300, Signal.ABSTAIN),
        ])
        m = compute_metrics(trades, preds)
        assert m.turnover == pytest.approx(1 / 3, abs=0.001)
        assert m.abstain_count == 2

    def test_brier_score_perfect(self):
        """prob_up=1.0 when actual is positive -> Brier = 0."""
        trades = [_trade(100, Signal.LONG, 50.0)]
        preds = predictions_to_series([
            _pred(100, Signal.LONG, prob_up=1.0),
        ])
        m = compute_metrics(trades, preds)
        assert m.brier_score == pytest.approx(0.0)

    def test_brier_score_worst(self):
        """prob_up=0.0 when actual is positive -> Brier = 1."""
        trades = [_trade(100, Signal.LONG, 50.0)]
        preds = predictions_to_series([
            _pred(100, Signal.LONG, prob_up=0.0),
        ])
        m = compute_metrics(trades, preds)
        assert m.brier_score == pytest.approx(1.0)

    def test_sharpe_nonzero(self):
        trades = [
            _trade(100, Signal.LONG, 50.0, cost=0),
            _trade(200, Signal.LONG, 60.0, cost=0),
            _trade(300, Signal.LONG, 40.0, cost=0),
        ]
        preds = predictions_to_series([
            _pred(100, Signal.LONG),
            _pred(200, Signal.LONG),
            _pred(300, Signal.LONG),
        ])
        m = compute_metrics(trades, preds)
        assert m.sharpe_ratio > 0
        assert m.trade_count == 3

    def test_metrics_to_dict(self):
        m = Metrics(
            direction_accuracy=0.5, brier_score=0.25, mae_bps=30.0,
            net_pnl_bps=100.0, max_drawdown_bps=50.0, sharpe_ratio=1.2,
            sortino_ratio=1.5, turnover=0.8, hit_rate=0.6,
            trade_count=10, abstain_count=2,
        )
        d = m.to_dict()
        assert d["direction_accuracy"] == 0.5
        assert d["trade_count"] == 10
        assert len(d) == 11
