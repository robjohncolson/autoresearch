"""Tests for backtest engine."""

import numpy as np
import pandas as pd
import pytest

from trading_eval.backtest import TradeResult, run_backtest
from trading_eval.candidate import Prediction, Signal, predictions_to_series
from trading_eval.config import EvalConfig


def _config(fee_bps=10.0, slippage_bps=5.0):
    return EvalConfig(data_dir="/tmp", fee_bps=fee_bps, slippage_bps=slippage_bps)


class TestRunBacktest:
    def test_basic_long_trade(self):
        """LONG on +50 bps gross -> net = 50 - 15 = 35, pnl = +35."""
        preds = predictions_to_series([
            Prediction(timestamp=100, signal=Signal.LONG, confidence=0.8),
        ])
        labels = pd.DataFrame({
            "timestamp": [100],
            "return_bps_6h": [50.0],
        })
        trades = run_backtest(preds, labels, _config())
        assert len(trades) == 1
        t = trades[0]
        assert t.actual_return_bps == 50.0
        assert t.cost_bps == 15.0
        assert t.net_return_bps == 35.0
        assert t.pnl_bps == 35.0

    def test_basic_short_trade(self):
        """SHORT on +50 bps gross -> net = 50 - 15 = 35, pnl = -35 (wrong direction)."""
        preds = predictions_to_series([
            Prediction(timestamp=100, signal=Signal.SHORT, confidence=0.8),
        ])
        labels = pd.DataFrame({
            "timestamp": [100],
            "return_bps_6h": [50.0],
        })
        trades = run_backtest(preds, labels, _config())
        assert len(trades) == 1
        assert trades[0].pnl_bps == -35.0

    def test_short_on_negative_return(self):
        """SHORT on -80 bps gross -> net = -80 - 15 = -95, pnl = -1 * -95 = +95."""
        preds = predictions_to_series([
            Prediction(timestamp=100, signal=Signal.SHORT, confidence=0.9),
        ])
        labels = pd.DataFrame({
            "timestamp": [100],
            "return_bps_6h": [-80.0],
        })
        trades = run_backtest(preds, labels, _config())
        assert len(trades) == 1
        assert trades[0].pnl_bps == pytest.approx(95.0)

    def test_abstain_skipped(self):
        preds = predictions_to_series([
            Prediction(timestamp=100, signal=Signal.ABSTAIN, confidence=0.5),
            Prediction(timestamp=200, signal=Signal.LONG, confidence=0.7),
        ])
        labels = pd.DataFrame({
            "timestamp": [100, 200],
            "return_bps_6h": [50.0, 30.0],
        })
        trades = run_backtest(preds, labels, _config())
        assert len(trades) == 1
        assert trades[0].timestamp == 200

    def test_nan_labels_skipped(self):
        preds = predictions_to_series([
            Prediction(timestamp=100, signal=Signal.LONG, confidence=0.7),
            Prediction(timestamp=200, signal=Signal.LONG, confidence=0.7),
        ])
        labels = pd.DataFrame({
            "timestamp": [100, 200],
            "return_bps_6h": [np.nan, 30.0],
        })
        trades = run_backtest(preds, labels, _config())
        assert len(trades) == 1
        assert trades[0].timestamp == 200

    def test_empty_predictions(self):
        preds = predictions_to_series([])
        labels = pd.DataFrame({"timestamp": [100], "return_bps_6h": [10.0]})
        trades = run_backtest(preds, labels, _config())
        assert trades == []

    def test_alignment_error(self):
        preds = predictions_to_series([
            Prediction(timestamp=999, signal=Signal.LONG, confidence=0.7),
        ])
        labels = pd.DataFrame({
            "timestamp": [100],
            "return_bps_6h": [50.0],
        })
        with pytest.raises(ValueError, match="not found in labels"):
            run_backtest(preds, labels, _config())

    def test_missing_label_column(self):
        preds = predictions_to_series([
            Prediction(timestamp=100, signal=Signal.LONG, confidence=0.7),
        ])
        labels = pd.DataFrame({"timestamp": [100], "return_bps_12h": [50.0]})
        with pytest.raises(ValueError, match="not found"):
            run_backtest(preds, labels, _config())

    def test_custom_fees(self):
        """Zero fees means pnl = signal * actual_return."""
        preds = predictions_to_series([
            Prediction(timestamp=100, signal=Signal.LONG, confidence=0.8),
        ])
        labels = pd.DataFrame({
            "timestamp": [100],
            "return_bps_6h": [50.0],
        })
        trades = run_backtest(preds, labels, _config(fee_bps=0, slippage_bps=0))
        assert trades[0].pnl_bps == 50.0
        assert trades[0].cost_bps == 0.0

    def test_multiple_trades(self):
        preds = predictions_to_series([
            Prediction(timestamp=100, signal=Signal.LONG, confidence=0.8),
            Prediction(timestamp=200, signal=Signal.SHORT, confidence=0.6),
            Prediction(timestamp=300, signal=Signal.LONG, confidence=0.9),
        ])
        labels = pd.DataFrame({
            "timestamp": [100, 200, 300],
            "return_bps_6h": [50.0, -30.0, 100.0],
        })
        trades = run_backtest(preds, labels, _config())
        assert len(trades) == 3
        # LONG +50 -> pnl = 1*(50-15) = 35
        assert trades[0].pnl_bps == pytest.approx(35.0)
        # SHORT -30 -> pnl = -1*(-30-15) = 45
        assert trades[1].pnl_bps == pytest.approx(45.0)
        # LONG +100 -> pnl = 1*(100-15) = 85
        assert trades[2].pnl_bps == pytest.approx(85.0)

    def test_bps_is_10000x(self):
        """Verify the convention: return_bps = 10_000 * fractional return.
        50 bps = 0.5% = 0.005 fractional return."""
        actual_bps = 50.0
        fractional_return = actual_bps / 10_000
        assert fractional_return == pytest.approx(0.005)
