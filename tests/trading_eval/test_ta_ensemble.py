"""Tests for standalone TA ensemble baseline."""

import numpy as np
import pandas as pd
import pytest

from trading_eval.baselines.ta_ensemble import (
    BOLLINGER_WINDOW,
    MIN_REQUIRED_BARS,
    TAEnsembleCandidate,
    TASignals,
    compute_signals,
    ema_crossover_signal,
    macd_histogram_positive_signal,
    momentum_signal,
    rsi_above_50_signal,
    signals_to_prediction,
)
from trading_eval.candidate import Signal


def _rising_close(n: int = 50, start: float = 0.08) -> pd.Series:
    """Generate a steadily rising close series."""
    return pd.Series(np.linspace(start, start * 1.5, n))


def _falling_close(n: int = 50, start: float = 0.12) -> pd.Series:
    """Generate a steadily falling close series."""
    return pd.Series(np.linspace(start, start * 0.7, n))


def _flat_close(n: int = 50, value: float = 0.09) -> pd.Series:
    """Generate a flat close series."""
    return pd.Series([value] * n)


class TestMomentumSignal:
    def test_rising_is_bullish(self):
        assert momentum_signal(_rising_close(), lookback=12) is True

    def test_falling_is_bearish(self):
        assert momentum_signal(_falling_close(), lookback=12) is False

    def test_insufficient_bars(self):
        assert momentum_signal(pd.Series([1.0, 2.0]), lookback=12) is False


class TestEMACrossover:
    def test_rising_crossover(self):
        assert ema_crossover_signal(_rising_close()) is True

    def test_falling_no_crossover(self):
        assert ema_crossover_signal(_falling_close()) is False


class TestRSI:
    def test_rising_above_50(self):
        assert rsi_above_50_signal(_rising_close()) is True

    def test_falling_below_50(self):
        assert rsi_above_50_signal(_falling_close()) is False


class TestMACDHistogram:
    def test_rising_positive(self):
        assert macd_histogram_positive_signal(_rising_close()) is True

    def test_falling_negative(self):
        assert macd_histogram_positive_signal(_falling_close()) is False


class TestComputeSignals:
    def test_rising_all_bullish(self):
        signals = compute_signals(_rising_close())
        assert signals.bullish_count >= 4
        assert signals.momentum_12h is True
        assert signals.ema_crossover is True

    def test_falling_all_bearish(self):
        signals = compute_signals(_falling_close())
        assert signals.bearish_count >= 4

    def test_agreement_count(self):
        signals = TASignals(True, True, True, True, False, False)
        assert signals.bullish_count == 4
        assert signals.bearish_count == 2
        assert signals.agreement_count == 4


class TestSignalsToPrediction:
    def test_bullish_long(self):
        signals = TASignals(True, True, True, True, False, False)
        pred = signals_to_prediction(signals, timestamp=100)
        assert pred.signal == Signal.LONG
        assert pred.confidence == round(4 / 6, 2)
        assert pred.prob_up == round(4 / 6, 4)

    def test_bearish_short(self):
        signals = TASignals(False, False, False, False, True, True)
        pred = signals_to_prediction(signals, timestamp=100)
        assert pred.signal == Signal.SHORT
        assert pred.confidence == round(4 / 6, 2)

    def test_neutral_abstain(self):
        signals = TASignals(True, True, True, False, False, False)
        pred = signals_to_prediction(signals, timestamp=100)
        assert pred.signal == Signal.ABSTAIN
        assert pred.confidence == 0.5

    def test_unanimous_bullish(self):
        signals = TASignals(True, True, True, True, True, True)
        pred = signals_to_prediction(signals, timestamp=100)
        assert pred.signal == Signal.LONG
        assert pred.confidence == 1.0
        assert pred.prob_up == 1.0


class TestTAEnsembleCandidate:
    def _make_market(self, close_series: pd.Series) -> pd.DataFrame:
        n = len(close_series)
        return pd.DataFrame({
            "timestamp": np.arange(1700000000, 1700000000 + n * 3600, 3600),
            "open": close_series.values,
            "high": close_series.values * 1.01,
            "low": close_series.values * 0.99,
            "close": close_series.values,
            "volume": np.ones(n) * 1000,
        })

    def test_predict_returns_series(self):
        c = TAEnsembleCandidate()
        c.reset()
        market = self._make_market(_rising_close(60))
        labels = pd.DataFrame({"return_bps_6h": np.zeros(60)})
        c.fit(market, labels)
        preds = c.predict(market)
        assert len(preds) == 60
        assert preds.index.name == "timestamp"

    def test_first_bars_abstain(self):
        """First MIN_REQUIRED_BARS - 1 rows should ABSTAIN due to insufficient history."""
        c = TAEnsembleCandidate()
        c.reset()
        market = self._make_market(_rising_close(60))
        labels = pd.DataFrame({"return_bps_6h": np.zeros(60)})
        c.fit(market, labels)
        preds = c.predict(market)

        # First 39 predictions should be ABSTAIN
        for i in range(MIN_REQUIRED_BARS - 1):
            assert preds.iloc[i].signal == Signal.ABSTAIN

        # At position 39 (0-indexed), we have exactly 40 bars
        assert preds.iloc[MIN_REQUIRED_BARS - 1].signal != Signal.ABSTAIN

    def test_rising_predicts_long(self):
        c = TAEnsembleCandidate()
        c.reset()
        market = self._make_market(_rising_close(60))
        labels = pd.DataFrame({"return_bps_6h": np.zeros(60)})
        c.fit(market, labels)
        preds = c.predict(market)

        # Later predictions (with enough history) should be LONG
        last_pred = preds.iloc[-1]
        assert last_pred.signal == Signal.LONG

    def test_reset_clears_threshold(self):
        c = TAEnsembleCandidate()
        c.reset()
        assert c._bollinger_threshold is None
