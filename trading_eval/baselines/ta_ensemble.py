"""Standalone port of kraken-bot-v4's 6-signal technical ensemble.

This is a self-contained copy — no imports from kraken-bot-v4.
Parity with the original is enforced by test_ta_ensemble_parity.py.

Constants and signal logic match:
  kraken-bot-v4/beliefs/technical_ensemble_source.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from trading_eval.candidate import (
    Candidate,
    Prediction,
    Signal,
    predictions_to_series,
)

# ---- Constants (identical to kraken-bot-v4) ----

TOTAL_SIGNALS: Final[int] = 6
MOMENTUM_12H_LOOKBACK: Final[int] = 12
MOMENTUM_6H_LOOKBACK: Final[int] = 6
EMA_FAST_SPAN: Final[int] = 7
EMA_SLOW_SPAN: Final[int] = 26
RSI_PERIOD: Final[int] = 8
MACD_FAST_SPAN: Final[int] = 14
MACD_SLOW_SPAN: Final[int] = 23
MACD_SIGNAL_SPAN: Final[int] = 9
BOLLINGER_WINDOW: Final[int] = 20
BOLLINGER_PERCENTILE: Final[float] = 85.0
MIN_REQUIRED_BARS: Final[int] = 40


# ---- Signal computation (pure functions) ----

def _coerce_close(close: pd.Series, min_bars: int) -> pd.Series:
    series = pd.to_numeric(close, errors="coerce").astype(float).reset_index(drop=True)
    if len(series) < min_bars:
        return pd.Series(dtype=float)
    return series


def momentum_signal(close: pd.Series, lookback: int) -> bool:
    series = _coerce_close(close, lookback + 1)
    if series.empty:
        return False
    return bool(series.iloc[-1] > series.iloc[-(lookback + 1)])


def ema_crossover_signal(close: pd.Series) -> bool:
    series = _coerce_close(close, EMA_SLOW_SPAN)
    if series.empty:
        return False
    ema_fast = series.ewm(span=EMA_FAST_SPAN, adjust=False).mean()
    ema_slow = series.ewm(span=EMA_SLOW_SPAN, adjust=False).mean()
    return bool(ema_fast.iloc[-1] > ema_slow.iloc[-1])


def rsi_above_50_signal(close: pd.Series) -> bool:
    series = _coerce_close(close, RSI_PERIOD + 1)
    if series.empty:
        return False
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = float(gains.rolling(window=RSI_PERIOD, min_periods=RSI_PERIOD).mean().iloc[-1])
    avg_loss = float(losses.rolling(window=RSI_PERIOD, min_periods=RSI_PERIOD).mean().iloc[-1])

    if avg_loss == 0.0:
        rsi = 100.0 if avg_gain > 0.0 else 50.0
    else:
        rsi = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

    return bool(rsi > 50.0)


def macd_histogram_positive_signal(close: pd.Series) -> bool:
    series = _coerce_close(close, MACD_SLOW_SPAN)
    if series.empty:
        return False
    ema_fast = series.ewm(span=MACD_FAST_SPAN, adjust=False).mean()
    ema_slow = series.ewm(span=MACD_SLOW_SPAN, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL_SPAN, adjust=False).mean()
    histogram = macd_line - signal_line
    return bool(histogram.iloc[-1] > 0.0)


def bollinger_width_compressed_signal(
    close: pd.Series,
    threshold: float | None = None,
) -> bool:
    """Check if current Bollinger width is below threshold.

    If threshold is None, compute 85th percentile from the series itself
    (original bot behavior). In walk-forward mode, threshold should be
    pre-computed from training data to avoid leakage.
    """
    series = _coerce_close(close, BOLLINGER_WINDOW)
    if series.empty:
        return False
    rolling_mean = series.rolling(window=BOLLINGER_WINDOW, min_periods=BOLLINGER_WINDOW).mean()
    rolling_std = series.rolling(window=BOLLINGER_WINDOW, min_periods=BOLLINGER_WINDOW).std(ddof=0)
    width = (4.0 * rolling_std) / rolling_mean.replace(0.0, np.nan)
    width = width.dropna()

    if width.empty:
        return False

    if threshold is None:
        threshold = float(np.percentile(width.to_numpy(dtype=float), BOLLINGER_PERCENTILE))

    return bool(float(width.iloc[-1]) < threshold)


@dataclass(frozen=True, slots=True)
class TASignals:
    """Result of computing all 6 signals."""
    momentum_12h: bool
    momentum_6h: bool
    ema_crossover: bool
    rsi_above_50: bool
    macd_histogram_positive: bool
    bollinger_width_compressed: bool

    @property
    def bullish_count(self) -> int:
        return sum([
            self.momentum_12h, self.momentum_6h, self.ema_crossover,
            self.rsi_above_50, self.macd_histogram_positive,
            self.bollinger_width_compressed,
        ])

    @property
    def bearish_count(self) -> int:
        return TOTAL_SIGNALS - self.bullish_count

    @property
    def agreement_count(self) -> int:
        return max(self.bullish_count, self.bearish_count)


def compute_signals(close: pd.Series, bollinger_threshold: float | None = None) -> TASignals:
    return TASignals(
        momentum_12h=momentum_signal(close, MOMENTUM_12H_LOOKBACK),
        momentum_6h=momentum_signal(close, MOMENTUM_6H_LOOKBACK),
        ema_crossover=ema_crossover_signal(close),
        rsi_above_50=rsi_above_50_signal(close),
        macd_histogram_positive=macd_histogram_positive_signal(close),
        bollinger_width_compressed=bollinger_width_compressed_signal(close, bollinger_threshold),
    )


def signals_to_prediction(signals: TASignals, timestamp: int) -> Prediction:
    bullish = signals.bullish_count
    bearish = signals.bearish_count

    if bullish >= 4:
        signal = Signal.LONG
    elif bearish >= 4:
        signal = Signal.SHORT
    else:
        signal = Signal.ABSTAIN

    confidence = round(signals.agreement_count / TOTAL_SIGNALS, 2)
    prob_up = round(bullish / TOTAL_SIGNALS, 4)

    return Prediction(
        timestamp=timestamp,
        signal=signal,
        confidence=confidence,
        prob_up=prob_up,
    )


# ---- Candidate implementation ----

class TAEnsembleCandidate(Candidate):
    """Walk-forward wrapper around the 6-signal TA ensemble."""

    _bollinger_threshold: float | None

    @property
    def name(self) -> str:
        return "ta_ensemble_6signal"

    @property
    def version(self) -> str:
        return "1.0"

    def reset(self) -> None:
        self._bollinger_threshold = None

    def fit(self, market: pd.DataFrame, labels: pd.DataFrame) -> None:
        """Pre-compute Bollinger threshold from training data."""
        close = pd.to_numeric(market["close"], errors="coerce").astype(float).reset_index(drop=True)
        if len(close) < BOLLINGER_WINDOW:
            self._bollinger_threshold = None
            return

        rolling_mean = close.rolling(window=BOLLINGER_WINDOW, min_periods=BOLLINGER_WINDOW).mean()
        rolling_std = close.rolling(window=BOLLINGER_WINDOW, min_periods=BOLLINGER_WINDOW).std(ddof=0)
        width = (4.0 * rolling_std) / rolling_mean.replace(0.0, np.nan)
        width = width.dropna()

        if width.empty:
            self._bollinger_threshold = None
        else:
            self._bollinger_threshold = float(
                np.percentile(width.to_numpy(dtype=float), BOLLINGER_PERCENTILE)
            )

    def predict(self, market: pd.DataFrame) -> pd.Series:
        """Generate predictions for validation window.

        For each row, uses up to MIN_REQUIRED_BARS preceding rows of close
        prices from the market DataFrame to compute signals.
        """
        close_all = pd.to_numeric(market["close"], errors="coerce").astype(float).reset_index(drop=True)
        timestamps = market["timestamp"].values
        predictions: list[Prediction] = []

        for i in range(len(market)):
            # Use all available history up to and including row i
            start = max(0, i - MIN_REQUIRED_BARS + 1)
            close_window = close_all.iloc[start:i + 1].reset_index(drop=True)

            if len(close_window) < MIN_REQUIRED_BARS:
                pred = Prediction(
                    timestamp=int(timestamps[i]),
                    signal=Signal.ABSTAIN,
                    confidence=0.5,
                    prob_up=0.5,
                )
            else:
                signals = compute_signals(close_window, self._bollinger_threshold)
                pred = signals_to_prediction(signals, int(timestamps[i]))

            predictions.append(pred)

        return predictions_to_series(predictions)
