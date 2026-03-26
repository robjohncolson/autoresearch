"""Parity test: standalone TA port vs kraken-bot-v4 original.

This test imports from kraken-bot-v4 and requires both repos to be present.
Mark with @pytest.mark.parity so it can be skipped in CI or when
kraken-bot-v4 is not available.

Run: pytest tests/trading_eval/test_ta_ensemble_parity.py -v -m parity
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add kraken-bot-v4 to sys.path for cross-repo import
KRAKEN_BOT_PATH = Path(__file__).resolve().parents[3] / "kraken-bot-v4"

# Try to import; skip all tests if kraken-bot-v4 not available
try:
    sys.path.insert(0, str(KRAKEN_BOT_PATH))
    from beliefs.technical_ensemble_source import TechnicalEnsembleSource
    from core.types import BeliefDirection, MarketRegime
    HAS_KRAKEN_BOT = True
except ImportError:
    HAS_KRAKEN_BOT = False
finally:
    if str(KRAKEN_BOT_PATH) in sys.path:
        sys.path.remove(str(KRAKEN_BOT_PATH))

from trading_eval.baselines.ta_ensemble import (
    compute_signals,
    signals_to_prediction,
)
from trading_eval.candidate import Signal

pytestmark = pytest.mark.parity


def _make_synthetic_bars(seed: int, n: int = 60) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV bars."""
    rng = np.random.default_rng(seed)
    close = 0.09 * np.cumprod(1 + rng.normal(0.0005, 0.003, n))
    return pd.DataFrame({
        "open": close * (1 + rng.uniform(-0.002, 0.002, n)),
        "high": close * (1 + rng.uniform(0, 0.005, n)),
        "low": close * (1 - rng.uniform(0, 0.005, n)),
        "close": close,
        "volume": rng.uniform(500, 5000, n),
    })


@pytest.mark.skipif(not HAS_KRAKEN_BOT, reason="kraken-bot-v4 not found")
class TestTAEnsembleParity:
    """Verify standalone port produces identical results to the original."""

    @pytest.mark.parametrize("seed", [42, 123, 777, 2024, 9999])
    def test_signal_parity(self, seed):
        """All 6 signals match on identical synthetic bars."""
        bars = _make_synthetic_bars(seed)
        close = pd.to_numeric(bars["close"], errors="coerce").astype(float).reset_index(drop=True)

        # Original (kraken-bot-v4)
        sys.path.insert(0, str(KRAKEN_BOT_PATH))
        try:
            original = TechnicalEnsembleSource(min_bars=40)
            orig_signals = original.compute_signals(bars)
        finally:
            sys.path.remove(str(KRAKEN_BOT_PATH))

        # Port (standalone)
        port_signals = compute_signals(close, bollinger_threshold=None)

        assert port_signals.momentum_12h == orig_signals.momentum_12h, f"momentum_12h mismatch (seed={seed})"
        assert port_signals.momentum_6h == orig_signals.momentum_6h, f"momentum_6h mismatch (seed={seed})"
        assert port_signals.ema_crossover == orig_signals.ema_crossover, f"ema_crossover mismatch (seed={seed})"
        assert port_signals.rsi_above_50 == orig_signals.rsi_above_50, f"rsi_above_50 mismatch (seed={seed})"
        assert port_signals.macd_histogram_positive == orig_signals.macd_histogram_positive, f"macd mismatch (seed={seed})"
        assert port_signals.bollinger_width_compressed == orig_signals.bollinger_width_compressed, f"bollinger mismatch (seed={seed})"

    @pytest.mark.parametrize("seed", [42, 123, 777])
    def test_direction_confidence_parity(self, seed):
        """Direction and confidence match the original."""
        bars = _make_synthetic_bars(seed)
        close = pd.to_numeric(bars["close"], errors="coerce").astype(float).reset_index(drop=True)

        # Original
        sys.path.insert(0, str(KRAKEN_BOT_PATH))
        try:
            original = TechnicalEnsembleSource(min_bars=40)
            orig_snapshot = original.analyze(pair="DOGE/USD", bars=bars)
        finally:
            sys.path.remove(str(KRAKEN_BOT_PATH))

        # Port
        port_signals = compute_signals(close, bollinger_threshold=None)
        port_pred = signals_to_prediction(port_signals, timestamp=0)

        # Map original direction to Signal
        direction_map = {
            BeliefDirection.BULLISH: Signal.LONG,
            BeliefDirection.BEARISH: Signal.SHORT,
            BeliefDirection.NEUTRAL: Signal.ABSTAIN,
        }
        expected_signal = direction_map[orig_snapshot.direction]

        assert port_pred.signal == expected_signal, (
            f"Direction mismatch (seed={seed}): "
            f"original={orig_snapshot.direction}, port={port_pred.signal}"
        )
        assert port_pred.confidence == orig_snapshot.confidence, (
            f"Confidence mismatch (seed={seed}): "
            f"original={orig_snapshot.confidence}, port={port_pred.confidence}"
        )

    @pytest.mark.parametrize("seed", [42, 123, 777])
    def test_regime_parity(self, seed):
        """Regime classification matches the original."""
        bars = _make_synthetic_bars(seed)
        close = pd.to_numeric(bars["close"], errors="coerce").astype(float).reset_index(drop=True)

        # Original
        sys.path.insert(0, str(KRAKEN_BOT_PATH))
        try:
            original = TechnicalEnsembleSource(min_bars=40)
            orig_snapshot = original.analyze(pair="DOGE/USD", bars=bars)
        finally:
            sys.path.remove(str(KRAKEN_BOT_PATH))

        # Port
        port_signals = compute_signals(close, bollinger_threshold=None)

        regime_map = {
            MarketRegime.RANGING: True,
            MarketRegime.TRENDING: False,
        }
        expected_compressed = regime_map[orig_snapshot.regime]

        assert port_signals.bollinger_width_compressed == expected_compressed, (
            f"Regime mismatch (seed={seed}): "
            f"original={orig_snapshot.regime}, "
            f"port_compressed={port_signals.bollinger_width_compressed}"
        )
