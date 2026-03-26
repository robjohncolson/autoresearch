"""Tests for candidate protocol."""

import numpy as np
import pandas as pd

from trading_eval.candidate import (
    Candidate,
    Prediction,
    Signal,
    predictions_to_series,
)


class DummyCandidate(Candidate):
    """Minimal implementation for testing."""

    @property
    def name(self) -> str:
        return "dummy"

    def fit(self, market: pd.DataFrame, labels: pd.DataFrame) -> None:
        pass

    def predict(self, market: pd.DataFrame) -> pd.Series:
        preds = [
            Prediction(
                timestamp=int(ts),
                signal=Signal.LONG,
                confidence=0.6,
                prob_up=0.6,
            )
            for ts in market["timestamp"]
        ]
        return predictions_to_series(preds)


class TestSignalEnum:
    def test_values(self):
        assert Signal.SHORT == -1
        assert Signal.ABSTAIN == 0
        assert Signal.LONG == 1

    def test_multiplication(self):
        assert Signal.LONG * 100 == 100
        assert Signal.SHORT * 100 == -100
        assert Signal.ABSTAIN * 100 == 0


class TestPrediction:
    def test_creation(self):
        p = Prediction(timestamp=1700000000, signal=Signal.LONG, confidence=0.8)
        assert p.timestamp == 1700000000
        assert p.signal == Signal.LONG
        assert p.confidence == 0.8
        assert p.prob_up == 0.5  # default

    def test_with_prob_up(self):
        p = Prediction(timestamp=1700000000, signal=Signal.LONG, confidence=0.8, prob_up=0.75)
        assert p.prob_up == 0.75

    def test_frozen(self):
        p = Prediction(timestamp=1700000000, signal=Signal.LONG, confidence=0.8)
        try:
            p.confidence = 0.5  # type: ignore
            assert False, "Should raise"
        except AttributeError:
            pass


class TestPredictionsToSeries:
    def test_creates_indexed_series(self):
        preds = [
            Prediction(timestamp=100, signal=Signal.LONG, confidence=0.6),
            Prediction(timestamp=200, signal=Signal.SHORT, confidence=0.7),
            Prediction(timestamp=300, signal=Signal.ABSTAIN, confidence=0.5),
        ]
        series = predictions_to_series(preds)
        assert len(series) == 3
        assert series.index.name == "timestamp"
        assert list(series.index) == [100, 200, 300]
        assert series.iloc[0].signal == Signal.LONG

    def test_empty_list(self):
        series = predictions_to_series([])
        assert len(series) == 0


class TestCandidateProtocol:
    def test_dummy_candidate(self):
        c = DummyCandidate()
        assert c.name == "dummy"
        assert c.version == "1.0"

        market = pd.DataFrame({
            "timestamp": [100, 200, 300],
            "close": [1.0, 1.1, 1.2],
        })
        labels = pd.DataFrame({"return_bps_6h": [10, 20, 30]})

        c.fit(market, labels)
        preds = c.predict(market)

        assert len(preds) == 3
        assert list(preds.index) == [100, 200, 300]

    def test_reset_is_optional(self):
        c = DummyCandidate()
        c.reset()  # should not raise
