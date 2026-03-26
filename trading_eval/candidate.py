"""Candidate protocol for walk-forward evaluation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

import pandas as pd


class Signal(IntEnum):
    SHORT = -1
    ABSTAIN = 0
    LONG = 1


@dataclass(frozen=True, slots=True)
class Prediction:
    """A single prediction for one timestamp."""

    timestamp: int
    signal: Signal
    confidence: float  # 0.0 to 1.0
    prob_up: float = 0.5  # calibrated P(price goes up), for Brier score


class Candidate(ABC):
    """Abstract base class for evaluation candidates."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def version(self) -> str:
        return "1.0"

    @abstractmethod
    def fit(self, market: pd.DataFrame, labels: pd.DataFrame) -> None:
        """Train on the given window. Called once per fold."""
        ...

    @abstractmethod
    def predict(self, market: pd.DataFrame) -> pd.Series:
        """Return a Series of Prediction objects indexed by timestamp.

        The index must match market['timestamp'] exactly.
        """
        ...

    def reset(self) -> None:
        """Reset internal state between folds. Override if needed."""
        pass


def predictions_to_series(predictions: list[Prediction]) -> pd.Series:
    """Convert a list of Predictions to a timestamp-indexed Series."""
    return pd.Series(
        predictions,
        index=pd.Index([p.timestamp for p in predictions], name="timestamp"),
    )
