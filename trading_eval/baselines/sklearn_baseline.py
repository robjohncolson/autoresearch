"""Sklearn-based baselines: logistic regression and gradient-boosted tree."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from trading_eval.candidate import (
    Candidate,
    Prediction,
    Signal,
    predictions_to_series,
)


def _build_features(market: pd.DataFrame) -> np.ndarray:
    """Engineer features from OHLCV data.

    Features (all point-in-time):
    - 1-bar return: (close - prev_close) / prev_close
    - 6-bar return: (close - close_6_ago) / close_6_ago
    - 12-bar return: (close - close_12_ago) / close_12_ago
    - high-low range: (high - low) / close
    - close-open range: (close - open) / open
    - volume ratio: volume / rolling_20_mean_volume
    - volatility: rolling 12-bar std of returns
    """
    close = market["close"].astype(float)
    open_ = market["open"].astype(float)
    high = market["high"].astype(float)
    low = market["low"].astype(float)
    volume = market["volume"].astype(float)

    ret_1 = close.pct_change(1)
    ret_6 = close.pct_change(6)
    ret_12 = close.pct_change(12)
    hl_range = (high - low) / close
    co_range = (close - open_) / open_
    vol_ratio = volume / volume.rolling(20, min_periods=1).mean()
    volatility = ret_1.rolling(12, min_periods=1).std()

    features = pd.DataFrame({
        "ret_1": ret_1,
        "ret_6": ret_6,
        "ret_12": ret_12,
        "hl_range": hl_range,
        "co_range": co_range,
        "vol_ratio": vol_ratio,
        "volatility": volatility,
    })

    return features.fillna(0.0).values


class LogisticBaselineCandidate(Candidate):
    """Logistic regression on engineered OHLCV features."""

    def __init__(self, horizon: str = "6h", threshold: float = 0.55):
        self._horizon = horizon
        self._threshold = threshold
        self._model: LogisticRegression | None = None
        self._scaler: StandardScaler | None = None

    @property
    def name(self) -> str:
        return "logistic_regression"

    @property
    def version(self) -> str:
        return "1.0"

    def reset(self) -> None:
        self._model = None
        self._scaler = None

    def fit(self, market: pd.DataFrame, labels: pd.DataFrame) -> None:
        X = _build_features(market)
        sign_col = f"return_sign_{self._horizon}"
        if sign_col not in labels.columns:
            return

        y_raw = labels[sign_col].values.astype(float)
        # Binary: 1 if positive, 0 otherwise
        y = (y_raw > 0).astype(int)

        # Drop NaN rows
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y_raw))
        if mask.sum() < 10:
            return

        X_clean = X[mask]
        y_clean = y[mask]

        self._scaler = StandardScaler().fit(X_clean)
        X_scaled = self._scaler.transform(X_clean)

        self._model = LogisticRegression(
            max_iter=1000, random_state=42, C=1.0,
        ).fit(X_scaled, y_clean)

    def predict(self, market: pd.DataFrame) -> pd.Series:
        timestamps = market["timestamp"].values
        X = _build_features(market)

        if self._model is None or self._scaler is None:
            return predictions_to_series([
                Prediction(int(ts), Signal.ABSTAIN, 0.5, 0.5)
                for ts in timestamps
            ])

        X_scaled = self._scaler.transform(X)
        probs = self._model.predict_proba(X_scaled)[:, 1]

        predictions = []
        for i, ts in enumerate(timestamps):
            prob_up = float(probs[i])
            if prob_up > self._threshold:
                signal = Signal.LONG
            elif prob_up < (1.0 - self._threshold):
                signal = Signal.SHORT
            else:
                signal = Signal.ABSTAIN

            confidence = abs(prob_up - 0.5) * 2  # 0-1 scale
            predictions.append(Prediction(
                timestamp=int(ts),
                signal=signal,
                confidence=round(float(confidence), 4),
                prob_up=round(prob_up, 4),
            ))

        return predictions_to_series(predictions)


class GBTBaselineCandidate(Candidate):
    """Gradient-boosted tree on engineered OHLCV features."""

    def __init__(self, horizon: str = "6h", threshold: float = 0.55):
        self._horizon = horizon
        self._threshold = threshold
        self._model: GradientBoostingClassifier | None = None
        self._scaler: StandardScaler | None = None

    @property
    def name(self) -> str:
        return "gradient_boosted_tree"

    @property
    def version(self) -> str:
        return "1.0"

    def reset(self) -> None:
        self._model = None
        self._scaler = None

    def fit(self, market: pd.DataFrame, labels: pd.DataFrame) -> None:
        X = _build_features(market)
        sign_col = f"return_sign_{self._horizon}"
        if sign_col not in labels.columns:
            return

        y_raw = labels[sign_col].values.astype(float)
        y = (y_raw > 0).astype(int)

        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y_raw))
        if mask.sum() < 10:
            return

        X_clean = X[mask]
        y_clean = y[mask]

        self._scaler = StandardScaler().fit(X_clean)
        X_scaled = self._scaler.transform(X_clean)

        self._model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        ).fit(X_scaled, y_clean)

    def predict(self, market: pd.DataFrame) -> pd.Series:
        timestamps = market["timestamp"].values
        X = _build_features(market)

        if self._model is None or self._scaler is None:
            return predictions_to_series([
                Prediction(int(ts), Signal.ABSTAIN, 0.5, 0.5)
                for ts in timestamps
            ])

        X_scaled = self._scaler.transform(X)
        probs = self._model.predict_proba(X_scaled)[:, 1]

        predictions = []
        for i, ts in enumerate(timestamps):
            prob_up = float(probs[i])
            if prob_up > self._threshold:
                signal = Signal.LONG
            elif prob_up < (1.0 - self._threshold):
                signal = Signal.SHORT
            else:
                signal = Signal.ABSTAIN

            confidence = abs(prob_up - 0.5) * 2
            predictions.append(Prediction(
                timestamp=int(ts),
                signal=signal,
                confidence=round(float(confidence), 4),
                prob_up=round(prob_up, 4),
            ))

        return predictions_to_series(predictions)
