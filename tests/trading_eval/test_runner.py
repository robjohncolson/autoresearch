"""Tests for experiment runner."""

import json

import numpy as np
import pandas as pd
import pytest

from trading_eval.candidate import (
    Candidate,
    Prediction,
    Signal,
    predictions_to_series,
)
from trading_eval.config import EvalConfig
from trading_eval.runner import ExperimentResult, run_experiment


class AlwaysLongCandidate(Candidate):
    """Test candidate that always predicts LONG with confidence 0.7."""

    @property
    def name(self) -> str:
        return "always_long"

    def fit(self, market: pd.DataFrame, labels: pd.DataFrame) -> None:
        pass

    def predict(self, market: pd.DataFrame) -> pd.Series:
        preds = [
            Prediction(
                timestamp=int(ts),
                signal=Signal.LONG,
                confidence=0.7,
                prob_up=0.7,
            )
            for ts in market["timestamp"]
        ]
        return predictions_to_series(preds)


@pytest.fixture
def dataset_dir(tmp_path):
    """Create a 30-day hourly dataset."""
    n = 30 * 24  # 720 hourly candles
    rng = np.random.default_rng(42)
    timestamps = np.arange(1700000000, 1700000000 + n * 3600, 3600)

    # Simulate a random walk for close prices
    returns = rng.normal(0, 0.001, n)
    close = 0.09 * np.cumprod(1 + returns)

    market = pd.DataFrame({
        "timestamp": timestamps,
        "open": close * (1 + rng.uniform(-0.005, 0.005, n)),
        "high": close * (1 + rng.uniform(0, 0.01, n)),
        "low": close * (1 - rng.uniform(0, 0.01, n)),
        "close": close,
        "volume": rng.uniform(1000, 5000, n),
    })

    # Compute actual return_bps labels (10_000 * fractional return)
    return_bps_6h = np.full(n, np.nan)
    return_sign_6h = np.full(n, np.nan)
    for i in range(n - 6):
        ret = 10_000 * (close[i + 6] - close[i]) / close[i]
        return_bps_6h[i] = ret
        return_sign_6h[i] = np.sign(ret)

    labels = pd.DataFrame({
        "return_bps_6h": return_bps_6h,
        "return_sign_6h": return_sign_6h,
        "return_bps_12h": np.full(n, np.nan),  # unused but required
        "return_sign_12h": np.full(n, np.nan),
        "regime_label": ["medium"] * n,
    })

    manifest = {
        "schema_version": "research-dataset/v1",
        "pair": "DOGE/USD",
        "interval_minutes": 60,
        "row_count": n,
        "timestamp_range": {"start": int(timestamps[0]), "end": int(timestamps[-1])},
        "market_columns": list(market.columns),
        "label_columns": list(labels.columns),
        "generated_at": "2026-03-25T14:30:45Z",
    }

    market.to_parquet(tmp_path / "market_v1.parquet", index=False)
    labels.to_parquet(tmp_path / "labels_v1.parquet", index=False)
    with open(tmp_path / "manifest_v1.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return tmp_path


class TestRunExperiment:
    def test_produces_result(self, dataset_dir):
        config = EvalConfig(
            data_dir=dataset_dir,
            train_days=10,
            val_days=1,
            step_days=1,
        )
        result = run_experiment(AlwaysLongCandidate(), config)

        assert isinstance(result, ExperimentResult)
        assert result.candidate_name == "always_long"
        assert len(result.fold_results) > 0
        assert result.aggregate_metrics.trade_count > 0

    def test_fold_count(self, dataset_dir):
        config = EvalConfig(
            data_dir=dataset_dir,
            train_days=10,
            val_days=1,
            step_days=5,
        )
        result = run_experiment(AlwaysLongCandidate(), config)
        # 30 days, 10 train + 1 val = 11 min, step 5 -> ~4 folds
        assert len(result.fold_results) >= 3

    def test_no_future_leakage_in_folds(self, dataset_dir):
        config = EvalConfig(
            data_dir=dataset_dir,
            train_days=10,
            val_days=1,
            step_days=3,
        )
        result = run_experiment(AlwaysLongCandidate(), config)
        for fr in result.fold_results:
            assert fr.train_end == fr.val_start
            assert fr.train_end <= fr.val_start

    def test_aggregate_metrics_sum_trades(self, dataset_dir):
        config = EvalConfig(
            data_dir=dataset_dir,
            train_days=10,
            val_days=1,
            step_days=5,
        )
        result = run_experiment(AlwaysLongCandidate(), config)
        total_fold_trades = sum(fr.metrics.trade_count for fr in result.fold_results)
        assert result.aggregate_metrics.trade_count == total_fold_trades

    def test_experiment_id_is_set(self, dataset_dir):
        config = EvalConfig(data_dir=dataset_dir, train_days=10, val_days=1, step_days=5)
        result = run_experiment(AlwaysLongCandidate(), config)
        assert len(result.experiment_id) == 8  # uuid[:8]

    def test_timestamps_are_set(self, dataset_dir):
        config = EvalConfig(data_dir=dataset_dir, train_days=10, val_days=1, step_days=5)
        result = run_experiment(AlwaysLongCandidate(), config)
        assert result.started_at < result.finished_at
