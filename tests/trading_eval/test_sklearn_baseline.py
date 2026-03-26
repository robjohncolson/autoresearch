"""Tests for sklearn baselines."""

import json

import numpy as np
import pandas as pd
import pytest

from trading_eval.baselines.sklearn_baseline import (
    GBTBaselineCandidate,
    LogisticBaselineCandidate,
    _build_features,
)
from trading_eval.candidate import Signal
from trading_eval.config import EvalConfig
from trading_eval.runner import run_experiment


def _make_dataset(n: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    close = 0.09 * np.cumprod(1 + rng.normal(0, 0.002, n))
    timestamps = np.arange(1700000000, 1700000000 + n * 3600, 3600)

    market = pd.DataFrame({
        "timestamp": timestamps,
        "open": close * (1 + rng.uniform(-0.002, 0.002, n)),
        "high": close * (1 + rng.uniform(0, 0.005, n)),
        "low": close * (1 - rng.uniform(0, 0.005, n)),
        "close": close,
        "volume": rng.uniform(1000, 5000, n),
    })

    return_bps_6h = np.full(n, np.nan)
    return_sign_6h = np.full(n, np.nan)
    for i in range(n - 6):
        ret = 10_000 * (close[i + 6] - close[i]) / close[i]
        return_bps_6h[i] = ret
        return_sign_6h[i] = np.sign(ret)

    labels = pd.DataFrame({
        "return_bps_6h": return_bps_6h,
        "return_sign_6h": return_sign_6h,
        "return_bps_12h": np.full(n, np.nan),
        "return_sign_12h": np.full(n, np.nan),
        "regime_label": ["medium"] * n,
    })

    return market, labels


class TestBuildFeatures:
    def test_shape(self):
        market, _ = _make_dataset(100)
        X = _build_features(market)
        assert X.shape == (100, 7)

    def test_no_nans(self):
        market, _ = _make_dataset(100)
        X = _build_features(market)
        assert not np.isnan(X).any()


class TestLogisticBaseline:
    def test_fit_predict(self):
        market, labels = _make_dataset(200)
        c = LogisticBaselineCandidate()
        c.fit(market[:150], labels[:150])
        preds = c.predict(market[150:].reset_index(drop=True))
        assert len(preds) == 50
        assert all(p.prob_up >= 0 and p.prob_up <= 1 for p in preds)

    def test_name_and_version(self):
        c = LogisticBaselineCandidate()
        assert c.name == "logistic_regression"
        assert c.version == "1.0"

    def test_reset_clears_model(self):
        market, labels = _make_dataset(200)
        c = LogisticBaselineCandidate()
        c.fit(market, labels)
        assert c._model is not None
        c.reset()
        assert c._model is None

    def test_predict_without_fit_abstains(self):
        market, _ = _make_dataset(50)
        c = LogisticBaselineCandidate()
        preds = c.predict(market)
        assert all(p.signal == Signal.ABSTAIN for p in preds)


class TestGBTBaseline:
    def test_fit_predict(self):
        market, labels = _make_dataset(200)
        c = GBTBaselineCandidate()
        c.fit(market[:150], labels[:150])
        preds = c.predict(market[150:].reset_index(drop=True))
        assert len(preds) == 50

    def test_name_and_version(self):
        c = GBTBaselineCandidate()
        assert c.name == "gradient_boosted_tree"
        assert c.version == "1.0"


class TestSklearnInRunner:
    """Integration test: run sklearn baseline through the full runner."""

    @pytest.fixture
    def dataset_dir(self, tmp_path):
        market, labels = _make_dataset(720)  # 30 days hourly
        manifest = {
            "schema_version": "research-dataset/v1",
            "pair": "DOGE/USD",
            "interval_minutes": 60,
            "row_count": 720,
            "timestamp_range": {
                "start": int(market["timestamp"].iloc[0]),
                "end": int(market["timestamp"].iloc[-1]),
            },
            "market_columns": list(market.columns),
            "label_columns": list(labels.columns),
            "generated_at": "2026-03-25T14:30:45Z",
        }
        market.to_parquet(tmp_path / "market_v1.parquet", index=False)
        labels.to_parquet(tmp_path / "labels_v1.parquet", index=False)
        with open(tmp_path / "manifest_v1.json", "w") as f:
            json.dump(manifest, f, indent=2)
        return tmp_path

    def test_logistic_through_runner(self, dataset_dir):
        config = EvalConfig(
            data_dir=dataset_dir,
            train_days=10,
            val_days=1,
            step_days=5,
        )
        result = run_experiment(LogisticBaselineCandidate(), config)
        assert result.aggregate_metrics.trade_count >= 0
        assert len(result.fold_results) > 0
