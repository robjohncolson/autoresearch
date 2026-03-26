"""Tests for experiment storage."""

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
from trading_eval.runner import run_experiment
from trading_eval.storage import (
    ComparisonTable,
    compare_experiments,
    format_comparison,
    list_experiments,
    load_experiment,
    save_experiment,
)


class StubCandidate(Candidate):
    @property
    def name(self) -> str:
        return "stub"

    @property
    def version(self) -> str:
        return "0.1"

    def fit(self, market, labels):
        pass

    def predict(self, market):
        return predictions_to_series([
            Prediction(int(ts), Signal.LONG, 0.6, 0.6)
            for ts in market["timestamp"]
        ])


@pytest.fixture
def experiment_result(tmp_path):
    n = 30 * 24
    rng = np.random.default_rng(42)
    timestamps = np.arange(1700000000, 1700000000 + n * 3600, 3600)
    close = 0.09 * np.cumprod(1 + rng.normal(0, 0.001, n))

    market = pd.DataFrame({
        "timestamp": timestamps,
        "open": close, "high": close, "low": close,
        "close": close, "volume": rng.uniform(1000, 5000, n),
    })

    return_bps = np.full(n, np.nan)
    return_sign = np.full(n, np.nan)
    for i in range(n - 6):
        ret = 10_000 * (close[i + 6] - close[i]) / close[i]
        return_bps[i] = ret
        return_sign[i] = np.sign(ret)

    labels = pd.DataFrame({
        "return_bps_6h": return_bps, "return_sign_6h": return_sign,
        "return_bps_12h": np.full(n, np.nan), "return_sign_12h": np.full(n, np.nan),
        "regime_label": ["medium"] * n,
    })

    manifest = {
        "schema_version": "research-dataset/v1",
        "pair": "DOGE/USD", "interval_minutes": 60, "row_count": n,
        "timestamp_range": {"start": int(timestamps[0]), "end": int(timestamps[-1])},
        "market_columns": list(market.columns),
        "label_columns": list(labels.columns),
        "generated_at": "2026-03-25T14:30:45Z",
    }

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    market.to_parquet(data_dir / "market_v1.parquet", index=False)
    labels.to_parquet(data_dir / "labels_v1.parquet", index=False)
    with open(data_dir / "manifest_v1.json", "w") as f:
        json.dump(manifest, f, indent=2)

    config = EvalConfig(data_dir=data_dir, train_days=10, val_days=1, step_days=5)
    return run_experiment(StubCandidate(), config)


class TestSaveAndLoad:
    def test_round_trip(self, experiment_result, tmp_path):
        out_dir = tmp_path / "experiments"
        path = save_experiment(experiment_result, out_dir)
        assert path.exists()

        record = load_experiment(path)
        assert record["record_version"] == "experiment/v1"
        assert record["candidate_name"] == "stub"
        assert record["candidate_version"] == "0.1"

    def test_record_has_reproducibility_fields(self, experiment_result, tmp_path):
        out_dir = tmp_path / "experiments"
        path = save_experiment(experiment_result, out_dir)
        record = load_experiment(path)

        assert "dataset_manifest_hash" in record
        assert len(record["dataset_manifest_hash"]) == 16
        assert "dataset_manifest" in record
        assert record["dataset_manifest"]["schema_version"] == "research-dataset/v1"
        assert "source_commit" in record
        assert "config" in record
        assert "fold_summaries" in record
        assert len(record["fold_summaries"]) == record["fold_count"]

    def test_fold_summaries_have_metrics(self, experiment_result, tmp_path):
        out_dir = tmp_path / "experiments"
        path = save_experiment(experiment_result, out_dir)
        record = load_experiment(path)

        for fold in record["fold_summaries"]:
            assert "train_start" in fold
            assert "val_end" in fold
            assert "metrics" in fold
            assert "direction_accuracy" in fold["metrics"]

    def test_invalid_version_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        with open(path, "w") as f:
            json.dump({"record_version": "wrong/v99"}, f)

        with pytest.raises(ValueError, match="Unsupported record version"):
            load_experiment(path)


class TestListExperiments:
    def test_lists_saved_experiments(self, experiment_result, tmp_path):
        out_dir = tmp_path / "experiments"
        save_experiment(experiment_result, out_dir)
        summaries = list_experiments(out_dir)
        assert len(summaries) == 1
        assert summaries[0]["candidate_name"] == "stub"
        assert "net_pnl_bps" in summaries[0]

    def test_empty_dir(self, tmp_path):
        assert list_experiments(tmp_path / "nonexistent") == []


class TestCompareExperiments:
    def test_comparison(self, experiment_result, tmp_path):
        out_dir = tmp_path / "experiments"
        path = save_experiment(experiment_result, out_dir)
        record = load_experiment(path)

        table = compare_experiments(record, record)
        assert isinstance(table, ComparisonTable)
        assert len(table.rows) > 0
        # Same experiment compared to itself -> all ties
        for row in table.rows:
            assert row.delta == 0.0

    def test_format_comparison(self, experiment_result, tmp_path):
        out_dir = tmp_path / "experiments"
        path = save_experiment(experiment_result, out_dir)
        record = load_experiment(path)
        table = compare_experiments(record, record)
        text = format_comparison(table)
        assert "Metric" in text
        assert "stub" in text
