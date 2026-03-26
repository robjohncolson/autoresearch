"""Tests for dataset loading and manifest validation."""

import json

import numpy as np
import pandas as pd
import pytest

from trading_eval.config import EvalConfig
from trading_eval.data import DatasetError, load_dataset


@pytest.fixture
def data_dir(tmp_path):
    """Create a valid dataset in a temp directory."""
    n = 100
    timestamps = np.arange(1700000000, 1700000000 + n * 3600, 3600)

    market = pd.DataFrame({
        "timestamp": timestamps,
        "open": np.random.default_rng(42).uniform(0.08, 0.10, n),
        "high": np.random.default_rng(42).uniform(0.09, 0.11, n),
        "low": np.random.default_rng(42).uniform(0.07, 0.09, n),
        "close": np.random.default_rng(42).uniform(0.08, 0.10, n),
        "volume": np.random.default_rng(42).uniform(1000, 5000, n),
    })

    labels = pd.DataFrame({
        "return_bps_6h": np.random.default_rng(42).uniform(-100, 100, n),
        "return_sign_6h": np.random.default_rng(42).choice([-1.0, 0.0, 1.0], n),
        "return_bps_12h": np.random.default_rng(42).uniform(-200, 200, n),
        "return_sign_12h": np.random.default_rng(42).choice([-1.0, 0.0, 1.0], n),
        "regime_label": np.random.default_rng(42).choice(["low", "medium", "high"], n),
    })

    manifest = {
        "schema_version": "research-dataset/v1",
        "pair": "DOGE/USD",
        "interval_minutes": 60,
        "row_count": n,
        "timestamp_range": {
            "start": int(timestamps[0]),
            "end": int(timestamps[-1]),
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


def _make_config(data_dir):
    return EvalConfig(data_dir=data_dir)


class TestLoadDatasetValid:
    def test_loads_successfully(self, data_dir):
        ds = load_dataset(_make_config(data_dir))
        assert len(ds.market) == 100
        assert len(ds.labels) == 100
        assert ds.manifest["schema_version"] == "research-dataset/v1"

    def test_market_columns_present(self, data_dir):
        ds = load_dataset(_make_config(data_dir))
        for col in ("timestamp", "open", "high", "low", "close", "volume"):
            assert col in ds.market.columns

    def test_label_columns_present(self, data_dir):
        ds = load_dataset(_make_config(data_dir))
        for col in ("return_bps_6h", "return_sign_6h", "return_bps_12h", "return_sign_12h"):
            assert col in ds.labels.columns

    def test_timestamps_monotonic(self, data_dir):
        ds = load_dataset(_make_config(data_dir))
        assert ds.market["timestamp"].is_monotonic_increasing


class TestLoadDatasetMissingManifest:
    def test_raises_on_missing_manifest(self, tmp_path):
        with pytest.raises(DatasetError, match="Missing manifest"):
            load_dataset(_make_config(tmp_path))


class TestLoadDatasetSchemaMismatch:
    def test_raises_on_wrong_schema_version(self, data_dir):
        manifest_path = data_dir / "manifest_v1.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["schema_version"] = "wrong/v2"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        with pytest.raises(DatasetError, match="Schema version mismatch"):
            load_dataset(_make_config(data_dir))


class TestLoadDatasetRowCountMismatch:
    def test_raises_on_row_count_mismatch(self, data_dir):
        # Overwrite labels with fewer rows
        labels = pd.DataFrame({
            "return_bps_6h": [1.0],
            "return_sign_6h": [1.0],
            "return_bps_12h": [1.0],
            "return_sign_12h": [1.0],
            "regime_label": ["low"],
        })
        labels.to_parquet(data_dir / "labels_v1.parquet", index=False)

        with pytest.raises(DatasetError, match="Row count mismatch"):
            load_dataset(_make_config(data_dir))

    def test_raises_on_manifest_row_count_mismatch(self, data_dir):
        manifest_path = data_dir / "manifest_v1.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["row_count"] = 999
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        with pytest.raises(DatasetError, match="Row count mismatch with manifest"):
            load_dataset(_make_config(data_dir))


class TestLoadDatasetMissingColumns:
    def test_raises_on_missing_market_columns(self, data_dir):
        market = pd.DataFrame({"timestamp": [1], "open": [1.0]})
        market.to_parquet(data_dir / "market_v1.parquet", index=False)

        # Also fix row count
        manifest_path = data_dir / "manifest_v1.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["row_count"] = 1
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        labels = pd.DataFrame({
            "return_bps_6h": [1.0],
            "return_sign_6h": [1.0],
            "return_bps_12h": [1.0],
            "return_sign_12h": [1.0],
            "regime_label": ["low"],
        })
        labels.to_parquet(data_dir / "labels_v1.parquet", index=False)

        with pytest.raises(DatasetError, match="Missing market columns"):
            load_dataset(_make_config(data_dir))

    def test_raises_on_missing_label_columns(self, data_dir):
        manifest_path = data_dir / "manifest_v1.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["label_columns"] = ["return_bps_6h", "nonexistent_col"]
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        with pytest.raises(DatasetError, match="Missing label columns"):
            load_dataset(_make_config(data_dir))


class TestLoadDatasetNoLabelColumns:
    def test_raises_on_empty_label_columns(self, data_dir):
        manifest_path = data_dir / "manifest_v1.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["label_columns"] = []
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        with pytest.raises(DatasetError, match="no label_columns"):
            load_dataset(_make_config(data_dir))
