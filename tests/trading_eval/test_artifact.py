"""Tests for artifact schema and promotion."""

import json

import numpy as np
import pandas as pd
import pytest

from trading_eval.artifact import (
    ArtifactManifest,
    list_artifacts,
    load_artifact,
    promote_candidate,
)
from trading_eval.candidate import (
    Candidate,
    Prediction,
    Signal,
    predictions_to_series,
)
from trading_eval.config import EvalConfig
from trading_eval.runner import run_experiment
from trading_eval.storage import save_experiment, load_experiment


class StubCandidate(Candidate):
    @property
    def name(self):
        return "stub_model"

    def fit(self, market, labels):
        pass

    def predict(self, market):
        return predictions_to_series([
            Prediction(int(ts), Signal.LONG, 0.6, 0.6)
            for ts in market["timestamp"]
        ])


@pytest.fixture
def experiment_record(tmp_path):
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
    result = run_experiment(StubCandidate(), config)
    exp_dir = tmp_path / "experiments"
    path = save_experiment(result, exp_dir)
    return load_experiment(path)


class TestPromoteCandidate:
    def test_creates_artifact_directory(self, experiment_record, tmp_path):
        art_dir = tmp_path / "artifacts"
        path = promote_candidate(experiment_record, artifacts_dir=art_dir)
        assert path.exists()
        assert (path / "manifest.json").exists()
        assert (path / "experiment.json").exists()
        assert (path / "model").is_dir()

    def test_manifest_fields(self, experiment_record, tmp_path):
        art_dir = tmp_path / "artifacts"
        path = promote_candidate(experiment_record, artifacts_dir=art_dir)
        manifest = load_artifact(path)

        assert isinstance(manifest, ArtifactManifest)
        assert manifest.model_family == "stub_model"
        assert manifest.input_schema_version == "market/v1"
        assert manifest.output_schema_version == "prediction/v1"
        assert manifest.label_horizon == "6h"
        assert "direction_accuracy" in manifest.evaluation_summary

    def test_artifact_version(self, experiment_record, tmp_path):
        art_dir = tmp_path / "artifacts"
        path = promote_candidate(experiment_record, artifact_version="2.0", artifacts_dir=art_dir)
        manifest = load_artifact(path)
        assert manifest.artifact_version == "2.0"


class TestListArtifacts:
    def test_lists_promoted_artifacts(self, experiment_record, tmp_path):
        art_dir = tmp_path / "artifacts"
        promote_candidate(experiment_record, artifacts_dir=art_dir)
        artifacts = list_artifacts(art_dir)
        assert len(artifacts) == 1
        assert artifacts[0].model_family == "stub_model"

    def test_empty_dir(self, tmp_path):
        assert list_artifacts(tmp_path / "nonexistent") == []


class TestArtifactManifestToDict:
    def test_round_trip(self, experiment_record, tmp_path):
        art_dir = tmp_path / "artifacts"
        path = promote_candidate(experiment_record, artifacts_dir=art_dir)
        manifest = load_artifact(path)
        d = manifest.to_dict()
        assert d["artifact_id"] == manifest.artifact_id
        assert d["model_family"] == "stub_model"
        assert len(d) == 11
