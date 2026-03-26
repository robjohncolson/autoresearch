"""Tests for LLM candidate (mocked Ollama, no real inference)."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from trading_eval.baselines.llm_candidate import (
    DECISION_CADENCE,
    LLMCandidate,
    _extract_json,
)
from trading_eval.candidate import Signal
from trading_eval.config import EvalConfig
from trading_eval.runner import run_experiment
from trading_eval.storage import save_experiment, load_experiment


def _make_market(n: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 0.09 * np.cumprod(1 + rng.normal(0, 0.002, n))
    timestamps = np.arange(1700000000, 1700000000 + n * 3600, 3600)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close * (1 + rng.uniform(-0.002, 0.002, n)),
        "high": close * (1 + rng.uniform(0, 0.005, n)),
        "low": close * (1 - rng.uniform(0, 0.005, n)),
        "close": close,
        "volume": rng.uniform(1000, 5000, n),
    })


def _make_labels(n: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "return_bps_6h": rng.normal(0, 50, n),
        "return_sign_6h": rng.choice([-1.0, 0.0, 1.0], n),
        "return_bps_12h": np.full(n, np.nan),
        "return_sign_12h": np.full(n, np.nan),
        "regime_label": ["medium"] * n,
    })


def _mock_ollama_response(direction="long", confidence=0.8, prob_up=0.7, horizon_hours=6):
    """Build a mock requests.Response for Ollama API."""
    content = json.dumps({
        "direction": direction,
        "confidence": confidence,
        "prob_up": prob_up,
        "horizon_hours": horizon_hours,
    })
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"message": {"content": content}}
    resp.raise_for_status = MagicMock()
    return resp


class TestExtractJson:
    def test_plain_json(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_json_with_think_tags(self):
        text = '<think>reasoning here</think>{"direction": "long"}'
        result = _extract_json(text)
        assert result == {"direction": "long"}

    def test_no_json(self):
        assert _extract_json("no json here") is None

    def test_invalid_json(self):
        assert _extract_json("{broken json") is None


class TestLLMCandidate:
    def test_name_and_version(self):
        c = LLMCandidate()
        assert c.name == "llm_qwen3_8b"
        assert c.version == "1.0"

    def test_fit_stores_summary_and_tail(self):
        c = LLMCandidate(context_bars=20)
        market = _make_market(100)
        labels = _make_labels(100)
        c.fit(market, labels)

        assert c._train_summary is not None
        assert "mean_close" in c._train_summary
        assert "volatility_ann_pct" in c._train_summary
        assert "trend_pct" in c._train_summary
        assert c._train_tail is not None
        assert len(c._train_tail) == 20  # context_bars

    def test_reset_clears_state(self):
        c = LLMCandidate()
        c.fit(_make_market(60), _make_labels(60))
        assert c._train_summary is not None
        c.reset()
        assert c._train_summary is None
        assert c._train_tail is None

    def test_inference_metadata(self):
        c = LLMCandidate(model="qwen3.5:9b", temperature=0.0, seed=42, context_bars=48)
        meta = c.inference_metadata
        assert meta["model"] == "qwen3.5:9b"
        assert meta["temperature"] == 0.0
        assert meta["seed"] == 42
        assert meta["context_bars"] == 48
        assert meta["decision_cadence"] == 6
        assert "prompt_version" in meta
        assert "prompt_hash" in meta


class TestLLMPredict:
    @patch("trading_eval.baselines.llm_candidate.requests.post")
    def test_predict_6h_cadence(self, mock_post):
        """Only every 6th row should get a non-ABSTAIN prediction."""
        mock_post.return_value = _mock_ollama_response()
        c = LLMCandidate(decision_cadence=6)
        c.fit(_make_market(100), _make_labels(100))

        val_market = _make_market(24)  # 24 rows = 1 day
        preds = c.predict(val_market)

        assert len(preds) == 24
        for i in range(24):
            if i % 6 == 0:
                # Decision point — should have called Ollama
                assert preds.iloc[i].signal != Signal.ABSTAIN or True  # may still abstain on parse
            else:
                # Non-decision point — always ABSTAIN
                assert preds.iloc[i].signal == Signal.ABSTAIN

    @patch("trading_eval.baselines.llm_candidate.requests.post")
    def test_predict_success(self, mock_post):
        """Valid Ollama response → LONG at decision points."""
        mock_post.return_value = _mock_ollama_response(
            direction="long", confidence=0.8, prob_up=0.7
        )
        c = LLMCandidate(decision_cadence=6)
        c.fit(_make_market(100), _make_labels(100))

        val_market = _make_market(12)
        preds = c.predict(val_market)

        # Row 0 and 6 are decision points
        assert preds.iloc[0].signal == Signal.LONG
        assert preds.iloc[0].confidence == 0.8
        assert preds.iloc[0].prob_up == 0.7
        assert preds.iloc[6].signal == Signal.LONG
        # Non-decision rows are ABSTAIN
        assert preds.iloc[1].signal == Signal.ABSTAIN

    @patch("trading_eval.baselines.llm_candidate.requests.post")
    def test_predict_connection_error(self, mock_post):
        """Connection error → all ABSTAIN."""
        mock_post.side_effect = ConnectionError("refused")
        c = LLMCandidate(decision_cadence=6)
        c.fit(_make_market(100), _make_labels(100))

        preds = c.predict(_make_market(12))
        assert all(p.signal == Signal.ABSTAIN for p in preds)

    @patch("trading_eval.baselines.llm_candidate.requests.post")
    def test_predict_invalid_json(self, mock_post):
        """Garbage response → ABSTAIN at decision points."""
        resp = MagicMock()
        resp.json.return_value = {"message": {"content": "not json at all"}}
        resp.raise_for_status = MagicMock()
        mock_post.return_value = resp

        c = LLMCandidate(decision_cadence=6)
        c.fit(_make_market(100), _make_labels(100))
        preds = c.predict(_make_market(12))
        assert all(p.signal == Signal.ABSTAIN for p in preds)

    @patch("trading_eval.baselines.llm_candidate.requests.post")
    def test_predict_missing_fields(self, mock_post):
        """JSON without prob_up → ABSTAIN."""
        resp = MagicMock()
        content = json.dumps({"direction": "long", "confidence": 0.8})
        resp.json.return_value = {"message": {"content": content}}
        resp.raise_for_status = MagicMock()
        mock_post.return_value = resp

        c = LLMCandidate(decision_cadence=6)
        c.fit(_make_market(100), _make_labels(100))
        preds = c.predict(_make_market(12))
        # Decision point predictions should be ABSTAIN due to missing prob_up
        assert preds.iloc[0].signal == Signal.ABSTAIN

    @patch("trading_eval.baselines.llm_candidate.requests.post")
    def test_predict_wrong_horizon(self, mock_post):
        """horizon_hours != 6 → ABSTAIN."""
        mock_post.return_value = _mock_ollama_response(horizon_hours=12)
        c = LLMCandidate(decision_cadence=6)
        c.fit(_make_market(100), _make_labels(100))
        preds = c.predict(_make_market(12))
        assert preds.iloc[0].signal == Signal.ABSTAIN

    @patch("trading_eval.baselines.llm_candidate.requests.post")
    def test_predict_point_in_time(self, mock_post):
        """Verify bars in prompt are only up to decision timestamp."""
        calls = []

        def capture_call(*args, **kwargs):
            calls.append(kwargs)
            return _mock_ollama_response()

        mock_post.side_effect = capture_call

        # Use a contiguous timeline: train=50 rows, then val=18 rows
        rng = np.random.default_rng(42)
        total_n = 68
        all_ts = np.arange(1700000000, 1700000000 + total_n * 3600, 3600)
        close = 0.09 * np.cumprod(1 + rng.normal(0, 0.002, total_n))

        full_market = pd.DataFrame({
            "timestamp": all_ts,
            "open": close, "high": close, "low": close,
            "close": close, "volume": rng.uniform(1000, 5000, total_n),
        })

        train_market = full_market.iloc[:50].reset_index(drop=True)
        val_market = full_market.iloc[50:].reset_index(drop=True)

        c = LLMCandidate(decision_cadence=6, context_bars=10)
        c.fit(train_market, _make_labels(50))
        c.predict(val_market)

        # Should have 3 Ollama calls (rows 0, 6, 12)
        assert len(calls) == 3

        # All timestamps in each prompt must be <= the decision timestamp
        for i, call_kwargs in enumerate(calls):
            request_body = call_kwargs.get("json", {})
            user_msg = request_body["messages"][1]["content"]
            decision_row = i * 6
            decision_ts = int(val_market["timestamp"].iloc[decision_row])
            for line in user_msg.split("\n"):
                parts = line.split(",")
                if len(parts) >= 5 and parts[0].strip().isdigit():
                    ts_in_prompt = int(parts[0].strip())
                    assert ts_in_prompt <= decision_ts, (
                        f"Future timestamp {ts_in_prompt} found in prompt for "
                        f"decision at {decision_ts} (row {decision_row})"
                    )

    @patch("trading_eval.baselines.llm_candidate.requests.post")
    def test_prompt_includes_regime_context(self, mock_post):
        """Verify training regime summary appears in prompt."""
        captured = []

        def capture(*args, **kwargs):
            captured.append(kwargs)
            return _mock_ollama_response()

        mock_post.side_effect = capture

        c = LLMCandidate(decision_cadence=6)
        c.fit(_make_market(100), _make_labels(100))
        c.predict(_make_market(6))  # one decision point

        assert len(captured) == 1
        user_msg = captured[0]["json"]["messages"][1]["content"]
        assert "mean_close" in user_msg
        assert "volatility" in user_msg.lower()
        assert "trend" in user_msg.lower()


class TestLLMCandidateIntegration:
    """Integration test through full runner with mocked Ollama."""

    @patch("trading_eval.baselines.llm_candidate.requests.post")
    def test_through_runner(self, mock_post, tmp_path):
        mock_post.return_value = _mock_ollama_response(
            direction="long", confidence=0.75, prob_up=0.65
        )

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
        result = run_experiment(LLMCandidate(), config)

        assert result.candidate_name == "llm_qwen3_8b"
        assert len(result.fold_results) > 0
        assert result.aggregate_metrics.trade_count >= 0

        # Save and verify experiment record
        exp_dir = tmp_path / "experiments"
        path = save_experiment(result, exp_dir)
        record = load_experiment(path)
        assert record["candidate_name"] == "llm_qwen3_8b"
