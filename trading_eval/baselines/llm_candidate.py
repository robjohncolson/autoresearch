"""LLM candidate via local Ollama API (Qwen3 8B).

Calls Ollama at a 6h decision cadence with point-in-time OHLCV context.
Deterministic: temperature=0, fixed seed. Any failure → ABSTAIN.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import Final

import pandas as pd
import requests

from trading_eval.candidate import (
    Candidate,
    Prediction,
    Signal,
    predictions_to_series,
)

logger = logging.getLogger(__name__)

PROMPT_VERSION: Final[str] = "v1"
DECISION_CADENCE: Final[int] = 6  # predict every 6 rows (= 6h with hourly candles)

SYSTEM_PROMPT: Final[str] = """\
You are a cryptocurrency trading analyst. The user will give you \
recent hourly price candles for a crypto pair. Analyze the price \
action, momentum, and volatility to predict the most likely price \
direction over the next 6 hours. You must respond with a JSON \
prediction object and nothing else."""


def _prompt_hash() -> str:
    """Stable hash of the system prompt template for reproducibility."""
    return hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]


def _extract_json(text: str) -> dict | None:
    """Extract JSON from text that may contain <think> tags or preamble."""
    # Strip Qwen3 thinking tags
    brace_pos = text.find("{")
    if brace_pos == -1:
        return None
    # Find the matching closing brace
    depth = 0
    for i in range(brace_pos, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[brace_pos : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


class LLMCandidate(Candidate):
    """Ollama-backed LLM candidate with 6h decision cadence."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
        context_bars: int = 48,
        timeout: float = 120.0,
        temperature: float = 0.0,
        seed: int = 42,
        decision_cadence: int = DECISION_CADENCE,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._context_bars = context_bars
        self._timeout = timeout
        self._temperature = temperature
        self._seed = seed
        self._num_predict = 100  # cap output tokens to prevent verbose generation
        self._decision_cadence = decision_cadence
        self._train_summary: dict | None = None
        self._train_tail: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return "llm_qwen3_8b"

    @property
    def version(self) -> str:
        return "1.0"

    @property
    def inference_metadata(self) -> dict:
        """Reproducibility metadata for experiment records."""
        return {
            "model": self._model,
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": _prompt_hash(),
            "temperature": self._temperature,
            "seed": self._seed,
            "context_bars": self._context_bars,
            "decision_cadence": self._decision_cadence,
            "base_url": self._base_url,
        }

    def reset(self) -> None:
        self._train_summary = None
        self._train_tail = None

    def fit(self, market: pd.DataFrame, labels: pd.DataFrame) -> None:
        """Compute regime summary from training data only."""
        close = market["close"].astype(float)
        volume = market["volume"].astype(float)

        first_close = float(close.iloc[0])
        last_close = float(close.iloc[-1])
        mean_close = float(close.mean())
        returns = close.pct_change().dropna()
        volatility_ann = float(returns.std() * math.sqrt(8760)) if len(returns) > 1 else 0.0
        trend_pct = 100.0 * (last_close - first_close) / first_close if first_close > 0 else 0.0
        avg_volume = float(volume.mean())
        price_range_pct = (
            100.0 * (float(close.max()) - float(close.min())) / mean_close
            if mean_close > 0
            else 0.0
        )

        self._train_summary = {
            "mean_close": round(mean_close, 6),
            "volatility_ann_pct": round(volatility_ann * 100, 1),
            "trend_pct": round(trend_pct, 2),
            "avg_volume": round(avg_volume, 0),
            "price_range_pct": round(price_range_pct, 1),
        }

        # Store tail of training data for point-in-time context
        tail_n = min(self._context_bars, len(market))
        self._train_tail = market.iloc[-tail_n:].copy().reset_index(drop=True)

    def predict(self, market: pd.DataFrame) -> pd.Series:
        """Generate predictions at 6h decision cadence.

        Non-decision rows get ABSTAIN. Decision rows call Ollama with
        all bars up to that timestamp (training tail + validation prefix).
        """
        timestamps = market["timestamp"].values
        predictions: list[Prediction] = []

        for i in range(len(market)):
            ts = int(timestamps[i])

            # Only predict at decision points
            if i % self._decision_cadence != 0:
                predictions.append(
                    Prediction(timestamp=ts, signal=Signal.ABSTAIN, confidence=0.5, prob_up=0.5)
                )
                continue

            # Build point-in-time context: training tail + validation rows up to i
            val_prefix = market.iloc[: i + 1]
            if self._train_tail is not None:
                context = pd.concat(
                    [self._train_tail, val_prefix], ignore_index=True
                )
            else:
                context = val_prefix.copy()

            # Take last context_bars rows
            if len(context) > self._context_bars:
                context = context.iloc[-self._context_bars :]

            system_msg, user_msg = self._build_prompt(context)
            raw = self._call_ollama(system_msg, user_msg)

            if raw is None:
                predictions.append(
                    Prediction(timestamp=ts, signal=Signal.ABSTAIN, confidence=0.5, prob_up=0.5)
                )
                continue

            parsed = self._parse_response(raw)
            if parsed is None:
                predictions.append(
                    Prediction(timestamp=ts, signal=Signal.ABSTAIN, confidence=0.5, prob_up=0.5)
                )
                continue

            signal, confidence, prob_up = parsed
            predictions.append(
                Prediction(timestamp=ts, signal=signal, confidence=confidence, prob_up=prob_up)
            )

        return predictions_to_series(predictions)

    def _build_prompt(self, bars: pd.DataFrame) -> tuple[str, str]:
        """Build system and user prompts from point-in-time bars."""
        # Format OHLCV as compact CSV
        lines = ["timestamp,open,high,low,close,volume"]
        for _, row in bars.iterrows():
            lines.append(
                f"{int(row['timestamp'])},"
                f"{float(row['open']):.6f},"
                f"{float(row['high']):.6f},"
                f"{float(row['low']):.6f},"
                f"{float(row['close']):.6f},"
                f"{int(float(row['volume']))}"
            )
        ohlcv_table = "\n".join(lines)

        regime_text = ""
        if self._train_summary:
            s = self._train_summary
            regime_text = (
                f"Training period summary (from prior data, not shown):\n"
                f"  mean_close: {s['mean_close']}\n"
                f"  annualized_volatility: {s['volatility_ann_pct']}%\n"
                f"  trend: {s['trend_pct']:+.2f}% over training window\n"
                f"  avg_hourly_volume: {int(s['avg_volume'])}\n"
                f"  price_range: {s['price_range_pct']}%\n"
            )

        user_msg = (
            f"Pair: DOGE/USD\n"
            f"{regime_text}\n"
            f"Here are the most recent hourly price candles (newest last):\n"
            f"{ohlcv_table}\n\n"
            f"Based on the price action above, predict DOGE/USD direction "
            f"for the next 6 hours. Respond with this exact JSON format:\n"
            f'{{"direction":"long","confidence":0.75,"prob_up":0.75,"horizon_hours":6}}\n'
            f"Use \"long\" if price will rise, \"short\" if fall, \"abstain\" if uncertain."
        )

        return SYSTEM_PROMPT, user_msg

    def _call_ollama(self, system: str, user: str) -> dict | None:
        """Call Ollama chat API and return parsed JSON response."""
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "seed": self._seed,
                "num_predict": self._num_predict,
            },
        }

        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except (requests.RequestException, ConnectionError, OSError) as exc:
            logger.warning("Ollama call failed: %s", exc)
            return None

        try:
            body = resp.json()
        except (json.JSONDecodeError, ValueError):
            logger.warning("Ollama response not JSON")
            return None

        content = body.get("message", {}).get("content", "")
        if not content:
            logger.warning("Ollama response has no content")
            return None

        result = _extract_json(content)
        if result is None:
            logger.warning("Could not extract JSON from Ollama response: %s", content[:200])
        return result

    def _parse_response(self, raw: dict) -> tuple[Signal, float, float] | None:
        """Validate and parse structured model output.

        Returns (signal, confidence, prob_up) or None if invalid.
        """
        # Required: direction
        direction_str = raw.get("direction")
        if not isinstance(direction_str, str):
            logger.warning("Missing or non-string 'direction'")
            return None

        direction_map = {
            "long": Signal.LONG,
            "short": Signal.SHORT,
            "abstain": Signal.ABSTAIN,
        }
        signal = direction_map.get(direction_str.lower())
        if signal is None:
            logger.warning("Invalid direction: %r", direction_str)
            return None

        # Required: confidence
        try:
            confidence = float(raw["confidence"])
        except (KeyError, TypeError, ValueError):
            logger.warning("Missing or invalid 'confidence'")
            return None

        # Required: prob_up
        try:
            prob_up = float(raw["prob_up"])
        except (KeyError, TypeError, ValueError):
            logger.warning("Missing or invalid 'prob_up'")
            return None

        # Optional: horizon_hours — if present, must == 6
        horizon = raw.get("horizon_hours")
        if horizon is not None:
            try:
                if int(horizon) != 6:
                    logger.warning("Wrong horizon_hours: %s (expected 6)", horizon)
                    return None
            except (TypeError, ValueError):
                logger.warning("Invalid horizon_hours: %r", horizon)
                return None

        # Clamp to valid ranges
        confidence = max(0.0, min(1.0, confidence))
        prob_up = max(0.0, min(1.0, prob_up))

        return signal, round(confidence, 4), round(prob_up, 4)
