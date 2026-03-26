"""Experiment runner: orchestrates walk-forward evaluation."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from trading_eval.backtest import TradeResult, run_backtest
from trading_eval.candidate import Candidate
from trading_eval.config import EvalConfig
from trading_eval.data import Dataset, load_dataset
from trading_eval.metrics import Metrics, compute_metrics
from trading_eval.splitter import Fold, walk_forward_splits


@dataclass(frozen=True)
class FoldResult:
    fold_index: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    train_rows: int
    val_rows: int
    trades: list[TradeResult]
    metrics: Metrics


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    candidate_name: str
    candidate_version: str
    config: EvalConfig
    dataset: Dataset
    fold_results: list[FoldResult]
    aggregate_metrics: Metrics
    started_at: str
    finished_at: str


def run_experiment(
    candidate: Candidate,
    config: EvalConfig,
) -> ExperimentResult:
    """Run a complete walk-forward evaluation.

    1. Load dataset (with manifest validation)
    2. Generate walk-forward folds
    3. Per fold: reset -> fit -> predict -> backtest -> metrics
    4. Aggregate metrics across all folds
    """
    started_at = datetime.now(timezone.utc).isoformat()

    dataset = load_dataset(config)
    market = dataset.market
    labels = dataset.labels
    timestamps = market["timestamp"].values

    folds = walk_forward_splits(
        timestamps,
        train_days=config.train_days,
        val_days=config.val_days,
        step_days=config.step_days,
    )

    all_trades: list[TradeResult] = []
    all_predictions: list[pd.Series] = []
    fold_results: list[FoldResult] = []

    for fold in folds:
        candidate.reset()

        train_market = market.iloc[fold.train_idx].reset_index(drop=True)
        train_labels = labels.iloc[fold.train_idx].reset_index(drop=True)
        val_market = market.iloc[fold.val_idx].reset_index(drop=True)
        val_labels = labels.iloc[fold.val_idx].reset_index(drop=True)

        # Add timestamp column to val_labels if not present (for backtest join)
        if "timestamp" not in val_labels.columns:
            val_labels = val_labels.copy()
            val_labels["timestamp"] = val_market["timestamp"].values

        candidate.fit(train_market, train_labels)
        predictions = candidate.predict(val_market)

        trades = run_backtest(predictions, val_labels, config)
        fold_metrics = compute_metrics(trades, predictions)

        fold_results.append(FoldResult(
            fold_index=fold.fold_index,
            train_start=fold.train_start,
            train_end=fold.train_end,
            val_start=fold.val_start,
            val_end=fold.val_end,
            train_rows=len(fold.train_idx),
            val_rows=len(fold.val_idx),
            trades=trades,
            metrics=fold_metrics,
        ))

        all_trades.extend(trades)
        all_predictions.append(predictions)

    # Aggregate predictions for metrics
    combined_preds = pd.concat(all_predictions) if all_predictions else pd.Series(dtype=object)
    aggregate_metrics = compute_metrics(all_trades, combined_preds)

    finished_at = datetime.now(timezone.utc).isoformat()

    return ExperimentResult(
        experiment_id=str(uuid.uuid4())[:8],
        candidate_name=candidate.name,
        candidate_version=candidate.version,
        config=config,
        dataset=dataset,
        fold_results=fold_results,
        aggregate_metrics=aggregate_metrics,
        started_at=started_at,
        finished_at=finished_at,
    )
