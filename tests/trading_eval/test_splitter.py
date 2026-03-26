"""Tests for walk-forward splitter."""

import numpy as np
import pytest

from trading_eval.splitter import SECONDS_PER_DAY, Fold, walk_forward_splits


def _make_hourly_timestamps(days: int, start: int = 1700000000) -> np.ndarray:
    """Create hourly timestamps spanning `days` days."""
    n = days * 24
    return np.arange(start, start + n * 3600, 3600)


class TestWalkForwardSplits:
    def test_basic_split(self):
        ts = _make_hourly_timestamps(days=10)
        folds = walk_forward_splits(ts, train_days=5, val_days=1, step_days=1)
        assert len(folds) > 0
        assert all(isinstance(f, Fold) for f in folds)

    def test_fold_count(self):
        ts = _make_hourly_timestamps(days=10)
        folds = walk_forward_splits(ts, train_days=5, val_days=1, step_days=1)
        # 10 days total, 5 train + 1 val = 6 days minimum
        # Step 1 day: folds at day 0,1,2,3,4 (val ends at 6,7,8,9,10)
        assert len(folds) == 5

    def test_no_train_val_overlap(self):
        ts = _make_hourly_timestamps(days=30)
        folds = walk_forward_splits(ts, train_days=10, val_days=2, step_days=3)
        for fold in folds:
            train_set = set(fold.train_idx.tolist())
            val_set = set(fold.val_idx.tolist())
            assert train_set.isdisjoint(val_set), f"Fold {fold.fold_index} has overlap"

    def test_no_future_leakage(self):
        ts = _make_hourly_timestamps(days=30)
        folds = walk_forward_splits(ts, train_days=10, val_days=2, step_days=3)
        for fold in folds:
            max_train_ts = ts[fold.train_idx[-1]]
            min_val_ts = ts[fold.val_idx[0]]
            assert max_train_ts < min_val_ts, (
                f"Fold {fold.fold_index}: train max {max_train_ts} >= val min {min_val_ts}"
            )

    def test_train_end_equals_val_start(self):
        ts = _make_hourly_timestamps(days=20)
        folds = walk_forward_splits(ts, train_days=10, val_days=2, step_days=2)
        for fold in folds:
            assert fold.train_end == fold.val_start

    def test_consecutive_fold_indices(self):
        ts = _make_hourly_timestamps(days=20)
        folds = walk_forward_splits(ts, train_days=5, val_days=1, step_days=1)
        for i, fold in enumerate(folds):
            assert fold.fold_index == i

    def test_empty_timestamps(self):
        folds = walk_forward_splits(np.array([]), train_days=5, val_days=1, step_days=1)
        assert folds == []

    def test_data_too_short(self):
        ts = _make_hourly_timestamps(days=3)
        folds = walk_forward_splits(ts, train_days=5, val_days=1, step_days=1)
        assert folds == []

    def test_invalid_params(self):
        ts = _make_hourly_timestamps(days=10)
        with pytest.raises(ValueError):
            walk_forward_splits(ts, train_days=0, val_days=1, step_days=1)
        with pytest.raises(ValueError):
            walk_forward_splits(ts, train_days=5, val_days=-1, step_days=1)

    def test_large_step(self):
        ts = _make_hourly_timestamps(days=100)
        folds = walk_forward_splits(ts, train_days=30, val_days=5, step_days=30)
        # Each fold uses 35 days; stepping 30 days at a time
        assert len(folds) >= 2
        # Folds should not share validation windows
        for i in range(1, len(folds)):
            assert folds[i].val_start >= folds[i - 1].val_end

    def test_window_boundaries_in_seconds(self):
        ts = _make_hourly_timestamps(days=10)
        folds = walk_forward_splits(ts, train_days=5, val_days=1, step_days=1)
        f0 = folds[0]
        assert f0.train_end - f0.train_start == 5 * SECONDS_PER_DAY
        assert f0.val_end - f0.val_start == 1 * SECONDS_PER_DAY
