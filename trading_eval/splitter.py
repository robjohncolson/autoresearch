"""Walk-forward time-series splitter."""

from dataclasses import dataclass

import numpy as np

SECONDS_PER_DAY = 86400


@dataclass(frozen=True)
class Fold:
    """One train/validate window in a walk-forward split."""

    fold_index: int
    train_start: int  # epoch seconds (inclusive)
    train_end: int    # epoch seconds (exclusive)
    val_start: int    # epoch seconds (inclusive)
    val_end: int      # epoch seconds (exclusive)
    train_idx: np.ndarray  # integer indices into the source array
    val_idx: np.ndarray


def walk_forward_splits(
    timestamps: np.ndarray,
    train_days: int,
    val_days: int,
    step_days: int,
) -> list[Fold]:
    """Generate walk-forward train/validate folds from a timestamp array.

    Args:
        timestamps: Monotonically increasing Unix epoch seconds.
        train_days: Length of training window in days.
        val_days: Length of validation window in days.
        step_days: How far to advance the window each fold.

    Returns:
        List of Fold objects, each with non-overlapping train/val windows.
        Empty list if the data is too short for even one fold.
    """
    if len(timestamps) == 0:
        return []
    if train_days <= 0 or val_days <= 0 or step_days <= 0:
        raise ValueError("train_days, val_days, and step_days must be positive")

    ts = np.asarray(timestamps)
    t_min = int(ts[0])
    t_max = int(ts[-1])

    train_seconds = train_days * SECONDS_PER_DAY
    val_seconds = val_days * SECONDS_PER_DAY
    step_seconds = step_days * SECONDS_PER_DAY

    folds: list[Fold] = []
    fold_index = 0
    cursor = t_min

    while True:
        train_start = cursor
        train_end = cursor + train_seconds
        val_start = train_end
        val_end = val_start + val_seconds

        # Stop if validation window extends beyond data
        if val_start > t_max:
            break

        train_mask = (ts >= train_start) & (ts < train_end)
        val_mask = (ts >= val_start) & (ts < val_end)

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        # Skip folds with empty train or val sets
        if len(train_idx) == 0 or len(val_idx) == 0:
            cursor += step_seconds
            continue

        folds.append(Fold(
            fold_index=fold_index,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            train_idx=train_idx,
            val_idx=val_idx,
        ))
        fold_index += 1
        cursor += step_seconds

    return folds
