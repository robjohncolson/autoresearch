"""Dataset loader with manifest-based contract validation."""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trading_eval.config import EvalConfig

EXPECTED_SCHEMA_VERSION = "research-dataset/v1"

REQUIRED_MARKET_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}

REQUIRED_LABEL_PREFIXES = {"return_bps_", "return_sign_"}


class DatasetError(Exception):
    pass


@dataclass(frozen=True)
class Dataset:
    """Loaded and validated research dataset."""

    market: pd.DataFrame
    labels: pd.DataFrame
    manifest: dict


def load_dataset(config: EvalConfig) -> Dataset:
    """Load market + labels Parquet files, validated against manifest_v1.json."""
    data_dir = Path(config.data_dir)

    # --- Load and validate manifest ---
    manifest_path = data_dir / "manifest_v1.json"
    if not manifest_path.exists():
        raise DatasetError(f"Missing manifest: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    schema_version = manifest.get("schema_version")
    if schema_version != EXPECTED_SCHEMA_VERSION:
        raise DatasetError(
            f"Schema version mismatch: expected {EXPECTED_SCHEMA_VERSION!r}, "
            f"got {schema_version!r}"
        )

    # --- Load Parquet files ---
    market_path = data_dir / "market_v1.parquet"
    labels_path = data_dir / "labels_v1.parquet"

    if not market_path.exists():
        raise DatasetError(f"Missing market file: {market_path}")
    if not labels_path.exists():
        raise DatasetError(f"Missing labels file: {labels_path}")

    market = pd.read_parquet(market_path)
    labels = pd.read_parquet(labels_path)

    # --- Validate row count alignment ---
    if len(market) != len(labels):
        raise DatasetError(
            f"Row count mismatch: market has {len(market)} rows, "
            f"labels has {len(labels)} rows"
        )

    manifest_row_count = manifest.get("row_count")
    if manifest_row_count is not None and len(market) != manifest_row_count:
        raise DatasetError(
            f"Row count mismatch with manifest: expected {manifest_row_count}, "
            f"got {len(market)}"
        )

    # --- Validate market columns ---
    missing_market = REQUIRED_MARKET_COLUMNS - set(market.columns)
    if missing_market:
        raise DatasetError(f"Missing market columns: {sorted(missing_market)}")

    # --- Validate label columns from manifest ---
    manifest_label_cols = manifest.get("label_columns", [])
    if not manifest_label_cols:
        raise DatasetError("Manifest has no label_columns defined")

    missing_labels = set(manifest_label_cols) - set(labels.columns)
    if missing_labels:
        raise DatasetError(f"Missing label columns: {sorted(missing_labels)}")

    # --- Validate required label prefixes exist ---
    label_cols = set(labels.columns)
    for prefix in REQUIRED_LABEL_PREFIXES:
        if not any(col.startswith(prefix) for col in label_cols):
            raise DatasetError(
                f"No label column with prefix {prefix!r} found in labels"
            )

    # --- Validate timestamps are monotonically increasing ---
    if "timestamp" in market.columns:
        ts = market["timestamp"]
        if not ts.is_monotonic_increasing:
            raise DatasetError("Market timestamps are not monotonically increasing")

    return Dataset(market=market, labels=labels, manifest=manifest)
