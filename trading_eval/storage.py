"""Experiment storage with structured reproducibility metadata."""

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from trading_eval.backtest import TradeResult
from trading_eval.candidate import Signal
from trading_eval.metrics import Metrics
from trading_eval.runner import ExperimentResult, FoldResult

RECORD_VERSION = "experiment/v1"


@dataclass(frozen=True)
class ComparisonRow:
    metric: str
    value_a: float
    value_b: float
    delta: float
    winner: str  # "A", "B", or "tie"


@dataclass(frozen=True)
class ComparisonTable:
    name_a: str
    name_b: str
    rows: list[ComparisonRow]


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()[:12] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _manifest_hash(manifest: dict) -> str:
    raw = json.dumps(manifest, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _metrics_to_dict(m: Metrics) -> dict:
    return m.to_dict()


def _fold_summary(fr: FoldResult) -> dict:
    return {
        "fold_index": fr.fold_index,
        "train_start": fr.train_start,
        "train_end": fr.train_end,
        "val_start": fr.val_start,
        "val_end": fr.val_end,
        "train_rows": fr.train_rows,
        "val_rows": fr.val_rows,
        "trade_count": fr.metrics.trade_count,
        "metrics": _metrics_to_dict(fr.metrics),
    }


def _build_record(result: ExperimentResult) -> dict:
    return {
        "record_version": RECORD_VERSION,
        "experiment_id": result.experiment_id,
        "candidate_name": result.candidate_name,
        "candidate_version": result.candidate_version,
        "config": result.config.to_dict(),
        "dataset_manifest_hash": _manifest_hash(result.dataset.manifest),
        "dataset_manifest": result.dataset.manifest,
        "source_commit": _get_git_commit(),
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "fold_count": len(result.fold_results),
        "fold_summaries": [_fold_summary(fr) for fr in result.fold_results],
        "aggregate_metrics": _metrics_to_dict(result.aggregate_metrics),
    }


def save_experiment(result: ExperimentResult, output_dir: Path) -> Path:
    """Save experiment result as a structured JSON record."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    record = _build_record(result)
    filename = f"{result.candidate_name}_{result.experiment_id}.json"
    path = output_dir / filename

    with open(path, "w") as f:
        json.dump(record, f, indent=2)

    return path


def load_experiment(path: Path) -> dict:
    """Load an experiment record from JSON."""
    with open(path) as f:
        record = json.load(f)

    if record.get("record_version") != RECORD_VERSION:
        raise ValueError(
            f"Unsupported record version: {record.get('record_version')}"
        )

    return record


def list_experiments(output_dir: Path) -> list[dict]:
    """List all experiments in directory with summary info."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []

    summaries = []
    for path in sorted(output_dir.glob("*.json")):
        try:
            record = load_experiment(path)
            summaries.append({
                "file": path.name,
                "experiment_id": record["experiment_id"],
                "candidate_name": record["candidate_name"],
                "candidate_version": record.get("candidate_version", "?"),
                "fold_count": record["fold_count"],
                "started_at": record["started_at"],
                "net_pnl_bps": record["aggregate_metrics"]["net_pnl_bps"],
                "direction_accuracy": record["aggregate_metrics"]["direction_accuracy"],
                "sharpe_ratio": record["aggregate_metrics"]["sharpe_ratio"],
                "trade_count": record["aggregate_metrics"]["trade_count"],
            })
        except (ValueError, KeyError, json.JSONDecodeError):
            continue

    return summaries


# Metrics where higher is better
_HIGHER_IS_BETTER = {
    "direction_accuracy", "net_pnl_bps", "sharpe_ratio",
    "sortino_ratio", "hit_rate",
}
# Metrics where lower is better
_LOWER_IS_BETTER = {
    "brier_score", "mae_bps", "max_drawdown_bps",
}


def compare_experiments(record_a: dict, record_b: dict) -> ComparisonTable:
    """Side-by-side comparison of two experiment records."""
    metrics_a = record_a["aggregate_metrics"]
    metrics_b = record_b["aggregate_metrics"]
    name_a = record_a["candidate_name"]
    name_b = record_b["candidate_name"]

    rows = []
    for key in metrics_a:
        va = metrics_a[key]
        vb = metrics_b.get(key, 0)

        if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
            continue

        delta = vb - va
        if key in _HIGHER_IS_BETTER:
            winner = "B" if delta > 0 else ("A" if delta < 0 else "tie")
        elif key in _LOWER_IS_BETTER:
            winner = "B" if delta < 0 else ("A" if delta > 0 else "tie")
        else:
            winner = "tie"

        rows.append(ComparisonRow(
            metric=key,
            value_a=float(va),
            value_b=float(vb),
            delta=float(delta),
            winner=winner,
        ))

    return ComparisonTable(name_a=name_a, name_b=name_b, rows=rows)


def format_comparison(table: ComparisonTable) -> str:
    """Format a comparison table as a human-readable string."""
    lines = [
        f"{'Metric':<25} {'A (' + table.name_a + ')':>15} {'B (' + table.name_b + ')':>15} {'Delta':>10} {'Winner':>8}",
        "-" * 75,
    ]
    for row in table.rows:
        lines.append(
            f"{row.metric:<25} {row.value_a:>15.4f} {row.value_b:>15.4f} "
            f"{row.delta:>+10.4f} {row.winner:>8}"
        )
    return "\n".join(lines)
