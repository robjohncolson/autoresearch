"""Run all baselines and print comparison table.

Usage:
    python -m trading_eval.baselines.run_baselines --data-dir <path>
"""

import argparse
import sys
from pathlib import Path

from trading_eval.baselines.sklearn_baseline import (
    GBTBaselineCandidate,
    LogisticBaselineCandidate,
)
from trading_eval.baselines.ta_ensemble import TAEnsembleCandidate
from trading_eval.config import EvalConfig
from trading_eval.runner import run_experiment
from trading_eval.storage import (
    compare_experiments,
    format_comparison,
    load_experiment,
    save_experiment,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run all baselines and compare results",
    )
    parser.add_argument("--data-dir", required=True, type=str)
    parser.add_argument("--train-days", type=int, default=90)
    parser.add_argument("--val-days", type=int, default=1)
    parser.add_argument("--step-days", type=int, default=1)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--horizon", default="6h")
    parser.add_argument("--output-dir", type=str, default="experiments")
    args = parser.parse_args(argv)

    config = EvalConfig(
        data_dir=Path(args.data_dir),
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        train_days=args.train_days,
        val_days=args.val_days,
        step_days=args.step_days,
        label_horizon=args.horizon,
    )
    output_dir = Path(args.output_dir)

    candidates = [
        TAEnsembleCandidate(),
        LogisticBaselineCandidate(horizon=args.horizon),
        GBTBaselineCandidate(horizon=args.horizon),
    ]

    paths: list[Path] = []
    records: list[dict] = []

    for candidate in candidates:
        print(f"\n{'='*60}")
        print(f"Running: {candidate.name} v{candidate.version}")
        print(f"{'='*60}")

        result = run_experiment(candidate, config)
        path = save_experiment(result, output_dir)
        record = load_experiment(path)

        paths.append(path)
        records.append(record)

        m = result.aggregate_metrics
        print(f"  Folds: {len(result.fold_results)}")
        print(f"  Trades: {m.trade_count}")
        print(f"  Direction accuracy: {m.direction_accuracy:.4f}")
        print(f"  Net P&L: {m.net_pnl_bps:+.2f} bps")
        print(f"  Max drawdown: {m.max_drawdown_bps:.2f} bps")
        print(f"  Sharpe: {m.sharpe_ratio:.4f}")
        print(f"  Hit rate: {m.hit_rate:.4f}")
        print(f"  Brier: {m.brier_score:.4f}")
        print(f"  Saved: {path}")

    # Print pairwise comparisons
    if len(records) >= 2:
        print(f"\n\n{'='*75}")
        print("BASELINE COMPARISON")
        print(f"{'='*75}")

        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                table = compare_experiments(records[i], records[j])
                print(f"\n--- {table.name_a} vs {table.name_b} ---\n")
                print(format_comparison(table))

    # Summary: best on each key metric
    if records:
        print(f"\n\n{'='*60}")
        print("SUMMARY: Best baseline per metric")
        print(f"{'='*60}")

        key_metrics = [
            ("direction_accuracy", True),   # higher is better
            ("net_pnl_bps", True),
            ("sharpe_ratio", True),
            ("hit_rate", True),
            ("brier_score", False),          # lower is better
            ("max_drawdown_bps", False),
        ]

        for metric, higher_is_better in key_metrics:
            values = [
                (r["candidate_name"], r["aggregate_metrics"].get(metric, 0))
                for r in records
            ]
            if higher_is_better:
                best = max(values, key=lambda x: x[1])
            else:
                best = min(values, key=lambda x: x[1])
            print(f"  {metric:<25} -> {best[0]} ({best[1]:.4f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
