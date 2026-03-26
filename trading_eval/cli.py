"""CLI for walk-forward trading evaluation."""

import argparse
import sys
from pathlib import Path

from trading_eval.config import EvalConfig
from trading_eval.storage import (
    compare_experiments,
    format_comparison,
    list_experiments,
    load_experiment,
    save_experiment,
)


def _get_candidate(name: str):
    """Lazy-import candidate by name."""
    if name == "ta_ensemble":
        from trading_eval.baselines.ta_ensemble import TAEnsembleCandidate
        return TAEnsembleCandidate()
    elif name == "logistic_regression":
        from trading_eval.baselines.sklearn_baseline import LogisticBaselineCandidate
        return LogisticBaselineCandidate()
    elif name == "gradient_boosted_tree":
        from trading_eval.baselines.sklearn_baseline import GBTBaselineCandidate
        return GBTBaselineCandidate()
    else:
        raise ValueError(f"Unknown candidate: {name!r}")


CANDIDATE_NAMES = ["ta_ensemble", "logistic_regression", "gradient_boosted_tree"]


def cmd_run(args) -> int:
    from trading_eval.runner import run_experiment

    config = EvalConfig(
        data_dir=Path(args.data_dir),
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        train_days=args.train_days,
        val_days=args.val_days,
        step_days=args.step_days,
        label_horizon=args.horizon,
    )

    candidate = _get_candidate(args.candidate)
    print(f"Running {candidate.name} v{candidate.version} ...")
    print(f"  data_dir: {config.data_dir}")
    print(f"  train={config.train_days}d val={config.val_days}d step={config.step_days}d")
    print(f"  fees={config.fee_bps} bps  slippage={config.slippage_bps} bps")
    print()

    result = run_experiment(candidate, config)

    output_dir = Path(args.output_dir)
    path = save_experiment(result, output_dir)

    m = result.aggregate_metrics
    print(f"Experiment {result.experiment_id} complete")
    print(f"  Folds: {len(result.fold_results)}")
    print(f"  Trades: {m.trade_count}")
    print(f"  Direction accuracy: {m.direction_accuracy:.4f}")
    print(f"  Net P&L: {m.net_pnl_bps:+.2f} bps")
    print(f"  Max drawdown: {m.max_drawdown_bps:.2f} bps")
    print(f"  Sharpe: {m.sharpe_ratio:.4f}")
    print(f"  Hit rate: {m.hit_rate:.4f}")
    print(f"  Brier: {m.brier_score:.4f}")
    print(f"  Turnover: {m.turnover:.4f}")
    print(f"  Saved: {path}")
    return 0


def cmd_list(args) -> int:
    summaries = list_experiments(Path(args.output_dir))
    if not summaries:
        print("No experiments found.")
        return 0

    print(f"{'ID':<10} {'Candidate':<25} {'Trades':>7} {'P&L(bps)':>10} {'Accuracy':>10} {'Sharpe':>8}")
    print("-" * 72)
    for s in summaries:
        print(
            f"{s['experiment_id']:<10} {s['candidate_name']:<25} "
            f"{s['trade_count']:>7} {s['net_pnl_bps']:>+10.2f} "
            f"{s['direction_accuracy']:>10.4f} {s['sharpe_ratio']:>8.4f}"
        )
    return 0


def cmd_compare(args) -> int:
    record_a = load_experiment(Path(args.exp_a))
    record_b = load_experiment(Path(args.exp_b))
    table = compare_experiments(record_a, record_b)
    print(format_comparison(table))
    return 0


def cmd_promote(args) -> int:
    from trading_eval.artifact import load_artifact, promote_candidate

    record = load_experiment(Path(args.experiment))
    path = promote_candidate(
        record,
        artifact_version=args.artifact_version,
        artifacts_dir=Path(args.artifacts_dir),
    )
    manifest = load_artifact(path)
    print(f"Promoted: {manifest.artifact_id}")
    print(f"  Model family: {manifest.model_family}")
    print(f"  Artifact version: {manifest.artifact_version}")
    print(f"  Label horizon: {manifest.label_horizon}")
    print(f"  Input schema: {manifest.input_schema_version}")
    print(f"  Output schema: {manifest.output_schema_version}")
    print(f"  Experiment: {manifest.experiment_id}")
    print(f"  Directory: {path}")
    return 0


def cmd_artifacts(args) -> int:
    from trading_eval.artifact import list_artifacts

    artifacts = list_artifacts(Path(args.artifacts_dir))
    if not artifacts:
        print("No artifacts found.")
        return 0

    print(f"{'ID':<45} {'Family':<25} {'Horizon':>8} {'Version':>8}")
    print("-" * 88)
    for a in artifacts:
        print(f"{a.artifact_id:<45} {a.model_family:<25} {a.label_horizon:>8} {a.artifact_version:>8}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trading_eval",
        description="Walk-forward trading evaluation harness",
    )
    subparsers = parser.add_subparsers(dest="command")

    # run
    run_p = subparsers.add_parser("run", help="Run a walk-forward experiment")
    run_p.add_argument("--candidate", required=True, choices=CANDIDATE_NAMES)
    run_p.add_argument("--data-dir", required=True, type=str)
    run_p.add_argument("--train-days", type=int, default=90)
    run_p.add_argument("--val-days", type=int, default=1)
    run_p.add_argument("--step-days", type=int, default=1)
    run_p.add_argument("--fee-bps", type=float, default=10.0)
    run_p.add_argument("--slippage-bps", type=float, default=5.0)
    run_p.add_argument("--horizon", default="6h")
    run_p.add_argument("--output-dir", type=str, default="experiments")

    # list
    list_p = subparsers.add_parser("list", help="List saved experiments")
    list_p.add_argument("--output-dir", type=str, default="experiments")

    # compare
    cmp_p = subparsers.add_parser("compare", help="Compare two experiments")
    cmp_p.add_argument("exp_a", type=str, help="Path to first experiment JSON")
    cmp_p.add_argument("exp_b", type=str, help="Path to second experiment JSON")

    # promote
    promote_p = subparsers.add_parser("promote", help="Promote experiment to artifact")
    promote_p.add_argument("experiment", type=str, help="Path to experiment JSON")
    promote_p.add_argument("--artifact-version", default="1.0")
    promote_p.add_argument("--artifacts-dir", type=str, default="artifacts")

    # artifacts
    art_p = subparsers.add_parser("artifacts", help="List all artifacts")
    art_p.add_argument("--artifacts-dir", type=str, default="artifacts")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    handlers = {
        "run": cmd_run, "list": cmd_list, "compare": cmd_compare,
        "promote": cmd_promote, "artifacts": cmd_artifacts,
    }
    return handlers[args.command](args)
