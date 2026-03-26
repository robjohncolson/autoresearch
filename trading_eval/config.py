"""Evaluation configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalConfig:
    """All configurable parameters for a walk-forward evaluation run."""

    data_dir: Path
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    train_days: int = 90
    val_days: int = 1
    step_days: int = 1
    label_horizon: str = "6h"
    seed: int = 42

    @property
    def cost_bps(self) -> float:
        return self.fee_bps + self.slippage_bps

    @property
    def return_bps_col(self) -> str:
        return f"return_bps_{self.label_horizon}"

    @property
    def return_sign_col(self) -> str:
        return f"return_sign_{self.label_horizon}"

    def to_dict(self) -> dict:
        return {
            "data_dir": str(self.data_dir),
            "fee_bps": self.fee_bps,
            "slippage_bps": self.slippage_bps,
            "train_days": self.train_days,
            "val_days": self.val_days,
            "step_days": self.step_days,
            "label_horizon": self.label_horizon,
            "seed": self.seed,
        }
