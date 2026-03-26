"""Artifact schema and promotion workflow."""

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    artifact_id: str
    artifact_version: str
    model_family: str
    input_schema_version: str   # "market/v1"
    output_schema_version: str  # "prediction/v1"
    label_horizon: str
    calibration: dict
    evaluation_summary: dict
    source_commit: str
    experiment_id: str
    created_at: str

    def to_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "artifact_version": self.artifact_version,
            "model_family": self.model_family,
            "input_schema_version": self.input_schema_version,
            "output_schema_version": self.output_schema_version,
            "label_horizon": self.label_horizon,
            "calibration": self.calibration,
            "evaluation_summary": self.evaluation_summary,
            "source_commit": self.source_commit,
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
        }


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()[:12] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def promote_candidate(
    experiment_record: dict,
    artifact_version: str = "1.0",
    artifacts_dir: Path = Path("artifacts"),
) -> Path:
    """Create a versioned artifact directory from a completed experiment.

    Creates:
        artifacts/<artifact_id>/
            manifest.json       - artifact manifest with evaluation summary
            experiment.json     - full experiment record
            model/              - directory for model weights/config (empty for TA)
    """
    candidate_name = experiment_record["candidate_name"]
    experiment_id = experiment_record["experiment_id"]
    config = experiment_record["config"]
    label_horizon = config.get("label_horizon", "6h")

    now = datetime.now(timezone.utc)
    artifact_id = f"{candidate_name}_{now.strftime('%Y%m%d')}_{experiment_id}"

    artifact_dir = Path(artifacts_dir) / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_dir = artifact_dir / "model"
    model_dir.mkdir(exist_ok=True)

    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_version=artifact_version,
        model_family=candidate_name,
        input_schema_version="market/v1",
        output_schema_version="prediction/v1",
        label_horizon=label_horizon,
        calibration={"method": "none"},
        evaluation_summary=experiment_record["aggregate_metrics"],
        source_commit=_get_git_commit(),
        experiment_id=experiment_id,
        created_at=now.isoformat(),
    )

    with open(artifact_dir / "manifest.json", "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    with open(artifact_dir / "experiment.json", "w") as f:
        json.dump(experiment_record, f, indent=2)

    return artifact_dir


def load_artifact(artifact_dir: Path) -> ArtifactManifest:
    """Load artifact manifest from a directory."""
    manifest_path = Path(artifact_dir) / "manifest.json"
    with open(manifest_path) as f:
        data = json.load(f)

    return ArtifactManifest(
        artifact_id=data["artifact_id"],
        artifact_version=data["artifact_version"],
        model_family=data["model_family"],
        input_schema_version=data["input_schema_version"],
        output_schema_version=data["output_schema_version"],
        label_horizon=data["label_horizon"],
        calibration=data["calibration"],
        evaluation_summary=data["evaluation_summary"],
        source_commit=data["source_commit"],
        experiment_id=data["experiment_id"],
        created_at=data["created_at"],
    )


def list_artifacts(artifacts_dir: Path = Path("artifacts")) -> list[ArtifactManifest]:
    """List all artifacts in the directory."""
    artifacts_dir = Path(artifacts_dir)
    if not artifacts_dir.exists():
        return []

    results = []
    for d in sorted(artifacts_dir.iterdir()):
        if d.is_dir() and (d / "manifest.json").exists():
            try:
                results.append(load_artifact(d))
            except (json.JSONDecodeError, KeyError):
                continue
    return results
