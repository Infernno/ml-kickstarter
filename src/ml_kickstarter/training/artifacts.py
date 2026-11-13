import json
import shutil
import subprocess
from pathlib import Path

import joblib
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.compose import ColumnTransformer


def git_metadata() -> dict:
    """Identify the source revision and whether the working tree has changes."""
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
    }


def save_artifacts(
    checkpoint_path: Path,
    preprocessor: ColumnTransformer,
    history: list[dict],
    config: DictConfig,
    run_id: str,
) -> dict:
    """Save the selected checkpoint and everything needed to reproduce predictions."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    best_epoch = checkpoint["epoch"] + 1
    metrics = next(record for record in history if record["epoch"] == best_epoch)
    metadata = {
        **git_metadata(),
        "mlflow_run_id": run_id,
        "feature_names": preprocessor.get_feature_names_out().tolist(),
        "torch_version": str(torch.__version__),
        "lightning_version": L.__version__,
    }
    directory = Path(config.train.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint_path, directory / "model.ckpt")
    joblib.dump(preprocessor, directory / "preprocessor.joblib")
    save_training_report(directory, config, history, metrics, metadata)
    return metrics


def save_training_report(
    directory: Path,
    config: DictConfig,
    history: list[dict],
    metrics: dict,
    metadata: dict,
) -> None:
    """Save the configuration, learning curves and selected model's metrics."""
    OmegaConf.save(config=config, f=directory / "config.yaml", resolve=True)
    for name, content in {
        "history": history,
        "metrics": metrics,
        "metadata": metadata,
    }.items():
        (directory / f"{name}.json").write_text(
            json.dumps(content, indent=2) + "\n", encoding="utf-8"
        )
