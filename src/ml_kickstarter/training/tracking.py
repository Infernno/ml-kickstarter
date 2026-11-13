from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from lightning.pytorch.loggers import MLFlowLogger
from mlflow.environment_variables import MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT
from omegaconf import DictConfig, OmegaConf

from ml_kickstarter.training.artifacts import git_metadata


@contextmanager
def track_training(config: DictConfig) -> Iterator[MLFlowLogger]:
    """Track one training run, including failures while saving its results."""
    # MLflow's URL messages contain emoji unsupported by Windows legacy consoles.
    MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT.set(True)
    metadata = git_metadata()
    logger = MLFlowLogger(
        tracking_uri=config.logging.web_url,
        experiment_name=config.logging.experiment_name,
        run_name=config.logging.run_name,
        tags={
            "mlflow.source.git.commit": metadata["git_commit"],
            "git_dirty": str(metadata["git_dirty"]).lower(),
        },
        synchronous=True,
    )
    try:
        logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))
        yield logger
    except BaseException:
        logger.finalize("failed")
        raise
    else:
        logger.finalize("success")


def log_results(
    logger: MLFlowLogger,
    model_dir: str | Path,
    plots: list[Path],
    metrics: dict,
) -> None:
    """Log the best epoch and small artifacts; model files are managed by DVC."""
    best_metrics = {
        f"best/{split}/{name}": value
        for split in ("train", "validation")
        for name, value in metrics[split].items()
    }
    logger.log_metrics(
        {"best_epoch": metrics["epoch"], **best_metrics}, step=metrics["epoch"]
    )
    directory = Path(model_dir)
    for name in ("config.yaml", "history.json", "metrics.json", "metadata.json"):
        logger.experiment.log_artifact(logger.run_id, str(directory / name))
    for path in plots:
        logger.experiment.log_artifact(logger.run_id, str(path), artifact_path="plots")
