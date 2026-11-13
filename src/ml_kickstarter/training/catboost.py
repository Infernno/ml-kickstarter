import logging
from pathlib import Path

import catboost
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from ml_kickstarter.data.catboost import prepare_pools
from ml_kickstarter.training.artifacts import git_metadata, save_training_report
from ml_kickstarter.training.plots import save_training_plots
from ml_kickstarter.training.tracking import log_results, track_training

METRIC_NAMES = {
    "Logloss": "log_loss",
    "AUC": "roc_auc",
    "F1": "f1",
    "Accuracy": "accuracy",
}
EVALUATION_SETS = {"train": "validation_0", "validation": "validation_1"}


def iteration_metrics(iteration: int, scores: dict) -> dict:
    """Read CatBoost's native metrics using the names shared with the baseline."""
    return {
        "iteration": iteration,
        **{
            split: {
                name: scores[dataset][native_name][iteration - 1]
                for native_name, name in METRIC_NAMES.items()
            }
            for split, dataset in EVALUATION_SETS.items()
        },
    }


class TrainingLog:
    """Send native iteration metrics to MLflow while CatBoost is training."""

    def __init__(self, logger: MLFlowLogger, interval: int) -> None:
        self.logger = logger
        self.interval = interval

    def after_iteration(self, info) -> bool:
        if info.iteration == 1 or info.iteration % self.interval == 0:
            record = iteration_metrics(info.iteration, info.metrics)
            self.logger.log_metrics(
                {
                    f"{split}/{name}": value
                    for split in EVALUATION_SETS
                    for name, value in record[split].items()
                },
                step=info.iteration,
            )
            logging.getLogger(__name__).info(
                "Iteration %d: validation Logloss=%.4f, ROC-AUC=%.4f",
                info.iteration,
                record["validation"]["log_loss"],
                record["validation"]["roc_auc"],
            )
        return True


def save_artifacts(model, history: list[dict], config: DictConfig, run_id: str) -> dict:
    """Keep the best model and its native categorical-feature processing together."""
    directory = Path(config.train.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_model(directory / "model.cbm")
    metrics = history[model.get_best_iteration()]
    metadata = {
        **git_metadata(),
        "mlflow_run_id": run_id,
        "feature_names": model.feature_names_,
        "catboost_version": catboost.__version__,
        "tree_count": model.tree_count_,
    }
    save_training_report(directory, config, history, metrics, metadata)
    return metrics


def train_catboost(
    train_path: str | Path, validation_path: str | Path, config: DictConfig
) -> dict:
    """Train on the existing split and select the model by validation Logloss."""
    if config.logging.log_every_n_steps < 1:
        raise ValueError("log_every_n_steps must be positive")
    train, validation = prepare_pools(train_path, validation_path)
    parameters = OmegaConf.to_container(config.train.params, resolve=True)
    model = catboost.CatBoostClassifier(**parameters)
    with track_training(config) as logger:
        # Evaluating train as a separate pool makes its curves match saved-model
        # predictions. The last evaluation pool controls best-model selection.
        model.fit(
            train,
            eval_set=[train, validation],
            callbacks=[TrainingLog(logger, config.logging.log_every_n_steps)],
        )
        scores = model.get_evals_result()
        history = [
            iteration_metrics(iteration, scores)
            for iteration in range(1, len(scores["validation_1"]["Logloss"]) + 1)
        ]
        metrics = save_artifacts(model, history, config, logger.run_id)
        plots = save_training_plots(
            history, config.logging.plots_dir, step_name="iteration"
        )
        log_results(
            logger, config.train.output_dir, plots, metrics, step_name="iteration"
        )
    return metrics
