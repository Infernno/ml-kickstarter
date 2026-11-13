from pathlib import Path
from tempfile import TemporaryDirectory

import lightning
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from ml_kickstarter.data.encoding import prepare_datasets
from ml_kickstarter.models.logistic_regression import LogisticRegression
from ml_kickstarter.training.artifacts import save_artifacts
from ml_kickstarter.training.metrics import TrainingHistory
from ml_kickstarter.training.plots import save_training_plots
from ml_kickstarter.training.tracking import log_results, track_training


def create_loader(
    dataset: TensorDataset, config: DictConfig, *, shuffle: bool
) -> DataLoader:
    """Keep shuffling reproducible and load in the main process on Windows."""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(config.seed),
    )


def fit_model(
    train: TensorDataset,
    validation: TensorDataset,
    config: DictConfig,
    checkpoint_dir: str,
    logger: MLFlowLogger,
    log_every_n_steps: int,
) -> tuple[Path, list[dict]]:
    """Run Lightning and return the best checkpoint and epoch metrics."""
    input_dim = train.tensors[0].shape[1]
    model = LogisticRegression(input_dim, config.learning_rate, config.weight_decay)
    history = TrainingHistory(train, validation, config.threshold)
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="model",
        monitor="validation_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
    )
    trainer = lightning.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=config.epochs,
        deterministic=True,
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[history, checkpoint],
    )
    trainer.fit(
        model,
        create_loader(train, config, shuffle=True),
        create_loader(validation, config, shuffle=False),
    )
    return Path(checkpoint.best_model_path), history.history


def train_baseline(
    train_path: str | Path, validation_path: str | Path, config: DictConfig
) -> dict:
    """Prepare data, train the model and save the selected checkpoint."""
    if config.train.epochs < 1 or not 0 < config.train.threshold < 1:
        raise ValueError(
            "epochs must be positive and threshold must be between 0 and 1"
        )
    lightning.seed_everything(config.train.seed, workers=True)
    torch.set_num_threads(config.train.num_threads)
    train, validation, preprocessor = prepare_datasets(
        train_path,
        validation_path,
        handle_unknown=config.encoding.handle_unknown,
    )
    with track_training(config) as logger:
        with TemporaryDirectory(prefix="kickstarter-checkpoints-") as checkpoint_dir:
            checkpoint_path, history = fit_model(
                train,
                validation,
                config.train,
                checkpoint_dir,
                logger,
                config.logging.log_every_n_steps,
            )
            metrics = save_artifacts(
                checkpoint_path, preprocessor, history, config, logger.run_id
            )
        plots = save_training_plots(history, config.logging.plots_dir)
        log_results(logger, config.train.output_dir, plots, metrics)
    return metrics
