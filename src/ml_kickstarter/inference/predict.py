from pathlib import Path

import numpy as np
import polars as pl
import torch
from omegaconf import DictConfig

from ml_kickstarter.data.download import ensure_data, ensure_model
from ml_kickstarter.data.encoding import transform_features
from ml_kickstarter.data.features import select_model_features
from ml_kickstarter.data.inference import read_inference_data
from ml_kickstarter.models.catboost import load_catboost
from ml_kickstarter.models.logistic_regression import load_baseline

MODEL_FILES = {
    "baseline": ("model.ckpt", "preprocessor.joblib"),
    "catboost": ("model.cbm",),
}


@torch.inference_mode()
def baseline_probabilities(data: pl.DataFrame, config: DictConfig) -> np.ndarray:
    """Apply the saved encoder and linear model in batches, without fitting."""
    torch.set_num_threads(config.num_threads)
    model, preprocessor = load_baseline(config.model_dir)
    probabilities = []
    for batch in data.iter_slices(config.batch_size):
        features = torch.from_numpy(transform_features(preprocessor, batch))
        probabilities.append(torch.sigmoid(model(features).double()).numpy())
    return np.concatenate(probabilities)


def catboost_probabilities(data: pl.DataFrame, config: DictConfig) -> np.ndarray:
    """Apply the saved model with its native categorical-feature processing."""
    model = load_catboost(config.model_dir)
    probabilities = [
        model.predict_proba(
            select_model_features(batch), thread_count=config.num_threads
        )[:, 1]
        for batch in data.iter_slices(config.batch_size)
    ]
    return np.concatenate(probabilities)


def predict_csv(config: DictConfig) -> Path:
    """Restore missing inputs/model artifacts and write one prediction per row."""
    if config.model not in MODEL_FILES:
        raise ValueError(f"Unknown model: {config.model}; choose baseline or catboost")
    if not 0 < config.threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    if config.batch_size < 1 or config.num_threads < 1:
        raise ValueError("batch_size and num_threads must be positive")
    source = Path(config.input_path)
    destination = Path(config.output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output CSV paths must be different")

    ensure_data(source, config.data_remote)
    data = read_inference_data(source)
    ensure_model(config.model_dir, config.model_remote, MODEL_FILES[config.model])
    if config.model == "baseline":
        probabilities = baseline_probabilities(data, config)
    else:
        probabilities = catboost_probabilities(data, config)
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("Model produced invalid probabilities")
    result = data.select("id").with_columns(
        pl.Series("success_probability", probabilities),
        pl.Series("prediction", (probabilities >= config.threshold).astype(np.int8)),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(destination)
    return destination
