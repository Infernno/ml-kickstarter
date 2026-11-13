from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from torch.utils.data import TensorDataset

from ml_kickstarter.constants import (
    CATEGORY_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    TARGET_BY_STATE,
)
from ml_kickstarter.data.features import select_model_features


def read_labeled_data(path: str | Path) -> pl.DataFrame:
    """Read only the active model inputs, labels and IDs used for split checks."""
    data = pl.read_csv(path, columns=["id", *MODEL_FEATURE_COLUMNS, "target"])
    if data["id"].null_count() or data["id"].n_unique() != data.height:
        raise ValueError("Training and validation data must have unique, non-null IDs")
    if set(data["target"].unique()) != set(TARGET_BY_STATE.values()):
        raise ValueError(
            "Each dataset must contain both binary classes and no missing targets"
        )
    return data


def create_preprocessor(handle_unknown: str) -> ColumnTransformer:
    """Encode categories, log-transform goal and scale duration; fit on train only."""
    return ColumnTransformer(
        [
            (
                "categories",
                OneHotEncoder(
                    handle_unknown=handle_unknown, sparse_output=False, dtype=np.float32
                ),
                CATEGORY_COLUMNS,
            ),
            (
                "goal",
                FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                ["goal"],
            ),
            ("duration", StandardScaler(), ["duration_days"]),
        ]
    )


def transform_features(
    preprocessor: ColumnTransformer, data: pl.DataFrame
) -> np.ndarray:
    """Apply the fitted transformations without learning from new data."""
    features = preprocessor.transform(select_model_features(data)).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Encoded features contain missing or non-finite values")
    return features


def encode_dataset(
    preprocessor: ColumnTransformer, data: pl.DataFrame
) -> TensorDataset:
    """Convert model inputs and labels to the tensors expected by PyTorch."""
    features = torch.from_numpy(transform_features(preprocessor, data))
    targets = torch.from_numpy(data["target"].to_numpy().astype(np.float32))
    return TensorDataset(features, targets)


def prepare_datasets(
    train_path: str | Path,
    validation_path: str | Path,
    *,
    handle_unknown: str,
) -> tuple[TensorDataset, TensorDataset, ColumnTransformer]:
    """Validate the split, fit preprocessing on train, and encode both datasets."""
    train = read_labeled_data(train_path)
    validation = read_labeled_data(validation_path)
    if train.select("id").join(validation.select("id"), on="id").height:
        raise ValueError("Training and validation project IDs overlap")
    preprocessor = create_preprocessor(handle_unknown)
    preprocessor.fit(select_model_features(train))
    return (
        encode_dataset(preprocessor, train),
        encode_dataset(preprocessor, validation),
        preprocessor,
    )
