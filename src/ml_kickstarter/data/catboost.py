from pathlib import Path

from catboost import Pool

from ml_kickstarter.constants import CATEGORY_COLUMNS
from ml_kickstarter.data.encoding import read_labeled_data
from ml_kickstarter.data.features import select_model_features


def prepare_pools(
    train_path: str | Path, validation_path: str | Path
) -> tuple[Pool, Pool]:
    """Keep native categorical columns and validate the same split as the baseline."""
    train = read_labeled_data(train_path)
    validation = read_labeled_data(validation_path)
    if train.select("id").join(validation.select("id"), on="id").height:
        raise ValueError("Training and validation project IDs overlap")
    return (
        Pool(
            select_model_features(train),
            label=train["target"].to_numpy(),
            cat_features=CATEGORY_COLUMNS,
        ),
        Pool(
            select_model_features(validation),
            label=validation["target"].to_numpy(),
            cat_features=CATEGORY_COLUMNS,
        ),
    )
