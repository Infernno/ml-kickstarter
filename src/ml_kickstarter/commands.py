import logging
from pathlib import Path

import fire
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from ml_kickstarter.data.download import ensure_data
from ml_kickstarter.data.preprocess import prepare_csv
from ml_kickstarter.data.split import split_csv
from ml_kickstarter.training import baseline


def _load_config(overrides: tuple[str, ...]) -> DictConfig:
    config_dir = str(Path("configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name="config", overrides=list(overrides))


def download(*overrides: str) -> None:
    """Restore the tracked raw dataset from DVC."""
    config = _load_config(overrides)
    ensure_data(config.data.raw_path, config.data.remote)


def preprocess(*overrides: str) -> None:
    """Ensure the raw dataset is available and prepare the CSV."""
    config = _load_config(overrides)
    ensure_data(config.data.raw_path, config.data.remote)
    prepare_csv(config.data.raw_path, config.data.prepared_path)


def split(*overrides: str) -> None:
    """Split the prepared CSV using the configured proportions and seed."""
    config = _load_config(overrides)
    split_csv(
        config.data.prepared_path,
        config.data.split_dir,
        train_size=config.split.train_size,
        validation_size=config.split.validation_size,
        test_size=config.split.test_size,
        seed=config.split.seed,
    )


def train_baseline(*overrides: str) -> dict:
    """Ensure data is available, then train and validate the tabular baseline."""
    config = _load_config(overrides)
    split_dir = Path(config.data.split_dir)
    train_path = split_dir / "train.csv"
    validation_path = split_dir / "validation.csv"
    if not train_path.is_file() or not validation_path.is_file():
        if not Path(config.data.prepared_path).is_file():
            preprocess(*overrides)
        split(*overrides)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return baseline.train_baseline(train_path, validation_path, config)


def main() -> None:
    fire.Fire(
        {
            "download": download,
            "preprocess": preprocess,
            "split": split,
            "train_baseline": train_baseline,
        }
    )


if __name__ == "__main__":
    main()
