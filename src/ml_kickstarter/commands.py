import fire
from hydra import compose, initialize
from omegaconf import DictConfig

from ml_kickstarter.data.download import ensure_data
from ml_kickstarter.data.preprocess import prepare_csv
from ml_kickstarter.data.split import split_csv


def _load_config(overrides: tuple[str, ...]) -> DictConfig:
    with initialize(version_base=None, config_path="../../configs"):
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


def main() -> None:
    fire.Fire({"download": download, "preprocess": preprocess, "split": split})


if __name__ == "__main__":
    main()
