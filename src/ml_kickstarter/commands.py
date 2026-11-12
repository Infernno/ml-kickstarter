import fire
from hydra import compose, initialize
from omegaconf import DictConfig

from ml_kickstarter.data.download import ensure_data
from ml_kickstarter.data.preprocess import prepare_csv


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


def main() -> None:
    fire.Fire({"download": download, "preprocess": preprocess})


if __name__ == "__main__":
    main()
