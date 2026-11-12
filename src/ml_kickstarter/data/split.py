from math import isclose
from pathlib import Path

import polars as pl
from sklearn.model_selection import train_test_split

from ml_kickstarter.constants import TARGET_BY_STATE


def split_data(
    data: pl.DataFrame,
    *,
    train_size: float,
    validation_size: float,
    test_size: float,
    seed: int,
) -> dict[str, pl.DataFrame]:
    """Split campaigns reproducibly while keeping class proportions."""
    sizes = (train_size, validation_size, test_size)
    if not all(0 < size < 1 for size in sizes) or not isclose(sum(sizes), 1):
        raise ValueError("Split proportions must be positive and sum to 1")
    if data["id"].null_count() or data["id"].n_unique() != data.height:
        raise ValueError("Project IDs must be present and unique before splitting")
    if set(data["target"].unique()) != set(TARGET_BY_STATE.values()):
        raise ValueError(
            "Splitting requires both binary classes and no missing targets"
        )

    holdout_size = validation_size + test_size
    train, holdout = train_test_split(
        data,
        test_size=holdout_size,
        stratify=data["target"],
        random_state=seed,
    )
    validation, test = train_test_split(
        holdout,
        test_size=test_size / holdout_size,
        stratify=holdout["target"],
        random_state=seed,
    )
    return {"train": train, "validation": validation, "test": test}


def split_csv(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    train_size: float,
    validation_size: float,
    test_size: float,
    seed: int,
) -> None:
    """Read prepared data and save train, validation and test CSV files."""
    source_path = Path(input_path)
    destination_dir = Path(output_dir)
    data = pl.read_csv(source_path, empty_string_is_null=False)
    splits = split_data(
        data,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        seed=seed,
    )
    output_paths = {name: destination_dir / f"{name}.csv" for name in splits}
    if source_path.resolve() in [path.resolve() for path in output_paths.values()]:
        raise ValueError("Input CSV must not be overwritten by a split")

    destination_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
        subset.write_csv(output_paths[name])
