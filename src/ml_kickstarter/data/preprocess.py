from pathlib import Path

import polars as pl

from ml_kickstarter.constants import INPUT_COLUMNS, TARGET_BY_STATE
from ml_kickstarter.data.cleaning import (
    clean_columns,
    fill_missing_values,
    valid_campaigns,
)
from ml_kickstarter.data.features import build_features


def filter_campaigns(data: pl.DataFrame) -> pl.DataFrame:
    """Keep labeled campaigns with usable goals, IDs, currencies and dates."""
    return data.filter(
        valid_campaigns(),
        pl.col("state").is_in(list(TARGET_BY_STATE)),
        pl.col("created_at") > 0,
        pl.col("created_at") <= pl.col("launched_at"),
    )


def add_target(data: pl.DataFrame) -> pl.DataFrame:
    """Replace the outcome status with a binary target."""
    return data.with_columns(
        pl.col("state")
        .replace_strict(TARGET_BY_STATE, return_dtype=pl.Int8)
        .alias("target")
    ).drop("state")


def prepare_csv(input_path: str | Path, output_path: str | Path) -> None:
    """Prepare campaign features and labels; keep ID only as project metadata."""
    source_path = Path(input_path)
    destination_path = Path(output_path)
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("Input and output CSV paths must be different")

    data = pl.read_csv(source_path, columns=INPUT_COLUMNS, infer_schema=False).select(
        INPUT_COLUMNS
    )
    data = clean_columns(data)
    data = filter_campaigns(data)
    data = fill_missing_values(data)
    data = add_target(data)
    data = data.unique(maintain_order=True)

    if data.is_empty():
        raise ValueError("No valid successful or failed campaigns remain")
    if data["id"].is_duplicated().any():
        raise ValueError("Conflicting rows found for the same project ID")

    data = build_features(data)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    data.write_csv(destination_path)
