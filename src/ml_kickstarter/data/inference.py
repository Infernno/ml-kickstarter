from pathlib import Path

import polars as pl

from ml_kickstarter.constants import INFERENCE_INPUT_COLUMNS
from ml_kickstarter.data.cleaning import (
    clean_columns,
    fill_missing_values,
    valid_campaigns,
)
from ml_kickstarter.data.features import build_features


def read_inference_data(path: str | Path) -> pl.DataFrame:
    """Prepare unlabeled campaigns without dropping, sorting or deduplicating rows."""
    data = pl.read_csv(path, columns=INFERENCE_INPUT_COLUMNS, infer_schema=False)
    if data.is_empty():
        raise ValueError("Input CSV contains no campaigns")
    data = clean_columns(data)
    valid = data.select(valid_campaigns()).to_series()
    if not valid.all():
        raise ValueError(
            f"Input CSV contains {(~valid).sum()} invalid campaigns: "
            "IDs and goals must be positive, currency is required, "
            "and positive Unix timestamps must satisfy launched_at < deadline"
        )
    if data["id"].is_duplicated().any():
        raise ValueError("Input CSV contains duplicate project IDs")
    return build_features(fill_missing_values(data))
