import polars as pl

from ml_kickstarter.constants import MODEL_FEATURE_COLUMNS, SECONDS_PER_DAY


def build_features(data: pl.DataFrame) -> pl.DataFrame:
    """Add duration from cleaned Unix timestamps; preserve labels and metadata."""
    return data.with_columns(
        ((pl.col("deadline") - pl.col("launched_at")) / SECONDS_PER_DAY).alias(
            "duration_days"
        )
    )


def select_model_features(data: pl.DataFrame) -> pl.DataFrame:
    """Select model inputs, excluding target, IDs and source dates."""
    return data.select(MODEL_FEATURE_COLUMNS)
