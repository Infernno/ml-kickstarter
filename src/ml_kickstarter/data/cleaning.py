import polars as pl

from ml_kickstarter.constants import (
    CATEGORY_COLUMNS,
    DATE_COLUMNS,
    MISSING_CATEGORY,
    TEXT_COLUMNS,
)


def clean_columns(data: pl.DataFrame) -> pl.DataFrame:
    """Trim whitespace, normalize blanks and parse numbers and Unix timestamps."""
    date_columns = [column for column in DATE_COLUMNS if column in data.columns]
    return data.with_columns(pl.all().str.strip_chars().replace("", None)).with_columns(
        pl.col(["id", *date_columns]).cast(pl.Int64, strict=False),
        pl.col("goal").cast(pl.Float64, strict=False),
    )


def valid_campaigns() -> pl.Expr:
    """Build the shared validity mask for campaign inputs; nulls are invalid."""
    return pl.all_horizontal(
        pl.col("id") > 0,
        pl.col("goal").is_finite() & (pl.col("goal") > 0),
        pl.col("currency").is_not_null(),
        pl.col("launched_at") > 0,
        pl.col("launched_at") < pl.col("deadline"),
    ).fill_null(False)


def fill_missing_values(data: pl.DataFrame) -> pl.DataFrame:
    """Keep campaigns with missing text or categories without fitting statistics."""
    text_columns = [column for column in TEXT_COLUMNS if column in data.columns]
    return data.with_columns(
        pl.col(text_columns).fill_null(""),
        pl.col(CATEGORY_COLUMNS).fill_null(MISSING_CATEGORY),
    )
