# Kickstarter campaign success prediction

MLOps course project for predicting whether a Kickstarter campaign will reach its funding goal.

## Setup

Requires Python 3.14+ and uv. Run commands from the repository root:

```powershell
uv sync --locked
uv run pre-commit install
```

## Prepare data

Download the source dataset and run preprocessing:

```powershell
uv run ml-kickstarter download
uv run dvc repro
```

Source: `data/raw/dataset.csv`. Result: `data/processed/prepared.csv`.
DVC reruns preprocessing when its inputs or code change.

Preprocessing uses Polars to:

- Keep successful (`target = 1`) and failed (`target = 0`) campaigns.
- Keep names, descriptions, funding goals, currencies, countries, categories and dates.
- Remove outcome-revealing fields, clean missing values and exclude invalid rows.

[features.py](src/ml_kickstarter/data/features.py) adds `duration_days` from launch to deadline.
`select_model_features()` selects model inputs; text, `target`, `id` and source dates are excluded.
The prepared CSV keeps IDs and dates for tracking projects and splitting by time.
Rules are in [preprocess.py](src/ml_kickstarter/data/preprocess.py),
with column lists in [constants.py](src/ml_kickstarter/constants.py).
Data paths are configured in [configs/data/kickstarter.yaml](configs/data/kickstarter.yaml).

## Data versions

DVC stores versions in Cloudflare R2: `data` is the dataset remote, `models` is for models.
Downloading published versions requires no account or access keys.

To download both raw and prepared data without running preprocessing:

```powershell
uv run dvc pull -r data
```

The `data/` directory is ignored by Git. Git tracks `dataset.dvc`, `dvc.yaml` and
`dvc.lock`, which describe the data versions and processing steps.

To publish a finished data version (owner only):

```powershell
uv run dvc push -r data
```

Uploading requires R2 credentials and S3 remote settings in `.dvc/config.local`.
This local file is ignored by Git. Preprocessing does not upload data automatically.

## TODO

- [ ] Use text features
- [ ] Try a time-based train/validation/test split, including class balance and label availability
