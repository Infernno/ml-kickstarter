# Kickstarter campaign success prediction

Predict if a Kickstarter campaign will reach its funding goal. This can help creators
estimate their chances at launch. Output: a probability of success and a class:
`1` for successful, `0` for failed.

## Data

Source: [Kickstarter dataset on Kaggle](https://www.kaggle.com/datasets/domingosun/kickstarter-dataset-629147-projects-may-2025/data).
This dataset version has **694,062 campaigns**, 43 columns and about 455 MB of CSV data.

- **Target:** successful or failed campaigns only. Cleaning leaves **604,630** rows.
- **Features:** funding goal in the campaign's currency, currency, country, main category,
  subcategory and campaign duration in days.
- **Preprocessing:** Polars cleans data, removes invalid rows and duplicates, and calculates
  duration from launch to deadline. Fields that reveal the result, such as money raised and
  the number of backers, are left out. Text features will be added later.
- **Split:** random 70% train / 15% validation / 15% test, keeping the class balance, with seed `42`.

Some details may have changed after launch. We plan a split by launch date to check
performance on future campaigns.

## Models and results

Both models use the same features and data split. Logistic regression uses PyTorch Lightning,
one-hot encoding for categories, `log1p(goal)` and scaled duration. CatBoost handles categories directly
and uses numeric features without scaling.

**Validation results:** 90,694 campaigns, classification threshold `0.5`.

| Model               | Accuracy ↑ |  ROC-AUC ↑ |       F1 ↑ |  Logloss ↓ |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| Logistic regression |     68.89% |     0.7589 |     0.6810 |     0.5799 |
| CatBoost            | **70.79%** | **0.7844** | **0.7028** | **0.5541** |

ROC-AUC shows how well the model ranks successful campaigns above failed ones.
Our first goal is **0.80**. Logloss measures probability errors; the model with the
lowest validation Logloss is saved. F1 combines precision and recall; accuracy is
the share of correct predictions.

## Setup

You need **Python 3.14+** and **uv**. Run commands from the main project folder.
Both models run on CPU.

```powershell
uv sync --locked
uv run pre-commit install
uv run pre-commit run --all-files
```

## Train

Start MLflow in a separate terminal:

```powershell
uv run mlflow server --host 127.0.0.1 --port 8080 --workers 1 --backend-store-uri sqlite:///mlflow.db
```

Train one of the models. Missing data is downloaded and prepared automatically:

```powershell
uv run ml-kickstarter train_baseline
uv run ml-kickstarter train_catboost
```

Models and reports are saved in `models/baseline/` and `models/catboost/`.
Logloss, ROC-AUC and F1 plots go to `plots/<model>/`.
[MLflow](http://127.0.0.1:8080) saves metrics, hyperparameters and the Git commit.

Settings are in [configs/](configs): data, split, encoding, training,
logging and inference. For example, `uv run ml-kickstarter train_baseline train.epochs=10`
sets training to 10 epochs.

## Predict

CatBoost is the default model. You can also choose the baseline.
DVC downloads missing data and model files. You do not need to start MLflow.

```powershell
uv run ml-kickstarter infer
uv run ml-kickstarter infer infer.model=baseline
```

Default input: `data/splits/validation.csv`. Output: `data/predictions/<model>.csv`,
with columns `id`, `success_probability` and `prediction`. Rows keep their input order.

For your own campaigns:

```powershell
uv run ml-kickstarter infer infer.input_path=campaigns.csv infer.output_path=predictions.csv
```

Input CSV example (all eight columns are needed; dates are Unix seconds):

```csv
id,goal,currency,country,category_parent_name,category_name,launched_at,deadline
1,5000,USD,US,Games,Tabletop Games,1761955200,1764547200
```

Use unique IDs greater than zero, positive finite goals, a currency and launch dates
before deadlines. Labels are optional. Invalid rows cause an error. The saved
preprocessing is reused without fitting again.

## Data pipeline and storage

DVC stores versions in two public Cloudflare R2 remotes: `data` and `models`.
Downloads need no account or access keys. Git stores version details.
Data and model files stay out of Git.

To download raw data and rebuild the prepared CSV and splits:

```powershell
uv run ml-kickstarter download
uv run dvc repro split
```

Data folders: `data/raw/` → `data/processed/` → `data/splits/`.
The `preprocess` and `split` commands can also be run separately with
`uv run ml-kickstarter <command>`.
Run all stages with `uv run dvc repro`. Keep MLflow running.

<details>
<summary>Upload data and models (owner only)</summary>

Set up S3 access in `.dvc/config.local` (ignored by Git), then run:

```powershell
uv run dvc push -r data
uv run dvc push -r models
```

Training and preprocessing do not upload files automatically.

</details>

## TODO

- [ ] Use text features to improve metrics
- [ ] Add GitHub Actions
- [ ] Add a Telegram bot
- [ ] Add tests
- [ ] Try time based split
