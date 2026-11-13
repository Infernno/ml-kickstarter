TARGET_BY_STATE = {"successful": 1, "failed": 0}

TEXT_COLUMNS = ["name", "blurb"]
CATEGORY_COLUMNS = ["currency", "country", "category_parent_name", "category_name"]
DATE_COLUMNS = ["created_at", "launched_at", "deadline"]
INPUT_COLUMNS = [
    "id",
    *TEXT_COLUMNS,
    "goal",
    *CATEGORY_COLUMNS,
    *DATE_COLUMNS,
    "state",
]
INFERENCE_INPUT_COLUMNS = [
    "id",
    "goal",
    *CATEGORY_COLUMNS,
    "launched_at",
    "deadline",
]
MISSING_CATEGORY = "unknown"

SECONDS_PER_DAY = 24 * 60 * 60
PLOT_SIZE = (7, 4)
PLOT_DPI = 150
PLOT_GRID_ALPHA = 0.25

MODEL_FEATURE_COLUMNS = [
    "goal",
    *CATEGORY_COLUMNS,
    "duration_days",
]
