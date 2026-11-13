from pathlib import Path

from catboost import CatBoostClassifier


def load_catboost(directory: str | Path) -> CatBoostClassifier:
    """Restore a classifier, including its feature names and categorical processing."""
    model = CatBoostClassifier()
    model.load_model(Path(directory) / "model.cbm")
    return model
