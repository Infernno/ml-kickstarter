from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

PLOT_METRICS = {"log_loss": "Logloss", "roc_auc": "ROC-AUC", "f1": "F1"}


def save_training_plots(
    history: list[dict], directory: str | Path, *, step_name: str = "epoch"
) -> list[Path]:
    """Save train/validation curves without opening graphical windows."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    steps = [record[step_name] for record in history]
    paths = []
    for metric, label in PLOT_METRICS.items():
        figure = Figure(figsize=(7, 4), layout="constrained")
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        for split in ("train", "validation"):
            values = [record[split][metric] for record in history]
            axis.plot(steps, values, label=split)
        axis.set(
            xlabel=step_name.title(), ylabel=label, title=f"{label} during training"
        )
        axis.legend()
        axis.grid(alpha=0.25)
        path = directory / f"{metric}.png"
        figure.savefig(path, dpi=150)
        paths.append(path)
    return paths
