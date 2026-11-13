import logging
from math import log

import lightning
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import TensorDataset

from ml_kickstarter.models.logistic_regression import LogisticRegression


@torch.inference_mode()
def evaluate(
    model: LogisticRegression,
    features: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    """Measure one fixed model on a complete dataset."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    model.eval()
    logits = model(features.to(model.device))
    if not torch.isfinite(logits).all():
        raise ValueError("Model produced non-finite logits")
    scores = logits.cpu().numpy()
    loss = binary_cross_entropy_with_logits(
        logits.double(), targets.to(device=model.device, dtype=torch.float64)
    )
    labels = targets.cpu().numpy()
    logit_threshold = log(threshold / (1 - threshold))
    predictions = scores >= logit_threshold
    return {
        "log_loss": loss.item(),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


class TrainingHistory(lightning.Callback):
    """Record train and validation metrics using the weights at each epoch's end."""

    def __init__(
        self,
        train_data: TensorDataset,
        validation_data: TensorDataset,
        threshold: float,
    ) -> None:
        self.train_data = train_data
        self.validation_data = validation_data
        self.threshold = threshold
        self.history: list[dict] = []

    def on_validation_end(
        self, trainer: lightning.Trainer, pl_module: LogisticRegression
    ) -> None:
        train_metrics = evaluate(pl_module, *self.train_data.tensors, self.threshold)
        validation_metrics = evaluate(
            pl_module, *self.validation_data.tensors, self.threshold
        )
        epoch = trainer.current_epoch + 1
        self.history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        )
        trainer.logger.log_metrics(
            {
                f"{split}/{name}": value
                for split, metrics in (
                    ("train", train_metrics),
                    ("validation", validation_metrics),
                )
                for name, value in metrics.items()
            },
            step=epoch,
        )
        logging.getLogger(__name__).info(
            "Epoch %d/%d: train Logloss=%.4f, validation Logloss=%.4f, ROC-AUC=%.4f",
            epoch,
            trainer.max_epochs,
            train_metrics["log_loss"],
            validation_metrics["log_loss"],
            validation_metrics["roc_auc"],
        )
