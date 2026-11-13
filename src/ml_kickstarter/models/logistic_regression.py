from pathlib import Path

import joblib
import lightning
import torch
from sklearn.compose import ColumnTransformer
from torch import nn


class LogisticRegression(lightning.LightningModule):
    """A linear classifier with Lightning training and validation steps."""

    def __init__(
        self, input_dim: int, learning_rate: float, weight_decay: float
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.linear = nn.Linear(input_dim, 1)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        features, targets = batch
        loss = self.criterion(self(features), targets)
        if not torch.isfinite(loss):
            raise ValueError("Training loss became non-finite")
        self.log_dict(
            {
                "batch/log_loss": loss,
                "batch/learning_rate": self.optimizers().param_groups[0]["lr"],
            },
            on_step=True,
            on_epoch=False,
        )
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        features, targets = batch
        loss = self.criterion(self(features), targets)
        self.log(
            "validation_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(targets),
            logger=False,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )


def load_baseline(
    directory: str | Path,
) -> tuple[LogisticRegression, ColumnTransformer]:
    """Restore the best Lightning checkpoint and fitted preprocessing."""
    directory = Path(directory)
    model = LogisticRegression.load_from_checkpoint(
        directory / "model.ckpt", map_location="cpu", weights_only=True
    )
    model.eval()
    preprocessor = joblib.load(directory / "preprocessor.joblib")
    return model, preprocessor
