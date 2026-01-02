"""PyTorch Lightning module for OCT classification."""

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Precision, Recall

from src.models.classifier import create_model


class OCTLightningModule(pl.LightningModule):
    """Lightning module for training OCT classifier."""

    def __init__(self, config: dict):
        """Initialize the lightning module.

        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # Create model
        self.model = create_model(config)

        # Loss function with class weights for imbalanced data
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        num_classes = config.get("model", {}).get("num_classes", 4)

        # Training metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        # Validation metrics
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_precision = Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_recall = Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        # Test metrics
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_precision = Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_recall = Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_confusion = ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )

        # Store predictions for visualization
        self.test_predictions = []
        self.test_labels = []
        self.test_probs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/acc",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        self.val_accuracy(preds, labels)
        self.val_f1(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        self.test_accuracy(preds, labels)
        self.test_f1(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_confusion(preds, labels)

        # Store for later visualization
        self.test_predictions.extend(preds.cpu().numpy())
        self.test_labels.extend(labels.cpu().numpy())
        self.test_probs.extend(probs.cpu().numpy())

        self.log("test/acc", self.test_accuracy, on_step=False, on_epoch=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        training_config = self.config.get("training", {})
        lr = training_config.get("learning_rate", 1e-4)
        weight_decay = training_config.get("weight_decay", 0.01)
        scheduler_type = training_config.get("scheduler", "cosine")
        epochs = training_config.get("epochs", 20)
        warmup_epochs = training_config.get("warmup_epochs", 2)

        # Use AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Learning rate scheduler
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - warmup_epochs, eta_min=lr * 0.01
            )
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=epochs // 3, gamma=0.1
            )
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
            }
        else:
            return optimizer

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_test_results(self) -> Dict[str, Any]:
        """Get test results for visualization."""
        return {
            "predictions": self.test_predictions,
            "labels": self.test_labels,
            "probabilities": self.test_probs,
            "confusion_matrix": self.test_confusion.compute().cpu().numpy(),
        }
