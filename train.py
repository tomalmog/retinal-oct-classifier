#!/usr/bin/env python3
"""Training script for Retinal OCT Classification."""

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.data import get_dataloaders
from src.training import OCTLightningModule
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Train Retinal OCT Classifier")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set up paths
    save_dir = Path(config.get("logging", {}).get("save_dir", "outputs"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Print class distribution
    print("\nClass distribution (train):")
    for class_name, count in train_loader.dataset.get_class_distribution().items():
        print(f"  {class_name}: {count}")

    # Create model
    print("\nCreating model...")
    model = OCTLightningModule(config)
    print(f"Backbone: {config.get('model', {}).get('backbone', 'efficientnet_b3')}")

    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=save_dir / "checkpoints",
            filename="oct-{epoch:02d}-{val_acc:.4f}",
            monitor="val/acc",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(monitor="val/acc", patience=5, mode="max"),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]

    # Set up logger
    logging_config = config.get("logging", {})
    if logging_config.get("use_wandb", False):
        logger = WandbLogger(
            project=logging_config.get("project_name", "retinal-oct-classification"),
            save_dir=save_dir,
        )
    else:
        # Use CSVLogger to avoid TensorBoard/TensorFlow conflicts
        from pytorch_lightning.loggers import CSVLogger

        logger = CSVLogger(save_dir=save_dir, name="logs")

    # Training configuration
    training_config = config.get("training", {})

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=training_config.get("epochs", 20),
        accelerator="auto",
        devices=1,
        precision="16-mixed" if training_config.get("mixed_precision", True) else 32,
        gradient_clip_val=training_config.get("gradient_clip", 1.0),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=logging_config.get("log_every_n_steps", 10),
        deterministic=True,
    )

    # Train
    print("\nStarting training...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume,
    )

    # Test
    print("\nRunning test evaluation...")
    trainer.test(model, dataloaders=test_loader)

    print(f"\nTraining complete! Checkpoints saved to: {save_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
