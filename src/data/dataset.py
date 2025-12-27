"""Dataset classes for Retinal OCT classification."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .augmentations import get_train_transforms, get_val_transforms


class OCTDataset(Dataset):
    """PyTorch Dataset for Retinal OCT images.

    Expects data organized as:
        data_dir/
            train/
                CNV/
                DME/
                DRUSEN/
                NORMAL/
            test/
                CNV/
                DME/
                DRUSEN/
                NORMAL/
    """

    CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
        val_split: float = 0.2,
        random_state: int = 42,
    ):
        """Initialize dataset.

        Args:
            data_dir: Root directory containing train/ and test/ folders
            split: One of "train", "val", or "test"
            transform: Albumentations transform pipeline
            val_split: Fraction of training data to use for validation
            random_state: Random seed for train/val split
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.val_split = val_split
        self.random_state = random_state

        self.class_to_idx = {name: idx for idx, name in enumerate(self.CLASS_NAMES)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load image paths and labels."""
        samples = []

        if self.split in ["train", "val"]:
            split_dir = self.data_dir / "train"
        else:
            split_dir = self.data_dir / "test"

        if not split_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {split_dir}")

        for class_name in self.CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue

            for img_path in class_dir.glob("*.jpeg"):
                # Skip Mac metadata files that start with ._
                if img_path.name.startswith("._"):
                    continue
                samples.append((str(img_path), self.class_to_idx[class_name]))

            # Also check for other common extensions
            for ext in ["*.jpg", "*.png"]:
                for img_path in class_dir.glob(ext):
                    if img_path.name.startswith("._"):
                        continue
                    samples.append((str(img_path), self.class_to_idx[class_name]))

        # For train/val, split the training data
        if self.split in ["train", "val"]:
            train_samples, val_samples = train_test_split(
                samples,
                test_size=self.val_split,
                random_state=self.random_state,
                stratify=[s[1] for s in samples],
            )
            samples = train_samples if self.split == "train" else val_samples

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        distribution = {name: 0 for name in self.CLASS_NAMES}
        for _, label in self.samples:
            class_name = self.idx_to_class[label]
            distribution[class_name] += 1
        return distribution


def get_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    data_dir = data_config.get("data_dir", "data/OCT2017")
    batch_size = training_config.get("batch_size", 32)
    num_workers = data_config.get("num_workers", 4)
    val_split = 1 - data_config.get("train_split", 0.8)

    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)

    train_dataset = OCTDataset(
        data_dir=data_dir, split="train", transform=train_transform, val_split=val_split
    )

    val_dataset = OCTDataset(
        data_dir=data_dir, split="val", transform=val_transform, val_split=val_split
    )

    test_dataset = OCTDataset(data_dir=data_dir, split="test", transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
