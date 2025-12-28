"""Data augmentation pipelines using albumentations."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(config: dict) -> A.Compose:
    """Get training augmentation pipeline.

    Args:
        config: Configuration dictionary with augmentation settings

    Returns:
        Albumentations compose object for training
    """
    aug_config = config.get("augmentation", {}).get("train", {})
    image_size = config.get("data", {}).get("image_size", 224)

    transforms = [
        A.Resize(image_size, image_size),
    ]

    if aug_config.get("horizontal_flip", True):
        transforms.append(A.HorizontalFlip(p=0.5))

    if aug_config.get("vertical_flip", False):
        transforms.append(A.VerticalFlip(p=0.5))

    rotation = aug_config.get("rotation", 15)
    if rotation > 0:
        transforms.append(A.Rotate(limit=rotation, p=0.5))

    brightness = aug_config.get("brightness", 0.2)
    contrast = aug_config.get("contrast", 0.2)
    if brightness > 0 or contrast > 0:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=brightness, contrast_limit=contrast, p=0.5
            )
        )

    # Additional augmentations for medical imaging
    transforms.extend(
        [
            A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms)


def get_val_transforms(config: dict) -> A.Compose:
    """Get validation/test augmentation pipeline (no augmentation, only resize and normalize).

    Args:
        config: Configuration dictionary

    Returns:
        Albumentations compose object for validation
    """
    image_size = config.get("data", {}).get("image_size", 224)

    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
