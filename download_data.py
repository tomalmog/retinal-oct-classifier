#!/usr/bin/env python3
"""Script to download the OCT dataset from Kaggle."""

import os
import shutil
import sys
import zipfile
from pathlib import Path


def download_dataset(data_dir: str = "data"):
    """Download the Kermany2018 OCT dataset from Kaggle.

    Prerequisites:
        1. Install kaggle: pip install kaggle
        2. Set up Kaggle API credentials:
           - Go to kaggle.com -> Account -> Create New API Token
           - Save kaggle.json to ~/.kaggle/kaggle.json
           - chmod 600 ~/.kaggle/kaggle.json

    Args:
        data_dir: Directory to download data to
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    oct_dir = data_dir / "OCT2017"

    if oct_dir.exists() and (oct_dir / "train").exists():
        print(f"Dataset already exists at {oct_dir}")
        print("Skipping download.")
        return oct_dir

    print("Downloading OCT2017 dataset from Kaggle...")
    print("This may take a few minutes (~5GB)...")

    try:
        import kaggle

        kaggle.api.authenticate()
    except Exception as e:
        print(f"\nError: Could not authenticate with Kaggle API.")
        print(f"Details: {e}")
        print("\nPlease set up Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token' to download kaggle.json")
        print("3. Place it in ~/.kaggle/kaggle.json")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)

    # Download dataset
    try:
        kaggle.api.dataset_download_files(
            "paultimothymooney/kermany2018", path=str(data_dir), unzip=False
        )
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        sys.exit(1)

    # Extract
    zip_path = data_dir / "kermany2018.zip"
    if zip_path.exists():
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

        # Clean up zip file
        zip_path.unlink()
        print("Removed zip file.")

    # The dataset structure is:
    # data/OCT2017/train/CNV, DME, DRUSEN, NORMAL
    # data/OCT2017/test/CNV, DME, DRUSEN, NORMAL

    # Sometimes it extracts to a nested folder, fix if needed
    potential_paths = [
        data_dir / "OCT2017",
        data_dir / "kermany2018" / "OCT2017",
        data_dir / "oct2017" / "OCT2017",
    ]

    for path in potential_paths:
        if path.exists() and (path / "train").exists():
            if path != oct_dir:
                shutil.move(str(path), str(oct_dir))
            break

    if oct_dir.exists():
        print(f"\nDataset downloaded and extracted to: {oct_dir}")

        # Print dataset statistics
        train_dir = oct_dir / "train"
        test_dir = oct_dir / "test"

        print("\nDataset Statistics:")
        for split, split_dir in [("Train", train_dir), ("Test", test_dir)]:
            if split_dir.exists():
                total = 0
                print(f"\n{split}:")
                for class_dir in sorted(split_dir.iterdir()):
                    if class_dir.is_dir():
                        count = len(list(class_dir.glob("*.jpeg")))
                        print(f"  {class_dir.name}: {count:,} images")
                        total += count
                print(f"  Total: {total:,} images")
    else:
        print("\nError: Could not find extracted dataset.")
        print("Please check the download and try again.")
        sys.exit(1)

    return oct_dir


def create_sample_dataset(data_dir: str = "data", samples_per_class: int = 100):
    """Create a smaller sample dataset for quick testing.

    Args:
        data_dir: Data directory containing full dataset
        samples_per_class: Number of samples per class for the sample dataset
    """
    data_dir = Path(data_dir)
    full_dir = data_dir / "OCT2017"
    sample_dir = data_dir / "OCT2017_sample"

    if not full_dir.exists():
        print("Full dataset not found. Please download it first.")
        return

    if sample_dir.exists():
        print(f"Sample dataset already exists at {sample_dir}")
        return sample_dir

    print(f"Creating sample dataset with {samples_per_class} images per class...")

    import random

    for split in ["train", "test"]:
        split_src = full_dir / split
        split_dst = sample_dir / split

        for class_name in ["CNV", "DME", "DRUSEN", "NORMAL"]:
            class_src = split_src / class_name
            class_dst = split_dst / class_name
            class_dst.mkdir(parents=True, exist_ok=True)

            images = list(class_src.glob("*.jpeg"))
            sample_count = min(samples_per_class, len(images))
            sampled = random.sample(images, sample_count)

            for img_path in sampled:
                shutil.copy(img_path, class_dst / img_path.name)

            print(f"  {split}/{class_name}: {sample_count} images")

    print(f"\nSample dataset created at: {sample_dir}")
    return sample_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download OCT Dataset")
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Directory to download data to"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Also create a small sample dataset for testing",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=100,
        help="Number of samples per class for sample dataset",
    )
    args = parser.parse_args()

    download_dataset(args.data_dir)

    if args.sample:
        create_sample_dataset(args.data_dir, args.samples_per_class)
