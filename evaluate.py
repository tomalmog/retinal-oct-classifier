#!/usr/bin/env python3
"""Evaluation script with visualizations for Retinal OCT Classification."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import get_dataloaders, get_val_transforms
from src.data.dataset import OCTDataset
from src.training import OCTLightningModule
from src.utils import load_config
from src.visualization import (
    GradCAMVisualizer,
    generate_classification_report,
    plot_confusion_matrix,
    plot_roc_curves,
)
from src.visualization.metrics import plot_class_distribution


def evaluate_model(checkpoint_path: str, config_path: str, output_dir: str):
    """Run full evaluation with visualizations.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to configuration file
        output_dir: Directory to save outputs
    """
    # Setup
    config = load_config(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = config.get("class_names", ["CNV", "DME", "DRUSEN", "NORMAL"])

    # Load model
    print("Loading model from checkpoint...")
    model = OCTLightningModule.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Get test data
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(config)

    # Collect predictions
    print("Running inference on test set...")
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Calculate accuracy
    accuracy = (all_preds == all_labels).mean()
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Confusion Matrix
    from sklearn.metrics import confusion_matrix as sklearn_cm

    cm = sklearn_cm(all_labels, all_preds)
    plot_confusion_matrix(
        cm, class_names, save_path=output_dir / "confusion_matrix.png"
    )
    plt.close()

    # 2. ROC Curves
    plot_roc_curves(
        all_labels, all_probs, class_names, save_path=output_dir / "roc_curves.png"
    )
    plt.close()

    # 3. Class Distribution
    data_config = config.get("data", {})
    test_dataset = OCTDataset(
        data_dir=data_config.get("data_dir", "data/OCT2017"),
        split="test",
        transform=get_val_transforms(config),
    )
    plot_class_distribution(
        test_dataset.get_class_distribution(),
        save_path=output_dir / "class_distribution.png",
    )
    plt.close()

    # 4. Classification Report
    report = generate_classification_report(
        all_labels,
        all_preds,
        class_names,
        save_path=output_dir / "classification_report.txt",
    )
    print("\nClassification Report:")
    print(report)

    # 5. Grad-CAM visualizations (sample images)
    print("\nGenerating Grad-CAM visualizations...")
    try:
        gradcam = GradCAMVisualizer(model)

        # Get sample images from each class
        sample_images = []
        sample_tensors = []
        sample_labels = []

        for class_idx, class_name in enumerate(class_names):
            # Find an image of this class
            for i, (img_path, label) in enumerate(test_dataset.samples):
                if label == class_idx:
                    # Load original image
                    from PIL import Image

                    orig_img = np.array(Image.open(img_path).convert("RGB"))

                    # Get transformed tensor
                    tensor, _ = test_dataset[i]

                    sample_images.append(orig_img)
                    sample_tensors.append(tensor)
                    sample_labels.append(class_name)
                    break

        if sample_images:
            gradcam.plot_gradcam_grid(
                sample_images,
                torch.stack(sample_tensors).to(device),
                class_names,
                save_path=output_dir / "gradcam_samples.png",
            )
            plt.close()

        gradcam.remove_hooks()
    except Exception as e:
        print(f"Warning: Could not generate Grad-CAM: {e}")

    print(f"\nAll visualizations saved to: {output_dir}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCT Classifier")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to save evaluation outputs",
    )
    args = parser.parse_args()

    evaluate_model(args.checkpoint, args.config, args.output_dir)


if __name__ == "__main__":
    main()
